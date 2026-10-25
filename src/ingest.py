import logging
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
LOG_PATH = PROJECT_ROOT / "logs" / "ingest.log"

logger = logging.getLogger(__name__)

# Raw names the general snake_case rule gets wrong
COLUMN_RENAMES = {"CustomerID": "customer_id"}


def setup_logging(log_path: Path) -> None:
    """Log to both a file and the terminal."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def load_config(config_path: Path) -> dict:
    """Read the YAML config file into a dict."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_raw(raw_path: Path) -> pd.DataFrame:
    """Read the raw Telco Excel file into a DataFrame."""
    if not raw_path.is_file():
        logger.error("Raw data not found: %s", raw_path)
        raise FileNotFoundError(f"Raw data not found: {raw_path}")
    if raw_path.stat().st_size == 0:
        logger.error("Raw data file is empty: %s", raw_path)
        raise ValueError(f"Raw data file is empty: {raw_path}")

    logger.info("Loading raw file: %s", raw_path)
    df = pd.read_excel(raw_path)
    logger.info("Shape of the raw data: %s", df.shape)
    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to snake_case, e.g. 'Tenure Months' -> 'tenure_months'."""
    df = df.rename(columns=COLUMN_RENAMES)
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9_]+", "_", regex=True)
        .str.strip("_")
    )
    logger.info("Cleaned column names: %s", list(df.columns))
    return df

def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Convert total_charges to numeric and fill blanks with 0.

    Blanks belong to new customers (tenure 0) who have not been billed yet,
    so their true total is 0.
    """
    df = df.copy()
    df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")

    missing = df["total_charges"].isna()
    n_missing = int(missing.sum())

    # Our fill rule is only valid for new customers; stop if that ever changes
    if not (df.loc[missing, "tenure_months"] == 0).all():
        raise ValueError(
            "total_charges is blank for customers with tenure > 0; "
            "filling with 0 would be wrong"
        )

    df["total_charges"] = df["total_charges"].fillna(0)
    logger.info(
        "total_charges: %d blank values, all with tenure 0 -> filled with 0",
        n_missing,
    )
    return df

def drop_columns(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Drop constant, leakage and geo columns listed in the config."""
    cols_to_drop = []
    for reason, cols in cfg["drop_cols"].items():
        logger.info("Dropping %s columns: %s", reason, cols)
        cols_to_drop.extend(cols)

    df = df.drop(columns=cols_to_drop)
    logger.info("Shape after dropping columns: %s", df.shape)
    return df

def validate(df: pd.DataFrame, cfg: dict) -> None:
    """Fail fast if the cleaned data breaks any basic rule."""
    target = cfg["target_col"]
    id_col = cfg["id_cols"][0]
    errors = []

    if len(df) == 0:
        errors.append("dataset has no rows")

    leaked = [c for c in cfg["drop_cols"]["leakage"] if c in df.columns]
    if leaked:
        errors.append(f"leakage columns still present: {leaked}")

    n_missing = int(df.isna().sum().sum())
    if n_missing > 0:
        missing_cols = df.columns[df.isna().any()].tolist()
        errors.append(f"{n_missing} missing values in columns: {missing_cols}")

    n_dup_ids = int(df[id_col].duplicated().sum())
    if n_dup_ids > 0:
        errors.append(f"{n_dup_ids} duplicate values in {id_col}")

    if target not in df.columns:
        errors.append(f"target column '{target}' is missing")
    elif not set(df[target].unique()) <= {0, 1}:
        errors.append(f"target has values other than 0/1: {sorted(df[target].unique().tolist())}")

    if not pd.api.types.is_float_dtype(df["total_charges"]):
        errors.append(f"total_charges is {df['total_charges'].dtype}, expected float")
    elif (df["total_charges"] < 0).any():
        errors.append("total_charges has negative values")

    if errors:
        for e in errors:
            logger.error("Validation failed: %s", e)
        raise ValueError(f"Data validation failed with {len(errors)} error(s): {errors}")

    logger.info(
        "Validation passed: %d rows, %d columns, churn rate %.3f",
        len(df), df.shape[1], df[target].mean(),
    )

def save(df: pd.DataFrame, out_path: Path) -> None:
    """Save the cleaned data to Parquet and check it reads back the same."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    reloaded = pd.read_parquet(out_path)
    if reloaded.shape != df.shape:
        raise ValueError(f"Saved shape {reloaded.shape} != original {df.shape}")
    logger.info("Saved %s rows to %s", len(reloaded), out_path)


def main() -> None:
    setup_logging(LOG_PATH)
    cfg = load_config(CONFIG_PATH)
    raw_path = PROJECT_ROOT / cfg["paths"]["raw_data"]
    out_path = PROJECT_ROOT / cfg["paths"]["processed_data"]

    df = load_raw(raw_path)
    df = clean_column_names(df)
    df = fix_total_charges(df)
    df = drop_columns(df, cfg)
    validate(df, cfg)
    save(df, out_path)


if __name__ == "__main__":
    main()
