"""Batch scoring: score current customers and write a campaign list for the retention team.

Usage:
    python -m src.score                              # scores the default input from config
    python -m src.score --input path/to/customers.parquet
"""
import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.ingest import PROJECT_ROOT, load_config, setup_logging
from src.policy import choose_actions, customer_value, expected_profit_of
from src.train import get_feature_lists

LOG_PATH = PROJECT_ROOT / "logs" / "score.log"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- inputs

def load_customers(path: Path) -> pd.DataFrame:
    """Read customers to score (cleaned format, same columns as ingest output)."""
    if not path.is_file():
        raise FileNotFoundError(f"Scoring input not found: {path}")
    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    logger.info("Loaded %d customers from %s", len(df), path)
    return df


def validate_input(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Fail fast on input the model cannot score; never use the label if present."""
    numeric, categorical = get_feature_lists(cfg)
    id_col = cfg["id_cols"][0]
    target = cfg["target_col"]

    missing = [c for c in [id_col] + numeric + categorical if c not in df.columns]
    if missing:
        raise ValueError(f"Scoring input is missing columns: {missing}")

    if target in df.columns:
        logger.warning("Input contains '%s'; it is dropped and never used for scoring", target)
        df = df.drop(columns=[target])

    errors = []
    if df.empty:
        errors.append("no customers to score")
    if df[id_col].duplicated().any():
        errors.append(f"{int(df[id_col].duplicated().sum())} duplicate {id_col} values")
    n_missing = int(df[numeric + categorical].isna().sum().sum())
    if n_missing:
        errors.append(f"{n_missing} missing values in model inputs")
    if errors:
        raise ValueError(f"Scoring input failed validation: {errors}")
    return df


# ---------------------------------------------------------------- explanations

def risk_reasons(model, X: pd.DataFrame, n: int = 3) -> list[str]:
    """Top features pushing each customer's churn risk UP (for the retention team).

    contribution = coefficient x transformed feature value, averaged over the
    calibration folds. Only positive contributions are reasons for risk.
    """
    pipes = [c.estimator for c in model.calibrated_classifiers_]
    contributions = np.mean(
        [pipe[:-1].transform(X) * pipe[-1].coef_[0] for pipe in pipes], axis=0
    )
    names = [readable(name) for name in pipes[0][:-1].get_feature_names_out()]
    scaled = pipes[0][:-1].transform(X)

    reasons = []
    for row, values in zip(contributions, scaled):
        top = [i for i in np.argsort(-row)[:n] if row[i] > 0]
        reasons.append("; ".join(label(names[i], values[i]) for i in top))
    return reasons


def readable(feature_name: str) -> str:
    """'cat__contract_Month-to-month' -> 'contract = Month-to-month'.

    Category values never contain '_', so the last '_' separates column and value.
    """
    kind, _, name = feature_name.partition("__")
    if kind != "cat":
        return name
    column, _, value = name.rpartition("_")
    return f"{column} = {value}"


def label(name: str, scaled_value: float) -> str:
    """Add 'low'/'high' to numeric features (scaled value < 0 means below average)."""
    if "=" in name:
        return name
    return f"{name} ({'low' if scaled_value < 0 else 'high'})"


# ---------------------------------------------------------------- outputs

def model_fingerprint(path: Path) -> str:
    """Short hash of the model file, so every score can be traced to a model."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def build_campaign(df, p, model, cfg) -> pd.DataFrame:
    """One row per customer: probability, value, action, expected profit, reasons."""
    biz = cfg["business"]
    numeric, categorical = get_feature_lists(cfg)
    value = customer_value(df["monthly_charges"], biz)
    actions = choose_actions(p, value, biz)

    campaign = pd.DataFrame({
        cfg["id_cols"][0]: df[cfg["id_cols"][0]].to_numpy(),
        "churn_probability": np.round(p, 4),
        "customer_value": np.round(value, 2),
        "action": actions,
        "expected_profit": np.round(expected_profit_of(actions, p, value, biz), 2),
        "risk_reasons": risk_reasons(model, df[numeric + categorical]),
    })
    order = {"call": 0, "email": 1, "none": 2}
    return (campaign.sort_values(["action", "expected_profit"],
                                 key=lambda s: s.map(order) if s.name == "action" else -s)
                    .reset_index(drop=True))


def summarise(campaign: pd.DataFrame, p: np.ndarray, meta: dict) -> dict:
    """Run summary: what was scored, with which model, and what was decided."""
    counts = campaign["action"].value_counts()
    return {
        **meta,
        "customers_scored": len(campaign),
        "mean_churn_probability": round(float(p.mean()), 4),
        "probability_quantiles": {q: round(float(np.quantile(p, q)), 4)
                                  for q in [0.1, 0.25, 0.5, 0.75, 0.9]},
        "actions": {a: int(counts.get(a, 0)) for a in ["call", "email", "none"]},
        "total_expected_profit": round(float(campaign["expected_profit"].sum()), 2),
    }


# ---------------------------------------------------------------- main

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score customers and build a campaign list.")
    parser.add_argument("--input", type=Path, help="parquet or csv file of customers to score")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(LOG_PATH)
    cfg = load_config(PROJECT_ROOT / "config" / "config.yaml")

    input_path = args.input or PROJECT_ROOT / cfg["paths"]["scoring_input"]
    model_path = PROJECT_ROOT / cfg["paths"]["model"]
    model = joblib.load(model_path)

    df = validate_input(load_customers(input_path), cfg)
    p = model.predict_proba(df)[:, 1]
    campaign = build_campaign(df, p, model, cfg)

    scored_at = datetime.now(timezone.utc)
    meta = {
        "scored_at": scored_at.isoformat(timespec="seconds"),
        "input": str(input_path),
        "model_fingerprint": model_fingerprint(model_path),
    }
    summary = summarise(campaign, p, meta)

    out_dir = PROJECT_ROOT / cfg["paths"]["scored_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = scored_at.strftime("%Y%m%d_%H%M%S")
    campaign.to_csv(out_dir / f"campaign_{stamp}.csv", index=False)
    (out_dir / f"summary_{stamp}.json").write_text(json.dumps(summary, indent=2))

    logger.info("Actions: %s | total expected profit $%.0f",
                summary["actions"], summary["total_expected_profit"])
    logger.info("Top 5 customers to call:\n%s",
                campaign.head(5).to_string(index=False))
    logger.info("Saved campaign_%s.csv and summary_%s.json to %s", stamp, stamp, out_dir)


if __name__ == "__main__":
    main()
