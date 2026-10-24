import logging
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
LOG_PATH = PROJECT_ROOT / "logs" / "ingest.log"

logger = logging.getLogger(__name__)


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

def main() -> None:
    setup_logging(LOG_PATH)
    cfg = load_config(CONFIG_PATH)
    raw_path = PROJECT_ROOT / cfg["paths"]["raw_data"]
    df = load_raw(raw_path)

if __name__ == "__main__":
    main()
