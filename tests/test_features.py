"""Unit tests for src/features.py, using small hand-made customers."""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features import ADDON_COLS, ENGINEERED_FEATURES, add_features

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def make_customers() -> pd.DataFrame:
    """4 customers covering the edge cases.

    0: brand new (tenure 0), no internet, no family, electronic check
    1: tenure 6 (boundary, still new), partner only, all 6 add-ons, bank transfer
    2: tenure 7 (no longer new), dependents only, 2 add-ons, credit card
    3: tenure 24, no family, no add-ons but has internet, mailed check
    """
    df = pd.DataFrame({
        "tenure_months": [0, 6, 7, 24],
        "monthly_charges": [50.0, 60.0, 80.0, 100.0],
        "total_charges": [0.0, 360.0, 490.0, 1800.0],
        "partner": ["No", "Yes", "No", "No"],
        "dependents": ["No", "No", "Yes", "No"],
        "payment_method": [
            "Electronic check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Mailed check",
        ],
    })
    df[ADDON_COLS] = "No"
    df.loc[0, ADDON_COLS] = "No internet service"
    df.loc[1, ADDON_COLS] = "Yes"
    df.loc[2, ["online_security", "tech_support"]] = "Yes"
    return df


def test_adds_all_engineered_features():
    out = add_features(make_customers())
    assert set(ENGINEERED_FEATURES) <= set(out.columns)


def test_is_new_customer_boundary():
    out = add_features(make_customers())
    # tenure 0 and 6 are new, 7 and 24 are not
    assert out["is_new_customer"].tolist() == [1, 1, 0, 0]


def test_num_addon_services():
    out = add_features(make_customers())
    # "No internet service" must count as 0, not as a service
    assert out["num_addon_services"].tolist() == [0, 6, 2, 0]


def test_has_auto_payment():
    out = add_features(make_customers())
    assert out["has_auto_payment"].tolist() == [0, 1, 1, 0]


def test_has_family():
    out = add_features(make_customers())
    # partner only -> 1, dependents only -> 1, neither -> 0
    assert out["has_family"].tolist() == [0, 1, 1, 0]


def test_avg_monthly_spend_tenure_zero():
    out = add_features(make_customers())
    # tenure 0 falls back to monthly_charges instead of dividing by zero
    assert out.loc[0, "avg_monthly_spend"] == 50.0
    assert np.isfinite(out["avg_monthly_spend"]).all()


def test_avg_monthly_spend_and_charge_increase():
    out = add_features(make_customers())
    # customer 2: 490 / 7 = 70 per month on average, now pays 80 -> +10
    assert out.loc[2, "avg_monthly_spend"] == 70.0
    assert out.loc[2, "charge_increase"] == 10.0


def test_does_not_modify_input():
    df = make_customers()
    before = df.columns.tolist()
    add_features(df)
    assert df.columns.tolist() == before


def test_row_count_unchanged():
    df = make_customers()
    assert len(add_features(df)) == len(df)


def test_engineered_features_listed_in_config():
    # if a feature is missing from config, the pipeline would silently drop it
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert set(ENGINEERED_FEATURES) <= set(cfg["features"]["numeric"])
