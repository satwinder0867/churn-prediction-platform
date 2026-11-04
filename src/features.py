"""Feature engineering: row-wise, deterministic features derived from EDA."""
import numpy as np
import pandas as pd

# Customers at or below this tenure are "new" (EDA: 52.9% churn in months 0-6)
NEW_CUSTOMER_MONTHS = 6

# Internet add-on services; each is "Yes", "No" or "No internet service"
ADDON_COLS = [
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]

# Names of the features this module creates (must also be listed in config features.numeric)
ENGINEERED_FEATURES = [
    "is_new_customer",
    "num_addon_services",
    "has_auto_payment",
    "has_family",
    "avg_monthly_spend",
    "charge_increase",
]


def is_new_customer(df: pd.DataFrame) -> pd.Series:
    """1 if the customer has been with us 6 months or less, else 0."""
    return (df["tenure_months"] <= NEW_CUSTOMER_MONTHS).astype(int)


def num_addon_services(df: pd.DataFrame) -> pd.Series:
    """Number of internet add-on services the customer has (0-6)."""
    return (df[ADDON_COLS] == "Yes").sum(axis=1)


def has_auto_payment(df: pd.DataFrame) -> pd.Series:
    """1 if the customer pays by bank transfer or credit card (automatic), else 0."""
    return df["payment_method"].str.contains("automatic").astype(int)


def has_family(df: pd.DataFrame) -> pd.Series:
    """1 if the customer has a partner or dependents, else 0."""
    return ((df["partner"] == "Yes") | (df["dependents"] == "Yes")).astype(int)


def avg_monthly_spend(df: pd.DataFrame) -> pd.Series:
    """Average amount paid per month so far.

    New customers (tenure 0) have not been billed yet, so their current
    monthly charge is used instead of dividing by zero.
    """
    tenure = df["tenure_months"].replace(0, np.nan)
    avg = df["total_charges"] / tenure
    return avg.fillna(df["monthly_charges"])


def charge_increase(df: pd.DataFrame) -> pd.Series:
    """Current monthly charge minus the historical average (positive = paying more now)."""
    return df["monthly_charges"] - avg_monthly_spend(df)


def select_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Keep only the model's input columns, in a fixed order.

    Makes the pipeline independent of column order and ignores extra columns
    (e.g. customer_id) sent at scoring time; a missing column raises KeyError.
    """
    return df[columns]


def selected_names_out(transformer, _input_features) -> list:
    """Column names after select_columns: exactly the columns it keeps."""
    return list(transformer.kw_args["columns"])


def feature_names_out(_transformer, input_features) -> list:
    """Column names after add_features, so the pipeline can report feature names.

    scikit-learn passes the transformer as the first argument; it is not needed here.
    """
    return list(input_features) + ENGINEERED_FEATURES


def get_feature_lists(cfg: dict) -> tuple[list, list]:
    """Return (raw numeric, categorical) columns.

    Engineered features are listed in the config but created inside the
    pipeline, so they are not expected in the input data.
    """
    numeric = [c for c in cfg["features"]["numeric"] if c not in ENGINEERED_FEATURES]
    categorical = cfg["features"]["categorical"]
    return numeric, categorical


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features. Row-wise, deterministic, never uses the target."""
    df = df.copy()
    df["is_new_customer"] = is_new_customer(df)
    df["num_addon_services"] = num_addon_services(df)
    df["has_auto_payment"] = has_auto_payment(df)
    df["has_family"] = has_family(df)
    df["avg_monthly_spend"] = avg_monthly_spend(df)
    df["charge_increase"] = charge_increase(df)
    return df
