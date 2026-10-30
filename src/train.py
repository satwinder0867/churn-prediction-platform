"""Train churn models: split, compare with cross-validation, track in MLflow, save the best."""
import logging
import os
import warnings

# Silence an MLflow startup hint and its artifact progress bars
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.features import (
    ENGINEERED_FEATURES,
    add_features,
    feature_names_out,
    select_columns,
    selected_names_out,
)
from src.ingest import PROJECT_ROOT, load_config, setup_logging

LOG_PATH = PROJECT_ROOT / "logs" / "train.log"

# Our own functions used inside the pipeline. MLflow saves models with skops,
# which refuses to load any code it has not been told to trust.
PIPELINE_FUNCTIONS = [select_columns, selected_names_out, add_features, feature_names_out]
TRUSTED_TYPES = [f"{f.__module__}.{f.__name__}" for f in PIPELINE_FUNCTIONS]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- data

def load_processed(path) -> pd.DataFrame:
    """Read the cleaned data produced by ingest.py."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `python -m src.ingest` first")
    df = pd.read_parquet(path)
    logger.info("Loaded processed data: %s", df.shape)
    return df


def get_feature_lists(cfg: dict) -> tuple[list, list]:
    """Return (raw numeric, categorical) columns.

    Engineered features are listed in the config but created inside the
    pipeline, so they are not expected in the input data.
    """
    numeric = [c for c in cfg["features"]["numeric"] if c not in ENGINEERED_FEATURES]
    categorical = cfg["features"]["categorical"]
    return numeric, categorical


def check_columns(df: pd.DataFrame, cols: list) -> None:
    """Fail fast if any expected input column is missing."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from data: {missing}")


def split_data(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split; the test set is kept aside for evaluate.py."""
    target = cfg["target_col"]
    split = cfg["split"]
    train_df, test_df = train_test_split(
        df,
        test_size=split["test_size"],
        random_state=split["random_state"],
        stratify=df[target] if split["stratify"] else None,
    )
    for name, part in [("train", train_df), ("test", test_df)]:
        logger.info("%s: %s rows, churn rate %.3f", name, len(part), part[target].mean())
    return train_df, test_df


# ---------------------------------------------------------------- pipeline

def build_pipeline(model, numeric: list, categorical: list, use_features: bool) -> Pipeline:
    """Select inputs -> feature engineering -> preprocessing -> model, as one object."""
    select = FunctionTransformer(
        select_columns,
        kw_args={"columns": numeric + categorical},
        feature_names_out=selected_names_out,
    )
    if use_features:
        numeric = numeric + ENGINEERED_FEATURES
        feature_step = FunctionTransformer(add_features, feature_names_out=feature_names_out)
    else:
        feature_step = "passthrough"  # no engineered features

    preprocess = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical),
    ])
    return Pipeline([
        ("select", select),
        ("features", feature_step),
        ("preprocess", preprocess),
        ("model", model),
    ])


def get_models(random_state: int) -> dict:
    """Candidate models, ordered from simplest to most complex."""
    return {
        "dummy": DummyClassifier(strategy="prior"),
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "lgbm": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=15,
            class_weight="balanced",
            random_state=random_state,
            verbose=-1,
        ),
    }


# ---------------------------------------------------------------- experiments

def rule_baseline(X: pd.DataFrame, y: pd.Series) -> float:
    """PR-AUC of the rule 'every month-to-month customer will churn'."""
    score = (X["contract"] == "Month-to-month").astype(int)
    return average_precision_score(y, score)


def evaluate_cv(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, cv) -> dict:
    """Cross-validated PR-AUC and ROC-AUC (mean and std across folds)."""
    scores = cross_validate(pipe, X, y, cv=cv, scoring=["average_precision", "roc_auc"])
    return {
        "cv_pr_auc": scores["test_average_precision"].mean(),
        "cv_pr_auc_std": scores["test_average_precision"].std(),
        "cv_roc_auc": scores["test_roc_auc"].mean(),
    }


def run_experiments(X, y, numeric, categorical, cfg) -> pd.DataFrame:
    """Train every model with and without engineered features; log each run to MLflow."""
    random_state = cfg["split"]["random_state"]
    cv = StratifiedKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=random_state)

    rows = []
    for use_features in [False, True]:
        feature_set = "engineered" if use_features else "raw"
        for name, model in get_models(random_state).items():
            pipe = build_pipeline(model, numeric, categorical, use_features)
            metrics = evaluate_cv(pipe, X, y, cv)

            with mlflow.start_run(run_name=f"{name}-{feature_set}"):
                mlflow.log_params({"model": name, "feature_set": feature_set,
                                   **model.get_params()})
                mlflow.log_metrics(metrics)

            rows.append({"model": name, "feature_set": feature_set, **metrics})
            logger.info("%-6s %-10s PR-AUC %.3f +/- %.3f | ROC-AUC %.3f", name, feature_set,
                        metrics["cv_pr_auc"], metrics["cv_pr_auc_std"], metrics["cv_roc_auc"])

    return pd.DataFrame(rows)


def pick_best(results: pd.DataFrame) -> pd.Series:
    """Simplest candidate whose PR-AUC is within one std of the best score."""
    best = results.loc[results["cv_pr_auc"].idxmax()]
    threshold = best["cv_pr_auc"] - best["cv_pr_auc_std"]
    good_enough = results[(results["model"] != "dummy") & (results["cv_pr_auc"] >= threshold)]
    # simpler = logreg before lgbm, raw features before engineered ("raw" sorts last, so descending)
    complexity = {"logreg": 0, "lgbm": 1}
    good_enough = good_enough.assign(order=good_enough["model"].map(complexity))
    chosen = good_enough.sort_values(["order", "feature_set"], ascending=[True, False]).iloc[0]
    logger.info("Best PR-AUC: %s-%s %.3f; chosen (simplest within 1 std): %s-%s %.3f",
                best["model"], best["feature_set"], best["cv_pr_auc"],
                chosen["model"], chosen["feature_set"], chosen["cv_pr_auc"])
    return chosen


# ---------------------------------------------------------------- main

def main() -> None:
    setup_logging(LOG_PATH)
    cfg = load_config(PROJECT_ROOT / "config" / "config.yaml")
    paths = cfg["paths"]
    target = cfg["target_col"]

    # 1. data
    df = load_processed(PROJECT_ROOT / paths["processed_data"])
    numeric, categorical = get_feature_lists(cfg)
    check_columns(df, numeric + categorical + [target])

    train_df, test_df = split_data(df, cfg)
    train_df.to_parquet(PROJECT_ROOT / paths["train_data"], index=False)
    test_df.to_parquet(PROJECT_ROOT / paths["test_data"], index=False)
    logger.info("Saved train/test splits to data/processed/")

    X_train = train_df[numeric + categorical]
    y_train = train_df[target]

    # 2. experiments (cross-validation on the training set only)
    mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")
    experiment = cfg["mlflow"]["experiment_name"]
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(PROJECT_ROOT / "mlruns").as_uri())
    mlflow.set_experiment(experiment)

    logger.info("Rule baseline (month-to-month) PR-AUC: %.3f", rule_baseline(X_train, y_train))
    results = run_experiments(X_train, y_train, numeric, categorical, cfg)

    # 3. final model: refit the chosen pipeline on the full training set and save it
    chosen = pick_best(results)
    model = get_models(cfg["split"]["random_state"])[chosen["model"]]
    final_pipe = build_pipeline(model, numeric, categorical, chosen["feature_set"] == "engineered")
    final_pipe.fit(X_train, y_train)

    model_path = PROJECT_ROOT / paths["model"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipe, model_path)

    with mlflow.start_run(run_name=f"final-{chosen['model']}-{chosen['feature_set']}"):
        mlflow.log_params({"model": chosen["model"], "feature_set": chosen["feature_set"]})
        mlflow.log_metrics({k: chosen[k] for k in ["cv_pr_auc", "cv_pr_auc_std", "cv_roc_auc"]})
        # The example keeps the real dtypes (ints stay ints) so the logged schema matches
        # real input. MLflow warns that int columns can't hold missing values; our input
        # never has any (ingest validates that), so that one warning is silenced.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*integer column", category=UserWarning)
            mlflow.sklearn.log_model(
                final_pipe,
                name="model",
                input_example=X_train.head(3),
                skops_trusted_types=TRUSTED_TYPES,
            )

    logger.info("Saved final model to %s", model_path)


if __name__ == "__main__":
    main()
