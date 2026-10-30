"""Evaluate the saved model once on the held-out test set: ranking, calibration, business value."""
import json
import logging
import os

# Silence an MLflow startup hint and its artifact progress bars
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

import joblib
import matplotlib

matplotlib.use("Agg")  # draw plots to files, no window needed

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.ingest import PROJECT_ROOT, load_config, setup_logging
from src.policy import choose_actions, customer_value

LOG_PATH = PROJECT_ROOT / "logs" / "evaluate.log"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- 1. ranking quality

def ranking_metrics(y: pd.Series, p: np.ndarray, contract: pd.Series) -> dict:
    """PR-AUC and ROC-AUC for the model and the two baselines."""
    rule = (contract == "Month-to-month").astype(int)
    return {
        "test_pr_auc": average_precision_score(y, p),
        "test_roc_auc": roc_auc_score(y, p),
        "baseline_random_pr_auc": y.mean(),
        "baseline_rule_pr_auc": average_precision_score(y, rule),
    }


# ---------------------------------------------------------------- 2. calibration

def calibration_report(y: pd.Series, p: np.ndarray, out_path) -> dict:
    """Do predicted probabilities match real churn rates? Saves a reliability plot."""
    prob_true, prob_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfectly calibrated")
    ax.plot(prob_pred, prob_true, "o-", label="model")
    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel("Actual churn rate")
    ax.set_title("Calibration on test set")
    ax.legend()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return {
        "brier": brier_score_loss(y, p),
        "brier_baseline": brier_score_loss(y, np.full(len(y), y.mean())),
        "mean_predicted": float(p.mean()),
        "actual_churn_rate": float(y.mean()),
        "max_calibration_gap": float(np.abs(prob_true - prob_pred).max()),
    }


# ---------------------------------------------------------------- 3. business decisions

def campaign_value(actions: np.ndarray, churned: np.ndarray, value: np.ndarray,
                   biz: dict) -> dict:
    """Estimated net savings of a set of actions, using the real test labels."""
    saved, cost = 0.0, 0.0
    for action in ["email", "call"]:
        mask = actions == action
        saved += (churned[mask] * biz[action]["save_rate"] * value[mask]).sum()
        cost += mask.sum() * biz[action]["cost"]
    return {
        "emails": int((actions == "email").sum()),
        "calls": int((actions == "call").sum()),
        "value_saved": round(saved, 2),
        "cost": round(cost, 2),
        "net_savings": round(saved - cost, 2),
    }


def compare_strategies(p, test: pd.DataFrame, target: str, biz: dict) -> pd.DataFrame:
    """Net savings of the model policy vs simple alternatives."""
    churned = test[target].to_numpy()
    value = customer_value(test["monthly_charges"], biz)
    n = len(test)

    strategies = {
        "do nothing": np.full(n, "none", dtype=object),
        "email everyone": np.full(n, "email", dtype=object),
        "rule: email month-to-month": np.where(
            test["contract"] == "Month-to-month", "email", "none").astype(object),
        "model policy": choose_actions(p, value, biz),
        "perfect foresight (upper bound)": choose_actions(churned.astype(float), value, biz),
    }
    rows = [{"strategy": name, **campaign_value(a, churned, value, biz)}
            for name, a in strategies.items()]
    return pd.DataFrame(rows).set_index("strategy")


def tier_metrics(actions: np.ndarray, churned: np.ndarray) -> dict:
    """How well each tier targets real churners."""
    called = actions == "call"
    contacted = actions != "none"
    return {
        "call_precision": float(churned[called].mean()) if called.any() else 0.0,
        "contacted_recall": float(churned[contacted].sum() / churned.sum()),
    }


# ---------------------------------------------------------------- 4. explainability

def top_drivers(model, n: int = 10) -> pd.DataFrame:
    """Logistic-regression coefficients averaged over the calibration folds."""
    pipes = [c.estimator for c in model.calibrated_classifiers_]
    coefs = np.mean([pipe[-1].coef_[0] for pipe in pipes], axis=0)
    names = pipes[0][:-1].get_feature_names_out()
    drivers = pd.DataFrame({"feature": names, "coef": coefs})
    drivers["odds_ratio"] = np.exp(drivers["coef"])
    return drivers.reindex(drivers["coef"].abs().sort_values(ascending=False).index).head(n)


# ---------------------------------------------------------------- main

def main() -> None:
    setup_logging(LOG_PATH)
    cfg = load_config(PROJECT_ROOT / "config" / "config.yaml")
    target = cfg["target_col"]
    biz = cfg["business"]
    reports = PROJECT_ROOT / cfg["reports_dir"]
    reports.mkdir(parents=True, exist_ok=True)

    model = joblib.load(PROJECT_ROOT / cfg["paths"]["model"])
    test = pd.read_parquet(PROJECT_ROOT / cfg["paths"]["test_data"])
    y = test[target]
    p = model.predict_proba(test)[:, 1]
    logger.info("Scored %d test customers (used once, never for training)", len(test))

    # 1. ranking
    ranking = ranking_metrics(y, p, test["contract"])
    logger.info("PR-AUC %.3f (random %.3f, rule %.3f) | ROC-AUC %.3f",
                ranking["test_pr_auc"], ranking["baseline_random_pr_auc"],
                ranking["baseline_rule_pr_auc"], ranking["test_roc_auc"])

    # 2. calibration
    calib = calibration_report(y, p, reports / "calibration.png")
    logger.info("Brier %.4f (baseline %.4f) | mean predicted %.3f vs actual %.3f | "
                "max bin gap %.3f", calib["brier"], calib["brier_baseline"],
                calib["mean_predicted"], calib["actual_churn_rate"],
                calib["max_calibration_gap"])

    # 3. business value
    strategies = compare_strategies(p, test, target, biz)
    logger.info("Campaign on test set:\n%s", strategies.to_string())
    value = customer_value(test["monthly_charges"], biz)
    tiers = tier_metrics(choose_actions(p, value, biz), y.to_numpy())
    logger.info("Called customers who really churned: %.1f%% | churners reached: %.1f%%",
                100 * tiers["call_precision"], 100 * tiers["contacted_recall"])

    # 4. explainability
    drivers = top_drivers(model)
    logger.info("Top churn drivers (odds ratio > 1 = more churn):\n%s",
                drivers.round(3).to_string(index=False))

    # save and track
    metrics = {**ranking, **calib, **tiers,
               "net_savings_model": strategies.loc["model policy", "net_savings"]}
    (reports / "evaluation.json").write_text(json.dumps(metrics, indent=2, default=float))
    strategies.to_csv(reports / "strategies.csv")
    drivers.to_csv(reports / "drivers.csv", index=False)

    mlflow.set_tracking_uri(f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name="evaluation-test"):
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        mlflow.log_artifacts(str(reports))
    logger.info("Saved reports to %s", reports)


if __name__ == "__main__":
    main()
