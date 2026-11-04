"""Churn prediction API: score customers on request with the saved pipeline.

Run locally:
    uvicorn api.main:app --reload
Then open http://127.0.0.1:8000/docs for interactive documentation.
"""
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request

from api.schemas import BatchRequest, BatchResponse, Customer, ModelInfo, Prediction
from src.features import get_feature_lists
from src.ingest import PROJECT_ROOT, load_config, setup_logging
from src.policy import customer_value, expected_profits, recommend
from src.score import model_fingerprint, risk_reasons

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load config and model once at startup, not on every request."""
    setup_logging(PROJECT_ROOT / "logs" / "api.log")
    cfg = load_config(PROJECT_ROOT / "config" / "config.yaml")
    model_path = Path(os.environ.get("MODEL_PATH", PROJECT_ROOT / cfg["paths"]["model"]))
    if not model_path.is_file():
        raise RuntimeError(f"Model not found at {model_path}; run `python -m src.train` first")

    numeric, categorical = get_feature_lists(cfg)
    app.state.cfg = cfg
    app.state.model = joblib.load(model_path)
    app.state.features = numeric + categorical
    app.state.info = ModelInfo(
        model_fingerprint=model_fingerprint(model_path),
        model_type=type(app.state.model).__name__,
        loaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        features=app.state.features,
        note="recommended_action ignores call capacity; batch scoring (src.score) applies it",
    )
    logger.info("Loaded model %s from %s", app.state.info.model_fingerprint, model_path)
    yield


app = FastAPI(
    title="Churn Prediction API",
    description="Churn probability, expected retention profit and risk reasons per customer.",
    version="1.0.0",
    lifespan=lifespan,
)


def predict_customers(customers: list[Customer], request: Request) -> list[Prediction]:
    """Shared logic for single and batch requests."""
    state = request.app.state
    biz = state.cfg["business"]

    df = pd.DataFrame([c.model_dump() for c in customers])
    p = state.model.predict_proba(df)[:, 1]
    value = customer_value(df["monthly_charges"], biz)
    ev_email, ev_call = expected_profits(p, value, biz)
    actions = recommend(p, value, biz)
    reasons = risk_reasons(state.model, df[state.features])

    logger.info("Scored %d customer(s), mean probability %.3f", len(df), p.mean())
    return [
        Prediction(
            customer_id=c.customer_id,
            churn_probability=round(float(p[i]), 4),
            customer_value=round(float(value[i]), 2),
            expected_profit_email=round(float(ev_email[i]), 2),
            expected_profit_call=round(float(ev_call[i]), 2),
            recommended_action=actions[i],
            risk_reasons=reasons[i].split("; ") if reasons[i] else [],
        )
        for i, c in enumerate(customers)
    ]


@app.get("/health")
def health(request: Request) -> dict:
    """Liveness check for Docker / load balancers."""
    return {"status": "ok", "model_loaded": hasattr(request.app.state, "model")}


@app.get("/model/info", response_model=ModelInfo)
def model_info(request: Request) -> ModelInfo:
    """Which model is serving, so every prediction can be traced."""
    return request.app.state.info


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer, request: Request) -> Prediction:
    """Score one customer."""
    return predict_customers([customer], request)[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest, request: Request) -> BatchResponse:
    """Score up to 1,000 customers in one request."""
    return BatchResponse(
        predictions=predict_customers(batch.customers, request),
        model_fingerprint=request.app.state.info.model_fingerprint,
    )
