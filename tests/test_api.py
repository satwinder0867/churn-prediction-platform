"""API tests: start the real app with the saved model and send requests to it."""
import pytest
from fastapi.testclient import TestClient

from api.main import app
from src.ingest import PROJECT_ROOT

if not (PROJECT_ROOT / "models" / "churn_model.pkl").is_file():
    pytest.skip("no trained model; run `python -m src.train` first", allow_module_level=True)

# A high-risk profile: new, month-to-month, fiber, electronic check, no add-ons
RISKY = {
    "customer_id": "TEST-001",
    "gender": "Female",
    "senior_citizen": "No",
    "partner": "No",
    "dependents": "No",
    "tenure_months": 2,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 85.0,
    "total_charges": 170.0,
}

# A low-risk profile: long tenure, two-year contract, family, automatic payment
SAFE = {
    **RISKY,
    "customer_id": "TEST-002",
    "partner": "Yes",
    "dependents": "Yes",
    "tenure_months": 60,
    "internet_service": "DSL",
    "online_security": "Yes",
    "tech_support": "Yes",
    "contract": "Two year",
    "payment_method": "Credit card (automatic)",
    "monthly_charges": 60.0,
    "total_charges": 3600.0,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # "with" runs startup, so the model is loaded
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    info = client.get("/model/info").json()
    assert len(info["model_fingerprint"]) == 12
    assert len(info["features"]) == 19


def test_predict_risky_vs_safe(client):
    risky = client.post("/predict", json=RISKY).json()
    safe = client.post("/predict", json=SAFE).json()
    assert 0 <= safe["churn_probability"] < risky["churn_probability"] <= 1
    assert risky["recommended_action"] == "call"
    assert risky["risk_reasons"], "a risky customer should have reasons"


def test_field_order_does_not_matter(client):
    reversed_fields = dict(reversed(list(RISKY.items())))
    a = client.post("/predict", json=RISKY).json()
    b = client.post("/predict", json=reversed_fields).json()
    assert a["churn_probability"] == b["churn_probability"]


def test_batch_matches_single(client):
    batch = client.post("/predict/batch", json={"customers": [RISKY, SAFE]}).json()
    single = client.post("/predict", json=SAFE).json()
    assert len(batch["predictions"]) == 2
    assert batch["predictions"][1]["churn_probability"] == single["churn_probability"]


@pytest.mark.parametrize("change, reason", [
    ({"contract": "Weekly"}, "unknown category"),
    ({"tenure_months": -1}, "negative tenure"),
    ({"monthly_charges": "a lot"}, "not a number"),
    ({"churn_value": 1}, "label sent as input"),
    ({"internet_service": "No"}, "add-ons say internet exists"),
    ({"phone_service": "No"}, "multiple_lines says phone exists"),
])
def test_invalid_input_rejected(client, change, reason):
    response = client.post("/predict", json={**RISKY, **change})
    assert response.status_code == 422, reason


def test_missing_field_rejected(client):
    incomplete = {k: v for k, v in RISKY.items() if k != "contract"}
    assert client.post("/predict", json=incomplete).status_code == 422


def test_empty_batch_rejected(client):
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422
