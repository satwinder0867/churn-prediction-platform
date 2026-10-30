"""Request and response schemas: the API's contract with its callers.

Allowed values come from the training data. Anything else is rejected with a
422 error before it reaches the model.
"""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

YesNo = Literal["Yes", "No"]
PhoneAddon = Literal["Yes", "No", "No phone service"]
InternetAddon = Literal["Yes", "No", "No internet service"]

INTERNET_ADDONS = [
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]


class Customer(BaseModel):
    """One customer, in the same cleaned format the model was trained on."""

    model_config = ConfigDict(extra="forbid")  # unknown fields (typos, the label) are errors

    customer_id: Optional[str] = None
    gender: Literal["Male", "Female"]
    senior_citizen: YesNo
    partner: YesNo
    dependents: YesNo
    tenure_months: int = Field(ge=0, le=120)
    phone_service: YesNo
    multiple_lines: PhoneAddon
    internet_service: Literal["DSL", "Fiber optic", "No"]
    online_security: InternetAddon
    online_backup: InternetAddon
    device_protection: InternetAddon
    tech_support: InternetAddon
    streaming_tv: InternetAddon
    streaming_movies: InternetAddon
    contract: Literal["Month-to-month", "One year", "Two year"]
    paperless_billing: YesNo
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    monthly_charges: float = Field(gt=0, le=500)
    total_charges: float = Field(ge=0)

    @model_validator(mode="after")
    def check_consistency(self) -> "Customer":
        """Business rules between fields, the same ones that hold in the training data."""
        no_internet = self.internet_service == "No"
        for addon in INTERNET_ADDONS:
            if (getattr(self, addon) == "No internet service") != no_internet:
                raise ValueError(
                    f"{addon} must be 'No internet service' exactly when internet_service is 'No'"
                )
        if (self.multiple_lines == "No phone service") != (self.phone_service == "No"):
            raise ValueError(
                "multiple_lines must be 'No phone service' exactly when phone_service is 'No'"
            )
        return self


class Prediction(BaseModel):
    customer_id: Optional[str]
    churn_probability: float
    customer_value: float
    expected_profit_email: float
    expected_profit_call: float
    recommended_action: Literal["none", "email", "call"]
    risk_reasons: list[str]


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    model_fingerprint: str


class ModelInfo(BaseModel):
    model_fingerprint: str
    model_type: str
    loaded_at: str
    features: list[str]
    note: str
