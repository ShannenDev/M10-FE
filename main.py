import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Initialize FastAPI application
app = FastAPI(
    title="Bank Loan Default Prediction API",
    description="REST API for real-time customer default probability scoring.",
)


# Load trained Random Forest model artifact (500 trees)
MODEL_PATH = "random_forest_500_trees.pkl"

try:
    rf_model = joblib.load(MODEL_PATH)
except Exception:
    rf_model = None


# Define Pydantic Schema for Input Feature Validation
class LoanApplicant(BaseModel):
    AGE: float = Field(
        ...,
        example=35.0,
        description="Age of the applicant",
    )

    EMPLOY: float = Field(
        ...,
        example=10.0,
        description="Years with current employer",
    )

    ADDRESS: float = Field(
        ...,
        example=5.0,
        description="Years at current address",
    )

    DEBTINC: float = Field(
        ...,
        example=12.5,
        description="Debt-to-income ratio (%)",
    )

    CREDDEBT: float = Field(
        ...,
        example=4.2,
        description="Credit card debt ($k)",
    )

    OTHDEBT: float = Field(
        ...,
        example=6.1,
        description="Other outstanding debt ($k)",
    )


# Health Check Endpoint
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": rf_model is not None,
    }


# Task 7: Define /predict Endpoint
@app.get("/predict")
def predict_default(applicant: LoanApplicant):
    """
    Accepts customer attributes via JSON body and returns
    the estimated default probability and classification label.
    """

    if rf_model is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Model file not loaded. Ensure "
                "'random_forest_500_trees.pkl' exists."
            ),
        )

    # Construct DataFrame matching exact model feature names & order
    input_data = pd.DataFrame(
        [
            {
                "AGE": applicant.AGE,
                "EMPLOY": applicant.EMPLOY,
                "ADDRESS": applicant.ADDRESS,
                "DEBTINC": applicant.DEBTINC,
                "CREDDEBT": applicant.CREDDEBT,
                "OTHDEBT": applicant.OTHDEBT,
            }
        ]
    )

    # Estimate probability of default (Class 1)
    default_prob = float(
        np.round(rf_model.predict_proba(input_data)[0, 1], 4)
    )

    # Binary classification decision
    # 0 = Non-Defaulter, 1 = Defaulter
    predicted_class = int(rf_model.predict(input_data)[0])

    return {
        "probability_of_default": default_prob,
        "predicted_defaulter": predicted_class,
        "risk_status": (
            "Defaulter"
            if predicted_class == 1
            else "Non-Defaulter"
        ),
    }