from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib

# Load models
model = joblib.load("artifacts/log_reg_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")
encoder = joblib.load("artifacts/fault_type_encoder.pkl")
feature_order = joblib.load("artifacts/feature_order.pkl")

# Create FastAPI app
app = FastAPI(
    title="Fault Prediction API",
    description="Prediction of Machine Fault Type using Logistic Regression Model",
    version="1.0"
)

# Input shcema
class SensorInput(BaseModel):
    vibration_x: float
    vibration_y: float
    vibration_z: float
    temperature_c: float
    current_a: float
    rpm: float
    pressure_bar: float
    wavelet_feature_1: float
    wavelet_feature_2: float
    wavelet_feature_3: float
    wavelet_feature_4: float
    wavelet_feature_5: float
    maintenance_required: float


# Predict
@app.post("/api/predict")
def predict_fault(data: SensorInput):

    # Convert input to ordered array
    x = np.array([[getattr(data, f) for f in feature_order]])

    # Scale
    x_scaled = scaler.transform(x)

    # Predict probabilities
    probs = model.predict_proba(x_scaled)[0]
    class_idx = np.argmax(probs)

    return {
        "predicted_fault_type": encoder.inverse_transform([class_idx])[0],
        "confidence": round(float(probs[class_idx]), 3)
    }

# Trend
@app.post("/api/trend")
def trend(data: List[SensorInput]):
    results = []

    for record in data:
        x = np.array([[getattr(record, f) for f in feature_order]])
        x_scaled = scaler.transform(x)

        probs = model.predict_proba(x_scaled)[0]
        idx = np.argmax(probs)

        results.append({
            "fault_type": encoder.inverse_transform([idx])[0],
            "confidence": round(float(probs[idx]), 3)
        })

    return {"trend": results}