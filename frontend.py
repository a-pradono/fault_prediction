import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title="Fault Diagnosis Dashboard", layout="wide")

st.title("Machine Fault Prediction")

# Load dataset
df = pd.read_csv("data/petrochemical_maintenance.csv") 

# EDA
st.header("Exploratory Data Analysis")

if st.checkbox("Show Correlation Heatmap"):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots()
    sns.heatmap(corr, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

if st.checkbox("Show Fault Type Distribution"):
    st.bar_chart(df['fault_type'].value_counts())

# Prediction
st.header("Fault Prediction")

inputs = {}
features = [
    "vibration_x", "vibration_y", "vibration_z",
    "temperature_c", "current_a", "rpm", "pressure_bar",
    "wavelet_feature_1", "wavelet_feature_2", "wavelet_feature_3",
    "wavelet_feature_4", "wavelet_feature_5", "maintenance_required"
]

for f in features:
    inputs[f] = st.number_input(f, value=0.0)

if st.button("Predict Fault"):
    response = requests.post(
        "http://127.0.0.1:8000/api/predict",
        json=inputs
    )

    if response.status_code == 200:
        result = response.json()
        st.success(f"Predicted Fault: {result['fault_type']}")
        st.info(f"Confidence: {result['confidence']}")
    else:
        st.error("Prediction failed")