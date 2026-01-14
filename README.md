## Summmary
This project focuses on machine learning for machine fault prediction using sensor data. Dataset contains 5000 rows and 16 columns, some key insights from this project are:
- Strong correlations occurred between vibration signals and machine faults
- RPM illustrated an inverse relationship with machine faults
- Wavelet features shows minimum contribution to raw vibration signals
- Logistic regression model was performed better compared to other supersived machine learning models

## Backend
- Logistic regression model has been used to predict machine fault
- The model showed better metrics compared other models and has interpretable coefficients
- Backend has been built using FastAPI and consists of:
  1. api/predict: predict fault type for a single input
  2. api/trend: predict fault trends for multiple inputs
  
## Frontend
- Frontent has been built using Streamlit
- Visualize some charts for EDA
- Display predicted machine fault type and confidence
