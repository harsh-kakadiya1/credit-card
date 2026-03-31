# ================================
# Import Libraries
# ================================
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ================================
# Load Model & Files
# ================================
model = joblib.load("best_fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# ================================
# Page Config
# ================================
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection System")
st.write("Enter transaction details to check if it is Fraud or Legitimate")

# ================================
# Input Section
# ================================
st.subheader("Enter Transaction Details")

input_data = {}

# Create inputs dynamically
for feature in features:
    input_data[feature] = st.number_input(f"{feature}", value=0.0)

# ================================
# Convert Input
# ================================
input_df = pd.DataFrame([input_data])

# ================================
# Prediction
# ================================
if st.button("🔍 Predict Transaction"):

    try:
        # Ensure correct column order
        input_df = input_df[features]

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Convert back to DataFrame (optional but safe)
        input_scaled_df = pd.DataFrame(input_scaled, columns=features)

        # Predict
        prediction = model.predict(input_scaled_df)[0]
        prob = model.predict_proba(input_scaled_df)[0][1]

        # Output
        if prediction == 1:
            st.error(f"🚨 Fraudulent Transaction Detected!")
            st.write(f"Fraud Probability: **{prob:.4f}**")
        else:
            st.success(f"✅ Legitimate Transaction")
            st.write(f"Fraud Probability: **{prob:.4f}**")

    except Exception as e:
        st.error(f"Error: {e}")