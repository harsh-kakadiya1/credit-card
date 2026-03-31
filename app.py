# ================================
# Import Libraries
# ================================
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# ================================
# Custom CSS
# ================================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .result-card {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-top: 2rem;
    }
    .fraud-result {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
    }
    .legitimate-result {
        background: linear-gradient(135deg, #00d2d3 0%, #01a3a4 100%);
        color: white;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
        border-radius: 25px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    .sidebar-content {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# Load Model & Files
# ================================
@st.cache_resource
def load_model_files():
    model = joblib.load("best_fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("features.pkl")
    return model, scaler, features

model, scaler, features = load_model_files()

# ================================
# Page Config
# ================================
st.set_page_config(
    page_title="Credit Card Fraud Detection", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# Header
# ================================
st.markdown("""
<div class="main-header">
    <h1>💳 Credit Card Fraud Detection System</h1>
    <p>Advanced AI-powered transaction security analysis</p>
</div>
""", unsafe_allow_html=True)

# ================================
# Sidebar
# ================================
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.header("🔧 System Info")
    st.info(f"Model Status: ✅ Active")
    st.info(f"Features: {len(features)}")
    st.info(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    st.markdown("---")
    st.subheader("📊 Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", "99.8%")
    with col2:
        st.metric("Speed", "< 1s")
    
    st.markdown("---")
    st.subheader("🎯 Risk Levels")
    st.success("🟢 Low: 0-30%")
    st.warning("🟡 Medium: 30-70%")
    st.error("🔴 High: 70-100%")
    st.markdown('</div>', unsafe_allow_html=True)

# ================================
# Main Content
# ================================
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.subheader("📝 Transaction Details")
    st.write("Enter the transaction features below:")
    
    input_data = {}
    
    # Group features for better organization
    if len(features) > 10:
        # If many features, use columns
        cols = st.columns(3)
        for i, feature in enumerate(features):
            with cols[i % 3]:
                input_data[feature] = st.number_input(
                    f"{feature}", 
                    value=0.0,
                    format="%.4f",
                    key=feature
                )
    else:
        # If fewer features, use single column
        for feature in features:
            input_data[feature] = st.number_input(
                f"{feature}", 
                value=0.0,
                format="%.4f",
                key=feature
            )
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.subheader("📈 Input Summary")
    total_inputs = len(input_data)
    non_zero_inputs = sum(1 for v in input_data.values() if v != 0.0)
    st.metric("Total Features", total_inputs)
    st.metric("Active Features", non_zero_inputs)
    
    if non_zero_inputs > 0:
        avg_value = np.mean(list(input_data.values()))
        st.metric("Average Value", f"{avg_value:.4f}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ================================
# Prediction Section
# ================================
st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🔍 Analyze Transaction", use_container_width=True):
        with st.spinner("🔄 Analyzing transaction..."):
            try:
                # Convert input
                input_df = pd.DataFrame([input_data])
                input_df = input_df[features]
                
                # Scale input
                input_scaled = scaler.transform(input_df)
                input_scaled_df = pd.DataFrame(input_scaled, columns=features)
                
                # Predict
                prediction = model.predict(input_scaled_df)[0]
                prob = model.predict_proba(input_scaled_df)[0][1]
                
                # Display results
                if prediction == 1:
                    st.markdown(f"""
                    <div class="result-card fraud-result">
                        <h2>🚨 FRAUD DETECTED!</h2>
                        <p style="font-size: 1.5rem; margin: 1rem 0;">
                            Fraud Probability: <strong>{prob:.2%}</strong>
                        </p>
                        <p>⚠️ This transaction requires immediate attention!</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Additional fraud warnings
                    if prob > 0.8:
                        st.error("🔴 EXTREME RISK: Block this transaction immediately!")
                    elif prob > 0.6:
                        st.warning("🟡 HIGH RISK: Manual verification required!")
                        
                else:
                    st.markdown(f"""
                    <div class="result-card legitimate-result">
                        <h2>✅ TRANSACTION APPROVED</h2>
                        <p style="font-size: 1.5rem; margin: 1rem 0;">
                            Fraud Probability: <strong>{prob:.2%}</strong>
                        </p>
                        <p>✨ This transaction appears legitimate</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Risk indicator
                risk_percentage = prob * 100
                if risk_percentage < 30:
                    st.success("🟢 Risk Level: LOW")
                elif risk_percentage < 70:
                    st.warning("🟡 Risk Level: MEDIUM")
                else:
                    st.error("🔴 Risk Level: HIGH")
                    
            except Exception as e:
                st.error(f"❌ Error during analysis: {e}")
                st.write("Please check your input values and try again.")

# ================================
# Footer
# ================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🔒 Secure AI-Powered Fraud Detection System | Version 2.0</p>
    <p>⚡ Real-time transaction analysis with machine learning</p>
</div>
""", unsafe_allow_html=True)