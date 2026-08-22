import streamlit as st
import pandas as pd
import joblib

# 1. Configure the Website Layout
st.set_page_config(
    page_title="Agri-Price Prediction Engine",
    page_icon="🌾",
    layout="centered"
)

# Website Header UI
st.title("🌾 Smart Agri-Demand: Price Prediction Dashboard")
st.markdown("---")
st.write("Adjust the seasonal and economic market factors on the left sidebar to generate a live retail price forecast powered by your trained machine learning engine.")

# 2. Securely Load the Saved Data Science Model and Scaler Artifacts
@st.cache_resource
def load_artifacts():
    try:
        loaded_model = joblib.load('xgb_pricing_model.pkl')
        loaded_scaler = joblib.load('feature_scaler.pkl')
        return loaded_model, loaded_scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_artifacts()

if model is None or scaler is None:
    st.error("⚠️ Critical System Error: Production model artifacts (`.pkl` files) were not found in the current root directory. Please verify that your Jupyter pipeline exported them to this same folder.")
else:
    # 3. Sidebar UI Form for collecting user inputs
    st.sidebar.header("⚙️ Market Analytics Inputs")
    st.sidebar.write("Modify the indicators below:")
    
    # Matching the exact data ranges of your CSV variables
    month = st.sidebar.slider("Timeline (Select Month)", min_value=1, max_value=12, value=5)
    rainfall = st.sidebar.slider("Seasonal Rainfall Volume (mm)", min_value=0.0, max_value=400.0, value=120.5)
    supply = st.sidebar.slider("Active Market Supply (Tons)", min_value=10.0, max_value=500.0, value=50.0)
    transport_cost = st.sidebar.slider("Logistics & Transportation Cost (INR)", min_value=5.0, max_value=100.0, value=45.0)

    # 4. Input Processing & Live Forecasting Pipeline
    st.subheader("📊 Algorithmic Assessment Panel")
    
    if st.button("🚀 Forecast Retail Market Price"):
        # Map user input into structured dictionary format
        live_input = {
            'Month': month,
            'Rainfall_mm': rainfall,
            'Market_Supply_Tons': supply,
            'Transportation_Cost': transport_cost
        }
        
        # Convert dictionary to structurally aligned DataFrame object
        live_df = pd.DataFrame([live_input])
        
        # Normalize the incoming vector via our saved mathematical scaling matrix
        live_scaled = scaler.transform(live_df)
        
        # Execute live prediction matrix
        predicted_yield = model.predict(live_scaled)[0]
        
        # Display the custom styled prediction card response
        st.markdown("---")
        st.success(f"### 🎯 Predicted Retail Market Price: **{predicted_yield:.2f} INR / KG**")
        st.balloons()
        
        # Contextual insights card
        st.info("💡 **Operational Market Insight:** Drastic drops in overall volume supply combined with spikes in regional transport logistics directly impact standard pricing curves. Consider optimizing supply chain dispatch intervals.")
