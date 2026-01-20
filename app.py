import os
import joblib
import pandas as pd
import streamlit as st

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

st.title("🏠 House Price Prediction System")
st.write("Enter the house features below to predict the house price.")

# -----------------------------
# Load Trained Model
# -----------------------------
MODEL_PATH = os.path.join("models", "house_price_model.pkl")

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please run model/model_development.py first to create house_price_model.pkl")
    st.stop()

model = joblib.load(MODEL_PATH)

# -----------------------------
# User Inputs
# -----------------------------
overall_qual = st.slider("Overall Quality (1–10)", 1, 10, 5)
gr_liv_area = st.number_input("Above Ground Living Area (GrLivArea) in sq ft", min_value=100, value=1500, step=10)
total_bsmt_sf = st.number_input("Total Basement Area (TotalBsmtSF) in sq ft", min_value=0, value=800, step=10)
garage_cars = st.slider("Garage Capacity (GarageCars)", 0, 5, 2)
full_bath = st.slider("Number of Full Bathrooms (FullBath)", 0, 5, 2)
neighborhood = st.text_input("Neighborhood (e.g., NAmes, CollgCr, OldTown)", value="NAmes")

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict House Price"):
    input_df = pd.DataFrame({
        "OverallQual": [overall_qual],
        "GrLivArea": [gr_liv_area],
        "TotalBsmtSF": [total_bsmt_sf],
        "GarageCars": [garage_cars],
        "FullBath": [full_bath],
        "Neighborhood": [neighborhood]
    })

    try:
        prediction = model.predict(input_df)[0]
        st.success(f"✅ Predicted House Price: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Built by Osumuo Ikechukwu Charles (22cd032194) — Random Forest + Joblib + Streamlit")
