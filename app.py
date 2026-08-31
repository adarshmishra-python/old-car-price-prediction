
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "model" / "car_price_model.pkl"
DATA_PATH = BASE / "dataset" / "car_data.csv"

model = joblib.load(MODEL_PATH)
data = pd.read_csv(DATA_PATH)

st.set_page_config(page_title="Old Car Price Predictor", page_icon="🚗", layout="centered")

st.title("🚗 Old Car Price Prediction")
st.write("Enter the car details and the machine-learning model will estimate its resale price.")

col1, col2 = st.columns(2)
with col1:
    brand = st.selectbox("Car Brand", sorted(data["brand"].unique()))
    year = st.number_input("Manufacturing Year", min_value=int(data.year.min()),
                           max_value=2026, value=2018, step=1)
    km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000,
                                value=50000, step=1000)
with col2:
    fuel = st.selectbox("Fuel Type", sorted(data["fuel"].unique()))
    transmission = st.selectbox("Transmission", sorted(data["transmission"].unique()))
    owner = st.selectbox("Owner Type", sorted(data["owner"].unique()))

if st.button("Predict Price", type="primary", use_container_width=True):
    input_data = pd.DataFrame([{
        "brand": brand,
        "year": year,
        "km_driven": km_driven,
        "fuel": fuel,
        "transmission": transmission,
        "owner": owner
    }])
    price = model.predict(input_data)[0]
    st.success(f"Estimated Resale Price: ₹{price:,.0f}")
    st.caption("This is an ML estimate for educational/demo purposes. Actual market price may differ.")

with st.expander("About this project"):
    st.write("Model: Random Forest Regression")
    st.write("Input features: brand, year, kilometers driven, fuel, transmission and owner type.")
