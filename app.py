import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and features
model = joblib.load("decision_tree_model_final.pkl")
feature_names = joblib.load("feature_names.pkl")

st.title("Property Price Prediction")
st.write("Enter property details to predict price (in lakhs)")

# User inputs
bhk_no = st.number_input("Number of Bedrooms", 1, 10, 2)
square_ft = st.number_input("Square Feet", 200, 10000, 1000)

under_construction = st.selectbox("Under Construction", [0, 1])
rera = st.selectbox("RERA Approved", [0, 1])
ready_to_move = st.selectbox("Ready To Move", [0, 1])
resale = st.selectbox("Resale", [0, 1])

latitude = st.number_input("Latitude", value=12.97)
longitude = st.number_input("Longitude", value=77.59)

if st.button("Predict Price"):

    input_df = pd.DataFrame({
        "BHK_NO.": [bhk_no],
        "UNDER_CONSTRUCTION": [under_construction],
        "RERA": [rera],
        "READY_TO_MOVE": [ready_to_move],
        "RESALE": [resale],
        "LATITUDE": [latitude],
        "LONGITUDE": [longitude],
        "Log_SQUARE_FT": [np.log1p(square_ft)]
    })

    # Encode + align
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # Predict
    log_price = model.predict(input_df)
    price_lacs = np.expm1(log_price)[0]

    st.success(f" Estimated Property Price: Rs {price_lacs:.2f} Lakhs")
