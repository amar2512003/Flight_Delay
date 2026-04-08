import streamlit as st
import numpy as np
import joblib

# Load
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

st.title("✈️ Flight Delay Predictor")

# Dropdowns (important for good UI)
carrier = st.selectbox("Airline", encoders["OP_CARRIER"].classes_)
origin = st.selectbox("Origin Airport", encoders["ORIGIN"].classes_)
dest = st.selectbox("Destination Airport", encoders["DEST"].classes_)

crs_dep_time = st.number_input("Scheduled Departure Time (HHMM)", 0, 2359)
distance = st.number_input("Distance", 0)
crs_elapsed = st.number_input("Scheduled Duration (minutes)", 0)

if st.button("Predict"):

    # Encode
    carrier = encoders["OP_CARRIER"].transform([carrier])[0]
    origin = encoders["ORIGIN"].transform([origin])[0]
    dest = encoders["DEST"].transform([dest])[0]

    input_data = np.array([[carrier, origin, dest, crs_dep_time, distance, crs_elapsed]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Flight will be DELAYED")
    else:
        st.success("✅ Flight will be ON TIME")