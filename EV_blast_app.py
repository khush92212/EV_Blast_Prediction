import streamlit as st
import pandas as pd
import joblib

model = joblib.load("dtc_ev_model.pkl")

st.title("EV Blast Prediction")

options = ["Yes", "No"]

Battery_Type = st.selectbox("Battery_Type", options)
Poor_Cell_Design = st.selectbox("Poor_Cell_Design", options)
External_Abuse = st.selectbox("External_Abuse", options)
Poor_Battery_Design = st.selectbox("Poor_Battery_Design", options)
Short_Circuits = st.selectbox("Short_Circuits", options)
Temperature = st.selectbox("Temperature", options)
Overcharge_Overdischarge = st.selectbox("Overcharge_Overdischarge", options)
Battery_Maintenance = st.selectbox("Battery_Maintenance", options)
Battery_Health = st.selectbox("Battery_Health", options)

# Convert Yes/No to 1/0
mapping = {"Yes": 1, "No": 0}

input_data = pd.DataFrame({
    "Battery_Type": [mapping[Battery_Type]],
    "Poor_Cell_Design": [mapping[Poor_Cell_Design]],
    "External_Abuse": [mapping[External_Abuse]],
    "Poor_Battery_Design": [mapping[Poor_Battery_Design]],
    "Short_Circuits": [mapping[Short_Circuits]],
    "Temperature": [mapping[Temperature]],
    "Overcharge_Overdischarge": [mapping[Overcharge_Overdischarge]],
    "Battery_Maintenance": [mapping[Battery_Maintenance]],
    "Battery_Health": [mapping[Battery_Health]]
})

# Ensure correct column order
input_data = input_data[model.feature_names_in_]

if st.button("Predict"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ High Risk: Blast")
    else:
        st.success("✅ Moderate Condition")
