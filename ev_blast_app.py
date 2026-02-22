import streamlit as st
import pandas as pd
import joblib

model = joblib.load("dtc_ev_model.pkl")

st.title("EV Blast Prediction")

options = [0, 1]   # if your dataset uses 0 and 1

Battery_Type = st.selectbox("Battery Type", options)
Poor_Cell_Design = st.selectbox("Poor Cell Design", options)
External_Abuse = st.selectbox("External Abuse", options)
Poor_Battery_Design = st.selectbox("Poor Battery Design", options)
Short_Circuits = st.selectbox("Short Circuits", options)
Temperature = st.selectbox("High Temperature", options)
Overcharge_Overdischarge = st.selectbox("Overcharge/Overdischarge", options)
Battery_Maintenance = st.selectbox("Battery Maintenance", options)
Battery_Health = st.selectbox("Battery Health", options)

input_data = pd.DataFrame({
    "Battery_Type": [Battery_Type],
    "Poor_Cell_Design": [Poor_Cell_Design],
    "External_Abuse": [External_Abuse],
    "Poor_Battery_Design": [Poor_Battery_Design],
    "Short_Circuits": [Short_Circuits],
    "Temperature": [Temperature],
    "Overcharge_Overdischarge": [Overcharge_Overdischarge],
    "Battery_Maintenance": [Battery_Maintenance],
    "Battery_Health": [Battery_Health]
})

if st.button("Predict"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ High Risk: Blast")
    else:
        st.success("✅ Moderate / Safe Condition")
