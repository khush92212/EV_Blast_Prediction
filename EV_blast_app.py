import streamlit as st
import pandas as pd
import joblib

model = joblib.load("dtc_ev_model.pkl")
encoders = joblib.load("label_encoder_ev.pkl")   # dictionary

st.title("EV_Blast_prediction !")

Battery_Type = st.selectbox("Battery_Type", encoders["Battery_Type"].classes_)
Poor_Cell_Design = st.selectbox("Poor_Cell_Design", encoders["Poor_Cell_Design"].classes_)
External_Abuse = st.selectbox("External_Abuse", encoders["External_Abuse"].classes_)
Poor_Battery_Design = st.selectbox("Poor_Battery_Design", encoders["Poor_Battery_Design"].classes_)
Short_Circuits = st.selectbox("Short_Circuits", encoders["Short_Circuits"].classes_)
Temperature = st.selectbox("Temperature", encoders["Temperature"].classes_)
Overcharge_Overdischarge = st.selectbox("Overcharge_Overdischarge", encoders["Overcharge_Overdischarge"].classes_)
Battery_Maintenance = st.selectbox("Battery_Maintenance", encoders["Battery_Maintenance"].classes_)
Battery_Health = st.selectbox("Battery_Health", encoders["Battery_Health"].classes_)

input_data = pd.DataFrame({
    "Battery_Type": [encoders["Battery_Type"].transform([Battery_Type])[0]],
    "Poor_Cell_Design": [encoders["Poor_Cell_Design"].transform([Poor_Cell_Design])[0]],
    "External_Abuse": [encoders["External_Abuse"].transform([External_Abuse])[0]],
    "Poor_Battery_Design": [encoders["Poor_Battery_Design"].transform([Poor_Battery_Design])[0]],
    "Short_Circuits": [encoders["Short_Circuits"].transform([Short_Circuits])[0]],
    "Temperature": [encoders["Temperature"].transform([Temperature])[0]],
    "Overcharge_Overdischarge": [encoders["Overcharge_Overdischarge"].transform([Overcharge_Overdischarge])[0]],
    "Battery_Maintenance": [encoders["Battery_Maintenance"].transform([Battery_Maintenance])[0]],
    "Battery_Health": [encoders["Battery_Health"].transform([Battery_Health])[0]]
})

if st.button("predict"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("Blast")
    else:
        st.error("Moderate")
