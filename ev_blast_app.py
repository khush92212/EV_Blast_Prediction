import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("dtc_ev_model.pkl")
encoder = joblib.load("label_encoder_ev.pkl")

st.title("EV_Blast_prediction !")
options = ["Yes", "No"]

Battery_Type = st.selectbox("Battery_Type", options)
Poor_Cell_Design = st.selectbox("Poor_Cell_Design", options)
External_Abuse = st.selectbox("External_Abuse", options)
Poor_Battery_Design = st.selectbox("Poor_Battery_Design", options)
Short_Circuits = st.selectbox("Short_Circuits", options)
Temperature = st.selectbox("Temperature", options)
Overcharge_Overdischarge =  st.selectbox("Overcharge_Overdischarge", options)
Battery_Maintenance = st.selectbox("Battery_Maintenance", options)
Battery_Health = st.selectbox("Battery_Health", options)

input_data = pd.DataFrame({
    "Battery_Type" : [Battery_Type],
    "Poor_Cell_Design" : [Poor_Cell_Design],
    "External_Abuse" : [External_Abuse],
    "Poor_Battery_Design" : [Poor_Battery_Design],
    "Short_Circuits" : [Short_Circuits],
    "Temperature" : [Temperature],
    "Overcharge_Overdischarge" : [Overcharge_Overdischarge],
    "Battery_Maintenance" : [Battery_Maintenance],
    "Battery_Health" : [Battery_Health]
})

if st.button("predict"):
  prediction = model.predict(input_data)
  if prediction == "Blast":
    st.success("Blast")
  else:
    st.error("Moderate")
