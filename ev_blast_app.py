import streamlit as st
import pandas as pd
import joblib

# Load model and training columns
model = joblib.load("dtc_ev_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("EV Battery Health Prediction")

# ---------- USER INPUTS ----------

Temperature = st.number_input("Temperature", value=25.0)
Poor_Cell_Design = st.selectbox("Poor Cell Design", [0, 1])
Poor_Battery_Design = st.selectbox("Poor Battery Design", [0, 1])
Short_Circuits = st.selectbox("Short Circuits", [0, 1])

Battery_Type = st.selectbox("Battery Type", ["Li-ion", "NiMH", "Lead Acid"])
External_Abuse = st.selectbox("External Abuse", ["Yes", "No"])
Overcharge_Overdischarge = st.selectbox("Overcharge/Overdischarge", ["Yes", "No"])
Battery_Maintenance = st.selectbox("Battery Maintenance", ["Good", "Poor"])

# ---------- CREATE DATAFRAME ----------

input_dict = {
    "Temperature": Temperature,
    "Poor_Cell_Design": Poor_Cell_Design,
    "Poor_Battery_Design": Poor_Battery_Design,
    "Short_Circuits": Short_Circuits,
    "Battery_Type": Battery_Type,
    "External_Abuse": External_Abuse,
    "Overcharge_Overdischarge": Overcharge_Overdischarge,
    "Battery_Maintenance": Battery_Maintenance
}

input_df = pd.DataFrame([input_dict])

# Convert categorical columns
input_df = pd.get_dummies(input_df)

# Match training columns safely
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# ---------- PREDICT ----------

if st.button("Predict"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Battery Health: {prediction[0]}")

