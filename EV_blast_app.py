import streamlit as st
import pandas as pd
import joblib

# ---------------- LOAD MODEL ----------------
model = joblib.load("dtc_ev_model.pkl")
encoders = joblib.load("label_encoder_ev.pkl")

st.title("EV Blast Prediction App")

# --------------- INPUT FIELDS ----------------
Battery_Type = st.selectbox("Battery Type", encoder["Battery_Type"].classes_)
Poor_Cell_Design = st.selectbox("Poor Cell Design", encoder["Poor_Cell_Design"].classes_)
External_Abuse = st.selectbox("External Abuse", encoder["External_Abuse"].classes_)
Poor_Battery_Design = st.selectbox("Poor Battery Design", encoder["Poor_Battery_Design"].classes_)
Short_Circuits = st.selectbox("Short Circuits", encoder["Short_Circuits"].classes_)
Temperature = st.selectbox("Temperature", encoder["Temperature"].classes_)
Overcharge_Overdischarge = st.selectbox("Overcharge/Overdischarge", encoder["Overcharge_Overdischarge"].classes_)
Battery_Maintenance = st.selectbox("Battery Maintenance", encoder["Battery_Maintenance"].classes_)

# --------------- CREATE INPUT DATAFRAME ----------------
input_data = pd.DataFrame([[
    Battery_Type,
    Poor_Cell_Design,
    External_Abuse,
    Poor_Battery_Design,
    Short_Circuits,
    Temperature,
    Overcharge_Overdischarge,
    Battery_Maintenance
]],columns=model.feature_names_in_)

# --------------- ENCODE INPUT ----------------
for col in input_data.columns:
    input_data[col] = encoders[col].transform(input_data[col])

# --------------- PREDICTION ----------------
if st.button("Predict"):

    prediction = model.predict(input_data)[0]

    # If your model output is numeric (0/1)
    if prediction == 1:
        st.error("⚠️ Blast Risk Detected!")
    else:
        st.success("✅ Moderate / Safe")
