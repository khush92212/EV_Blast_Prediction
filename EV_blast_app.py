import streamlit as st
import pandas as pd
import joblib

# ---------------- LOAD MODEL & ENCODERS ----------------
model = joblib.load("dtc_ev_model.pkl")
encoders = joblib.load("label_encoder_ev.pkl")

st.title("EV Blast Prediction")

# ---------------- CREATE INPUT FIELDS DYNAMICALLY ----------------
inputs = {}

# Create dropdowns based on model features
for col in model.feature_names_in_:
    if col in encoders:
        inputs[col] = st.selectbox(col, encoders[col].classes_)
    else:
        st.error(f"Encoder missing for column: {col}")
        st.stop()

# ---------------- CREATE INPUT DATAFRAME ----------------
input_data = pd.DataFrame([inputs])

# ---------------- ENCODE INPUT DATA ----------------
for col in input_data.columns:
    input_data[col] = encoders[col].transform(input_data[col])

# ---------------- PREDICTION ----------------
if st.button("Predict"):

    prediction = model.predict(input_data)[0]

    # If your model was trained with 1 = Blast
    if prediction == 1:
        st.error("⚠️ High Risk: Blast")
    else:
        st.success("✅ Moderate / Safe Condition")
