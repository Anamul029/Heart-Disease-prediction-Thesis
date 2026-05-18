import streamlit as st
import pandas as pd
import numpy as np
import joblib


# -------------------------------------------------------------------------
# 1. Load the Trained Model and Scaler
# -------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("heart_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error(
        "Error: 'heart_model.pkl' or 'scaler.pkl' not found. Please run your Jupyter notebook first."
    )
    st.stop()

# -------------------------------------------------------------------------
# 2. UI Configuration and Header
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Predictor", page_icon="❤️", layout="centered"
)

st.title("❤️ Heart Disease Prediction Interface")
st.write(
    "Input patient health metrics below to predict the likelihood of heart disease."
)
st.markdown("---")

# -------------------------------------------------------------------------
# 3. User Inputs (Matching Training Feature Mapping)
# -------------------------------------------------------------------------
st.subheader("📋 Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    gender = st.selectbox("Gender", options=["Female", "Male"])
    cholesterol = st.number_input(
        "Cholesterol Level (mg/dL)", min_value=100, max_value=500, value=200
    )
    blood_pressure = st.number_input(
        "Blood Pressure (systolic)", min_value=80, max_value=250, value=120
    )
    heart_rate = st.number_input(
        "Heart Rate (bpm)", min_value=40, max_value=220, value=75
    )
    blood_sugar = st.number_input(
        "Blood Sugar Level (mg/dL)", min_value=50, max_value=300, value=100
    )

with col2:
    smoking = st.selectbox("Smoking Status", options=["Current", "Former", "Never"])
    alcohol = st.selectbox("Alcohol Intake", options=["Heavy", "Moderate", "Never"])
    family_history = st.selectbox(
        "Family History of Heart Disease", options=["No", "Yes"]
    )
    diabetes = st.selectbox("Diabetes", options=["No", "Yes"])
    obesity = st.selectbox("Obesity", options=["No", "Yes"])
    stress_level = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=5)

st.markdown("---")
st.subheader("🫀 Clinical Symptoms")

col3, col4 = st.columns(2)

with col3:
    chest_pain = st.selectbox(
        "Chest Pain Type",
        options=[
            "Asymptomatic",
            "Atypical Angina",
            "Non-anginal Pain",
            "Typical Angina",
        ],
    )

with col4:
    angina = st.selectbox("Exercise Induced Angina", options=["No", "Yes"])

# -------------------------------------------------------------------------
# 4. Data Preprocessing (Encoding & Scaling)
# -------------------------------------------------------------------------
# Manual Label Encoding Map based exactly on your Notebook output
mapping_gender = {"Female": 0, "Male": 1}
mapping_smoking = {"Current": 0, "Former": 1, "Never": 2}
mapping_alcohol = {"Heavy": 0, "Moderate": 1, "Never": 2}
mapping_binary = {"No": 0, "Yes": 1}
mapping_chest_pain = {
    "Asymptomatic": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Typical Angina": 3,
}

# Create a DataFrame of raw inputs matching original column order
input_data = pd.DataFrame(
    [
        {
            "Age": age,
            "Gender": mapping_gender[gender],
            "Cholesterol": cholesterol,
            "Blood Pressure": blood_pressure,
            "Heart Rate": heart_rate,
            "Smoking": mapping_smoking[smoking],
            "Alcohol Intake": mapping_alcohol[alcohol],
            "Exercise Hours": 0,  # Hidden baseline if missing from inputs but part of your features
            "Family History": mapping_binary[family_history],
            "Diabetes": mapping_binary[diabetes],
            "Obesity": mapping_binary[obesity],
            "Stress Level": stress_level,
            "Blood Sugar": blood_sugar,
            "Exercise Induced Angina": mapping_binary[angina],
            "Chest Pain Type": mapping_chest_pain[chest_pain],
        }
    ]
)

# Extract numerical columns for scaling
numerical_cols = [
    "Age",
    "Cholesterol",
    "Blood Pressure",
    "Heart Rate",
    "Stress Level",
    "Blood Sugar",
]

# Apply the pre-trained scaler
input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

# Ensure exact column order matches X_train feature layout
# (Excluding 'Heart Disease' target column)
features_order = [
    "Age",
    "Gender",
    "Cholesterol",
    "Blood Pressure",
    "Heart Rate",
    "Smoking",
    "Alcohol Intake",
    "Exercise Hours",
    "Family History",
    "Diabetes",
    "Obesity",
    "Stress Level",
    "Blood Sugar",
    "Exercise Induced Angina",
    "Chest Pain Type",
]
input_features = input_data[features_order]

# -------------------------------------------------------------------------
# 5. Prediction Execution
# -------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button(
    "🔍 Predict Heart Disease Status", type="primary", use_container_width=True
):
    prediction = model.predict(input_features)[0]

    # Optional: Get prediction probabilities if the model supports it
    try:
        probabilities = model.predict_proba(input_features)[0]
        confidence = probabilities[prediction] * 100
        prob_str = f" (Confidence: {confidence:.2f}%)"
    except AttributeError:
        prob_str = ""

    st.markdown("### **Prediction Results:**")
    if prediction == 1:
        st.error(
            f"🚨 **High Risk Detected:** The model indicates a presence of Heart Disease.{prob_str}"
        )
    else:
        st.success(
            f"✅ **Low Risk Detected:** The model indicates no current sign of Heart Disease.{prob_str}"
        )
