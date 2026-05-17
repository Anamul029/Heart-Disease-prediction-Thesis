import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ১. মডেল এবং স্কেলার লোড করা
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("❤️ Heart Disease Prediction App")
st.write("দয়া করে নিচের তথ্যগুলো পূরণ করুন:")

# ২. ইউজারের কাছ থেকে ইনপুট নেওয়া (Form তৈরি করা)
age = st.number_input("Age (বয়স)", min_value=1, max_value=120, value=30)
gender = st.selectbox("Gender", options=["Male", "Female"])
chest_pain = st.selectbox("Chest Pain Type", options=["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
blood_pressure = st.number_input("Blood Pressure (রক্তচাপ)", min_value=50, max_value=250, value=120)
cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=600, value=200)
blood_sugar = st.selectbox("Blood Sugar > 120 mg/dl?", options=["No", "Yes"])
heart_rate = st.number_input("Maximum Heart Rate", min_value=60, max_value=220, value=150)
smoking = st.selectbox("Smoking Status", options=["Never", "Former", "Current"])
alcohol = st.selectbox("Alcohol Intake", options=["Never", "Moderate", "Heavy"])
family_history = st.selectbox("Family History of Heart Disease?", options=["No", "Yes"])
diabetes = st.selectbox("Diabetes Status", options=["No", "Yes"])
obesity = st.selectbox("Obesity Status", options=["No", "Yes"])
stress_level = st.slider("Stress Level (১-১০)", 1, 10, 5)
angina = st.selectbox("Exercise Induced Angina?", options=["No", "Yes"])

# ৩. 'Predict' বাটনে ক্লিক করলে যা হবে
if st.button("Predict"):
    # আপনার মডেলের এনকোডিং ম্যাপিং অনুযায়ী ইনপুট ডেটাকে সংখ্যায় রূপান্তর (আপনার নোটবুকের ম্যাপিং অনুযায়ী)
    input_data = {
        'Age': age,
        'Gender': 1 if gender == "Male" else 0,
        'Chest Pain Type': ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(chest_pain),
        'Blood Pressure': blood_pressure,
        'Cholesterol': cholesterol,
        'Blood Sugar': 1 if blood_sugar == "Yes" else 0,
        'Heart Rate': heart_rate,
        'Smoking': ["Never", "Former", "Current"].index(smoking),
        'Alcohol Intake': ["Never", "Moderate", "Heavy"].index(alcohol),
        'Family History': 1 if family_history == "Yes" else 0,
        'Diabetes': 1 if diabetes == "Yes" else 0,
        'Obesity': 1 if obesity == "Yes" else 0,
        'Stress Level': stress_level,
        'Exercise Induced Angina': 1 if angina == "Yes" else 0
    }
    
    # ডেটাফ্রেমে রূপান্তর
    df_input = pd.DataFrame([input_data])
    
    # নিউমেরিকাল কলামগুলোকে স্কেল করা
    numerical_cols = ['Age', 'Cholesterol', 'Blood Pressure', 'Heart Rate', 'Stress Level', 'Blood Sugar']
    df_input[numerical_cols] = scaler.transform(df_input[numerical_cols])
    
    # প্রেডিকশন
    prediction = model.predict(df_input)
    
    # ফলাফল দেখানো
    if prediction[0] == 1:
        st.error("⚠️ সতর্কবার্তা: মডেলের প্রেডিকশন অনুযায়ী হার্ট ডিজিজের ঝুঁকি রয়েছে। চিকিৎসকের পরামর্শ নিন।")
    else:
        st.success("✅ অভিনন্দন! মডেল অনুযায়ী আপনার হার্ট ডিজিজের ঝুঁকি কম।")