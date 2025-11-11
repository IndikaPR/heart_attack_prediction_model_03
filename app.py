app_code = '''import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

@st.cache_resource
def load_resources():
    model = load_model('heart_attack_model.h5')
    scaler = joblib.load('scaler.pkl')
    encoders = joblib.load('encoders.pkl')
    cols = joblib.load('columns.pkl')
    return model, scaler, encoders, cols

model, scaler, encoders, columns = load_resources()

st.title("Heart Attack Risk Predictor (Sri Lanka)")
st.write("Enter patient details – **probability 0‑100%**")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 60, 30)
    gender = st.selectbox("Gender", ["Male","Female"])
    region = st.selectbox("Region", ["North","South","East","West","Central","North-East"])
    urban_rural = st.selectbox("Urban/Rural", ["Urban","Rural"])
    ses = st.selectbox("SES", ["Low","Middle","High"])
    smoking = st.selectbox("Smoking", ["Never","Occasionally","Regularly"])
    alcohol = st.selectbox("Alcohol", ["Never","Occasionally","Regularly"])
    diet = st.selectbox("Diet", ["Vegetarian","Non-Vegetarian","Vegan"])

with col2:
    activity = st.selectbox("Activity", ["Sedentary","Moderate","High"])
    screen_time = st.slider("Screen Time (hrs)", 0, 16, 6)
    sleep = st.slider("Sleep (hrs)", 3, 12, 7)
    family_hx = st.selectbox("Family Hx", ["Yes","No"])
    diabetes = st.selectbox("Diabetes", ["Yes","No"])
    hypertension = st.selectbox("Hypertension", ["Yes","No"])
    cholesterol = st.slider("Cholesterol (mg/dL)", 100, 400, 180)
    bmi = st.slider("BMI", 15.0, 50.0, 24.0, step=0.1)

st.markdown("---")
col3, col4 = st.columns(2)
with col3:
    resting_hr = st.slider("Resting HR (bpm)", 50, 120, 75)
    ecg = st.selectbox("ECG", ["Normal","Abnormal"])
    chest_pain = st.selectbox("Chest Pain", ["Typical","Atypical","Non-anginal","Asymptomatic"])

with col4:
    max_hr = st.slider("Max HR", 80, 220, 150)
    angina = st.selectbox("Angina", ["Yes","No"])
    spo2 = st.slider("SpO2 (%)", 85.0, 100.0, 96.0, step=0.1)
    triglycerides = st.slider("Triglycerides (mg/dL)", 50, 500, 150)
    systolic = st.slider("Systolic BP", 90, 200, 120)
    diastolic = st.slider("Diastolic BP", 60, 120, 80)

stress = None
if 'Stress Level' in columns:
    stress = st.selectbox("Stress Level", ["Low","Medium","High"])

if st.button("Predict Risk", type="primary"):
    data = {
        'Age':age,'Gender':gender,'Region':region,'Urban/Rural':urban_rural,
        'SES':ses,'Smoking Status':smoking,'Alcohol Consumption':alcohol,
        'Diet Type':diet,'Physical Activity Level':activity,
        'Screen Time (hrs/day)':screen_time,'Sleep Duration (hrs/day)':sleep,
        'Family History of Heart Disease':family_hx,'Diabetes':diabetes,
        'Hypertension':hypertension,'Cholesterol Levels (mg/dL)':cholesterol,
        'BMI (kg/m²)':bmi,'Resting Heart Rate (bpm)':resting_hr,
        'ECG Results':ecg,'Chest Pain Type':chest_pain,
        'Maximum Heart Rate Achieved':max_hr,'Exercise Induced Angina':angina,
        'Blood Oxygen Levels (SpO2%)':spo2,'Triglyceride Levels (mg/dL)':triglycerides,
        'Systolic_BP':systolic,'Diastolic_BP':diastolic
    }
    if stress is not None:
        data['Stress Level'] = stress

    df_in = pd.DataFrame([data])
    df_in = df_in.reindex(columns=columns, fill_value=0)

    for col in df_in.select_dtypes('object').columns:
    le = encoders.get(col)  # ← Fixed: added parentheses
    if le:
        df_in[col] = df_in[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    X_in = scaler.transform(df_in)
    prob = float(model.predict(X_in, verbose=0)[0][0])
    prob = np.clip(prob, 0.0, 1.0)

    risk = "HIGH RISK" if prob > 0.5 else "LOW RISK"
    st.markdown(f"### **Risk Probability: {prob:.1%}**")
    st.markdown(f"### **Prediction: {risk}**")
    if prob > 0.5:
        st.error("**HIGH RISK – URGENT CARDIOLOGY REFERRAL!**")
    else:
        st.success("Low risk – continue monitoring.")
'''

with open('app.py', 'w') as f:
    f.write(app_code)

files.download('app.py')

print("app.py generated and downloaded")
