import streamlit as st
import joblib
import pandas as pd

# loading

model_path = r'RandomForest_FraudModel.pkl'
loaded_model = joblib.load(model_path)

# app title

st.title("Healthcare Fraud Detection")

# user details input form

st.header("Enter Patient Information")
op_deductible = st.number_input("Outpatient Annual Deductible Amount", min_value=0.0)
ip_deductible = st.number_input("Inpatient Annual Deductible Amount", min_value=0.0)
renal_disease = st.selectbox("Renal Disease Indicator (1 = Yes, 0 = No)", [0, 1])
kidney_disease = st.selectbox("Chronic Kidney Disease (1 = Yes, 0 = No)", [0, 1])
county = st.number_input("County", min_value=0, step=1)
age = st.number_input("Age", min_value=0, step=1)
state = st.number_input("State", min_value=0, step=1)
heart_failure = st.selectbox("Chronic Heart Failure (1 = Yes, 0 = No)", [0, 1])
diabetes = st.selectbox("Chronic Diabetes (1 = Yes, 0 = No)", [0, 1])

# button for predicting fraud

if st.button("Predict Fraud"):

    # input data creation
    
    input_data = pd.DataFrame({
        'OPAnnualDeductibleAmt': [op_deductible],
        'IPAnnualDeductibleAmt': [ip_deductible],
        'RenalDiseaseIndicator': [renal_disease],
        'ChronicCond_KidneyDisease': [kidney_disease],
        'County': [county],
        'Age': [age],
        'State': [state],
        'ChronicCond_Heartfailure': [heart_failure],
        'ChronicCond_Diabetes': [diabetes]
    })

# prediction creation

    prediction = loaded_model.predict(input_data)
    probability = loaded_model.predict_proba(input_data)

# results query

    if prediction[0] == 1:
        st.error(f"The patient is likely involved in fraud with a probability of {probability[0][1]:.2f}.")
    else:
        st.success(f"The patient is not involved in fraud with a probability of {probability[0][0]:.2f}.")
