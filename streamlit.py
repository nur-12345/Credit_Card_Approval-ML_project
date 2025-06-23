import streamlit as st
import requests

st.title("Credit Approval Prediction")

# User input form
with st.form("credit_form"):
    Gender = st.selectbox("Gender", [0, 1])
    Age = st.number_input("Age", min_value=0.0)
    Debt = st.number_input("Debt", min_value=0.0)
    Married = st.selectbox("Married", [0, 1])
    BankCustomer = st.selectbox("Bank Customer", [0, 1])
    Industry = st.selectbox("Industry", [0, 1, 2, 3])  # Change if needed
    Ethnicity = st.selectbox("Ethnicity", [0, 1, 2])   # Change if needed
    YearsEmployed = st.number_input("Years Employed", min_value=0.0)
    PriorDefault = st.selectbox("Prior Default", [0, 1])
    Employed = st.selectbox("Employed", [0, 1])
    CreditScore = st.number_input("Credit Score", min_value=0)
    DriversLicense = st.selectbox("Driver's License", [0, 1])
    Citizen = st.selectbox("Citizen", [0, 1, 2])
    ZipCode = st.number_input("Zip Code", min_value=0)
    Income = st.number_input("Income", min_value=0)

    submit = st.form_submit_button("Predict")

# Call API on form submission
if submit:
    input_data = {
        "Gender": Gender,
        "Age": Age,
        "Debt": Debt,
        "Married": Married,
        "BankCustomer": BankCustomer,
        "Industry": Industry,
        "Ethnicity": Ethnicity,
        "YearsEmployed": YearsEmployed,
        "PriorDefault": PriorDefault,
        "Employed": Employed,
        "CreditScore": CreditScore,
        "DriversLicense": DriversLicense,
        "Citizen": Citizen,
        "ZipCode": ZipCode,
        "Income": Income
    }

    try:
        response = requests.post("http://localhost:8000/predict", json=input_data)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Prediction: {result['prediction']}")
        else:
            st.error("Something went wrong with the prediction.")
    except Exception as e:
        st.error(f"Error: {e}")
