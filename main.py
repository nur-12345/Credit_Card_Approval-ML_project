from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

# Load pre-trained model and scaler
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

class CreditApprovalRequest(BaseModel):
    Gender: int
    Age: float
    Debt: float
    Married: int
    BankCustomer: int
    Industry: int
    Ethnicity: int
    YearsEmployed: float
    PriorDefault: int
    Employed: int
    CreditScore: int
    DriversLicense: int
    Citizen: int
    ZipCode: int
    Income: int

@app.post("/predict")
def predict_approval(data: CreditApprovalRequest):
    input_data = np.array([[data.Gender, data.Age, data.Debt, data.Married, data.BankCustomer,
                            data.Industry, data.Ethnicity, data.YearsEmployed, data.PriorDefault,
                            data.Employed, data.CreditScore, data.DriversLicense,
                            data.Citizen, data.ZipCode, data.Income]])
    
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    result = "Approved" if prediction[0] == 1 else "Not Approved"
    return {"prediction": result}



