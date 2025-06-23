# Credit_Card_Approval-ML_project
# 💳 Credit Approval Prediction System  > A complete end-to-end Machine Learning + Web App project that predicts whether a person will be approved for credit based on their input information.
---

## 🧠 Use Case

Credit card companies and banks need a reliable system to evaluate whether an applicant is **eligible for credit approval**.  
This project automates the process using a trained **Machine Learning model** and provides predictions via a clean **Streamlit frontend** and a **FastAPI backend**.

---

## 🚀 Tech Stack

| Layer      | Tech Used                         |
|------------|----------------------------------|
| ML Model   | Scikit-learn, Pandas, Joblib     |
| Backend    | FastAPI, NumPy                   |
| Frontend   | Streamlit, Requests              |
| Deployment | Localhost (Dev)                  |

---

## 📁 Project Structure

credit-approval-prediction/
├── app.py # ML model training & evaluation
├── main.py # FastAPI backend to serve predictions
├── streamlit.py # Streamlit frontend UI
├── rf_model.pkl # Saved trained Random Forest model
├── scaler.pkl # Saved StandardScaler object
├── clean_dataset.csv # Cleaned dataset used for training
├── requirements.txt # All required Python dependencies
└── README.md # You're reading it 😄


---

## 🎯 Features

- 🔍 Clean dataset processing & visualizations
- 🧠 ML Models:
  - Logistic Regression
  - Random Forest Classifier ✅ (Best Accuracy)
  - Support Vector Machine
- 🧪 Accuracy comparison of models
- 🔗 REST API endpoint for predictions
- 🖥️ User-friendly Streamlit frontend
- 💡 JSON-based request/response format

---

## 🛠️ How to Run This Project

### ✅ Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/credit-approval-prediction.git
cd credit-approval-prediction
✅ Step 2: Create & Activate Virtual Environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
✅ Step 3: Install Dependencies
pip install -r requirements.txt
✅ Step 4: Train the Model
python3 app.py
This will generate:

rf_model.pkl (trained model)
scaler.pkl (standard scaler)
✅ Step 5: Run FastAPI Backend
uvicorn main:app --reload
Runs on: http://localhost:8000
Swagger docs: http://localhost:8000/docs
✅ Step 6: Run Streamlit Frontend
streamlit run streamlit.py
Runs on: http://localhost:8501
Fill in the form and get the prediction 🚀
🔁 API Example Request

📍 Endpoint: POST http://localhost:8000/predict

🔹 JSON Request Body
{
  "Gender": 1,
  "Age": 28.5,
  "Debt": 3.4,
  "Married": 0,
  "BankCustomer": 1,
  "Industry": 1,
  "Ethnicity": 1,
  "YearsEmployed": 2.8,
  "PriorDefault": 0,
  "Employed": 1,
  "CreditScore": 680,
  "DriversLicense": 0,
  "Citizen": 0,
  "ZipCode": 400001,
  "Income": 55000
}
🔹 Sample Response
{
  "prediction": "Approved"
}
📊 Accuracy Comparison of ML Models

Model	Accuracy (%)
Logistic Regression	✅ ~83%
Random Forest	⭐ Best ~91%
Support Vector Machine	~85%
Bar chart visualization is generated in app.py.

🖼️ UI Preview (Streamlit)

📦 requirements.txt

pandas
numpy
seaborn
matplotlib
scikit-learn
joblib
fastapi
uvicorn
streamlit
requests

🙋‍♀️ Author
Nupur Shivani
✨ Passionate about Machine Learning and DevOps.

📄 License

This project is free to use for learning, showcasing, and educational purposes.


---

