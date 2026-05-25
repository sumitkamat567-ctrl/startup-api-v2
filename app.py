from fastapi import FastAPI
import joblib
import pandas as pd

# Load model
model = joblib.load("model/startup_success_model.pkl")

# Create app
app = FastAPI()

# Home route
@app.get("/")
def home():
    return {"message": "Startup Success Prediction API Running Successfully"}

# Prediction route
@app.post("/predict")
def predict(
    amount_log: float,
    year: int,
    month: int,
    num_investors: int,
    has_top_vc: int,
    City: str,
    SubVertical: str,
    Industry: str
):

    data = pd.DataFrame({
        "amount_log": [amount_log],
        "year": [year],
        "month": [month],
        "num_investors": [num_investors],
        "has_top_vc": [has_top_vc],
        "City": [City],
        "SubVertical": [SubVertical],
        "Industry": [Industry]
    })

    prediction = model.predict(data)[0]

    return {
        "prediction": int(prediction)
    }