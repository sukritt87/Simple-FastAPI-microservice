from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(
    title="California Housing Prediction API",
    description="Predicts median housing price using a regression model",
    version="1.0"
)

# Load model + scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

class HousingInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.get("/")
def home():
    return {"message": "Service is running. Go to /docs"}

@app.post("/predict")
def predict(data: HousingInput):
    input_data = np.array([[
        data.MedInc,
        data.HouseAge,
        data.AveRooms,
        data.AveBedrms,
        data.Population,
        data.AveOccup,
        data.Latitude,
        data.Longitude
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    return {
        "predicted_median_house_value": round(float(prediction), 4)
    }