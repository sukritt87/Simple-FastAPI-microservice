from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import joblib
import numpy as np

app = FastAPI(
    title="California Housing Price Prediction Service",
    description="Microservice that predicts median California housing value using a PyTorch regression model.",
    version="1.0"
)

# Define the same model architecture used during training
class HousingRegressionModel(nn.Module):
    def __init__(self):
        super(HousingRegressionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

# Load model and scaler
model = HousingRegressionModel()
model.load_state_dict(torch.load("housing_model.pth", map_location=torch.device("cpu")))
model.eval()

scaler = joblib.load("scaler.pkl")

# Request schema
class HousingInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# Response schema
class PredictionOutput(BaseModel):
    predicted_median_house_value: float

@app.get("/")
def home():
    return {
        "message": "California Housing Price Prediction Service is running.",
        "usage": "Send a POST request to /predict with 8 housing features as JSON.",
        "docs": "/docs"
    }

@app.post("/predict", response_model=PredictionOutput)
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
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    with torch.no_grad():
        prediction = model(input_tensor).item()

    return PredictionOutput(predicted_median_house_value=round(prediction, 4))