import pandas as pd
import numpy as np
from joblib import load
import requests
from datetime import datetime

# Load models
regressor = load("models/weather_predictor.pkl")
classifier = load("models/flight_condition_model.pkl")
scaler = load("models/scaler.pkl")

# Example: Current drone input (could be from ROS2 later)
drone_data = {
    "temperature": 33.0,
    "humidity": 55.0,
    "cloudcover": 20.0,
    "precipitation": 0.0,
    "wind_speed": 12.0,
    "wind_direction": 90.0,
    "latitude": 28.6139,
    "longitude": 77.2090,
    "altitude": 120.0
}

# Convert to DataFrame
X_input = pd.DataFrame([drone_data])
X_scaled = scaler.transform(X_input)

# Make predictions
forecast = regressor.predict(X_scaled)[0]
flight_status = classifier.predict(X_scaled)[0]

# Format forecast output
forecast_dict = {
    "future_temperature": round(forecast[0], 2),
    "future_humidity": round(forecast[1], 2),
    "future_wind_speed": round(forecast[2], 2),
}

# Display result
print("\n📡 Drone Weather Forecast (next 3 hours):")
for key, val in forecast_dict.items():
    print(f"{key}: {val}")

print(f"\n🚁 Flight Recommendation: {flight_status.upper()}")
