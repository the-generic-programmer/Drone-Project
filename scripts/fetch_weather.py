import requests
import pandas as pd
from datetime import datetime, timedelta
import os

# Config
LAT = 28.6139   # Example: New Delhi
LON = 77.2090
START_DATE = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
END_DATE = datetime.now().strftime('%Y-%m-%d')

# Open-Meteo API URL
URL = (
    "https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={LAT}&longitude={LON}"
    f"&start_date={START_DATE}&end_date={END_DATE}"
    "&hourly=temperature_2m,relative_humidity_2m,cloudcover,precipitation,"
    "windspeed_10m,winddirection_10m"
    "&timezone=Asia%2FKolkata"
)

def fetch_and_save():
    print(f"Fetching weather data for {START_DATE} to {END_DATE}...")
    response = requests.get(URL)

    if response.status_code != 200:
        print("Failed to fetch weather data.")
        print("Status:", response.status_code)
        return

    data = response.json()

    df = pd.DataFrame({
        'time': data['hourly']['time'],
        'temperature': data['hourly']['temperature_2m'],
        'humidity': data['hourly']['relative_humidity_2m'],
        'cloudcover': data['hourly']['cloudcover'],
        'precipitation': data['hourly']['precipitation'],
        'wind_speed': data['hourly']['windspeed_10m'],
        'wind_direction': data['hourly']['winddirection_10m']
    })

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/weather_history.csv", index=False)
    print("✅ Weather data saved to data/weather_history.csv")

if __name__ == "__main__":
    fetch_and_save()