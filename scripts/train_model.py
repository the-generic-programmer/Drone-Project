import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

print("📄 Loading input CSV...")

# Load input data
try:
    df = pd.read_csv("data/merged_data.csv", parse_dates=['time'])
except FileNotFoundError:
    print("❌ Error: File not found. Please ensure 'data/merged_data.csv' exists.")
    exit()

print(f"📊 CSV loaded — Columns are: {df.columns.tolist()}")
print(f"📊 First 5 rows:\n{df.head()}\n")

# Sort by time just in case
df = df.sort_values('time')

# Rename for consistency
df.rename(columns={
    'temperature_x': 'temperature',
    'humidity_x': 'humidity',
    'wind_speed_x': 'wind_speed',
    'wind_direction_x': 'wind_direction'
}, inplace=True)

# Create future labels for prediction (12 hours ahead)
print("🕒 Creating future target columns (12-hour shift)...")
shift_period = 12  # assuming hourly data

df['future_temperature'] = df['temperature'].shift(-shift_period)
df['future_humidity'] = df['humidity'].shift(-shift_period)
df['future_wind_speed'] = df['wind_speed'].shift(-shift_period)

# Drop rows with missing future values
df.dropna(subset=['future_temperature', 'future_humidity', 'future_wind_speed'], inplace=True)

print(f"🔍 Columns in dataset: {df.columns.tolist()}")
print(f"🔍 First few rows:\n{df.head()}\n")

# Define features and labels
feature_cols = [
    'latitude', 'longitude', 'altitude',
    'wind_speed', 'wind_direction',
    'temperature', 'humidity',
    'cloudcover', 'precipitation'
]

target_cols = [
    'future_temperature',
    'future_humidity',
    'future_wind_speed'
]

# Check that all required columns exist
missing_features = [col for col in feature_cols if col not in df.columns]
missing_targets = [col for col in target_cols if col not in df.columns]

if missing_features or missing_targets:
    print("❌ Missing columns:")
    if missing_features:
        print(f"  - Features: {missing_features}")
    if missing_targets:
        print(f"  - Targets: {missing_targets}")
    exit()

# Split data
X = df[feature_cols]
y = df[target_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🚀 Training model...")
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

print("✅ Model trained.")

y_pred = model.predict(X_test)

# Evaluate
print("\n📈 Evaluation:")
for i, col in enumerate(target_cols):
    mse = mean_squared_error(y_test[col], y_pred[:, i])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test[col], y_pred[:, i])
    print(f"{col} — RMSE: {rmse:.2f}, MAE: {mae:.2f}")

# Save model
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "weather_predictor.pkl")
joblib.dump(model, model_path)

print(f"\n💾 Model saved to: {model_path}")