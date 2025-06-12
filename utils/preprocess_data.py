import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# Paths
INPUT_CSV = "data/latest_drone_log.csv"
OUTPUT_CSV = "data/processed_drone_features.csv"

# 🎯 Columns used during training
features = ['latitude', 'longitude', 'altitude',
            'wind_speed', 'wind_direction', 'temperature',
            'humidity', 'cloudcover', 'precipitation']

print("📄 Loading input data...")
df = pd.read_csv(INPUT_CSV, parse_dates=["time"])

# 🔧 Rename columns if needed (match training)
df = df.rename(columns={
    'wind_speed_x': 'wind_speed',
    'wind_direction_x': 'wind_direction',
    'temperature_x': 'temperature',
    'humidity_x': 'humidity'
})

# ✅ Filter to required features
df_features = df[features].copy()

print("⚖️ Standardizing features...")
scaler = StandardScaler()
scaled_array = scaler.fit_transform(df_features)
df_scaled = pd.DataFrame(scaled_array, columns=features)

# 💾 Save processed features
df_scaled.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Processed features saved to: {OUTPUT_CSV}")
