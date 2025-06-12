import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

# === Part 1: Wind Vector Estimation ===

def rotate_body_to_world(v, q):
    # Quaternion rotation of vector v (body to world frame)
    x, y, z, w = q
    q_vec = np.array([x, y, z])
    uv = np.cross(q_vec, v)
    uuv = np.cross(q_vec, uv)
    return v + 2 * (w * uv + uuv)

class WindEstimator:
    def __init__(self, alpha=0.85):
        self.estimated = np.zeros(3)
        self.alpha = alpha

    def update(self, gps_vel, imu_acc, quat, dt):
        air_vel_body = imu_acc * dt
        air_vel_world = rotate_body_to_world(air_vel_body, quat)
        raw_wind = gps_vel - air_vel_world
        self.estimated = self.alpha * self.estimated + (1 - self.alpha) * raw_wind
        return self.estimated

# === Part 2: Synthetic Dataset Generation for Training ===

def generate_synthetic_data(n_samples=1000, random_seed=42):
    np.random.seed(random_seed)
    
    # Random temperature [°C], Humidity [%]
    temp = np.random.uniform(-10, 40, n_samples)
    humidity = np.random.uniform(10, 90, n_samples)
    
    # Random wind speed [0, 15 m/s], wind direction [0, 360 degrees]
    wind_speed = np.random.uniform(0, 15, n_samples)
    wind_dir = np.random.uniform(0, 360, n_samples)

    # Construct features matrix
    X = np.column_stack((temp, humidity, wind_speed, wind_dir))
    
    # Risk logic (simple heuristic): 
    # High wind speed (>10 m/s) or extreme temp (<0 or >35) or low humidity (<20) → unsafe (1), else safe (0)
    risk = ((wind_speed > 10) | (temp < 0) | (temp > 35) | (humidity < 20)).astype(int)
    
    return X, risk

# === Part 3: Predictive Model Training ===

def train_predictive_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]  # Risk score (probability of unsafe)
    
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.3f}")
    
    return model, scaler

# === Part 4: Real-time Prediction Example Using Drone Data ===

def drone_predict_safe_condition(temp, humidity, imu_acc, gps_vel, quat, dt, model, scaler, wind_estimator):
    wind_vector = wind_estimator.update(gps_vel, imu_acc, quat, dt)
    wind_speed = np.linalg.norm(wind_vector[:2])
    wind_dir = (math.degrees(math.atan2(wind_vector[1], wind_vector[0])) + 360) % 360
    
    features = np.array([[temp, humidity, wind_speed, wind_dir]])
    features_scaled = scaler.transform(features)
    
    risk_score = model.predict_proba(features_scaled)[0,1]
    safe = risk_score < 0.5
    
    return {
        "wind_vector": wind_vector,
        "wind_speed": wind_speed,
        "wind_direction": wind_dir,
        "risk_score": risk_score,
        "safe": safe
    }

# === Part 5: Main Flow ===

def main():
    # Generate training data
    X, y = generate_synthetic_data(n_samples=2000)
    
    # Train model
    model, scaler = train_predictive_model(X, y)
    
    # Initialize wind estimator
    wind_estimator = WindEstimator(alpha=0.85)
    
    # Simulated drone sensor input (example)
    temp = 30.0       # °C
    humidity = 25.0   # %
    imu_acc = np.array([0.1, 0.0, 0.0])  # m/s^2
    gps_vel = np.array([5.0, 1.0, 0.0])  # m/s
    quat = np.array([0.0, 0.0, 0.0, 1.0]) # No rotation quaternion
    dt = 1.0           # 1 second timestep
    
    # Predict safety and risk score
    result = drone_predict_safe_condition(temp, humidity, imu_acc, gps_vel, quat, dt, model, scaler, wind_estimator)
    
    print("\n=== Drone Safety Prediction ===")
    print(f"Estimated Wind Vector: {result['wind_vector']}")
    print(f"Wind Speed: {result['wind_speed']:.2f} m/s")
    print(f"Wind Direction: {result['wind_direction']:.2f}°")
    print(f"Risk Score (0=safe, 1=unsafe): {result['risk_score']:.3f}")
    print(f"Safe to Fly: {'YES' if result['safe'] else 'NO'}")

if __name__ == "__main__":
    main()
