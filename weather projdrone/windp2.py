import numpy as np
import math

# Rotate a vector from body to world frame using quaternion
def rotate_body_to_world(v, q):
    x, y, z, w = q
    q_vec = np.array([x, y, z])
    uv = np.cross(q_vec, v)
    uuv = np.cross(q_vec, uv)
    return v + 2 * (w * uv + uuv)

# Complementary filter wind estimator
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

# Helper to read a 3D vector
def parse_vec(prompt):
    return np.array([float(v.strip()) for v in input(prompt).split(",")])

def main():
    print("=== Drift-Method Wind Estimation ===")

    imu_acc = parse_vec("Enter IMU acceleration (ax, ay, az) in m/s^2: ")
    gps_vel = parse_vec("Enter GPS velocity (vx, vy, vz) in m/s: ")

    # Optional quaternion input
    use_default_q = input("Use default quaternion [0, 0, 0, 1]? (Y/n): ").strip().lower()
    if use_default_q in ["n", "no"]:
        quat = parse_vec("Enter quaternion (x, y, z, w): ")
    else:
        quat = np.array([0.0, 0.0, 0.0, 1.0])

    # Optional time step input
    use_default_dt = input("Use default time step Δt = 1.0s? (Y/n): ").strip().lower()
    if use_default_dt in ["n", "no"]:
        dt = float(input("Enter timestep Δt (s): "))
    else:
        dt = 1.0

    # Estimate wind
    estimator = WindEstimator(alpha=0.85)
    wind_vector = estimator.update(gps_vel, imu_acc, quat, dt)
    wind_speed = np.linalg.norm(wind_vector[:2])
    wind_dir = (math.degrees(math.atan2(wind_vector[1], wind_vector[0])) + 360) % 360

    # Output formatting
    np.set_printoptions(precision=2, floatmode="fixed", suppress=False)

    print("\n--- Wind Estimation ---")
    print(f"Estimated Wind Vector: {wind_vector}")
    print(f"Wind Speed: {wind_speed:.2f} m/s")
    print(f"Wind Direction: {wind_dir:.2f}° (from North, clockwise)")

if __name__ == "__main__":
    main()
