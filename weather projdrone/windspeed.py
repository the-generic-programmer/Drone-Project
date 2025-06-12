import numpy as np
import math

def rotate_vector_by_quaternion(v, q):
    """
    Rotate vector v using quaternion q.
    """
    x, y, z, w = q
    q_vec = np.array([x, y, z])
    uv = np.cross(q_vec, v)
    uuv = np.cross(q_vec, uv)
    return v + 2 * (w * uv + uuv)

def estimate_wind_vector(gps_velocity, imu_acceleration, orientation_quat, dt):
    """
    Estimate wind vector using GPS velocity and IMU-derived air velocity.
    """
    # Estimate air-relative velocity (body frame to world frame)
    air_velocity_body = imu_acceleration * dt
    air_velocity_world = rotate_vector_by_quaternion(air_velocity_body, orientation_quat)

    # Wind = Ground velocity (GPS) - Air-relative velocity (IMU)
    wind_vector = gps_velocity - air_velocity_world

    # Calculate horizontal wind speed (ignoring z)
    wind_speed = np.linalg.norm(wind_vector[:2])

    # Calculate wind direction (from North, clockwise)
    wind_direction = (math.degrees(math.atan2(wind_vector[1], wind_vector[0])) + 360) % 360

    return wind_vector, wind_speed, wind_direction

def get_vector_input(name):
    vals = input(f"Enter {name} components (comma-separated, e.g. x,y,z): ").split(",")
    return np.array([float(v.strip()) for v in vals])

def get_quaternion_input():
    vals = input("Enter orientation quaternion (x,y,z,w): ").split(",")
    return np.array([float(v.strip()) for v in vals])

np.set_printoptions(precision=2, suppress=False, floatmode='fixed')

def main():
    print("=== Drone Wind Estimation Using IMU and GPS ===")

    imu_accel = get_vector_input("IMU acceleration (m/s^2)")
    gps_velocity = get_vector_input("GPS velocity (m/s)")
    quat = get_quaternion_input()
    dt = float(input("Enter time step Δt (in seconds): "))

    wind_vector, wind_speed, wind_dir = estimate_wind_vector(gps_velocity, imu_accel, quat, dt)

    print("\n--- Results ---")
    print(f"Estimated Wind Vector: {wind_vector}")
    print(f"Wind Speed: {wind_speed:.2f} m/s")
    print(f"Wind Direction: {wind_dir:.2f}° (from North, clockwise)")

if __name__ == "__main__":
    main()
