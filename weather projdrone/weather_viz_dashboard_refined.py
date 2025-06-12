import streamlit as st
import plotly.graph_objects as go
import numpy as np
import math

# Streamlit app configuration
st.set_page_config(page_title="Drone Weather Forecast Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for improved aesthetics
st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .stButton>button {background-color: #1e90ff; color: white; border-radius: 8px;}
    .stSlider {background-color: #ffffff; padding: 10px; border-radius: 8px;}
    h1 {color: #2c3e50;}
    h3 {color: #34495e;}
    .summary-box {background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);}
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🛩️ Drone Weather Forecast Dashboard")
st.markdown("Monitor weather conditions and flight safety for drone operations in real-time.")

# Sidebar for user inputs
st.sidebar.header("🎮 Weather Control Center")
st.sidebar.markdown("""
    ### ⚡ Adjust Your Drone's Environment
    
    Control these critical flight parameters:
    
    🌡️ **Temperature** (-10°C to 40°C)
    - Below 0°C: Risk of icing
    - 0-35°C: Optimal flying zone
    - Above 35°C: Risk of overheating
    
    💧 **Humidity** (0% to 100%)
    - Below 20%: Too dry, static risk
    - 20-80%: Perfect flying conditions
    - Above 80%: Risk of condensation
    
    🌪️ **Wind Speed** (0 to 15 m/s)
    - 0-5 m/s: Smooth sailing
    - 5-10 m/s: Moderate challenge
    - Above 10 m/s: High risk zone
    
    🧭 **Wind Direction** (0° to 360°)
    - Navigate like a compass!
    - 0° = North, 90° = East
    - 180° = South, 270° = West
    """)

# Each slider represents a different weather parameter:
temp = st.sidebar.slider("Temperature (°C)", 
    min_value=-10.0,    # Minimum temperature
    max_value=40.0,     # Maximum temperature
    value=25.0,         # Default value
    step=0.5,           # Increment step
    help="Temperature affects battery life and drone performance. Below 0°C risks icing, above 35°C risks overheating."  # Hover tooltip
)

humidity = st.sidebar.slider("Humidity (%)", 
    min_value=0.0,      # Minimum humidity
    max_value=100.0,    # Maximum humidity
    value=50.0,         # Default value
    step=1.0,           # Increment step
    help="High humidity (>80%) risks condensation on electronics. Low humidity (<20%) increases static electricity risk."
)

wind_speed = st.sidebar.slider("Wind Speed (m/s)", 
    min_value=0.0,      # Minimum speed
    max_value=15.0,     # Maximum speed
    value=5.0,          # Default value
    step=0.1,           # Increment step
    help="Wind speed impacts drone stability and battery consumption. Above 10 m/s is dangerous for most drones."
)

wind_dir = st.sidebar.slider("Wind Direction (° from North)", 
    min_value=0.0,      # Minimum angle
    max_value=360.0,    # Maximum angle
    value=0.0,          # Default value
    step=1.0,           # Increment step
    help="Direction wind is coming from. 0° = North, 90° = East, 180° = South, 270° = West. Affects flight path planning."
)

risk_score = st.sidebar.slider("Risk Score (0=safe, 1=unsafe)", 
    min_value=0.0,      # Minimum risk
    max_value=1.0,      # Maximum risk
    value=0.5,          # Default value
    step=0.01,          # Increment step
    help="Overall flight risk assessment. Below 0.3 is safe, 0.3-0.6 requires caution, above 0.6 is high risk."
)

# Helper function to get color based on value thresholds
def get_color(value, thresholds, colors):
    for threshold, color in zip(thresholds, colors):
        if value <= threshold:
            return color
    return colors[-1]

# Layout: Two rows - gauges in the first row, polar plot and summary in the second
st.markdown("### Weather Metrics")
col1, col2, col3 = st.columns(3)

# Temperature gauge
with col1:
    st.subheader("Temperature")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=temp,
        title={'text': "°C"},
        gauge={
            'axis': {'range': [-10, 40]},
            'bar': {'color': get_color(temp, [0, 35, 40], ['#00ff00', '#ffa500', '#ff0000'])},
            'steps': [
                {'range': [-10, 0], 'color': '#e6f3ff'},
                {'range': [0, 35], 'color': '#ccffcc'},
                {'range': [35, 40], 'color': '#ffcccc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'value': 35
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

# Humidity gauge
with col2:
    st.subheader("Humidity")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=humidity,
        title={'text': "%"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': get_color(humidity, [20, 100], ['#ff0000', '#00b7eb'])},
            'steps': [
                {'range': [0, 20], 'color': '#ffcccc'},
                {'range': [20, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'value': 20
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

# Risk score gauge
with col3:
    st.subheader("Flight Risk")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        title={'text': "Risk Score"},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': get_color(risk_score, [0.3, 0.6, 1.0], ['#00ff00', '#ffa500', '#ff0000'])},
            'steps': [
                {'range': [0, 0.3], 'color': '#ccffcc'},
                {'range': [0.3, 0.6], 'color': '#fff3cc'},
                {'range': [0.6, 1.0], 'color': '#ffcccc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'value': 0.5
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

# Wind speed and direction polar plot
st.markdown("### Wind Vector")
col_polar, col_summary = st.columns([2, 1])

with col_polar:
    st.subheader("Wind Speed & Direction")
    fig = go.Figure()
    wind_angle_rad = math.radians(wind_dir)
    fig.add_trace(go.Scatterpolar(
        r=[0, wind_speed],
        theta=[wind_dir, wind_dir],
        mode='lines+markers',
        line=dict(color='blue', width=4),
        marker=dict(size=10, symbol='circle')
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 15], showticklabels=False),
            angularaxis=dict(
                rotation=90,
                direction="clockwise",
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                ticktext=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
            )
        ),
        showlegend=False,
        height=400,
        margin=dict(t=20, b=20),
        annotations=[
            dict(
                x=0.5,
                y=0.1,
                xref="paper",
                yref="paper",
                text=f"Speed: {wind_speed:.1f} m/s<br>Direction: {wind_dir:.1f}°",
                showarrow=False,
                font=dict(size=12)
            )
        ]
    )
    st.plotly_chart(fig, use_container_width=True)

# Summary section
with col_summary:
    st.subheader("Flight Safety Summary")
    safe = risk_score < 0.5
    st.markdown(
        f"""
        <div class="summary-box" style="color: black;">
            <b>Safe to Fly</b>: {'✅ YES' if safe else '❌ NO'}<br>
            <b>Temperature</b>: {temp:.1f}°C<br>
            <b>Humidity</b>: {humidity:.1f}%<br>
            <b>Wind Speed</b>: {wind_speed:.1f} m/s<br>
            <b>Wind Direction</b>: {wind_dir:.1f}° from North<br>
            <b>Risk Score</b>: {risk_score:.2f} ({'Safe' if safe else 'Unsafe'})
        </div>
        """, unsafe_allow_html=True
    )

# Button to simulate drone action
if st.button("Simulate Drone Action"):
    st.write(f"Drone action: {'Proceed to safer location' if not safe else 'Continue flight'} based on risk score {risk_score:.2f}.")

#Local URL: http://localhost:8501
#Network URL: http://10.2.0.2:8501
#python -m streamlit run "c:\Users\rahul\Downloads\weather projdrone\weather_viz_dashboard_refined.py"- this is to reach near the site
