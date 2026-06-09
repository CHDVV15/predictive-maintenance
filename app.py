import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import random
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

model = joblib.load("predictive_maintenance.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Predictive Maintenance System")
st.write(
    "Monitor equipment health and predict failures using machine learning."
)

st.sidebar.markdown("### Model Information")
st.sidebar.success("Random Forest Classifier")
st.sidebar.info("Accuracy: 93.6+%")

mode = st.sidebar.radio(
    "Select Mode",
    ["Manual Input", "Live Monitoring"]
)

if mode == "Live Monitoring":
    st_autorefresh(
        interval=5000,
        key="live_refresh"
    )

    footfall = random.randint(50, 500)
    tempMode = random.randint(0, 5)
    AQ = random.randint(50, 300)
    USS = random.randint(10, 100)
    CS = round(random.uniform(1, 20), 2)
    VOC = random.randint(20, 200)
    RP = random.randint(500, 3000)
    IP = random.randint(10, 100)
    Temperature = random.randint(20, 90)

else:

    st.sidebar.header("Sensor Inputs")

    footfall = st.sidebar.number_input(
        "Footfall",
        min_value=0,
        value=100
    )

    tempMode = st.sidebar.number_input(
        "Temperature Mode",
        value=0
    )

    AQ = st.sidebar.number_input(
        "Air Quality",
        value=100
    )

    USS = st.sidebar.number_input(
        "Ultrasonic Sensor",
        value=20
    )

    CS = st.sidebar.number_input(
        "Current Sensor",
        value=5.0
    )

    VOC = st.sidebar.number_input(
        "VOC",
        value=50
    )

    RP = st.sidebar.number_input(
        "RP",
        value=1000
    )

    IP = st.sidebar.number_input(
        "Input Pressure",
        value=20
    )

    Temperature = st.sidebar.number_input(
        "Operating Temperature",
        value=35
    )

input_data = pd.DataFrame({
    "footfall": [footfall],
    "tempMode": [tempMode],
    "AQ": [AQ],
    "USS": [USS],
    "CS": [CS],
    "VOC": [VOC],
    "RP": [RP],
    "IP": [IP],
    "Temperature": [Temperature]
})

st.write(
    f"Last Updated: {datetime.now().strftime('%H:%M:%S')}"
)

st.subheader("Current Sensor Readings")
st.dataframe(
    input_data,
    use_container_width=True
)

scaled_data = scaler.transform(input_data)

prediction = model.predict(scaled_data)[0]

probability = model.predict_proba(
    scaled_data
)[0][1]

health_score = (1 - probability) * 100

st.subheader("Equipment Health Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Health Score",
        f"{health_score:.1f}%"
    )

with col2:
    st.metric(
        "Failure Risk",
        f"{probability*100:.1f}%"
    )

with col3:
    st.metric(
        "Temperature",
        f"{Temperature} °C"
    )

if probability > 0.7:

    st.error(
        "🚨 CRITICAL ALERT: High Failure Risk Detected"
    )

elif probability > 0.4:

    st.warning(
        "⚠️ Maintenance Recommended"
    )

else:

    st.success(
        "✅ Equipment Operating Normally"
    )

if prediction == 1:

    st.error(
        "⚠️ Equipment Failure Likely"
    )

else:

    st.success(
        "✅ Equipment Healthy"
    )

st.subheader("Maintenance Recommendation")

if health_score >= 80:

    st.success(
        "Machine is healthy. Continue normal operation."
    )

elif health_score >= 50:

    st.warning(
        "Schedule preventive maintenance."
    )

else:

    st.error(
        "Immediate maintenance required."
    )

st.subheader("Health Meter")
st.progress(int(health_score))

st.subheader("Sensor Values")

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(
    input_data.columns,
    input_data.iloc[0]
)

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

st.subheader("Health Distribution")

fig2, ax2 = plt.subplots()

ax2.pie(
    [health_score, 100 - health_score],
    labels=["Healthy", "Risk"],
    autopct="%1.1f%%"
)

st.pyplot(fig2)

st.markdown("---")

st.caption(
    "AI-Based Predictive Maintenance Dashboard"
)