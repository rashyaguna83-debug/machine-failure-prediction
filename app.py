import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("machine_failure_random_forest.pkl")

# Page configuration
st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="⚙️",
    layout="centered"
)

# Title
st.title("⚙️ Machine Failure Prediction")

st.write(
    "Enter the machine details below to predict whether the machine is likely to fail."
)

st.divider()

# Input fields

temperature = st.number_input(
    "Temperature",
    min_value=0.0,
    max_value=200.0,
    value=75.0
)

vibration = st.number_input(
    "Vibration",
    min_value=0.0,
    max_value=20.0,
    value=2.5
)

pressure = st.number_input(
    "Pressure",
    min_value=0.0,
    max_value=200.0,
    value=101.0
)

humidity = st.number_input(
    "Humidity",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

operating_hours = st.number_input(
    "Operating Hours",
    min_value=0,
    max_value=50000,
    value=1500
)

machine_age = st.number_input(
    "Machine Age",
    min_value=0,
    max_value=100,
    value=3
)

st.divider()

# Prediction button
if st.button("🔍 Predict Machine Failure"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "Temperature": [temperature],
        "Vibration": [vibration],
        "Pressure": [pressure],
        "Humidity": [humidity],
        "Operating_Hours": [operating_hours],
        "Machine_Age": [machine_age]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Get probability
    probability = model.predict_proba(input_data)[0]

    failure_probability = probability[1] * 100
    normal_probability = probability[0] * 100

    st.divider()

    # Display result
    if prediction == 1:
        st.error("⚠️ Machine Failure Predicted!")

        st.write(
            f"Failure Probability: **{failure_probability:.2f}%**"
        )

    else:
        st.success("✅ Machine is Not Likely to Fail")

        st.write(
            f"Normal Operation Probability: **{normal_probability:.2f}%**"
        )

    # Show entered values
    st.subheader("📊 Input Details")
    st.dataframe(input_data)