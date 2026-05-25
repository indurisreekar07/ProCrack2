import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# App title
st.title("ProCrack AI")

st.subheader("Smart Deadline Prediction System")

# Load dataset
data = pd.read_csv("tasks_dataset.csv")

# Features
X = data[[
    'Difficulty(1-10)',
    'Available_Hours_Per_Day',
    'Days_Left',
    'Distraction_Level(1-10)'
]]

# Target
y = data['Estimated_Completion_Days']

# Train model
model = LinearRegression()
model.fit(X, y)

# User Inputs
difficulty = st.slider(
    "Task Difficulty",
    1,
    10,
    5
)

hours = st.slider(
    "Available Hours Per Day",
    1,
    10,
    3
)

days_left = st.slider(
    "Days Left",
    1,
    30,
    7
)

distraction = st.slider(
    "Distraction Level",
    1,
    10,
    5
)

# Prediction
if st.button("Predict Deadline"):

    prediction = model.predict([[
        difficulty,
        hours,
        days_left,
        distraction
    ]])

    st.success(
        f"Estimated Completion Time: {round(prediction[0], 2)} days"
    )