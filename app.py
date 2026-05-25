import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("ProCrack AI")

st.subheader("Smart Deadline Prediction System")

data = pd.read_csv("tasks_dataset.csv")

X = data[[
    'Difficulty(1-10)',
    'Available_Hours_Per_Day',
    'Days_Left',
    'Distraction_Level(1-10)'
]]

y = data['Estimated_Completion_Days']

model = LinearRegression()
model.fit(X, y)

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
