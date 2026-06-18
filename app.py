import streamlit as st
import pandas as pd
import json
import os
import hashlib
from datetime import datetime
from sklearn.linear_model import LinearRegression


USERS_FILE = "users.json"
HISTORY_FILE = "history.json"


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return {}

def save_json(filepath, data):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

def register_user(username, password):
    users = load_json(USERS_FILE)
    if username in users:
        return False, "Username already exists."
    users[username] = hash_password(password)
    save_json(USERS_FILE, users)
    return True, "Registration successful!"

def login_user(username, password):
    users = load_json(USERS_FILE)
    if username not in users:
        return False, "Username not found."
    if users[username] != hash_password(password):
        return False, "Incorrect password."
    return True, "Login successful!"

def save_history(username, inputs, prediction):
    history = load_json(HISTORY_FILE)
    if username not in history:
        history[username] = []
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "difficulty": inputs[0],
        "hours_per_day": inputs[1],
        "days_left": inputs[2],
        "distraction": inputs[3],
        "predicted_days": round(prediction, 2)
    }
    history[username].append(entry)
    save_json(HISTORY_FILE, history)

def get_history(username):
    history = load_json(HISTORY_FILE)
    return history.get(username, [])


@st.cache_resource
def train_model():
    data = pd.read_csv("tasks_dataset.csv")
    X = data[['Difficulty(1-10)', 'Available_Hours_Per_Day', 'Days_Left', 'Distraction_Level(1-10)']]
    y = data['Estimated_Completion_Days']
    model = LinearRegression()
    model.fit(X, y)
    return model

model = train_model()


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""


st.title("🧠 ProCrack AI")
st.subheader("Smart Deadline Prediction System")

if not st.session_state.logged_in:
    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        st.markdown("### Login")
        login_user_input = st.text_input("Username", key="login_user")
        login_pass_input = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            success, msg = login_user(login_user_input, login_pass_input)
            if success:
                st.session_state.logged_in = True
                st.session_state.username = login_user_input
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    with tab2:
        st.markdown("### Register")
        reg_user_input = st.text_input("Choose Username", key="reg_user")
        reg_pass_input = st.text_input("Choose Password", type="password", key="reg_pass")
        if st.button("Register"):
            success, msg = register_user(reg_user_input, reg_pass_input)
            if success:
                st.success(msg + " Please log in.")
            else:
                st.error(msg)


else:
    username = st.session_state.username
    st.markdown(f"👤 Logged in as **{username}**")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

    st.markdown("---")

    st.markdown("### Predict Your Deadline")
    difficulty = st.slider("Task Difficulty", 1, 10, 5)
    hours = st.slider("Available Hours Per Day", 1, 10, 3)
    days_left = st.slider("Days Left", 1, 30, 7)
    distraction = st.slider("Distraction Level", 1, 10, 5)

    if st.button("Predict Deadline"):
        inputs = [difficulty, hours, days_left, distraction]
        prediction = model.predict([inputs])[0]
        save_history(username, inputs, prediction)

        st.success(f"⏱️ Estimated Completion Time: **{round(prediction, 2)} days**")

        # ── Low Time Warning ──
        if round(prediction[0], 2) > days_left:
            st.error(f"🚨 WARNING: The task is estimated to take {round(prediction[0], 2)} days but you only have {days_left} days left! You need to increase hours or reduce distraction immediately.")
        elif round(prediction[0], 2) > days_left * 0.85:
            st.warning(f"⚠️ Tight deadline! Estimated {round(prediction[0], 2)} days vs {days_left} days available. Very little buffer.")

        history = get_history(username)
        if len(history) > 1:
            past_predictions = [h["predicted_days"] for h in history[:-1]]
            avg_past = sum(past_predictions) / len(past_predictions)
            if prediction > avg_past * 1.2:
                st.warning(f"⚠️ This task is estimated to take longer than your average ({round(avg_past,2)} days). Consider reducing distraction or increasing hours.")
            elif prediction < avg_past * 0.8:
                st.info(f"✅ Great! This task looks easier than your usual average of {round(avg_past,2)} days.")
            else:
                st.info(f"📊 This is consistent with your past average of {round(avg_past,2)} days.")

    st.markdown("---")
    st.markdown("### 📋 Your Task History")
    history = get_history(username)
    if history:
        df = pd.DataFrame(history)
        df = df.rename(columns={
            "timestamp": "Time",
            "difficulty": "Difficulty",
            "hours_per_day": "Hrs/Day",
            "days_left": "Days Left",
            "distraction": "Distraction",
            "predicted_days": "Predicted Days"
        })
        st.dataframe(df[::-1], use_container_width=True)  # newest first
    else:
        st.info("No predictions yet. Submit your first task above!")
