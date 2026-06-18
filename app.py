import streamlit as st
import pandas as pd
import hashlib
from datetime import datetime
from sklearn.linear_model import LinearRegression
from supabase import create_client


SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(username, password):
    existing = (
        supabase.table("users")
        .select("username")
        .eq("username", username)
        .execute()
    )

    if existing.data:
        return False, "Username already exists."

    supabase.table("users").insert({
        "username": username,
        "password": hash_password(password)
    }).execute()

    return True, "Registration successful!"


def login_user(username, password):
    try:
        result = (
            supabase.table("users")
            .select("password")
            .eq("username", username)
            .execute()
        )

        st.write(result)  # Temporary: shows the response

        if not result.data:
            return False, "Username not found."

        if result.data[0]["password"] != hash_password(password):
            return False, "Incorrect password."

        return True, "Login successful!"

    except Exception as e:
        st.exception(e)
        return False, str(e)


def save_history(username, inputs, prediction):
    supabase.table("history").insert({
        "username": username,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "difficulty": inputs[0],
        "hours_per_day": inputs[1],
        "days_left": inputs[2],
        "distraction": inputs[3],
        "predicted_days": round(prediction, 2)
    }).execute()


def get_history(username):
    result = (
        supabase.table("history")
        .select("*")
        .eq("username", username)
        .order("id", desc=True)
        .execute()
    )

    return result.data


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
        if round(prediction, 2) > days_left:
            st.error(f"🚨 WARNING: The task is estimated to take {round(prediction, 2)} days but you only have {days_left} days left! You need to increase hours or reduce distraction immediately.")
        elif round(prediction, 2) > days_left * 0.85:
            st.warning(f"⚠️ Tight deadline! Estimated {round(prediction, 2)} days vs {days_left} days available. Very little buffer.")

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
