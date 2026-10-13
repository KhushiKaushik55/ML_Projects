import streamlit as st
import pandas as pd
import joblib

from pathlib import Path
from datetime import datetime

# ----------------------------------
# APP CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="FocusFlow",
    page_icon="🎯",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "models" / "focusflow_model.joblib"
SESSIONS_FILE = DATA_DIR / "real_sessions.csv"

DATA_DIR.mkdir(exist_ok=True)

# ----------------------------------
# LOAD THE TRAINED MODEL
# ----------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("🎯 FocusFlow")
st.caption("AI-Powered Personal Productivity Analyzer")

tab1, tab2 = st.tabs([
    "🔮 Predict Duration",
    "📝 Log & History"
])

# ----------------------------------
# TAB 1: PREDICTION
# ----------------------------------

with tab1:

    st.subheader("Predict Your Task Duration")

    st.info(
        "Experimental prediction: this model was "
        "trained on simulated task sessions."
    )

    with st.form("prediction_form"):

        task_type = st.selectbox(
            "Task Type",
            ["coding", "studying", "writing", "research"],
            key="pred_task"
        )

        planned_minutes = st.number_input(
            "Planned Duration (minutes)",
            min_value=20,
            max_value=120,
            value=60,
            key="pred_planned"
        )

        difficulty = st.slider(
            "Difficulty",
            min_value=1,
            max_value=5,
            value=3,
            key="pred_difficulty"
        )

        sleep_hours = st.slider(
            "Sleep Hours",
            min_value=4.0,
            max_value=9.0,
            value=7.0,
            step=0.5,
            key="pred_sleep"
        )

        experience_level = st.slider(
            "Experience Level",
            min_value=1,
            max_value=5,
            value=3,
            key="pred_experience"
        )

        time_of_day = st.selectbox(
            "Time of Day",
            ["morning", "afternoon", "evening", "night"],
            key="pred_time"
        )

        predict_button = st.form_submit_button(
            "Predict Duration"
        )

    if predict_button:

        if not MODEL_PATH.exists():
            st.error(
                "Model file not found. "
                "Please run improve_model.py first."
            )

        else:
            model = load_model()

            input_data = pd.DataFrame([{
                "task_type": task_type,
                "planned_minutes": planned_minutes,
                "difficulty": difficulty,
                "sleep_hours": sleep_hours,
                "experience_level": experience_level,
                "time_of_day": time_of_day
            }])

            prediction = float(
                model.predict(input_data)[0]
            )

            prediction = max(5, prediction)

            overrun = prediction - planned_minutes

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Predicted Duration",
                    f"{prediction:.0f} min"
                )

            with col2:
                st.metric(
                    "Estimated Overrun",
                    f"{overrun:+.0f} min"
                )

            if overrun > 0:
                st.warning(
                    "This task may take longer "
                    "than your planned time."
                )
            else:
                st.success(
                    "This task may fit within "
                    "your planned time."
                )

# ----------------------------------
# TAB 2: TASK LOGGER
# ----------------------------------

with tab2:

    st.subheader("Log a Completed Task")

    with st.form("task_form"):

        log_task = st.selectbox(
            "Task Type",
            ["coding", "studying", "writing", "research"]
        )

        log_planned = st.number_input(
            "Planned Duration (minutes)",
            min_value=5,
            max_value=480,
            value=60
        )

        log_actual = st.number_input(
            "Actual Duration (minutes)",
            min_value=1,
            max_value=600,
            value=60
        )

        log_difficulty = st.slider(
            "Difficulty", 1, 5, 3
        )

        log_sleep = st.slider(
            "Sleep Hours", 0.0, 12.0, 7.0, 0.5
        )

        log_experience = st.slider(
            "Experience Level", 1, 5, 3
        )

        log_time = st.selectbox(
            "Time of Day",
            ["morning", "afternoon", "evening", "night"]
        )

        save_button = st.form_submit_button(
            "Save Session"
        )

    if save_button:

        new_session = pd.DataFrame([{
            "timestamp": datetime.now().isoformat(
                timespec="seconds"
            ),
            "task_type": log_task,
            "planned_minutes": log_planned,
            "actual_minutes": log_actual,
            "difficulty": log_difficulty,
            "sleep_hours": log_sleep,
            "experience_level": log_experience,
            "time_of_day": log_time
        }])

        new_session.to_csv(
            SESSIONS_FILE,
            mode="a",
            header=not SESSIONS_FILE.exists(),
            index=False
        )

        st.success("Session saved successfully!")

    st.divider()
    st.subheader("Session History")

    if SESSIONS_FILE.exists():

        history = pd.read_csv(SESSIONS_FILE)

        st.metric(
            "Total Sessions",
            len(history)
        )

        st.dataframe(
            history,
            use_container_width=True
        )

    else:
        st.info("No sessions logged yet.")