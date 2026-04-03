import streamlit as st
import joblib
import pandas as pd
import os
import sqlite3
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash

# ====================== Page Config ======================
st.set_page_config(
    page_title="DiseaseGuard - AI Health Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== LOAD MODEL & SYMPTOMS ======================
MODEL_PATH = "model/disease_model.pkl"
SYMPTOMS_PATH = "model/symptoms_list.pkl"

@st.cache_resource(show_spinner="Loading AI Model...")
def load_model_and_symptoms():
    model = joblib.load(MODEL_PATH)
    symptoms_list = joblib.load(SYMPTOMS_PATH)
    st.success("✅ AI Model loaded successfully!")
    return model, symptoms_list

model, symptoms_list = load_model_and_symptoms()

# ====================== LOAD DATA ======================
@st.cache_data
def load_data():
    base_dir = Path(__file__).parent.absolute()
    data_dir = base_dir / "data"
    
    return (
        pd.read_csv(data_dir / "descriptions.csv"),
        pd.read_csv(data_dir / "medications.csv"),
        pd.read_csv(data_dir / "precautions.csv"),
        pd.read_csv(data_dir / "diets.csv"),
        pd.read_csv(data_dir / "workouts.csv")
    )

descriptions, medications, precautions, diets, workouts = load_data()

# ====================== DATABASE ======================
def init_db():
    conn = sqlite3.connect('users.db')
    conn.execute('''CREATE TABLE IF NOT EXISTS users
                    (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT)''')
    conn.close()

init_db()

def register_user(username, password):
    hashed = generate_password_hash(password)
    try:
        conn = sqlite3.connect('users.db')
        conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()

def login_user(username, password):
    conn = sqlite3.connect('users.db')
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    if user and check_password_hash(user[2], password):
        return True
    return False

# ====================== SESSION STATE ======================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

# ====================== SIDEBAR ======================
st.sidebar.title("🩺 DiseaseGuard")
st.sidebar.markdown("**AI Symptom → Disease Detector**")

if st.session_state.logged_in:
    st.sidebar.success(f"Logged in as: **{st.session_state.username}**")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()
else:
    st.sidebar.info("Please login or register")

# ====================== MAIN APP ======================
st.title("🩺 DiseaseGuard")
st.markdown("**Professional AI Health Assistant**")

if not st.session_state.logged_in:
    tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])
    
    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", type="primary"):
            if login_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    with tab2:
        new_user = st.text_input("Choose Username", key="reg_user")
        new_pass = st.text_input("Choose Password", type="password", key="reg_pass")
        if st.button("Register", type="primary"):
            if register_user(new_user, new_pass):
                st.success("Account created! Now login.")
            else:
                st.error("Username already exists.")

else:
    st.subheader("Select Your Symptoms")
    
    search = st.text_input("🔍 Search symptoms", "")
    filtered_symptoms = [s for s in symptoms_list if search.lower() in s.lower()]
    
    selected_symptoms = st.multiselect(
        "Choose symptoms you are experiencing",
        options=filtered_symptoms,
        default=[],
        help="You can select multiple symptoms"
    )
    
    if st.button("🔮 Predict Disease", type="primary", use_container_width=True):
        if not selected_symptoms:
            st.warning("⚠️ Please select at least one symptom")
        else:
            with st.spinner("Analyzing symptoms..."):
                input_vec = [1 if sym in selected_symptoms else 0 for sym in symptoms_list]
                prediction = model.predict([input_vec])[0]
                
                # Safe way to get information (handles both 'disease' and 'diseases' column)
                def get_info(df, col_name, default="Information not available"):
                    if df.empty:
                        return default
                    # Try both possible column names
                    for possible_col in ['disease', 'diseases', 'Disease']:
                        if possible_col in df.columns:
                            match = df[df[possible_col] == prediction]
                            if not match.empty:
                                return match[col_name].values[0]
                    return default
                
                info = {
                    "disease": prediction,
                    "description": get_info(descriptions, "description"),
                    "medications": get_info(medications, "medications"),
                    "precautions": get_info(precautions, "precautions"),
                    "diets": get_info(diets, "diets"),
                    "workouts": get_info(workouts, "workouts")
                }
                
                st.success(f"**Predicted Disease: {info['disease']}**")
                
                st.subheader("📋 Description")
                st.write(info["description"])
                st.subheader("💊 Recommended Medications")
                st.write(info["medications"])
                st.subheader("⚠️ Precautions")
                st.write(info["precautions"])
                st.subheader("🥗 Recommended Diet")
                st.write(info["diets"])
                st.subheader("🏋️ Suggested Workouts / Exercises")
                st.write(info["workouts"])
                
                st.caption("⚠️ This is for educational purposes only. Always consult a qualified doctor.")

st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ using Streamlit + Scikit-learn\nEducational project only")
