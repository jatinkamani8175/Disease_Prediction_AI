import streamlit as st
import joblib
import pandas as pd
import sqlite3
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash

# ====================== Page Config ======================
st.set_page_config(
    page_title="DiseaseGuard - AI Health Assistant",
    page_icon="🩺",
    layout="wide"
)

# ====================== LOAD MODEL ======================
MODEL_PATH = "model/disease_model.pkl"
SYMPTOMS_PATH = "model/symptoms_list.pkl"

@st.cache_resource(show_spinner="Loading AI Model...")
def load_model():
    model = joblib.load(MODEL_PATH)
    symptoms = joblib.load(SYMPTOMS_PATH)
    return model, symptoms

model, symptoms_list = load_model()

# ====================== LOAD DATA ======================
@st.cache_data
def load_data():
    base_dir = Path(__file__).parent  # ✅ FIXED
    data_dir = base_dir / "data"

    try:
        descriptions = pd.read_csv(data_dir / "descriptions.csv")
        medications = pd.read_csv(data_dir / "medications.csv")
        precautions = pd.read_csv(data_dir / "precautions.csv")
        diets = pd.read_csv(data_dir / "diets.csv")
        workouts = pd.read_csv(data_dir / "workouts.csv")

        # Normalize columns
        for df in [descriptions, medications, precautions, diets, workouts]:
            df.columns = df.columns.str.strip().str.lower()
            if "disease" in df.columns:
                df["disease"] = df["disease"].astype(str).str.strip().str.lower()

        return descriptions, medications, precautions, diets, workouts

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

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

# ====================== SESSION ======================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

# ====================== SIDEBAR ======================
st.sidebar.title("🩺 DiseaseGuard")

if st.session_state.logged_in:
    st.sidebar.success(f"Logged in as {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()
else:
    st.sidebar.info("Login required")

# ====================== HELPER FUNCTION ======================
def get_info(df, disease, col_name):
    try:
        disease = disease.strip().lower()

        if df.empty or "disease" not in df.columns:
            return "Information not available"

        match = df[df["disease"] == disease]

        if not match.empty and col_name in df.columns:
            return match.iloc[0][col_name]

    except Exception as e:
        st.error(f"Error fetching info: {e}")

    return "Information not available"

# ====================== MAIN ======================
st.title("🩺 DiseaseGuard AI")

if not st.session_state.logged_in:

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Login"):
            if login_user(u, p):
                st.session_state.logged_in = True
                st.session_state.username = u
                st.rerun()
            else:
                st.error("Invalid login")

    with tab2:
        u = st.text_input("New Username")
        p = st.text_input("New Password", type="password")

        if st.button("Register"):
            if register_user(u, p):
                st.success("Registered successfully")
            else:
                st.error("Username exists")

# ====================== AFTER LOGIN ======================
else:
    st.subheader("Select Symptoms")

    search = st.text_input("Search symptoms")
    filtered = [s for s in symptoms_list if search.lower() in s.lower()]

    selected = st.multiselect("Symptoms", filtered)

    if st.button("Predict Disease"):

        if not selected:
            st.warning("Select symptoms")
        else:
            input_vec = [1 if s in selected else 0 for s in symptoms_list]

            prediction = model.predict([input_vec])[0]
            proba = model.predict_proba([input_vec])[0]
            confidence = max(proba) * 100

            st.success(f"Prediction: {prediction} ({confidence:.2f}%)")

            # DEBUG (remove later)
            st.write("Prediction raw:", prediction)

            # Fetch info
            desc = get_info(descriptions, prediction, "description")
            meds = get_info(medications, prediction, "medications")
            prec = get_info(precautions, prediction, "precautions")
            diet = get_info(diets, prediction, "diets")
            work = get_info(workouts, prediction, "workouts")

            st.subheader("📋 Description")
            st.write(desc)

            st.subheader("💊 Medications")
            st.write(meds)

            st.subheader("⚠️ Precautions")
            st.write(prec)

            st.subheader("🥗 Diet")
            st.write(diet)

            st.subheader("🏋️ Workouts")
            st.write(work)

            st.caption("Educational purpose only")
