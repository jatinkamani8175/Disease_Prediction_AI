import streamlit as st
import joblib
import pandas as pd
import os
import requests
import sqlite3
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash

# ====================== Page Config (Professional Look) ======================
st.set_page_config(
    page_title="Disease_Detection - AI Health Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== MODEL AUTO DOWNLOAD ======================
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "disease_model.pkl")
SYMPTOMS_PATH = os.path.join(MODEL_DIR, "symptoms_list.pkl")

# === REPLACE THESE WITH YOUR ACTUAL GOOGLE DRIVE FILE IDs ===
MODEL_ID = "1i18hJfpWk4BXdbwNUYIIsGPG-ohfvqUx"      # ← Change this
SYMPTOMS_ID = "1i18hJfpWk4BXdbwNUYIIsGPG-ohfvqUx"  # ← Change this

@st.cache_resource(show_spinner="Downloading model from Google Drive (first time only)...")
def load_model():
    def download_file(file_id, save_path):
        if os.path.exists(save_path):
            return
        st.info(f"Downloading {os.path.basename(save_path)}... This may take 30-90 seconds.")
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        session = requests.Session()
        response = session.get(url, stream=True)
        
        if "confirm=" in response.text:
            confirm_token = response.text.split("confirm=")[1].split("&")[0]
            url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={confirm_token}"
            response = session.get(url, stream=True)
        
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        st.success(f"✅ {os.path.basename(save_path)} downloaded!")
    
    download_file(MODEL_ID, MODEL_PATH)
    download_file(SYMPTOMS_ID, SYMPTOMS_PATH)
    
    model = joblib.load(MODEL_PATH)
    symptoms = joblib.load(SYMPTOMS_PATH)
    return model, symptoms

model, symptoms_list = load_model()

# ====================== LOAD DATA (Improved & More Robust) ======================
@st.cache_data
def load_data():
    base_dir = Path(__file__).parent.absolute()
    data_dir = base_dir / "data"
    
    required_files = ["descriptions.csv", "medications.csv", "precautions.csv", "diets.csv", "workouts.csv"]
    
    # Check if data folder and all files exist
    if not data_dir.exists():
        st.error("❌ **'data' folder not found!**")
        st.error("Please create a folder named **data** in your GitHub repo and add all CSV files inside it.")
        st.stop()
    
    missing = []
    for file in required_files:
        if not (data_dir / file).exists():
            missing.append(file)
    
    if missing:
        st.error("❌ **Some data files are missing!**")
        st.error(f"Missing files: {', '.join(missing)}")
        st.error("Make sure all 5 CSV files are inside the **data/** folder on GitHub.")
        st.stop()
    
    try:
        descriptions = pd.read_csv(data_dir / "descriptions.csv")
        medications   = pd.read_csv(data_dir / "medications.csv")
        precautions   = pd.read_csv(data_dir / "precautions.csv")
        diets         = pd.read_csv(data_dir / "diets.csv")
        workouts      = pd.read_csv(data_dir / "workouts.csv")
        return descriptions, medications, precautions, diets, workouts
    except Exception as e:
        st.error(f"❌ Error reading data files: {e}")
        st.stop()

descriptions, medications, precautions, diets, workouts = load_data()

# ====================== DATABASE SETUP ======================
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
st.sidebar.markdown("**AI-Powered Symptom to Disease Detector**")

if st.session_state.logged_in:
    st.sidebar.success(f"Logged in as: **{st.session_state.username}**")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()
else:
    st.sidebar.info("Please login or register to continue")

# ====================== MAIN APP ======================
st.title("🩺 DiseaseGuard")
st.markdown("**Professional AI Health Assistant** — Predict disease from symptoms and get full recommendations")

if not st.session_state.logged_in:
    tab1, tab2 = st.tabs(["🔑 Login", "📝 Register"])
    
    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", type="primary"):
            if login_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    with tab2:
        new_user = st.text_input("Choose Username", key="reg_user")
        new_pass = st.text_input("Choose Password", type="password", key="reg_pass")
        if st.button("Register", type="primary"):
            if register_user(new_user, new_pass):
                st.success("Account created successfully! Now login.")
            else:
                st.error("Username already exists. Try another one.")

else:
    st.subheader("Select Your Symptoms")
    
    search = st.text_input("🔍 Search symptoms", "")
    filtered_symptoms = [s for s in symptoms_list if search.lower() in s.lower()]
    
    selected_symptoms = st.multiselect(
        "Choose symptoms you are experiencing",
        options=filtered_symptoms,
        default=[],
        help="Start typing to search"
    )
    
    if st.button("🔮 Predict Disease", type="primary", use_container_width=True):
        if not selected_symptoms:
            st.warning("Please select at least one symptom")
        else:
            with st.spinner("Analyzing symptoms..."):
                input_vec = [1 if sym in selected_symptoms else 0 for sym in symptoms_list]
                prediction = model.predict([input_vec])[0]
                
                info = {
                    "disease": prediction,
                    "description": descriptions[descriptions["disease"] == prediction]["description"].values[0],
                    "medications": medications[medications["disease"] == prediction]["medications"].values[0],
                    "precautions": precautions[precautions["disease"] == prediction]["precautions"].values[0],
                    "diets": diets[diets["disease"] == prediction]["diets"].values[0],
                    "workouts": workouts[workouts["disease"] == prediction]["workouts"].values[0]
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
                
                st.caption("⚠️ **Important**: This is for educational purposes only. Always consult a qualified doctor.")

st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ using Streamlit + Scikit-learn\n\nEducational project only")
