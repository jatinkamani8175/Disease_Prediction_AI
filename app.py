import streamlit as st
import joblib
import pandas as pd
import sqlite3
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash

st.set_page_config(
    page_title="DiseaseGuard - AI Health Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== LOAD MODEL ======================
MODEL_PATH = "model/disease_model.pkl"
SYMPTOMS_PATH = "model/symptoms_list.pkl"

@st.cache_resource(show_spinner="Loading AI Model...")
def load_model_and_symptoms():
    model = joblib.load(MODEL_PATH)
    symptoms_list = joblib.load(SYMPTOMS_PATH)
    st.success("✅ Model loaded successfully!")
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
        help="Select multiple symptoms for better accuracy"
    )
    
    if st.button("🔮 Predict Disease", type="primary", use_container_width=True):
        if not selected_symptoms:
            st.warning("Please select at least one symptom")
        else:
            with st.spinner("Analyzing symptoms..."):
                input_vec = [1 if sym in selected_symptoms else 0 for sym in symptoms_list]
                
                # Get numeric prediction
                pred_index = model.predict([input_vec])[0]
                
                # Convert numeric prediction to actual disease name
                disease_name = model.classes_[pred_index]
                
                # Get probabilities
                proba = model.predict_proba([input_vec])[0]
                top3 = sorted(zip(model.classes_, proba), key=lambda x: x[1], reverse=True)[:3]
                
                st.info(f"**Top 3 Predictions:** {top3[0][0]} ({top3[0][1]*100:.1f}%), "
                        f"{top3[1][0]} ({top3[1][1]*100:.1f}%), {top3[2][0]} ({top3[2][1]*100:.1f}%)")
                
                # Safe info retrieval
                def get_info(df, column_name, default="Not available in dataset"):
                    for col in ['Disease', 'diseases', 'disease']:
                        if col in df.columns:
                            match = df[df[col] == disease_name]
                            if not match.empty:
                                return match[column_name].values[0]
                    return default
                
                info = {
                    "disease": disease_name,
                    "description": get_info(descriptions, "Description"),
                    "medications": get_info(medications, "Medication"),
                    "precautions": get_info(precautions, "Precaution_1"),
                    "diets": get_info(diets, "Diet"),
                    "workouts": get_info(workouts, "Workouts")
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
