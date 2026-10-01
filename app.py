import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sqlite3

st.set_page_config(page_title="AI Disease Predictor", page_icon="🩺", layout="wide")

# ---------- DATABASE SETUP ----------
conn = sqlite3.connect("history.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symptoms TEXT,
    prediction TEXT,
    confidence REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()

# LOAD MODEL ----------
model = pickle.load(open("model.pkl", "rb"))
le = pickle.load(open("encoder.pkl", "rb"))

# ---------- LOAD DATA ----------
data = pd.read_csv("Training.csv")
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
data.columns = data.columns.str.strip()

symptoms = list(data.columns[:-1])

# ---------- DISEASE INFO ----------
disease_info = {
    "Fungal infection": {
        "desc": "A fungal infection affects skin, hair or nails.",
        "precautions": ["Keep skin dry", "Use antifungal cream", "Maintain hygiene"]
    },
    "Allergy": {
        "desc": "Reaction to substances like pollen or dust.",
        "precautions": ["Avoid allergens", "Take antihistamines", "Keep surroundings clean"]
    },
    "GERD": {
        "desc": "Acid reflux affecting food pipe.",
        "precautions": ["Avoid spicy food", "Eat small meals", "Don't lie down after eating"]
    },
    "Diabetes": {
        "desc": "High blood sugar condition.",
        "precautions": ["Control diet", "Exercise", "Monitor sugar"]
    },
    "Hypertension": {
        "desc": "High blood pressure.",
        "precautions": ["Reduce salt", "Exercise", "Manage stress"]
    }
}

# ---------- DOCTOR MAP ----------
doctor_map = {
    "fungal infection": "Dermatologist",
    "allergy": "Allergist",
    "gerd": "Gastroenterologist",
    "chronic cholestasis": "Hepatologist",
    "drug reaction": "General Physician",
    "peptic ulcer disease": "Gastroenterologist",
    "diabetes": "Endocrinologist",
    "hypertension": "Cardiologist"
}

# ---------- UI ----------
st.title("🩺 AI Disease Prediction System")
st.warning("Educational use only. Not a medical diagnosis.")

min_symptoms = st.sidebar.slider("Minimum symptoms required", 1, 5, 2)

selected_symptoms = st.multiselect(
    "Select Symptoms",
    [s.replace("_", " ").title() for s in symptoms]
)

st.write(f"Selected Symptoms: {len(selected_symptoms)}")
st.progress(min(len(selected_symptoms)/10, 1.0))

st.divider()

# ---------- PREDICTION ----------
if st.button("Predict Disease"):

    if len(selected_symptoms) < min_symptoms:
        st.warning(f"Select at least {min_symptoms} symptoms")
        st.stop()

    input_data = [0] * len(symptoms)

    for symptom in selected_symptoms:
        s = symptom.lower().replace(" ", "_")
        if s in symptoms:
            input_data[symptoms.index(s)] = 1

    probs = model.predict_proba([input_data])[0]
    classes = le.inverse_transform(np.arange(len(probs)))

    top_indices = np.argsort(probs)[-3:][::-1]

    st.subheader("Top Predictions")

    cols = st.columns(3)
    for idx, i in enumerate(top_indices):
        cols[idx].metric(classes[i], f"{round(probs[i]*100,2)}%")

    st.subheader("Confidence")
    st.progress(float(max(probs)))

    top_disease = classes[top_indices[0]]
    disease_key = top_disease.strip().lower()

    # ---------- SAVE TO DB ----------
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO history (symptoms, prediction, confidence) VALUES (?, ?, ?)",
        (", ".join(selected_symptoms), top_disease, float(max(probs)))
    )

    conn.commit()
    conn.close()

    # ---------- DOCTOR (FIXED) ----------
    st.subheader("Recommended Doctor")
    doctor = doctor_map.get(disease_key, "General Physician")
    st.write(doctor)

    # ---------- DISEASE INFO ----------
    if top_disease in disease_info:
        st.subheader("About Disease")
        st.write(disease_info[top_disease]["desc"])

        st.subheader("Precautions")
        for p in disease_info[top_disease]["precautions"]:
            st.write("•", p)

    # ---------- DOWNLOAD REPORT ----------
    report = f"""
Symptoms: {', '.join(selected_symptoms)}
Prediction: {top_disease}
Confidence: {round(max(probs)*100,2)}%
Doctor: {doctor}
"""

    st.download_button("Download Report", report, file_name="report.txt")

# ---------- HISTORY ----------
st.subheader("Prediction History")

conn = sqlite3.connect("history.db")
df = pd.read_sql_query("SELECT * FROM history ORDER BY id DESC LIMIT 5", conn)
conn.close()

st.dataframe(df)