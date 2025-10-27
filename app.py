import streamlit as st
import librosa
import numpy as np
import joblib

# ==========================
# Load trained models
# ==========================
age_model = joblib.load("age_prediction_model_voice.pkl")
emotion_model = joblib.load("male_emotion_model.pkl")  # your male emotion model
gender_model = joblib.load("gender_prediction_model_voice.pkl")
# ==========================
# Gender detection using pitch (robust)
# ==========================
def detect_gender(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    
    # Estimate fundamental frequency (F0)
    f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr)
    f0 = f0[~np.isnan(f0)]  # Remove NaNs
    
    if len(f0) == 0:
        return "female"  # default if pitch not detected
    
    median_pitch = np.median(f0)
    
    # Male: 85-180 Hz, Female: >180 Hz
    if 85 <= median_pitch <= 180:
        return "male"
    else:
        return "female"

# ==========================
# Feature extraction for age/emotion
# ==========================
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# ==========================
# Streamlit GUI
# ==========================
st.title("🎤 Age & Emotion Detection from Voice (Male Only)")
st.write("Upload a male voice file (wav or mp3). Female voices will be rejected.")

uploaded_file = st.file_uploader("Choose audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    
    # Detect gender
    gender = detect_gender(temp_path)
    if gender != "male":
        st.error("Upload male voice.")
    else:
        # Extract features
        features = extract_features(temp_path).reshape(1, -1)
        
        # Predict age
        age_group_index = age_model.predict(features)[0]
        age_mapping = {
            0: "teens",
            1: "twenties",
            2: "thirties",
            3: "fourties",
            4: "fifties",
            5: "sixties",
            6: "seventies",
            7: "eighties"
        }
        predicted_age = age_mapping.get(age_group_index, "Unknown")
        st.success(f"Predicted Age Group: {predicted_age}")
        
        # If senior citizen, detect emotion
        if predicted_age in ["sixties", "seventies", "eighties"]:
            emotion = emotion_model.predict(features)[0]
            st.info(f"Detected Emotion: {emotion}")
