📘 Project Overview
This project predicts a person’s age and emotion from their voice recording. It automatically identifies the speaker’s gender using frequency-based logic — if the system detects a female voice, it shows a message: “Upload male voice.”
If the speaker is male and aged above 60, the system classifies their emotion (Happy, Sad, Angry, Neutral, or Surprised). For those under 60, it only predicts age.
The system includes an interactive Streamlit GUI that allows users to upload audio files and instantly view results.

📦 Model Downloads
Due to file size limits, the trained models are hosted on Google Drive:
👉 [Download Hair Length Model]
👉 [Download Age & Gender Model]
https://drive.google.com/drive/folders/1EyQWxX5bvyfAYLLzZpmFf_7jS04Wpdqn?usp=sharing
After downloading, place them in the /models folder.


⚙️ Features
• Predicts age from voice using a regression-based ML model.
• Detects emotion for senior males using a multi-class classification model.
• Identifies gender using frequency rate analysis (no deep learning required).
• Automatically rejects non-male voices.
• Simple, modern Streamlit interface for audio upload and prediction visualization.
• Real-time feedback with clear user messages and clean UI.

🧠 Machine Learning Models

🧩 Gender Detection (Frequency Logic)
• Approach: Rule-based using average frequency (Hz).
• Logic:
 → Frequency < 160 Hz → Male
 → Frequency ≥ 160 Hz → Female
• Purpose: Filter female voices automatically before age or emotion detection.

🧩 Age Prediction Model (Voice)
• Dataset: Male voice samples with corresponding age labels.
• Model Type: Regression Model (Random Forest / Linear Regression).
• Features Extracted: MFCCs, Chroma, Spectral Contrast, Zero Crossing Rate, Energy.
• Purpose: Estimate speaker’s approximate age.

🧩 Emotion Detection Model (Male)
• Dataset: Male subset labeled with 5 balanced emotions (Happy, Sad, Angry, Neutral, Surprised).
• Model Type: Multi-Class Classifier (SVM / Random Forest).
• Features Extracted: MFCCs, Pitch, Energy, Spectral Features.
• Purpose: Identify emotional state from voice tone and pitch.

🧩 Datasets Used
• Custom Male Voice Dataset: For age regression.
• Emotion Dataset: Male subset with 5 main emotion labels.
• Audio Format: .wav / .mp3 processed with librosa.

🖥️ Streamlit GUI Application
• Upload audio files directly from the browser.
• Plays back the uploaded voice sample.
• Displays detected gender, age, and emotion results.
• If a female voice is detected → shows “Upload male voice.”
• Clean, intuitive interface with modern design.

📊 Model Results (Example)
Age Model: MAE ≈ 3.5 years
Emotion Model: Accuracy ≈ 85%
Gender Logic: 95% precision using frequency-based rule

🧰 Technologies Used
Python | Librosa | NumPy | Pandas | Scikit-learn | Streamlit | Matplotlib | Joblib | SoundFile

🚀 How to Run
1️⃣ Save your trained models as .pkl files in the models/ folder:
 - gender_prediction_model_voice.pkl
 - age_prediction_model_voice.pkl
 - male_emotion_model.pkl

2️⃣ Install dependencies:

pip install streamlit librosa scikit-learn numpy pandas soundfile


3️⃣ Run the app:

streamlit run app.py


4️⃣ Upload an audio file (.wav or .mp3) and view results.

💬 Summary
This project combines rule-based gender logic and machine learning to predict age and emotions from voice. It focuses on male voice analysis and handles edge cases logically. The Streamlit app provides a smooth and interactive experience, merging AI prediction with human-like reasoning.
