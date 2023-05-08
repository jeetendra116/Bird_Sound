import streamlit as st
import librosa
import numpy as np
import joblib

# Load the trained SVM model
model = joblib.load('svm_model.pkl')
names = ["Australian Brushurkey_sound", "Bearded_Guan_sound", "Chaco Chachalaca_sound", "Dusky-legged Guan_sound", "Elegant Crested Tinamou_sound"]

st.title("Identification Of Birds Species")
audio_file = st.file_uploader("Upload an audio file", type=["wav"])

def app():
    if audio_file is not None:
        st.audio(audio_file)
        if st.button("Identify bird species"):
            audio_data, sample_rate = librosa.load(audio_file, res_type='kaiser_fast')
            mfccs_features = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=20)
            mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
            mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)

            # Predict the bird species as before
            predicted_label = model.predict(mfccs_scaled_features)[0]
            predicted_bird = names[predicted_label]

            # Add the content to display
            if predicted_label == 0:
                st.write(
                    "The Australian brushturkey, also known as the scrub turkey, is a large ground-dwelling bird found in Australia. They are known for their distinctive appearance, with black feathers and a bright red head and neck. Brushturkeys are omnivorous and feed on a variety of foods, including insects, fruits, and seeds.")
            elif predicted_label == 1:
                st.write("Content for Bearded_Guan_sound")
            elif predicted_label == 2:
                st.write("Content for Chaco Chachalaca_sound")
            elif predicted_label == 3:
                st.write("Content for Dusky-legged Guan_sound")
            elif predicted_label == 4:
                st.write("Content for Elegant Crested Tinamou_sound")

            st.header("The predicted bird species is: " + predicted_bird)

if __name__ == '_main_':
    app()