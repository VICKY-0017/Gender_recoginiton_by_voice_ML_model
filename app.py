import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
import librosa

# Load the model and weights
model = load_model("results/model.h5")

def extract_feature(file_path, **kwargs):
    try:
        # Load audio file using librosa
        y, sr = librosa.load(file_path, sr=None, mono=True)

        # Extract features based on the provided arguments
        mfcc = kwargs.get("mfcc")
        chroma = kwargs.get("chroma")
        mel = kwargs.get("mel")
        contrast = kwargs.get("contrast")
        tonnetz = kwargs.get("tonnetz")

        if chroma or contrast:
            stft = np.abs(librosa.stft(y))  # Take only the first channel for analysis

        result = np.array([])

        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))

        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
            result = np.hstack((result, chroma))

        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y, sr=sr).T, axis=0)
            result = np.hstack((result, mel))

        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)
            result = np.hstack((result, contrast))

        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
            result = np.hstack((result, tonnetz))

        # Pad or truncate to ensure the features have the correct shape
        result = result[:128] if len(result) > 128 else np.pad(result, (0, 128 - len(result)))

        return result
    except Exception as e:
        raise ValueError(f"Invalid file: {str(e)}")


def predict_gender(file):
    try:
        features = extract_feature(file.name)  # Use file.name to get the file path
        features = np.reshape(features, (1, -1))

        # Print extracted features for debugging
        print("Extracted Features:", features)

        male_prob = model.predict(features)[0][0]
        female_prob = 1 - male_prob
        gender = "male" if male_prob > female_prob else "female"
        result = f"Result: {gender}\nProbabilities: Male: {male_prob*100:.2f}%  Female: {female_prob*100:.2f}%"
        return result
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}


iface = gr.Interface(
    fn=predict_gender,
    inputs=gr.File(type="filepath", label="Upload an audio file"),
    outputs=gr.Textbox(placeholder="Result will be shown here", type="text")
)

iface.launch(share=True, debug=True)
