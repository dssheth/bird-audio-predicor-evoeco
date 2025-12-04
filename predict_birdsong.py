import streamlit as st
import joblib
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from io import BytesIO
import requests

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Birdsong Classifier", layout="centered")
st.title("🎵 Birdsong Audio Classifier")
st.caption("Upload audio files and get species predictions with images.")

# -------------------------------
# Load model
# -------------------------------
model_path = "birdsong_rf.joblib"
bundle = joblib.load(model_path)
model = bundle["model"]
le = bundle["label_encoder"]
sr_target = bundle["sr"]

# -------------------------------
# Audio processing & feature extraction
# -------------------------------
def extract_features(y, sr):
    if y.size == 0:
        return np.zeros(2 + 2 + 2 + 2 + 2 + (13 * 2), dtype=np.float32)
    
    def agg_stats(mat):
        return [float(np.mean(mat)), float(np.std(mat))]

    zcr = librosa.feature.zero_crossing_rate(y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    feats = []
    feats += agg_stats(zcr)
    feats += agg_stats(centroid)
    feats += agg_stats(bandwidth)
    feats += agg_stats(rolloff)
    feats += agg_stats(rms)
    for i in range(mfcc.shape[0]):
        feats.append(float(np.mean(mfcc[i])))
        feats.append(float(np.std(mfcc[i])))
    return np.array(feats, dtype=np.float32)

# -------------------------------
# Prediction for one audio file
# -------------------------------
def predict_audio_file(file_bytes, window_s, hop_s):
    y, sr = librosa.load(BytesIO(file_bytes), sr=sr_target)
    win = int(window_s * sr)
    hop = int(hop_s * sr)
    start = 0
    results = []
    while start < len(y):
        end = min(start + win, len(y))
        segment = y[start:end]
        feats = extract_features(segment, sr)
        if hasattr(model, "predict_proba"):
            p = model.predict_proba([feats])[0]
            idx = int(np.argmax(p))
            conf = float(p[idx])
            species = le.inverse_transform([idx])[0]
        else:
            idx = int(model.predict([feats])[0])
            conf = float("nan")
            species = le.inverse_transform([idx])[0]
        results.append((start/sr, end/sr, species, conf))
        if end == len(y):
            break
        start += hop
    return pd.DataFrame(results, columns=["start_s", "end_s", "species", "confidence"])

# -------------------------------
# Automatic image fetching
# -------------------------------
def get_species_image(species_name):
    folder = Path("images")
    folder.mkdir(exist_ok=True)
    img_path = folder / f"{species_name}.jpg"
    
    if img_path.exists():
        return img_path

    # Simple search using Wikimedia commons (adjust for your species)
    search_name = species_name.replace(" ", "_")
    url = f"https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/{search_name}.jpg/320px-{search_name}.jpg"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            image.save(img_path)
            return img_path
    except Exception as e:
        print(f"Could not download image for {species_name}: {e}")
    return None

# -------------------------------
# Sidebar controls
# -------------------------------
uploaded_files = st.file_uploader(
    "Upload audio files", type=["wav","mp3"], accept_multiple_files=True
)

window_seconds = st.slider("Window length (seconds)", 1.0, 10.0, 5.0)
hop_seconds = st.slider("Hop length (seconds)", 0.5, 10.0, 2.5)
threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.25)

# -------------------------------
# Run predictions
# -------------------------------
if uploaded_files:
    st.info("Running predictions...")
    all_results = []

    for f in uploaded_files:
        df = predict_audio_file(f.read(), window_seconds, hop_seconds)
        df["file"] = f.name
        all_results.append(df)

    df_all = pd.concat(all_results, ignore_index=True)
    st.success("✅ Predictions complete!")

    # Filter by confidence threshold
    df_display = df_all[df_all["confidence"] >= threshold]
    st.subheader("Predicted Species")
    st.dataframe(df_display)

    # Display species images
    for species_name in df_display["species"].unique():
        st.markdown(f"**{species_name}**")
        img_path = get_species_image(species_name)
        if img_path:
            st.image(img_path, width=150)
        else:
            st.write(f"No image available for {species_name}")

    # Download CSV
    csv_bytes = df_display.to_csv(index=False).encode()
    st.download_button(
        "📥 Download predictions CSV",
        data=csv_bytes,
        file_name="bird_predictions.csv"
    )
