import streamlit as st

import joblib
import numpy as np

# Load model bundle and extract components if needed
bundle = joblib.load("models/best_model_bundle.pkl")
if isinstance(bundle, dict):
    model = bundle["model"]
    le = bundle["le"] if "le" in bundle else joblib.load("models/label_encoder.pkl")
    scaler = bundle["scaler"] if "scaler" in bundle else joblib.load("models/scaler.pkl")
else:
    model = bundle
    le = joblib.load("models/label_encoder.pkl")
    scaler = joblib.load("models/scaler.pkl")



st.markdown("""
<style>
.main-title {font-size:2.5em; font-weight:700; color:#2e7d32; margin-bottom:0.5em;}
.stButton>button {background-color:#388e3c; color:white; font-weight:600; border-radius:8px;}
.result-box {
    background: linear-gradient(90deg, #e8f5e9 60%, #c8e6c9 100%);
    border-radius: 14px;
    padding: 1.5em 2em;
    margin-top: 1.5em;
    box-shadow: 0 2px 8px rgba(46,125,50,0.08);
    display: flex;
    align-items: center;
}
.result-label {
    font-size: 1.3em;
    font-weight: 700;
    color: #1b5e20;
    margin-right: 1em;
}
.result-crop {
    font-size: 2em;
    font-weight: 800;
    color: #388e3c;
    letter-spacing: 2px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🌾 Crop Recommendation System</div>', unsafe_allow_html=True)

st.write("Select the soil and climate parameters using the sliders below:")

col1, col2, col3 = st.columns(3)
with col1:
    N = st.slider("Nitrogen (N)", min_value=0, max_value=200, value=90, step=1)
    P = st.slider("Phosphorus (P)", min_value=0, max_value=200, value=40, step=1)
    K = st.slider("Potassium (K)", min_value=0, max_value=200, value=40, step=1)
with col2:
    temperature = st.slider("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
    humidity = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.1)
with col3:
    ph = st.slider("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    rainfall = st.slider("Rainfall (mm)", min_value=0.0, max_value=500.0, value=200.0, step=0.1)

features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

if hasattr(model, "predict_proba") and model.__class__.__name__ in ["LogisticRegression", "KNeighborsClassifier", "MLPClassifier"]:
    features = scaler.transform(features)

st.markdown("---")
st.write("Click the button below to get your recommended crop:")

recommend = st.button("🌱 Recommend Crop", use_container_width=True)
if recommend:
    prediction = model.predict(features)
    crop = le.inverse_transform(prediction)[0]
    st.markdown(f'''<div class="result-box">✅ <span class="result-label">Recommended Crop:</span> <span class="result-crop">{crop}</span></div>''', unsafe_allow_html=True)
