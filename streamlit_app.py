import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(
    page_title="🐟 Fish Image Classification",
    layout="wide"
)

# =================================================
# CSS — UNDERWATER THEME
# =================================================
st.markdown("""
<style>

/* -------- BACKGROUND -------- */
.stApp {
    background-image: url("https://static.vecteezy.com/system/resources/previews/029/735/496/non_2x/underwater-life-with-coral-reefs-and-shipwreck-illustration-ocean-bottom-of-sea-world-wildlife-underwater-landscape-for-background-wallpaper-or-landing-page-deep-sea-silhouette-vector.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* -------- TITLE -------- */
.title {
    text-align: center;
    font-size: 44px;
    font-weight: 900;
    color: #E0F2FE;
    text-shadow:
        0 0 10px rgba(56,189,248,0.8),
        0 0 22px rgba(14,165,233,0.6);
    margin-bottom: 6px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    font-weight: 600;
    color: #BAE6FD;
    text-shadow: 0 0 10px rgba(56,189,248,0.6);
    margin-bottom: 28px;
}

/* -------- IMAGE FIX -------- */
div[data-testid="stImage"] img {
    height: 280px !important;
    width: auto !important;
    object-fit: contain;
}

/* -------- RESULT WRAPPER -------- */
.result-wrapper {
    width: 420px;
    margin-top: 90px;
}

/* -------- RESULT BOX -------- */
.result-box {
    background: rgba(2, 18, 35, 0.88);
    padding: 24px;
    border-radius: 20px;
    border: 1px solid rgba(56,189,248,0.45);
    box-shadow:
        0 0 25px rgba(56,189,248,0.35),
        inset 0 0 12px rgba(14,165,233,0.25);
    backdrop-filter: blur(8px);
}

/* -------- RESULT TEXT -------- */
.result-label {
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 1px;
    color: #7DD3FC;
}

.result-value {
    font-size: 30px;
    font-weight: 900;
    color: #F0F9FF;
    margin: 10px 0 6px;
    text-shadow: 0 0 10px rgba(56,189,248,0.6);
}

.result-confidence {
    font-size: 15px;
    font-weight: 700;
    color: #A7F3D0;
}

/* -------- PROGRESS BAR -------- */
.result-wrapper div[data-testid="stProgress"] > div {
    height: 12px;
    border-radius: 10px;
    background: rgba(186,230,253,0.25);
}

.result-wrapper div[data-testid="stProgress"] > div > div {
    border-radius: 10px;
    background: linear-gradient(
        90deg,
        #22D3EE,
        #38BDF8,
        #0EA5E9
    );
    box-shadow: 0 0 12px rgba(56,189,248,0.8);
}

</style>
""", unsafe_allow_html=True)

# =================================================
# CONFIG
# =================================================
IMG_SIZE = 224
MODEL_PATH = "model/fish_image_classifier_VGG16_v1.keras"
CLASS_INDEX_PATH = "model/class_indices.json"

# =================================================
# LOAD MODEL
# =================================================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(CLASS_INDEX_PATH) as f:
        idx2class = {int(k): v for k, v in json.load(f).items()}
    return model, idx2class

model, idx2class = load_model()

# =================================================
# PREPROCESS
# =================================================
preprocess_input = tf.keras.applications.vgg16.preprocess_input
def preprocess(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.expand_dims(np.array(img), axis=0)
    return preprocess_input(arr)

# =================================================
# HEADER
# =================================================
st.markdown('<div class="title">🐟 Fish Image Classification 🌊</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">📤 Upload a fish image to get an instant prediction ⚡</div>',
    unsafe_allow_html=True
)

# =================================================
# UPLOAD
# =================================================
file = st.file_uploader(
    "📂 Upload Fish Image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if file:
    img = Image.open(file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(
            img,
            caption="🖼️ Uploaded Fish Image 🐠",
            use_container_width=True
        )

    with col2:
        preds = model.predict(preprocess(img), verbose=0)[0]
        idx = int(np.argmax(preds))
        conf = float(preds[idx])
        label = idx2class[idx]

        st.markdown('<div class="result-wrapper">', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">🔍 🐠 PREDICTED FISH CATEGORY</div>
            <div class="result-value">🐟 {label}</div>
            <div class="result-confidence">📊 Confidence Score: {conf:.2%}</div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(conf)
        st.markdown('</div>', unsafe_allow_html=True)

# =================================================
# FOOTER
# =================================================
st.markdown("""
<hr>
<p style="text-align:center; font-size:13px; font-weight:600; color:#BAE6FD;">
🌊 AI Vision Dashboard • 🐟 Fish Image Classification
</p>
""", unsafe_allow_html=True)