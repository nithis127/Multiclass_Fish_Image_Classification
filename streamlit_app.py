import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# -------------------------------------------------
# BACKGROUND IMAGE FUNCTION
# -------------------------------------------------
def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-attachment: fixed;
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="🐟 Fish Image Classification",
    layout="wide"
)

set_background(
    "https://static.vecteezy.com/system/resources/previews/029/735/496/non_2x/underwater-life-with-coral-reefs-and-shipwreck-illustration-ocean-bottom-of-sea-world-wildlife-underwater-landscape-for-background-wallpaper-or-landing-page-deep-sea-silhouette-vector.jpg"
)

# -------------------------------------------------
# DARK OVERLAY
# -------------------------------------------------
st.markdown(
    """
    <style>
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        background: rgba(3, 15, 30, 0.75);
        z-index: -1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# CLEAN OCEAN UI (NO GLOW)
# -------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        color: #BEEFFF;
    }

    .card {
        background: rgba(3, 15, 30, 0.78);
        backdrop-filter: blur(10px);
        padding: 16px;
        border-radius: 14px;
        border: 1px solid rgba(34,211,238,0.25);
        margin-bottom: 18px;
    }

    .title {
        text-align: center;
        font-size: 38px;
        font-weight: 800;
        color: #0F172A;
        text-shadow:
        0 0 6px rgba(56, 189, 248, 0.6),
        0 0 14px rgba(56, 189, 248, 0.45);
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #7DD3FC;
        margin-bottom: 22px;
    }

    h3, h4 {
        color: #9BDCFD;
        font-weight: 600;
    }

    button {
        background: linear-gradient(135deg, #38BDF8, #22D3EE) !important;
        color: #00131F !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        border: none !important;
    }

    button:hover {
        opacity: 0.9;
    }

    img {
        max-height: 260px;
        object-fit: contain;
        border-radius: 10px;
    }

    /* -------- PREDICTION RESULT BOX -------- */
    .prediction-box {
        background: rgba(2, 20, 40, 0.90);
        border-left: 6px solid #22D3EE;
        padding: 18px;
        border-radius: 10px;
        margin-top: 14px;
    }

    .prediction-title {
        font-size: 14px;
        font-weight: 600;
        color: #9BDCFD;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    .prediction-value {
        font-size: 26px;
        font-weight: 800;
        color: #E6FAFF;
        margin-bottom: 8px;
    }

    .prediction-confidence {
        font-size: 15px;
        color: #7DD3FC;
        font-weight: 600;
    }

    hr {
        border: 0;
        height: 1px;
        background: rgba(56,189,248,0.35);
        margin: 24px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
IMG_SIZE = 224
MODEL_PATH = "model/fish_classifier_model.keras"
CLASS_INDEX_PATH = "model/class_indices.json"

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_INDEX_PATH, "r") as f:
        idx2class = {int(k): v for k, v in json.load(f).items()}
    return model, idx2class

model, idx2class = load_model_and_classes()

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "image" not in st.session_state:
    st.session_state.image = None
if "predictions" not in st.session_state:
    st.session_state.predictions = None

# -------------------------------------------------
# IMAGE PREPROCESSING
# -------------------------------------------------
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image)
    arr = np.expand_dims(arr, axis=0)
    return arr

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown('<div class="title">🐟 Fish Image Classification</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">🌊 Upload a fish image and let AI identify the species</div>',
    unsafe_allow_html=True
)

# -------------------------------------------------
# IMAGE UPLOAD
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Fish Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------------------------
# IMAGE + PREDICTION
# -------------------------------------------------
if uploaded_file:
    st.session_state.image = Image.open(uploaded_file).convert("RGB")

    col_img, col_pred = st.columns(2)

    # IMAGE
    with col_img:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🖼️ Uploaded Image")
        st.image(st.session_state.image, caption="Fish Image", width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    # PREDICTION
    with col_pred:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🔍 Model Output")

        if st.button("🚀 Predict Fish", width="stretch"):
            with st.spinner("Analyzing image..."):
                st.session_state.predictions = model.predict(
                    preprocess_image(st.session_state.image),
                    verbose=0
                )[0]

        if st.session_state.predictions is not None:
            preds = st.session_state.predictions
            idx = int(np.argmax(preds))
            confidence = float(preds[idx])

            st.markdown(
                f"""
                <div class="prediction-box">
                    <div class="prediction-title">Predicted Fish Category</div>
                    <div class="prediction-value">🐟 {idx2class[idx]}</div>
                    <div class="prediction-confidence">
                        Confidence: {confidence:.2%}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(confidence)

        st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# TOP-3 + GRAPH
# -------------------------------------------------
if st.session_state.predictions is not None:
    preds = st.session_state.predictions
    col_left, col_right = st.columns([1, 1.8])

    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🥇 Top-3 Predictions")
        for r, i in enumerate(preds.argsort()[-3:][::-1], 1):
            st.write(f"**{r}. {idx2class[i]}** — {preds[i]:.2%}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📊 Class Probability Distribution")
        st.bar_chart({idx2class[i]: float(preds[i]) for i in range(len(preds))})
        st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown(
    """
    <hr>
    <p style="text-align:center; font-size:13px; color:#7DD3FC;">
    🌊 AI Vision Dashboard • 🐟 Fish Image Classification • Streamlit
    </p>
    """,
    unsafe_allow_html=True
)