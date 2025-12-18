import streamlit as st
import numpy as np
import tensorflow as tf
import os
import urllib.request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="WaAI Chest X-ray",
    page_icon="🩻",
    layout="centered"
)

st.title("🩻 WaAI Chest X-ray Classifier")
st.caption("AI-powered Pneumonia Detection from Chest X-rays")

# ----------------------------
# Load Model (from HuggingFace)
# ----------------------------
MODEL_PATH = "WaAI_Xray_Model.keras"
MODEL_URL = "https://huggingface.co/ososoo/PneumoniaClassifier/resolve/main/WaAI_Xray_Model.keras"

@st.cache_resource
def load_waai_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading AI model..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return load_model(MODEL_PATH)

model = load_waai_model()

st.success("✅ Model loaded successfully")

# ----------------------------
# File Upload
# ----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    with st.spinner("🧠 Analyzing X-ray..."):
        prediction = model.predict(img_array)[0][0]

    # Result
    st.subheader("🧪 Diagnosis Result")

    if prediction < 0.5:
        confidence = (1 - prediction) * 100
        st.success(f"🟢 NORMAL")
        st.metric(label="Confidence", value=f"{confidence:.2f}%")
    else:
        confidence = prediction * 100
        st.error(f"🔴 PNEUMONIA")
        st.metric(label="Confidence", value=f"{confidence:.2f}%")

    st.warning(
        "⚠️ This tool is for educational purposes only and **not** a medical diagnosis."
    )

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Built with ❤️ by **Osama Youssef | ننشر الرحمة من خلال العلم** — Version 1.0")
