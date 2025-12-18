import streamlit as st
import numpy as np
import tensorflow as tf
import os
import urllib.request
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
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
# Model Path & URL
# ----------------------------
MODEL_WEIGHTS_PATH = "WaAI_Xray_Model_weights.h5"
MODEL_WEIGHTS_URL = "https://huggingface.co/ososoo/PneumoniaClassifier/resolve/main/WaAI_Xray_Model.h5"

# ----------------------------
# Build Model (Functional-safe)
# ----------------------------
def build_waai_model():
    base_model = DenseNet121(include_top=False, input_shape=(224, 224, 3), weights=None)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # تحميل weights
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        with st.spinner("⬇️ Downloading AI model weights..."):
            urllib.request.urlretrieve(MODEL_WEIGHTS_URL, MODEL_WEIGHTS_PATH)
    model.load_weights(MODEL_WEIGHTS_PATH)
    return model

@st.cache_resource
def load_waai_model():
    try:
        model = build_waai_model()
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_waai_model()

if model:
    st.success("✅ Model loaded successfully")
else:
    st.stop()

# ----------------------------
# File Upload
# ----------------------------
uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    with st.spinner("🧠 Analyzing X-ray..."):
        try:
            prediction = model.predict(img_array)[0][0]
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            prediction = None

    if prediction is not None:
        st.subheader("🧪 Diagnosis Result")
        if prediction < 0.5:
            confidence = (1 - prediction) * 100
            st.success("🟢 NORMAL")
            st.metric(label="Confidence", value=f"{confidence:.2f}%")
        else:
            confidence = prediction * 100
            st.error("🔴 PNEUMONIA")
            st.metric(label="Confidence", value=f"{confidence:.2f}%")
        st.warning("⚠️ This tool is for educational purposes only and **not** a medical diagnosis.")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Built with ❤️ by **Osama Youssef | ننشر الرحمة من خلال العلم** — Version 1.0")
