import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("best_model.h5")

# Class names
class_names = [
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___healthy'
]

st.title("🌿 Crop Disease Detection System")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    result = class_names[np.argmax(pred)]
    confidence = np.max(pred) * 100

    st.success(f"🌱 Disease Detected: {result.replace('_', ' ')}")
    st.info(f"Confidence: {confidence:.2f}%")

    if confidence < 70:
        st.warning("⚠️ Low confidence prediction, try another image")
