import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load saved model
model = tf.keras.models.load_model("best_model.h5")

# Class names (same order as training)
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
    img = image.load_img(uploaded_file, target_size=(224,224))
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)

result = class_names[np.argmax(pred)]
confidence = np.max(pred) * 100

st.success(f"🌱 Disease Detected: {result.replace('_',' ')}")
st.info(f"Confidence: {confidence:.2f}%")

if confidence < 70:
    st.warning("⚠️ Low confidence prediction, try another image")