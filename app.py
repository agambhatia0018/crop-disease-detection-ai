import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Page config
st.set_page_config(page_title="AI Crop Doctor", layout="wide")

# Custom CSS (🔥 UI MAGIC)
st.markdown("""
<style>
.main {
    background-color: #f0f2f6;
}
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #2e7d32;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: gray;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="title">🌿 AI Crop Disease Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload leaf image and detect disease instantly</p>', unsafe_allow_html=True)

# Load model
model = tf.keras.models.load_model("best_model.h5")

class_names = [
    'Potato___Late_blight',
    'Potato___Early_blight',
    'Potato___healthy',
    'Tomato___Late_blight',
    'Tomato___Early_blight',
    'Tomato___healthy'
]

# Layout
col1, col2 = st.columns([1,1])

# LEFT SIDE (UPLOAD)
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📤 Upload Leaf Image")

    uploaded_file = st.file_uploader("", type=["jpg","png","jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# RIGHT SIDE (RESULT)
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Prediction Result")

    if uploaded_file is not None:
        img = image.load_img(uploaded_file, target_size=(224,224))
        img_array = image.img_to_array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)
        confidence = np.max(pred) * 100
        result = class_names[np.argmax(pred)]

        # Clean text
        result_clean = result.replace("___", " - ")

        st.success(f"🌱 Disease: {result_clean}")
        st.info(f"🔍 Confidence: {confidence:.2f}%")

        # Progress bar
        st.progress(int(confidence))

        # Treatment suggestion (🔥 EXTRA POWER)
        st.markdown("### 💊 Suggested Action")

        if "Early_blight" in result:
            st.warning("Use fungicide and remove infected leaves.")
        elif "Late_blight" in result:
            st.warning("Apply copper-based fungicide immediately.")
        else:
            st.success("Plant is healthy. No action needed.")

    else:
        st.info("Upload an image to see prediction")

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Developed using Deep Learning | CNN + Transfer Learning")
