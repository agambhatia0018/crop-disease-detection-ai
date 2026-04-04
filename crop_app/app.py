import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# ------------------------
# PAGE CONFIG
# ------------------------
st.set_page_config(page_title="Crop Disease Detection", page_icon="🌿")

# ------------------------
# LOAD MODEL
# ------------------------
model = tf.keras.models.load_model("best_model.h5")

class_names = [
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___healthy'
]

solutions = {
    "Tomato___Late_blight": "Use fungicide and remove infected leaves",
    "Tomato___Early_blight": "Apply proper irrigation and fungicide",
    "Potato___Early_blight": "Use disease-free seeds",
    "Potato___Late_blight": "Remove infected plants immediately"
}

# ------------------------
# SESSION STATE
# ------------------------
if "page" not in st.session_state:
    st.session_state.page = "upload"

# ------------------------
# GLOBAL STYLE
# ------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}

/* spacing for text */
.info-section {
    padding-right: 40px;
}

/* card style */
.box {
    background-color: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------
# PAGE 1: UPLOAD
# ------------------------
if st.session_state.page == "upload":

    st.title("🌿 Crop Disease Detection")
    st.write("Upload a leaf image to detect disease")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if uploaded_file is not None:
        st.session_state.image = uploaded_file
        st.session_state.page = "result"
        st.rerun()

# ------------------------
# PAGE 2: RESULT
# ------------------------
elif st.session_state.page == "result":

    st.title("🧠 Detection Result")

    img_file = st.session_state.image
    img = image.load_img(img_file, target_size=(224,224))

    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    result = class_names[np.argmax(pred)]
    confidence = np.max(pred) * 100
    solution = solutions.get(result, "No suggestion available")

    # Layout: LEFT TEXT | RIGHT IMAGE
    col1, col2 = st.columns([1,1])

    with col1:
        st.markdown('<div class="info-section">', unsafe_allow_html=True)

        st.subheader("🌱 Disease Info")
        st.write(f"Prediction: {result}")
        st.write(f"Confidence: {confidence:.2f}%")

        st.subheader("💊 Treatment")
        st.write(solution)

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.image(img_file, caption="Leaf Image", use_container_width=True)

    st.write("")

    # Back Button
    if st.button("⬅ Back"):
        st.session_state.page = "upload"
        st.rerun()