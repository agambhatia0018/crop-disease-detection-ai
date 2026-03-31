import os
import streamlit as st
import anthropic
import base64
from PIL import Image
import io

# =============================================
#   CROP DISEASE DETECTION - Claude AI Powered
# =============================================

st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Crop Disease Detection System")
st.caption("Powered by Claude AI — No model file needed!")

# ----- Sidebar: API Key Input -----
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Get your key from https://console.anthropic.com"
    )
    st.markdown("---")
    st.markdown("**Supported Classes:**")
    st.markdown("""
    - 🥔 Potato Early Blight  
    - 🥔 Potato Late Blight  
    - 🥔 Potato Healthy  
    - 🍅 Tomato Early Blight  
    - 🍅 Tomato Late Blight  
    - 🍅 Tomato Healthy  
    """)

# ----- Main: Image Upload -----
uploaded_file = st.file_uploader(
    "📤 Upload Leaf Image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image of a potato or tomato leaf"
)

if uploaded_file is not None:

    # Show uploaded image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(uploaded_file, caption="Uploaded Leaf Image", use_column_width=True)

    # Predict button
    if st.button("🔍 Detect Disease", use_container_width=True, type="primary"):

        if not api_key:
            st.error("❌ Please enter your Anthropic API Key in the sidebar.")
        else:
            with st.spinner("🤖 Claude is analyzing the leaf..."):

                # Convert image to base64
                img_bytes = uploaded_file.read()
                img_base64 = base64.standard_b64encode(img_bytes).decode("utf-8")

                # Detect media type
                file_ext = uploaded_file.name.split(".")[-1].lower()
                media_type_map = {
                    "jpg": "image/jpeg",
                    "jpeg": "image/jpeg",
                    "png": "image/png"
                }
                media_type = media_type_map.get(file_ext, "image/jpeg")

                # Claude API call
                try:
                    client = anthropic.Anthropic(api_key=api_key)

                    message = client.messages.create(
                        model="claude-opus-4-5",
                        max_tokens=512,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": media_type,
                                            "data": img_base64,
                                        },
                                    },
                                    {
                                        "type": "text",
                                        "text": """You are an expert agricultural plant disease detection AI.

Analyze this leaf image and classify it into EXACTLY ONE of these 6 classes:
1. Potato___Early_blight
2. Potato___Late_blight
3. Potato___healthy
4. Tomato___Early_blight
5. Tomato___Late_blight
6. Tomato___healthy

Respond ONLY in this exact format (no extra text):
CLASS: <class_name>
CONFIDENCE: <number between 0 and 100>
REASON: <one sentence explanation>

If the image is not a leaf or unclear, respond:
CLASS: Unknown
CONFIDENCE: 0
REASON: <explain why>"""
                                    }
                                ],
                            }
                        ],
                    )

                    # Parse response
                    response_text = message.content[0].text.strip()
                    lines = response_text.split("\n")

                    result_class = "Unknown"
                    confidence = 0.0
                    reason = "Could not parse response."

                    for line in lines:
                        if line.startswith("CLASS:"):
                            result_class = line.replace("CLASS:", "").strip()
                        elif line.startswith("CONFIDENCE:"):
                            try:
                                confidence = float(line.replace("CONFIDENCE:", "").strip())
                            except:
                                confidence = 0.0
                        elif line.startswith("REASON:"):
                            reason = line.replace("REASON:", "").strip()

                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Detection Result")

                    if result_class == "Unknown":
                        st.warning("⚠️ Could not identify a crop leaf in this image. Please upload a clear potato or tomato leaf photo.")
                    else:
                        # Color based on health
                        if "healthy" in result_class.lower():
                            st.success(f"✅ **{result_class.replace('_', ' ')}**")
                        else:
                            st.error(f"🦠 **Disease Detected: {result_class.replace('_', ' ')}**")

                        # Confidence bar
                        st.metric("Confidence", f"{confidence:.1f}%")
                        st.progress(confidence / 100)

                        # Reason
                        st.info(f"💬 **Analysis:** {reason}")

                        # Low confidence warning
                        if confidence < 70:
                            st.warning("⚠️ Low confidence prediction. Try uploading a clearer image.")

                        # Recommendations
                        st.markdown("---")
                        st.subheader("💡 Recommendations")

                        recommendations = {
                            "Potato___Early_blight": "Apply fungicide (chlorothalonil or mancozeb). Remove infected leaves. Ensure proper spacing for air circulation.",
                            "Potato___Late_blight": "Immediately apply copper-based fungicide. Remove and destroy infected plants. Avoid overhead irrigation.",
                            "Potato___healthy": "Your crop looks healthy! Continue regular monitoring and maintain proper watering schedule.",
                            "Tomato___Early_blight": "Apply fungicide spray. Remove lower infected leaves. Mulch around plants to prevent soil splash.",
                            "Tomato___Late_blight": "Apply systemic fungicide urgently. Destroy infected plants. Avoid wetting foliage during watering.",
                            "Tomato___healthy": "Your tomato crop is healthy! Keep monitoring weekly and maintain good nutrition.",
                        }

                        rec = recommendations.get(result_class, "Consult a local agricultural expert for advice.")
                        st.write(f"🌱 {rec}")

                except anthropic.AuthenticationError:
                    st.error("❌ Invalid API Key. Please check your Anthropic API key in the sidebar.")
                except anthropic.RateLimitError:
                    st.error("⏳ Rate limit reached. Please wait a moment and try again.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

else:
    st.info("👆 Please upload a leaf image to get started.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Claude AI | Detects Potato & Tomato diseases")
