# 🌿 AI-Based Crop Disease Detection System

## 📌 Overview

This project is an AI-based crop disease detection system that uses deep learning techniques to identify diseases from plant leaf images. The system is trained on the PlantVillage dataset and supports multiple models for better accuracy and performance comparison.

🚀 **Includes a working Streamlit web application for real-time prediction**

---

## 🎯 Objective

To build an intelligent system that can automatically detect crop diseases using image processing and deep learning models, helping farmers take early action.

---

## 🚀 Features

* 🌱 Image-based disease detection
* 🤖 Multiple model comparison (CNN, MobileNetV2, ResNet50)
* ⚡ Real-time image prediction
* 🌐 Web application using Streamlit
* 📊 Accuracy visualization using graphs
* 📁 Structured and clean project implementation

---

## 🌐 Web Application (Streamlit)

This project has been converted into a web application using Streamlit.

The user can upload a leaf image, and the model predicts the disease along with confidence.

### ▶️ How to Run the App

1. Install dependencies:

```
pip install streamlit tensorflow numpy matplotlib
```

2. Run the application:

```
streamlit run app.py
```

3. Open in browser:

```
http://localhost:8501
```

---

### ⚙️ How It Works

* The trained model (`best_model.h5`) is loaded
* User uploads an image through the interface
* Image is preprocessed
* Model predicts the disease
* Output + confidence score is displayed

---

## 📂 Dataset

* 📌 **Dataset Used:** PlantVillage Dataset
* 🍅 Crops Covered: Tomato
* 🥔 Crops Covered: Potato

### 🦠 Classes:

* Tomato Early Blight
* Tomato Late Blight
* Tomato Healthy
* Potato Early Blight
* Potato Late Blight
* Potato Healthy

---

## 🛠 Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Google Colab
* Streamlit

---

## 🧠 Models Used

* 🔹 Convolutional Neural Network (CNN)
* 🔹 MobileNetV2 (Transfer Learning)
* 🔹 ResNet50 (Transfer Learning)

---

## 📊 Results

The models were trained and compared based on validation accuracy.

* CNN: Good performance
* MobileNetV2: Faster and efficient
* ResNet50: High accuracy with deep architecture

📌 Comparison graph is included in the notebook.

---

## 🖼️ Output

**Input:** Leaf Image
**Output:** Disease Prediction + Confidence Score

Example:
👉 Potato Leaf → Predicted: Early Blight (Confidence: 92%)

---

## ▶️ How to Run the Project (Notebook)

1. Open the notebook in Google Colab
2. Upload the dataset (PlantVillage - Tomato & Potato)
3. Run all cells step by step
4. Upload a leaf image for prediction
5. View the predicted disease output

---

## 📁 Project Structure

* `crop_disease_detection.ipynb` → Main project file
* `app.py` → Streamlit web application
* `best_model.h5` → Trained model
* `README.md` → Project documentation
* `LICENSE` → License file
* `.gitignore` → Ignored files

---

## 🔮 Future Scope

* 📱 Mobile Application Integration
* 📷 Real-time camera-based detection
* 🌍 Deployment as a live web application
* 📈 Improved accuracy with larger datasets

---

## 💡 Key Learning

* Deep Learning for Image Classification
* Transfer Learning techniques
* Model comparison and evaluation
* Converting ML model into web application
* Real-world problem solving using AI

---

## 👨‍💻 Author

* **Agam Bhatia**
* **Rudraksh Singh**

---

## ⭐ Acknowledgment

Thanks to the PlantVillage dataset and TensorFlow community for providing valuable resources.

---

## 📌 Note

This project is for educational purposes and demonstrates the use of AI in agriculture.
