# 🌾 Crop Disease Prediction Using CNN

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 Introduction

Crop diseases significantly reduce yield and cause economic losses. Early detection is critical to maintaining healthy crops.  
This project uses **Convolutional Neural Networks (CNN)** to automatically detect diseases in crop leaves from images. Farmers can upload a leaf image, and the model predicts whether the leaf is **healthy** or affected by a specific disease, along with actionable suggestions.

---
<img src="IMG-20260405-WA0017.jpg" width=800/>
## 🧠 Features

- CNN-based deep learning model for disease detection.
- Supports multiple crops and disease categories.
- User-friendly interface (web app using Flask).
- Real-time prediction with fast processing.
- Actionable suggestions for disease management.

---

## 💾 Dataset

- **Source:** [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)  
- **Format:** JPG / PNG images  
- **Classes:** Healthy leaves + multiple disease categories (e.g., Early Blight, Powdery Mildew, Leaf Spot)

---

## 🛠️ Tech Stack

- Python 3
- TensorFlow / Keras
- OpenCV
- Flask (for web UI)
- NumPy & Pandas
- Matplotlib / Seaborn

---

## 📁 Project Structure

```bash
crop-disease-prediction/
│
├── dataset/                   # Images of crops
│   ├── Tomato/
│   │   ├── Healthy/
│   │   ├── Early_blight/
│   │   └── ...
│   └── Potato/
│       ├── Healthy/
│       └── ...
│
├── model/
│   └── crop_disease_cnn.h5    # Trained CNN model
│
├── src/
│   ├── train_model.py          # Training script
│   ├── predict.py              # Prediction script
│   └── utils.py                # Helper functions
│
├── app.py                      # Flask web app
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── requirements.txt
└── README.md
