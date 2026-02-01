import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# ✅ REQUIRED FOR RENDER
UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Prevent large upload crashes
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB

# Load model
model = load_model("crop_disease_model.h5")

# Load class names
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    index = np.argmax(preds)
    confidence = preds[0][index]

    return class_names[index], confidence

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None

    if request.method == "POST":
        if "image" not in request.files:
            return "No image uploaded", 400

        file = request.files["image"]
        if file.filename == "":
            return "No selected file", 400

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        result, confidence = predict_disease(file_path)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run()
