import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

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

# ✅ HOME + PREDICT (SAME PAGE)
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    img_path = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(save_path)

            result, confidence = predict_disease(save_path)
            img_path = save_path

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        img_path=img_path
    )

# ✅ RUN
if __name__ == "__main__":
    app.run(debug=True)
