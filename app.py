import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# ✅ Render-safe upload directory
UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ✅ Limit upload size (5MB)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

# ===============================
# LOAD TFLITE MODEL
# ===============================
interpreter = tf.lite.Interpreter(model_path="crop_disease_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ===============================
# LOAD CLASS NAMES
# ===============================
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# ===============================
# PREDICTION FUNCTION
# ===============================
def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]["index"])
    index = np.argmax(preds)
    confidence = float(preds[0][index])

    return class_names[index], confidence

# ===============================
# ROUTES
# ===============================
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

# ===============================
# START APP
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
