from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)

# Load your trained Braille recognition model
model = load_model("Braille.h5")

# Braille character mapping
braille_dict = {
    "100000": "A", "101000": "B", "110000": "C", "110100": "D",
    "100100": "E", "111000": "F", "111100": "G", "101100": "H",
    "011000": "I", "011100": "J", "100010": "K", "101010": "L",
    "110010": "M", "110110": "N", "100110": "O", "111010": "P",
    "111110": "Q", "101110": "R", "011010": "S", "011110": "T",
    "100011": "U", "101011": "V", "011101": "W", "110011": "X",
    "110111": "Y", "100111": "Z",
    "001111": "1", "011011": "2", "100101": "3", "100111": "4",
    "101101": "5", "101111": "6", "110101": "7", "110111": "8",
    "111101": "9", "111111": "0"
}

@app.route("/process", methods=["POST"])
def process_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    image = Image.open(image_file.stream).convert("L")
    image = np.array(image)
    image = cv2.resize(image, (28, 28)) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    predicted_label = np.argmax(prediction)

    braille_code = format(predicted_label, "06b")  # Convert label to Braille code
    text = braille_dict.get(braille_code, "?")  # Map to corresponding letter

    return jsonify({"text": text})

if __name__ == "__main__":
    app.run(debug=True)
