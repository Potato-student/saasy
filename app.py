import sys
print("Starting app...", flush=True)

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import base64
import io
import os
import urllib.request
import numpy as np

app = Flask(__name__, static_folder=".")
CORS(app)

MODEL_PATH = "/tmp/realesrgan_x4.onnx"
MODEL_URL = "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.onnx"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading AI model... this takes 30 seconds", flush=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded!", flush=True)

def upscale_with_ai(img):
    import onnxruntime as ort

    download_model()

    # Prepare image
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))  # HWC to CHW
    img_array = np.expand_dims(img_array, axis=0)   # Add batch dimension

    # Run AI model
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img_array})[0]

    # Convert back to image
    output = np.squeeze(output, axis=0)
    output = np.transpose(output, (1, 2, 0))  # CHW to HWC
    output = np.clip(output * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(output)

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/upscale", methods=["POST"])
def upscale():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    scale = int(request.form.get("scale", 2))

    if scale not in [2, 4, 8]:
        return jsonify({"error": "Scale must be 2, 4, or 8"}), 400

    try:
        img = Image.open(image_file).convert("RGB")

        # AI upscale (always 4x from model)
        result = upscale_with_ai(img)

        # If user wants 2x, downscale from 4x result
        # If user wants 8x, upscale further from 4x result
        if scale == 2:
            result = result.resize(
                (img.width * 2, img.height * 2), Image.LANCZOS
            )
        elif scale == 8:
            result = result.resize(
                (img.width * 8, img.height * 8), Image.LANCZOS
            )
        # scale == 4 is already perfect from the model

        buffer = io.BytesIO()
        result.save(buffer, format="PNG")
        buffer.seek(0)

        result_b64 = base64.b64encode(buffer.read()).decode("utf-8")
        return jsonify({"result_b64": result_b64})

    except Exception as e:
        print(f"Error: {e}", flush=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
