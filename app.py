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
MODEL_URL = "https://huggingface.co/bukuroo/RealESRGAN-ONNX/resolve/main/real-esrgan-x4plus-128.onnx"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading AI model...", flush=True)
        try:
            req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as response, open(MODEL_PATH, "wb") as out_file:
                out_file.write(response.read())
            print("Model downloaded!", flush=True)
        except Exception as e:
            print(f"Download failed: {e}", flush=True)
            raise Exception(f"Model download failed: {e}")

def upscale_with_ai(img):
    import onnxruntime as ort

    download_model()

    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name

    TILE_SIZE = 128
    img_width, img_height = img.size

    # Calculate output size (4x upscale)
    out_width = img_width * 4
    out_height = img_height * 4
    output = Image.new("RGB", (out_width, out_height))

    # Process in 128x128 tiles
    for y in range(0, img_height, TILE_SIZE):
        for x in range(0, img_width, TILE_SIZE):
            # Crop tile from input
            x_end = min(x + TILE_SIZE, img_width)
            y_end = min(y + TILE_SIZE, img_height)
            tile = img.crop((x, y, x_end, y_end))

            # Pad tile to exactly 128x128 if needed
            if tile.size != (TILE_SIZE, TILE_SIZE):
                padded = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (0, 0, 0))
                padded.paste(tile, (0, 0))
                tile = padded

            # Prepare for model
            tile_array = np.array(tile).astype(np.float32) / 255.0
            tile_array = np.transpose(tile_array, (2, 0, 1))
            tile_array = np.expand_dims(tile_array, axis=0)

            # Run AI
            result = session.run(None, {input_name: tile_array})[0]

            # Convert back
            result = np.squeeze(result, axis=0)
            result = np.transpose(result, (1, 2, 0))
            result = np.clip(result * 255, 0, 255).astype(np.uint8)
            result_tile = Image.fromarray(result)

            # Crop result to match original tile size (remove padding)
            actual_w = (x_end - x) * 4
            actual_h = (y_end - y) * 4
            result_tile = result_tile.crop((0, 0, actual_w, actual_h))

            # Paste into output
            output.paste(result_tile, (x * 4, y * 4))

    return output

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

        # AI always does 4x
        result = upscale_with_ai(img)

        # Adjust to requested scale
        if scale == 2:
            result = result.resize((img.width * 2, img.height * 2), Image.LANCZOS)
        elif scale == 8:
            result = result.resize((img.width * 8, img.height * 8), Image.LANCZOS)

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
