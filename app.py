from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import os
import tempfile
import base64
from PIL import Image
import io

app = Flask(__name__, static_folder=".")
CORS(app)

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/upscale", methods=["POST"])
def upscale():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]

    try:
        img = Image.open(image_file)
        
        new_width = img.width * 2
        new_height = img.height * 2
        upscaled = img.resize((new_width, new_height), Image.LANCZOS)

        buffer = io.BytesIO()
        upscaled.save(buffer, format="PNG")
        buffer.seek(0)

        result_b64 = base64.b64encode(buffer.read()).decode("utf-8")
        return jsonify({"result_b64": result_b64})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
