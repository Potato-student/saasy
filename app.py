from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder=".")
CORS(app)

HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/caidas/swin2SR-classical-sr-x2-64"

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/upscale", methods=["POST"])
def upscale():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]

    try:
        # Save uploaded file temporarily
        suffix = os.path.splitext(image_file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            image_file.save(tmp.name)
            tmp_path = tmp.name

        # Send to Hugging Face
        with open(tmp_path, "rb") as f:
            image_data = f.read()

        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        response = requests.post(API_URL, headers=headers, data=image_data)

        # Clean up temp file
        os.unlink(tmp_path)

        if response.status_code == 200:
            # Save result and return it as base64
            import base64
            result_b64 = base64.b64encode(response.content).decode("utf-8")
            return jsonify({"result_b64": result_b64})
        else:
            return jsonify({"error": "Model error: " + response.text}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)