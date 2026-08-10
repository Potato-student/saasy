import sys
import gc
import os
import io
import base64
import urllib.request
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

print("Starting app...", flush=True)

app = Flask(__name__, static_folder=".")
CORS(app)

MODEL_PATH = "/tmp/realesrgan_x4.onnx"
MODEL_URL = "https://huggingface.co/bukuroo/RealESRGAN-ONNX/resolve/main/real-esrgan-x4plus-128.onnx"

# Load session once at startup — saves 200MB RAM per request
session = None
input_name = None

def load_session():
    global session, input_name
    if session is not None:
        return
    import onnxruntime as ort

    if not os.path.exists(MODEL_PATH):
        print("Downloading AI model...", flush=True)
        req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r, open(MODEL_PATH, "wb") as f:
            f.write(r.read())
        print("Model downloaded!", flush=True)

    # Optimize ONNX session for low memory CPU
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1      # Don't fight over CPU threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_mem_pattern = True     # Reuse memory patterns
    opts.enable_mem_reuse = True       # Reuse memory buffers

    session = ort.InferenceSession(MODEL_PATH, sess_options=opts)
    input_name = session.get_inputs()[0].name
    print("Model loaded and optimized!", flush=True)

# Load model at startup so first user isn't waiting
try:
    load_session()
except Exception as e:
    print(f"Model preload failed: {e}", flush=True)

def process_tile(tile_img):
    """Process a single 128x128 tile through AI."""
    arr = np.array(tile_img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis]  # HWC -> NCHW
    out = session.run(None, {input_name: arr})[0]
    out = out[0].transpose(1, 2, 0)           # NCHW -> HWC
    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(out)

def upscale_with_ai(img, scale):
    TILE = 128
    w, h = img.size

    # Smart input cap based on scale
    # Free server: keep output under ~1200px to avoid memory crash
    MAX_PIXELS = {2: 600, 4: 300, 8: 150}
    max_in = MAX_PIXELS.get(scale, 300)

    if w > max_in or h > max_in:
        ratio = min(max_in / w, max_in / h)
        w = int(w * ratio)
        h = int(h * ratio)
        img = img.resize((w, h), Image.LANCZOS)
        print(f"Input capped to {w}x{h} for scale {scale}x", flush=True)

    # Output canvas at 4x (model always does 4x)
    out_img = Image.new("RGB", (w * 4, h * 4))

    for y in range(0, h, TILE):
        for x in range(0, w, TILE):
            x2, y2 = min(x + TILE, w), min(y + TILE, h)
            tile = img.crop((x, y, x2, y2))
            tw, th = tile.size

            # Pad to 128x128 if edge tile
            if tw < TILE or th < TILE:
                padded = Image.new("RGB", (TILE, TILE))
                padded.paste(tile, (0, 0))
                tile = padded

            result_tile = process_tile(tile)

            # Crop away padding from result
            result_tile = result_tile.crop((0, 0, tw * 4, th * 4))
            out_img.paste(result_tile, (x * 4, y * 4))

            # Aggressively free memory after each tile
            del tile, result_tile
            gc.collect()

    # Resize from 4x to requested scale
    final_w = int(w * scale)
    final_h = int(h * scale)
    if scale != 4:
        out_img = out_img.resize((final_w, final_h), Image.LANCZOS)

    return out_img

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/upscale", methods=["POST"])
def upscale():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    scale = int(request.form.get("scale", 2))
    if scale not in [2, 4, 8]:
        return jsonify({"error": "Scale must be 2, 4, or 8"}), 400

    try:
        img = Image.open(request.files["image"]).convert("RGB")
        print(f"Input: {img.size} scale: {scale}x", flush=True)

        result = upscale_with_ai(img, scale)

        # Save as JPEG for smaller file size and faster transfer
        buffer = io.BytesIO()
        result.save(buffer, format="JPEG", quality=92, optimize=True)
        buffer.seek(0)
        result_b64 = base64.b64encode(buffer.read()).decode("utf-8")

        del result, img
        gc.collect()

        print("Done!", flush=True)
        return jsonify({"result_b64": result_b64, "format": "jpeg"})

    except Exception as e:
        print(f"Error: {e}", flush=True)
        gc.collect()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
