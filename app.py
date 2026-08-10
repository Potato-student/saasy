import sys
print("Starting SaaSy...", flush=True)

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import base64
import io
import os
import fitz  # pymupdf

app = Flask(__name__, static_folder=".")
CORS(app)

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

# ─── IMAGE TOOLS ───────────────────────────────────────────

@app.route("/compress", methods=["POST"])
def compress():
    try:
        file = request.files["image"]
        quality = int(request.form.get("quality", 70))
        img = Image.open(file).convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode("utf-8")
        return jsonify({"result_b64": b64, "format": "jpeg"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/resize", methods=["POST"])
def resize():
    try:
        file = request.files["image"]
        width = int(request.form.get("width", 800))
        height = int(request.form.get("height", 600))
        img = Image.open(file).convert("RGB")
        img = img.resize((width, height), Image.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode("utf-8")
        return jsonify({"result_b64": b64, "format": "png"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/convert", methods=["POST"])
def convert():
    try:
        file = request.files["image"]
        fmt = request.form.get("format", "png").upper()
        if fmt == "JPG":
            fmt = "JPEG"
        img = Image.open(file).convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format=fmt)
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode("utf-8")
        return jsonify({"result_b64": b64, "format": fmt.lower()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─── PDF TOOLS ─────────────────────────────────────────────

@app.route("/merge-pdf", methods=["POST"])
def merge_pdf():
    try:
        files = request.files.getlist("pdfs")
        merged = fitz.open()
        for file in files:
            pdf_bytes = file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            merged.insert_pdf(doc)
            doc.close()
        buffer = io.BytesIO()
        merged.save(buffer)
        merged.close()
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode("utf-8")
        return jsonify({"result_b64": b64, "format": "pdf"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/compress-pdf", methods=["POST"])
def compress_pdf():
    try:
        file = request.files["pdf"]
        pdf_bytes = file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        buffer = io.BytesIO()
        doc.save(buffer, garbage=4, deflate=True, clean=True)
        doc.close()
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode("utf-8")
        return jsonify({"result_b64": b64, "format": "pdf"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/split-pdf", methods=["POST"])
def split_pdf():
    try:
        file = request.files["pdf"]
        page = int(request.form.get("page", 1)) - 1
        pdf_bytes = file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        if page < 0 or page >= len(doc):
            return jsonify({"error": f"Page must be between 1 and {len(doc)}"}), 400

        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=page, to_page=page)
        buffer = io.BytesIO()
        new_doc.save(buffer)
        new_doc.close()
        doc.close()
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode("utf-8")
        return jsonify({"result_b64": b64, "format": "pdf", "total_pages": len(doc)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
