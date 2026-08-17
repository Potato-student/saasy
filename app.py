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
        total_pages = len(doc)

        if page < 0 or page >= total_pages:
            return jsonify({"error": f"Page must be between 1 and {total_pages}"}), 400

        new_doc = fitz.open()
        new_doc.insert_pdf(doc, from_page=page, to_page=page)
        buffer = io.BytesIO()
        new_doc.save(buffer)
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode("utf-8")

        new_doc.close()
        doc.close()

        return jsonify({"result_b64": b64, "format": "pdf", "total_pages": total_pages})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/crop", methods=["POST"])
def crop():
    try:
        file = request.files["image"]
        x = int(request.form.get("x", 0))
        y = int(request.form.get("y", 0))
        width = int(request.form.get("width", 100))
        height = int(request.form.get("height", 100))

        img = Image.open(file).convert("RGB")

        # Make sure crop doesn't go outside image bounds
        x = max(0, min(x, img.width))
        y = max(0, min(y, img.height))
        width = max(1, min(width, img.width - x))
        height = max(1, min(height, img.height - y))

        cropped = img.crop((x, y, x + width, y + height))

        buffer = io.BytesIO()
        cropped.save(buffer, format="PNG")
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode("utf-8")
        return jsonify({"result_b64": b64, "format": "png"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/img-to-pdf", methods=["POST"])
def img_to_pdf():
    try:
        files = request.files.getlist("images")
        pdf = fitz.open()
        for file in files:
            img = Image.open(file).convert("RGB")
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)
            img_bytes = buffer.read()
            img_doc = fitz.open(stream=img_bytes, filetype="png")
            rect = img_doc[0].rect
            page = pdf.new_page(width=rect.width, height=rect.height)
            page.show_pdf_page(rect, img_doc, 0)
            img_doc.close()
        buffer = io.BytesIO()
        pdf.save(buffer)
        pdf.close()
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode("utf-8")
        return jsonify({"result_b64": b64, "format": "pdf"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/pdf-to-img", methods=["POST"])
def pdf_to_img():
    try:
        file = request.files["pdf"]
        page_num = int(request.form.get("page", 1)) - 1
        pdf_bytes = file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        if page_num < 0 or page_num >= total_pages:
            return jsonify({"error": f"Page must be between 1 and {total_pages}"}), 400
        page = doc[page_num]
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        doc.close()
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        return jsonify({"result_b64": b64, "format": "png", "total_pages": total_pages})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/docx-to-pdf", methods=["POST"])
def docx_to_pdf():
    try:
        from docx2pdf import convert
        import tempfile
        file = request.files["file"]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        out_path = tmp_path.replace(".docx", ".pdf")
        convert(tmp_path, out_path)
        with open(out_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        os.unlink(tmp_path)
        os.unlink(out_path)
        return jsonify({"result_b64": b64, "format": "pdf"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
