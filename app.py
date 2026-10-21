import os
import io
import datetime
from PIL import Image
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from src.inference import predict_image
from src.database import init_db, log_prediction, get_prediction_history

load_dotenv()

app = Flask(__name__)

# Settings from environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "models/champion_model.keras")
MODEL_IDENTIFIER = os.getenv("MODEL_IDENTIFIER", "VisionGuard_Classifier@champion")
DECISION_THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.4494"))
PORT = int(os.getenv("PORT", "5001"))
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 5 * 1024 * 1024))

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Initialize database schema at startup
init_db()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET"])
def index():
    return render_template("upload.html")

@app.route("/predict", methods=["POST"])
def handle_prediction():
    if "file" not in request.files:
        return render_template("error.html", status_code=400, message="No file payload provided."), 400

    file = request.files["file"]
    if file.filename == "":
        return render_template("error.html", status_code=400, message="No file selected for upload."), 400

    filename = secure_filename(file.filename)
    if not allowed_file(filename):
        return render_template("error.html", status_code=415, 
                               message=f"Unsupported file format '{filename}'. Allowed formats: PNG, JPG, JPEG, WEBP."), 415

    image_bytes = file.read()

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.verify()
    except Exception:
        return render_template("error.html", status_code=422, 
                               message="Corrupted payload. The uploaded file looks like an image but cannot be decoded."), 422

    # Model inference
    result = predict_image(image_bytes)

    # Task 5: Permanent SQLite audit record
    log_prediction(
        filename=filename,
        model_name=MODEL_IDENTIFIER,
        verdict=result["verdict"],
        confidence=result["confidence"],
        raw_score=result["raw_probability"],
        threshold=DECISION_THRESHOLD
    )

    return render_template("result.html", result=result, filename=filename)

@app.route("/history", methods=["GET"])
def history():
    # Task 5: Query permanent database records
    records = get_prediction_history(limit=50)
    return render_template("history.html", records=records)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "model_path": MODEL_PATH,
        "model_identifier": MODEL_IDENTIFIER,
        "decision_threshold": DECISION_THRESHOLD,
        "database": "sqlite://data/visionguard_audit.db",
        "timestamp": datetime.datetime.now().isoformat()
    }), 200

@app.errorhandler(413)
def request_entity_too_large(error):
    return render_template("error.html", status_code=413, 
                           message="File size exceeds the 5MB limit. Please upload a smaller image."), 413

if __name__ == "__main__":
    print(f"Starting VisionGuard Flask server on http://localhost:{PORT}")
    print("Debug mode: OFF (Production configuration)")
    app.run(host="0.0.0.0", port=PORT, debug=False)
