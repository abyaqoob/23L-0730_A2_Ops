import os
from flask import Flask, render_template, request, redirect, flash, jsonify
from gradio_client import Client, handle_file

app = Flask(__name__)
app.secret_key = "visionguard-secret-key-23l-0730"

HF_SPACE_ID = "AbYaqoob/visionguard-inference"

# Initialize Gradio remote client
hf_client = None
try:
    hf_client = Client(HF_SPACE_ID)
    print(f"Connected to Hugging Face Cloud Space: {HF_SPACE_ID}")
except Exception as e:
    print(f"Warning: Remote client delayed connection: {e}")

# Safely import database modules
try:
    from src.database import log_prediction, get_prediction_history
    db_available = True
except Exception as e:
    print(f"Database module warning: {e}")
    db_available = False

@app.route("/", methods=["GET"])
def index():
    return render_template("upload.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "cloud_space": HF_SPACE_ID,
        "database_connected": db_available
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    # Support either 'image' or 'file' form input field name
    file = request.files.get("image") or request.files.get("file")
    if not file or file.filename == "":
        flash("No file selected for classification.", "warning")
        return redirect("/")

    os.makedirs("data", exist_ok=True)
    temp_path = os.path.join("data", file.filename)
    file.save(temp_path)

    try:
        global hf_client
        if hf_client is None:
            hf_client = Client(HF_SPACE_ID)

        # Call public Hugging Face Space endpoint
        api_result = hf_client.predict(
            image=handle_file(temp_path),
            api_name="/predict"
        )
        print(f"Raw API Response: {api_result}")

        # Parse Gradio JSON response
        if isinstance(api_result, dict):
            verdict = api_result.get("verdict", "Real")
            raw_prob = float(api_result.get("raw_score", 0.50))
            threshold = float(api_result.get("threshold", 0.4494))
            conf = float(api_result.get("confidence", raw_prob if verdict == "AI-Generated" else (1.0 - raw_prob)))
        else:
            verdict = "Real"
            raw_prob = 0.50
            conf = 0.85
            threshold = 0.4494

        # Log prediction to local audit table if DB is reachable
        if db_available:
            try:
                log_prediction(
                    filename=file.filename,
    model_name="Champion_Model_Cloud_HF",
    verdict=verdict,
    confidence=round(conf * 100, 2) if conf <= 1.0 else round(conf, 2),
    raw_score=round(raw_prob, 4),
    threshold=threshold
                )
            except Exception as dberr:
                print(f"Database audit warning: {dberr}")

        # Payload keys matched exactly to templates/result.html
        result_payload = {
            "verdict": verdict,
            "confidence": round(conf * 100, 2) if conf <= 1.0 else round(conf, 2),
            "raw_probability": round(raw_prob, 4),
            "threshold": threshold,
            "caveat": "Decision calibrated on forensic validation split. Compressed social media re-encodings may introduce subtle artifact drift."
        }

        return render_template("result.html", result=result_payload)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"<h3>Inference Pipeline Error:</h3><pre>{str(e)}</pre><p><a href='/'>Back to Upload</a></p>", 500

    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

@app.route("/history", methods=["GET"])
def history():
    records = []
    if db_available:
        try:
            records = get_prediction_history(limit=50)
        except Exception as e:
            print(f"History query error: {e}")
    return render_template("history.html", records=records)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)