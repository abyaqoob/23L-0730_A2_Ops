import os
import io
import numpy as np
from PIL import Image
import tensorflow as tf
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "models/champion_model.keras")
THRESHOLD = float(os.getenv("DECISION_THRESHOLD", "0.4494"))
TARGET_SIZE = (224, 224)

# Load model once into memory
print(f"[INIT] Loading champion model once from: {MODEL_PATH} ...")
if not os.path.exists(MODEL_PATH):
    # Fallback to registered artifact staging if local path doesn't exist
    fallback_path = "artifacts_staging/trained_visionguard_model.keras"
    if os.path.exists(fallback_path):
        MODEL_PATH = fallback_path

model = tf.keras.models.load_model(MODEL_PATH)
print("[INIT] Model successfully loaded into memory.\n")

def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decodes image directly from memory and matches training preprocessing."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(TARGET_SIZE, Image.Resampling.BILINEAR)
    img_array = tf.keras.utils.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    # Identical to training pipeline (MobileNetV2: [-1, 1] scaling)
    processed_batch = tf.keras.applications.mobilenet_v2.preprocess_input(img_batch)
    return processed_batch

def predict_image(image_bytes: bytes):
    """Generates verdict, calibrated confidence score, and forensic caveat."""
    tensor = preprocess_image_bytes(image_bytes)
    raw_prob = float(model.predict(tensor, verbose=0)[0][0])
    
    # Label mapping: >= threshold is Authentic (1), else AI-Generated (0)
    if raw_prob >= THRESHOLD:
        verdict = "Authentic"
        # Confidence scaled relative to threshold boundary
        confidence = (raw_prob - THRESHOLD) / (1.0 - THRESHOLD)
    else:
        verdict = "AI-Generated"
        confidence = (THRESHOLD - raw_prob) / THRESHOLD
        
    confidence_pct = round(max(50.0, min(100.0, confidence * 100.0)), 2)

    caveat = (
        "Classification is based on high-frequency artifact analysis from MobileNetV2. "
        "Adversarial perturbations, strong compression, or out-of-distribution generators "
        "can influence boundary reliability."
    )

    return {
        "verdict": verdict,
        "confidence": confidence_pct,
        "raw_probability": round(raw_prob, 4),
        "threshold": THRESHOLD,
        "caveat": caveat
    }

if __name__ == "__main__":
    import glob
    print(">>> Running 4.1 standalone verification for two known images <<<")
    sample_images = glob.glob("data/**/*.jpg", recursive=True) + glob.glob("data/**/*.png", recursive=True)
    if len(sample_images) >= 2:
        for p in sample_images[:2]:
            with open(p, "rb") as f:
                res = predict_image(f.read())
            print(f"\nImage: {p}")
            print(f"Result: {res['verdict']} | Confidence: {res['confidence']}% | Raw: {res['raw_probability']}")
    else:
        print("Please supply at least two sample images in 'data/' to test directly.")
