import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import mlflow
from dotenv import load_dotenv

from pipeline import IMG_SIZE

# Load tracking URI
load_dotenv()
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if not tracking_uri:
    raise ValueError("MLFLOW_TRACKING_URI not configured in .env")

mlflow.set_tracking_uri(tracking_uri)

MODEL_URI = "models:/VisionGuard_Classifier@champion"
print(f"Connecting to MLflow Model Registry...")
print(f"Loading champion model from: {MODEL_URI}\n")

# Load model by alias
model = mlflow.tensorflow.load_model(model_uri=MODEL_URI)
print("Champion model loaded successfully!\n")

# Find an existing sample image from test split or data directory
sample_img_path = None
test_csv = 'data/processed/splits/test.csv'

if os.path.exists(test_csv):
    df_test = pd.read_csv(test_csv)
    # Check the first column for valid image files
    for col in df_test.columns:
        for path in df_test[col]:
            if isinstance(path, str) and os.path.isfile(path):
                sample_img_path = path
                break
        if sample_img_path:
            break

if not sample_img_path or not os.path.isfile(sample_img_path):
    for root, _, files in os.walk('data'):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                candidate = os.path.join(root, file)
                if os.path.isfile(candidate):
                    sample_img_path = candidate
                    break
        if sample_img_path:
            break

print(f"Sample Image Path: {sample_img_path}")

# Preprocess image
img = tf.keras.utils.load_img(sample_img_path, target_size=IMG_SIZE)
img_array = tf.keras.utils.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)

# Run Inference
probability = float(model.predict(img_batch, verbose=0)[0][0])
DECISION_THRESHOLD = 0.4494
predicted_label = "Authentic (1)" if probability >= DECISION_THRESHOLD else "AI-Generated (0)"

print("\n" + "="*50)
print("           CHAMPION MODEL INFERENCE RESULT")
print("="*50)
print(f"Sample Image Tested   : {sample_img_path}")
print(f"Predicted Probability : {probability:.4f}")
print(f"Decision Threshold    : {DECISION_THRESHOLD}")
print(f"Classification Result : {predicted_label}")
print("="*50)
