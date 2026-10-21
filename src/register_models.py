import os
import tensorflow as tf
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

from pipeline import build_dataset_from_csv, IMG_SIZE

load_dotenv()
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if not tracking_uri:
    raise ValueError("MLFLOW_TRACKING_URI is not set in .env")

mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient()

REGISTERED_MODEL_NAME = "VisionGuard_Classifier"

def build_model(unfrozen_layers=40):
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="input_image")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet"
    )
    backbone.trainable = True
    freeze_until = len(backbone.layers) - unfrozen_layers
    for layer in backbone.layers[:freeze_until]:
        layer.trainable = False
        
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

train_ds = build_dataset_from_csv('data/processed/splits/train.csv', is_training=True)
val_ds = build_dataset_from_csv('data/processed/splits/val.csv', is_training=False)

# -------------------------------------------------------------
# 1. VERSION 1: Baseline Candidate (unfreeze = 40)
# -------------------------------------------------------------
print("\n>>> Training and Logging Version 1 (Baseline: unfreeze=40) <<<")
mlflow.set_experiment("VisionGuard_Registry_Staging")
with mlflow.start_run(run_name="Candidate_v1_Baseline") as run1:
    model_v1 = build_model(unfrozen_layers=40)
    model_v1.fit(train_ds, validation_data=val_ds, epochs=1, verbose=1)
    
    # Log model artifact with registration
    mlflow.tensorflow.log_model(
        model=model_v1,
        artifact_path="model",
        registered_model_name=REGISTERED_MODEL_NAME
    )
    print("Version 1 registered successfully.")

# -------------------------------------------------------------
# 2. VERSION 2: Champion Architecture (unfreeze = 80)
# -------------------------------------------------------------
print("\n>>> Training and Logging Version 2 (Champion: unfreeze=80) <<<")
with mlflow.start_run(run_name="Champion_v2_Deep_Unfreeze") as run2:
    model_v2 = build_model(unfrozen_layers=80)
    model_v2.fit(train_ds, validation_data=val_ds, epochs=2, verbose=1)
    
    mlflow.tensorflow.log_model(
        model=model_v2,
        artifact_path="model",
        registered_model_name=REGISTERED_MODEL_NAME
    )
    print("Version 2 registered successfully.")

# -------------------------------------------------------------
# 3. SET ALIAS 'champion' ON VERSION 2
# -------------------------------------------------------------
print("\n>>> Assigning 'champion' alias to Version 2 <<<")
try:
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version="2"
    )
    print("Alias 'champion' successfully linked to Version 2.")
except Exception as e:
    print(f"Notice setting alias via API: {e}")
    print("You can also set the alias 'champion' directly in the DagsHub Model Registry UI.")

print("\nModel registry setup complete!")
