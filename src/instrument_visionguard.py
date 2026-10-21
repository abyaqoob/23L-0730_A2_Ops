import os
import subprocess
import random
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from pipeline import build_dataset_from_csv, IMG_SIZE

# Load environment credentials for DagsHub
load_dotenv()
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if not tracking_uri:
    raise ValueError("MLFLOW_TRACKING_URI not configured in .env.")

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("VisionGuard_Instrumentation")

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown_commit"

def get_dvc_hash():
    # Read the data.dvc file hash if present
    dvc_file = "data.dvc"
    if os.path.exists(dvc_file):
        with open(dvc_file, 'r') as f:
            for line in f:
                if 'md5:' in line or 'hash:' in line:
                    return line.strip().split(':')[-1].strip()
    return "v1.0_dvc_tracked"

GIT_COMMIT = get_git_commit()
DVC_VERSION = get_dvc_hash()

# Data splits
train_ds = build_dataset_from_csv('data/processed/splits/train.csv', is_training=True)
val_ds = build_dataset_from_csv('data/processed/splits/val.csv', is_training=False)
test_ds = build_dataset_from_csv('data/processed/splits/test.csv', is_training=False)

# Ground truth test labels
y_test = []
for _, labels in test_ds:
    y_test.extend(labels.numpy())
y_test = np.array(y_test).astype(int)

def build_model(learning_rate=5e-5, dropout=0.35, unfrozen_layers=40):
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
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense128")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="classifier")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_VisionGuard")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def evaluate_test(model):
    probs = []
    for imgs, _ in test_ds:
        p = model.predict(imgs, verbose=0)
        probs.extend(p.flatten())
    probs = np.array(probs)
    preds = (probs >= 0.4494).astype(int)
    
    return {
        "test_accuracy": float(accuracy_score(y_test, preds)),
        "test_precision": float(precision_score(y_test, preds)),
        "test_recall": float(recall_score(y_test, preds)),
        "test_f1": float(f1_score(y_test, preds)),
        "test_roc_auc": float(roc_auc_score(y_test, probs))
    }

class ManualMLflowCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for metric, val in logs.items():
            mlflow.log_metric(f"epoch_{metric}", float(val), step=epoch + 1)

# ==============================================================
# APPROACH A: MANUAL LOGGING VIA CUSTOM KERAS CALLBACK
# ==============================================================
print("\n>>> STARTING APPROACH A: MANUAL LOGGING (KERAS CALLBACK) <<<")
with mlflow.start_run(run_name="VisionGuard_Manual_Callback") as run:
    # 1. Log tags
    mlflow.set_tag("approach", "manual_callback")
    mlflow.set_tag("git_commit", GIT_COMMIT)
    mlflow.set_tag("dvc_data_version", DVC_VERSION)
    mlflow.set_tag("seed", str(SEED))
    mlflow.set_tag("student_id", "23L-0730")
    
    # 2. Log hyperparameters
    lr = 5e-5
    dropout = 0.35
    unfrozen = 40
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("dropout", dropout)
    mlflow.log_param("unfrozen_layers", unfrozen)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("epochs", 4)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("decision_threshold", 0.4494)

    # 3. Train model with custom manual callback
    model_manual = build_model(learning_rate=lr, dropout=dropout, unfrozen_layers=unfrozen)
    history_manual = model_manual.fit(
        train_ds,
        validation_data=val_ds,
        epochs=4,
        callbacks=[ManualMLflowCallback()],
        verbose=1
    )

    # 4. Log final test metrics
    test_metrics = evaluate_test(model_manual)
    for k, v in test_metrics.items():
        mlflow.log_metric(k, v)
    
    print(f"Manual Run Logged to DagsHub: {run.info.run_id}")

# ==============================================================
# APPROACH B: AUTOLOGGING VIA MLFLOW TENSORFLOW AUTOLOG
# ==============================================================
print("\n>>> STARTING APPROACH B: AUTOLOGGING (MLFLOW.TENSORFLOW.AUTOLOG) <<<")
# Enable autologging
mlflow.tensorflow.autolog(
    log_models=True,
    disable=False,
    exclusive=False
)

with mlflow.start_run(run_name="VisionGuard_TF_Autolog") as run:
    # 1. Log domain-specific tags that autolog misses
    mlflow.set_tag("approach", "tensorflow_autolog")
    mlflow.set_tag("git_commit", GIT_COMMIT)
    mlflow.set_tag("dvc_data_version", DVC_VERSION)
    mlflow.set_tag("seed", str(SEED))
    mlflow.set_tag("student_id", "23L-0730")
    mlflow.log_param("decision_threshold", 0.4494)

    # 2. Train model (Autolog records params, epoch metrics, and model artifact automatically)
    model_auto = build_model(learning_rate=lr, dropout=dropout, unfrozen_layers=unfrozen)
    history_auto = model_auto.fit(
        train_ds,
        validation_data=val_ds,
        epochs=4,
        verbose=1
    )

    # 3. Final test metrics
    test_metrics_auto = evaluate_test(model_auto)
    for k, v in test_metrics_auto.items():
        mlflow.log_metric(k, v)

    print(f"Autolog Run Logged to DagsHub: {run.info.run_id}")

# Disable autologging to prevent side-effects on subsequent tasks
mlflow.tensorflow.autolog(disable=True)
print("\nTask 3.3 Completed successfully!")
