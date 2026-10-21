import os
import random
import numpy as np
import tensorflow as tf
import mlflow
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from pipeline import build_dataset_from_csv, IMG_SIZE

# Load DagsHub credentials from .env
load_dotenv()
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if not tracking_uri:
    raise ValueError("MLFLOW_TRACKING_URI not configured in .env.")

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("VisionGuard_Comparison_Study")

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load data splits
train_ds = build_dataset_from_csv('data/processed/splits/train.csv', is_training=True)
val_ds = build_dataset_from_csv('data/processed/splits/val.csv', is_training=False)
test_ds = build_dataset_from_csv('data/processed/splits/test.csv', is_training=False)

# Collect true test labels
y_test = []
for _, labels in test_ds:
    y_test.extend(labels.numpy())
y_test = np.array(y_test).astype(int)

def build_custom_model(unfrozen_layers=40, dropout=0.35, learning_rate=5e-5):
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="input_image")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet"
    )
    
    if unfrozen_layers == 0:
        backbone.trainable = False
    else:
        backbone.trainable = True
        freeze_until = len(backbone.layers) - unfrozen_layers
        for layer in backbone.layers[:freeze_until]:
            layer.trainable = False
            
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
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
        "test_precision": float(precision_score(y_test, preds, zero_division=0)),
        "test_recall": float(recall_score(y_test, preds, zero_division=0)),
        "test_f1": float(f1_score(y_test, preds, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, probs))
    }

class MLflowEpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for k, v in logs.items():
            mlflow.log_metric(k, float(v), step=epoch + 1)

# Define 6 distinct configurations differing in ONE variable from baseline
experiments = [
    {
        "name": "Exp1_Baseline",
        "lr": 5e-5, "unfrozen": 40, "dropout": 0.35, "delta": "Baseline (lr=5e-5, unfreeze=40, drop=0.35)"
    },
    {
        "name": "Exp2_High_LR",
        "lr": 5e-4, "unfrozen": 40, "dropout": 0.35, "delta": "High Learning Rate (5e-4)"
    },
    {
        "name": "Exp3_Low_LR",
        "lr": 1e-5, "unfrozen": 40, "dropout": 0.35, "delta": "Low Learning Rate (1e-5)"
    },
    {
        "name": "Exp4_Fully_Frozen",
        "lr": 5e-5, "unfrozen": 0, "dropout": 0.35, "delta": "Fully Frozen Backbone (unfreeze=0)"
    },
    {
        "name": "Exp5_Deep_Unfreeze",
        "lr": 5e-5, "unfrozen": 80, "dropout": 0.35, "delta": "Deeper Fine-Tuning (unfreeze=80)"
    },
    {
        "name": "Exp6_High_Dropout",
        "lr": 5e-5, "unfrozen": 40, "dropout": 0.55, "delta": "Higher Dropout Regularization (0.55)"
    }
]

for idx, exp in enumerate(experiments, 1):
    print(f"\n=======================================================")
    print(f"[{idx}/6] RUNNING: {exp['name']} | Delta: {exp['delta']}")
    print(f"=======================================================")
    
    with mlflow.start_run(run_name=exp["name"]):
        mlflow.log_param("learning_rate", exp["lr"])
        mlflow.log_param("unfrozen_layers", exp["unfrozen"])
        mlflow.log_param("dropout", exp["dropout"])
        mlflow.log_param("batch_size", 32)
        mlflow.log_param("epochs", 3)
        mlflow.set_tag("variation", exp["delta"])
        mlflow.set_tag("student_id", "23L-0730")

        model = build_custom_model(
            unfrozen_layers=exp["unfrozen"],
            dropout=exp["dropout"],
            learning_rate=exp["lr"]
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=3,
            callbacks=[MLflowEpochLogger()],
            verbose=1
        )

        # Evaluate on test set
        test_metrics = evaluate_test(model)
        for metric_name, val in test_metrics.items():
            mlflow.log_metric(metric_name, val)
            
        print(f"Finished {exp['name']}: Val Loss={history.history['val_loss'][-1]:.4f}, Test AUC={test_metrics['test_roc_auc']:.4f}, Test F1={test_metrics['test_f1']:.4f}")

print("\nAll 6 comparison runs completed successfully!")
