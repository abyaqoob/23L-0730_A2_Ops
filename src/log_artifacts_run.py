import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import mlflow
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix, roc_curve, auc

from pipeline import build_dataset_from_csv, IMG_SIZE

# Load credentials
load_dotenv()
tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if not tracking_uri:
    raise ValueError("MLFLOW_TRACKING_URI not configured in .env.")

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("VisionGuard_Artifacts")

os.makedirs('screenshots', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('artifacts_staging', exist_ok=True)

# 1. Export package dependencies list
subprocess.run("pip freeze > artifacts_staging/requirements.txt", shell=True, check=True)

# 2. Copy data split lists into staging
for split in ['train.csv', 'val.csv', 'test.csv']:
    src_split = f"data/processed/splits/{split}"
    if os.path.exists(src_split):
        subprocess.run(f"cp {src_split} artifacts_staging/{split}", shell=True, check=True)

# 3. Load datasets
train_ds = build_dataset_from_csv('data/processed/splits/train.csv', is_training=True)
val_ds = build_dataset_from_csv('data/processed/splits/val.csv', is_training=False)
test_ds = build_dataset_from_csv('data/processed/splits/test.csv', is_training=False)

# Collect ground-truth labels from test set
y_true = []
for _, labels in test_ds:
    y_true.extend(labels.numpy())
y_true = np.array(y_true).astype(int)

# 4. Build and train candidate model
def build_model():
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="input_image")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet"
    )
    backbone.trainable = True
    for layer in backbone.layers[:-40]:
        layer.trainable = False
        
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="VisionGuard_Artifact_Model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

with mlflow.start_run(run_name="VisionGuard_Artifacts_Logging") as run:
    print(f"\n=================== RUNNING ARTIFACTS LOGGING ===================")
    mlflow.set_tag("student_id", "23L-0730")
    mlflow.set_tag("task", "3.4_artifacts")

    model = build_model()
    model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=1)

    # 5. Evaluate on Test Set
    probs = []
    for imgs, _ in test_ds:
        p = model.predict(imgs, verbose=0)
        probs.extend(p.flatten())
    probs = np.array(probs)
    threshold = 0.4494
    y_pred = (probs >= threshold).astype(int)

    # 6. Generate & Save Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['AI-Generated (0)', 'Authentic (1)'],
                yticklabels=['AI-Generated (0)', 'Authentic (1)'])
    plt.title('Confusion Matrix on Test Set')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    cm_path = 'artifacts_staging/confusion_matrix.png'
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # 7. Generate & Save ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    roc_path = 'artifacts_staging/roc_curve.png'
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close()

    # 8. Save Trained Keras Model Artifact
    model_path = 'artifacts_staging/trained_visionguard_model.keras'
    model.save(model_path)

    # 9. Log all staged artifacts to MLflow
    mlflow.log_artifact(cm_path, artifact_path="evaluation_plots")
    mlflow.log_artifact(roc_path, artifact_path="evaluation_plots")
    mlflow.log_artifact("artifacts_staging/requirements.txt", artifact_path="environment")
    mlflow.log_artifact("artifacts_staging/train.csv", artifact_path="data_splits")
    mlflow.log_artifact("artifacts_staging/val.csv", artifact_path="data_splits")
    mlflow.log_artifact("artifacts_staging/test.csv", artifact_path="data_splits")
    mlflow.log_artifact(model_path, artifact_path="model")

    print(f"\nAll artifacts successfully uploaded to DagsHub MLflow run: {run.info.run_id}")
