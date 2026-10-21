import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, average_precision_score

from pipeline import build_dataset_from_csv, IMG_SIZE, BATCH_SIZE

os.makedirs('models', exist_ok=True)
os.makedirs('screenshots', exist_ok=True)

def build_transfer_model(backbone_type="mobilenetv2"):
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="input_image")
    
    if backbone_type == "mobilenetv2":
        # MobileNetV2 expects inputs in [-1, 1]
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
            include_top=False,
            weights="imagenet"
        )
    elif backbone_type == "resnet50v2":
        # ResNet50V2 expects inputs in [-1, 1]
        x = tf.keras.applications.resnet_v2.preprocess_input(inputs)
        backbone = tf.keras.applications.ResNet50V2(
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
            include_top=False,
            weights="imagenet"
        )
    else:
        raise ValueError("Unsupported backbone")

    backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense128")(x)
    x = tf.keras.layers.Dropout(0.4, name="dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="classifier")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{backbone_type.upper()}_Classifier")
    return model, backbone

def train_and_finetune(backbone_name, train_ds, val_ds):
    print(f"\n==================== TRAINING {backbone_name.upper()} ====================")
    model, backbone = build_transfer_model(backbone_name)
    save_path = f"models/{backbone_name}_finetuned.keras"

    # Stage 1: Feature Extraction (Backbone Frozen)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )
    print(f"--- Stage 1: Training classifier head (Backbone Frozen) ---")
    model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=1)

    # Stage 2: Fine-Tuning (Unfreeze Top Layers)
    backbone.trainable = True
    # Freeze the first 70% of layers, fine-tune the top 30%
    fine_tune_at = int(len(backbone.layers) * 0.70)
    for layer in backbone.layers[:fine_tune_at]:
        layer.trainable = False

    print(f"--- Stage 2: Fine-tuning top {len(backbone.layers) - fine_tune_at} layers with low learning rate ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=save_path, monitor="val_loss", save_best_only=True, verbose=1)
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks, verbose=1)
    return tf.keras.models.load_model(save_path)

def evaluate_model(model, test_ds):
    y_true, y_probs = [], []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_probs.extend(probs.flatten())
    y_true = np.array(y_true).astype(int)
    y_probs = np.array(y_probs)
    y_pred = (y_probs >= 0.5).astype(int)
    return y_true, y_probs, y_pred

if __name__ == "__main__":
    train_ds = build_dataset_from_csv('data/processed/splits/train.csv', is_training=True)
    val_ds = build_dataset_from_csv('data/processed/splits/val.csv', is_training=False)
    test_ds = build_dataset_from_csv('data/processed/splits/test.csv', is_training=False)

    # 1. Train MobileNetV2
    mb_model = train_and_finetune("mobilenetv2", train_ds, val_ds)
    y_true, mb_probs, mb_pred = evaluate_model(mb_model, test_ds)
    print("\n--- MobileNetV2 Test Report ---")
    print(classification_report(y_true, mb_pred, target_names=['RealArt', 'AiArtData'], digits=4))

    # 2. Train ResNet50V2
    res_model = train_and_finetune("resnet50v2", train_ds, val_ds)
    _, res_probs, res_pred = evaluate_model(res_model, test_ds)
    print("\n--- ResNet50V2 Test Report ---")
    print(classification_report(y_true, res_pred, target_names=['RealArt', 'AiArtData'], digits=4))

    # 3. Baseline comparison
    baseline_model = tf.keras.models.load_model('models/baseline_cnn.keras')
    _, base_probs, base_pred = evaluate_model(baseline_model, test_ds)

    # 4. Generate Combined Comparison Figure (ROC Curves & Precision-Recall Curves)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('VisionGuard (23L-0730_A2_Ops) - Model Comparison on Test Set', fontsize=14, fontweight='bold')

    # ROC Curves
    for name, probs, color in [('Baseline CNN', base_probs, 'gray'), ('MobileNetV2', mb_probs, 'blue'), ('ResNet50V2', res_probs, 'green')]:
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', color=color, linewidth=2)
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0].set_title('Receiver Operating Characteristic (ROC) Curve')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Precision-Recall Curves
    for name, probs, color in [('Baseline CNN', base_probs, 'gray'), ('MobileNetV2', mb_probs, 'blue'), ('ResNet50V2', res_probs, 'green')]:
        prec, rec, _ = precision_recall_curve(y_true, probs)
        ap = average_precision_score(y_true, probs)
        axes[1].plot(rec, prec, label=f'{name} (AP = {ap:.3f})', color=color, linewidth=2)
    axes[1].set_title('Precision-Recall Curve')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    comparison_path = 'screenshots/T2_3_transfer_learning_comparison.png'
    plt.savefig(comparison_path, dpi=200)
    plt.close()
    print(f"\nSaved model comparison curves to: {comparison_path}")
    print("Task 2.3 complete.")
