import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from pipeline import build_dataset_from_csv, IMG_SIZE, BATCH_SIZE

os.makedirs('models', exist_ok=True)
os.makedirs('screenshots', exist_ok=True)

def build_baseline_cnn():
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="input_image")
    # Normalize pixels from [0, 255] to [0, 1]
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)

    # Conv Block 1
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', name="conv1")(x)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Conv Block 2
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', name="conv2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Conv Block 3
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu', name="conv3")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Classification Head
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dense(64, activation='relu', name="dense1")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name="classifier")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Baseline_CNN_23L_0730")
    return model

def plot_training_history(history, save_path="screenshots/T2_2_baseline_training_curves.png"):
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    epochs_range = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('VisionGuard (23L-0730_A2_Ops) - Baseline CNN Training Curves', fontsize=14, fontweight='bold')

    axes[0].plot(epochs_range, acc, label='Train Accuracy', marker='o')
    axes[0].plot(epochs_range, val_acc, label='Val Accuracy', marker='s')
    axes[0].set_title('Accuracy vs Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].plot(epochs_range, loss, label='Train Loss', marker='o', color='red')
    axes[1].plot(epochs_range, val_loss, label='Val Loss', marker='s', color='orange')
    axes[1].set_title('Loss vs Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Binary Crossentropy Loss')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved baseline training curves to: {save_path}")

def evaluate_and_plot_cm(model, test_ds, save_path="screenshots/T2_2_baseline_confusion_matrix.png"):
    y_true = []
    y_pred_probs = []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred_probs.extend(probs.flatten())

    y_true = np.array(y_true).astype(int)
    y_pred = (np.array(y_pred_probs) >= 0.5).astype(int)

    # Class names: 0 -> RealArt, 1 -> AiArtData
    target_names = ['RealArt', 'AiArtData']
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    print("\n=== BASELINE TEST SET EVALUATION REPORT ===")
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.title('Baseline CNN Confusion Matrix (23L-0730_A2_Ops)', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved baseline confusion matrix to: {save_path}")

if __name__ == '__main__':
    train_ds = build_dataset_from_csv('data/processed/splits/train.csv', is_training=True)
    val_ds = build_dataset_from_csv('data/processed/splits/val.csv', is_training=False)
    test_ds = build_dataset_from_csv('data/processed/splits/test.csv', is_training=False)

    model = build_baseline_cnn()
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='models/baseline_cnn.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    print("\nStarting Baseline CNN training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=callbacks
    )

    plot_training_history(history)

    print("\nEvaluating best checkpoint on untouched Test Set...")
    best_model = tf.keras.models.load_model('models/baseline_cnn.keras')
    evaluate_and_plot_cm(best_model, test_ds)
    print("Task 2.2 complete.")
