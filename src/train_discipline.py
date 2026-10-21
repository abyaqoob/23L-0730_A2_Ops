import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# 1. Deterministic Seeds
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from pipeline import build_dataset_from_csv, IMG_SIZE, BATCH_SIZE

os.makedirs('models', exist_ok=True)
os.makedirs('screenshots', exist_ok=True)

def build_model():
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="input_image")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet"
    )
    # Freeze bottom 70%, unfreeze top 30%
    backbone.trainable = True
    fine_tune_at = int(len(backbone.layers) * 0.70)
    for layer in backbone.layers[:fine_tune_at]:
        layer.trainable = False

    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense128")(x)
    x = tf.keras.layers.Dropout(0.4, name="dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="classifier")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_Discipline")
    return model

def plot_curves_with_best_epoch(history, save_path="screenshots/T2_4_training_discipline_curves.png"):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    # Best epoch based on minimum validation loss
    best_epoch_idx = int(np.argmin(val_loss))  # 0-indexed
    best_epoch = best_epoch_idx + 1
    best_val_loss = val_loss[best_epoch_idx]
    best_val_acc = val_acc[best_epoch_idx]

    print("\n" + "="*50)
    print(f"BEST EPOCH STATED: Epoch {best_epoch}")
    print(f"Minimum Validation Loss: {best_val_loss:.4f}")
    print(f"Validation Accuracy at Best Epoch: {best_val_acc*100:.2f}%")
    print("="*50 + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'VisionGuard (23L-0730_A2_Ops) - Training Discipline (Best Epoch: {best_epoch})', fontsize=14, fontweight='bold')

    # Accuracy Plot
    axes[0].plot(epochs, acc, 'b-o', label='Training Accuracy', linewidth=2)
    axes[0].plot(epochs, val_acc, 'g-s', label='Validation Accuracy', linewidth=2)
    axes[0].axvline(best_epoch, color='crimson', linestyle='--', linewidth=2, label=f'Best Epoch ({best_epoch})')
    axes[0].scatter([best_epoch], [best_val_acc], color='crimson', s=150, zorder=5, marker='*')
    axes[0].set_title('Accuracy vs. Epoch', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Loss Plot
    axes[1].plot(epochs, loss, 'r-o', label='Training Loss', linewidth=2)
    axes[1].plot(epochs, val_loss, 'darkorange', marker='s', label='Validation Loss', linewidth=2)
    axes[1].axvline(best_epoch, color='crimson', linestyle='--', linewidth=2, label=f'Best Epoch ({best_epoch})')
    axes[1].scatter([best_epoch], [best_val_loss], color='crimson', s=150, zorder=5, marker='*')
    axes[1].set_title('Binary Crossentropy Loss vs. Epoch', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved deliverable curve with marked best epoch to: {save_path}")
    return best_epoch, best_val_loss, best_val_acc

if __name__ == '__main__':
    train_ds = build_dataset_from_csv('data/processed/splits/train.csv', is_training=True)
    val_ds = build_dataset_from_csv('data/processed/splits/val.csv', is_training=False)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )

    # Rubric Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='models/best_disciplined_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]

    print("Training with deterministic seeds and training discipline callbacks...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=callbacks
    )

    plot_curves_with_best_epoch(history)
