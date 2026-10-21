import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from pipeline import build_dataset_from_csv, IMG_SIZE

os.makedirs('data/processed', exist_ok=True)
train_ds = build_dataset_from_csv('data/processed/splits/train.csv', is_training=True)
val_ds = build_dataset_from_csv('data/processed/splits/val.csv', is_training=False)

def build_visionguard_model(hp):
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="input_image")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet"
    )
    
    # Hyperparameter 1: Number of unfrozen layers
    unfreeze_count = hp.Choice('unfrozen_layers', values=[20, 40, 60])
    backbone.trainable = True
    freeze_until = len(backbone.layers) - unfreeze_count
    for layer in backbone.layers[:freeze_until]:
        layer.trainable = False

    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="dense128")(x)
    
    # Hyperparameter 2: Dropout rate
    dropout_rate = hp.Choice('dropout', values=[0.2, 0.35, 0.5])
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="classifier")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="Tuned_VisionGuard")

    # Hyperparameter 3: Learning rate
    lr = hp.Choice('learning_rate', values=[1e-4, 5e-5, 1e-5])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

tuner = kt.RandomSearch(
    build_visionguard_model,
    objective='val_accuracy',
    max_trials=8,
    executions_per_trial=1,
    directory='tuner_logs',
    project_name='visionguard_kt',
    seed=SEED,
    overwrite=True
)

print("\n=== STARTING VISIONGUARD KERAS TUNER (8 TRIALS) ===")
# 4 epochs per trial for fast parameter sampling on M4 GPU
tuner.search(train_ds, validation_data=val_ds, epochs=4, verbose=1)

# Extract and build VisionGuard results table
trials = tuner.oracle.get_best_trials(num_trials=8)
vg_records = []
for trial in trials:
    vg_records.append({
        'Trial ID': trial.trial_id,
        'Learning Rate': trial.hyperparameters.values['learning_rate'],
        'Dropout': trial.hyperparameters.values['dropout'],
        'Unfrozen Layers': trial.hyperparameters.values['unfrozen_layers'],
        'Val Accuracy': round(trial.score, 4)
    })

vg_df = pd.DataFrame(vg_records)
vg_df.to_csv('data/processed/task2_7_visionguard_results.csv', index=False)
print("\n=== TUNER RESULTS FOR VISIONGUARD (8 TRIALS) ===")
print(vg_df.to_string(index=False))

best_hp = tuner.get_best_hyperparameters()[0]
print(f"\nBest Settings: lr={best_hp.get('learning_rate')}, dropout={best_hp.get('dropout')}, unfrozen_layers={best_hp.get('unfrozen_layers')}")
