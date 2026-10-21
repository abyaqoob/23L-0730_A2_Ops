import os
import pandas as pd
import tensorflow as tf
import keras_tuner as kt

os.makedirs('data/processed', exist_ok=True)

# 1. Load built-in Fashion-MNIST
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train[:10000] / 255.0  # Subset for fast execution
y_train = y_train[:10000]
x_val = x_val[:2000] / 255.0
y_val = y_val[:2000]

def build_toy_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    
    # Hyperparameter 1: Number of dense units
    units = hp.Choice('units', values=[32, 64, 128])
    model.add(tf.keras.layers.Dense(units=units, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # Hyperparameter 2: Learning rate
    lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

tuner = kt.RandomSearch(
    build_toy_model,
    objective='val_accuracy',
    max_trials=6,
    executions_per_trial=1,
    directory='tuner_logs',
    project_name='fashion_mnist_toy',
    overwrite=True
)

print("\n=== STARTING TOY KERAS TUNER RUN (FASHION-MNIST) ===")
tuner.search(x_train, y_train, epochs=4, validation_data=(x_val, y_val), verbose=1)

# Extract and format trial results into a clean table
trials = tuner.oracle.get_best_trials(num_trials=6)
toy_records = []
for idx, trial in enumerate(trials):
    toy_records.append({
        'Trial ID': trial.trial_id,
        'Dense Units': trial.hyperparameters.values['units'],
        'Learning Rate': trial.hyperparameters.values['learning_rate'],
        'Val Accuracy': round(trial.score, 4)
    })

toy_df = pd.DataFrame(toy_records)
toy_df.to_csv('data/processed/task2_7_toy_results.csv', index=False)
print("\n=== TOY RESULTS TABLE (FASHION-MNIST) ===")
print(toy_df.to_string(index=False))

best_hp = tuner.get_best_hyperparameters()[0]
print(f"\nBest Settings: units={best_hp.get('units')}, learning_rate={best_hp.get('learning_rate')}")
