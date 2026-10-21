import os
import matplotlib.pyplot as plt
import tensorflow as tf
import mlflow
from dotenv import load_dotenv

# Load credentials from .env via environment variables
load_dotenv()

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if not tracking_uri:
    raise ValueError("MLFLOW_TRACKING_URI not found in environment variables (.env).")

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Toy_Fashion_MNIST_DagsHub")

os.makedirs('screenshots', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Prepare lightweight Fashion-MNIST
(x_train, y_train), (x_val, y_val) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train[:5000] / 255.0
y_train = y_train[:5000]
x_val = x_val[:1000] / 255.0
y_val = y_val[:1000]

class MLflowEpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for metric_name, val in logs.items():
            mlflow.log_metric(metric_name, float(val), step=epoch + 1)

def run_experiment(lr, run_name):
    with mlflow.start_run(run_name=run_name):
        print(f"\n=================== DAGSHUB TOY RUN: {run_name} (lr={lr}) ===================")
        
        # 1. Log parameters and tags
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("batch_size", 64)
        mlflow.log_param("epochs", 5)
        mlflow.log_param("optimizer", "Adam")
        mlflow.set_tag("platform", "DagsHub")
        mlflow.set_tag("dataset", "Fashion-MNIST")
        mlflow.set_tag("student_id", "23L-0730")

        # 2. Build model
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # 3. Train
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=5,
            batch_size=64,
            callbacks=[MLflowEpochLogger()],
            verbose=1
        )

        # 4. Generate & Log Plot
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        epochs_range = range(1, 6)
        
        axes[0].plot(epochs_range, history.history['accuracy'], label='Train Acc', marker='o')
        axes[0].plot(epochs_range, history.history['val_accuracy'], label='Val Acc', marker='s')
        axes[0].set_title(f'Accuracy (lr={lr})')
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.6)

        axes[1].plot(epochs_range, history.history['loss'], label='Train Loss', marker='o', color='crimson')
        axes[1].plot(epochs_range, history.history['val_loss'], label='Val Loss', marker='s', color='darkorange')
        axes[1].set_title(f'Loss (lr={lr})')
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.6)

        plot_path = f"screenshots/T3_2_dagshub_plot_{run_name.lower().replace(' ', '_')}.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

        mlflow.log_artifact(plot_path, artifact_path="plots")

        # 5. Save & Log Model
        local_model_path = f"models/dagshub_toy_{lr}.keras"
        model.save(local_model_path)
        mlflow.log_artifact(local_model_path, artifact_path="model")
        print(f"Logged run: {run_name} successfully to DagsHub.")

if __name__ == "__main__":
    run_experiment(lr=0.001, run_name="DagsHub_Toy_LR_0.001")
    print("\nTask 3.2 DagsHub toy run completed successfully.")
