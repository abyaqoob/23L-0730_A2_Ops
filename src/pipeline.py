import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# Map class names to numeric targets: AiArtData -> 1, RealArt -> 0
LABEL_MAP = {'RealArt': 0, 'AiArtData': 1}

def parse_image_and_label(path, label):
    image_bytes = tf.io.read_file(path)
    # Decode image to 3-channel RGB float32 in [0, 255]
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    # Resize with pad preserves the aspect ratio by adding symmetrical black bars
    image = tf.image.resize_with_pad(image, IMG_SIZE[0], IMG_SIZE[1])
    return image, label

def get_augmentation_layer():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.05),
    ], name="data_augmentation")

def build_dataset_from_csv(csv_path, is_training=False, shuffle_buffer=1000):
    df = pd.read_csv(csv_path)
    paths = df['path'].values
    labels = df['class'].map(LABEL_MAP).values.astype('float32')

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if is_training:
        ds = ds.shuffle(buffer_size=shuffle_buffer, seed=42)

    ds = ds.map(parse_image_and_label, num_parallel_calls=AUTOTUNE)

    if is_training:
        aug_layer = get_augmentation_layer()
        ds = ds.map(lambda x, y: (aug_layer(x, training=True), y), num_parallel_calls=AUTOTUNE)

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def visualize_augmented_batch(ds, save_path="screenshots/T2_1_augmented_batch.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    for images, labels in ds.take(1):
        plt.figure(figsize=(12, 12))
        for i in range(16):
            ax = plt.subplot(4, 4, i + 1)
            # Clip pixel values between 0 and 255 for clean rendering
            img_disp = tf.clip_by_value(images[i], 0.0, 255.0).numpy().astype("uint8")
            plt.imshow(img_disp)
            cls_name = "AI-Generated" if labels[i] == 1 else "Real-Art"
            plt.title(f"{cls_name}", fontsize=10)
            plt.axis("off")
        plt.suptitle("VisionGuard (23L-0730_A2_Ops) - Augmented Training Batch Sample", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"Saved augmented batch visualization to: {save_path}")
        break

if __name__ == '__main__':
    train_csv = 'data/processed/splits/train.csv'
    val_csv = 'data/processed/splits/val.csv'
    test_csv = 'data/processed/splits/test.csv'

    print("Building datasets...")
    train_ds = build_dataset_from_csv(train_csv, is_training=True)
    val_ds = build_dataset_from_csv(val_csv, is_training=False)
    test_ds = build_dataset_from_csv(test_csv, is_training=False)

    print(f"Train batches: {len(train_ds)}, Val batches: {len(val_ds)}, Test batches: {len(test_ds)}")
    visualize_augmented_batch(train_ds)
    print("Task 2.1 pipeline validation complete.")
