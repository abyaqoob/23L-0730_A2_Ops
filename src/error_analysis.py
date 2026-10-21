import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

from pipeline import parse_image_and_label, IMG_SIZE, BATCH_SIZE, LABEL_MAP

test_csv = 'data/processed/splits/test.csv'
model_path = 'models/mobilenetv2_finetuned.keras'

df_test = pd.read_csv(test_csv)
model = tf.keras.models.load_model(model_path)

print(f"Running detailed error analysis on {len(df_test)} test images...")

# Collect predictions image by image to track exact filenames and paths
results = []
for idx, row in df_test.iterrows():
    img_tensor, label = parse_image_and_label(row['path'], LABEL_MAP[row['class']])
    # Add batch dimension (1, 224, 224, 3)
    img_batch = tf.expand_dims(img_tensor, axis=0)
    prob = float(model.predict(img_batch, verbose=0)[0][0])
    pred_class = 1 if prob >= 0.5 else 0
    true_class = int(label)

    is_correct = (pred_class == true_class)
    # Error classification:
    # False Positive (FP): True = RealArt (0), Predicted = AiArtData (1)
    # False Negative (FN): True = AiArtData (1), Predicted = RealArt (0)
    error_type = 'Correct'
    if true_class == 0 and pred_class == 1:
        error_type = 'False Positive (Real flagged as AI)'
    elif true_class == 1 and pred_class == 0:
        error_type = 'False Negative (AI missed as Real)'

    results.append({
        'filename': row['filename'],
        'path': row['path'],
        'true_class_name': row['class'],
        'true_label': true_class,
        'pred_prob_ai': prob,
        'pred_label': pred_class,
        'is_correct': is_correct,
        'error_type': error_type
    })

res_df = pd.DataFrame(results)
os.makedirs('data/processed', exist_ok=True)
res_df.to_csv('data/processed/test_predictions_detailed.csv', index=False)

errors_df = res_df[~res_df['is_correct']]
print(f"\nTotal test samples: {len(res_df)}")
print(f"Correct classifications: {res_df['is_correct'].sum()} ({res_df['is_correct'].mean()*100:.2f}%)")
print(f"Total misclassifications: {len(errors_df)}")
print("\nBreakdown of Errors:")
print(errors_df['error_type'].value_counts())

# Generate Visual Failure Grid (Top 8 most confident mistakes)
errors_sorted = errors_df.copy()
# Confidence of error: how far probability is from 0.5 towards the wrong side
errors_sorted['error_confidence'] = np.where(
    errors_sorted['true_label'] == 0,
    errors_sorted['pred_prob_ai'],        # FP: higher prob AI = bigger blunder
    1.0 - errors_sorted['pred_prob_ai']    # FN: lower prob AI = bigger blunder
)
top_errors = errors_sorted.sort_values(by='error_confidence', ascending=False).head(8)

fig, axes = plt.subplots(2, 4, figsize=(15, 8))
fig.suptitle('VisionGuard (23L-0730_A2_Ops) - MobileNetV2 Failure Case Analysis', fontsize=15, fontweight='bold')

for idx, (_, row) in enumerate(top_errors.iterrows()):
    ax = axes[idx // 4, idx % 4]
    img = Image.open(row['path'])
    ax.imshow(img)
    tag = "FP (Real marked AI)" if row['true_label'] == 0 else "FN (AI marked Real)"
    color = "red" if row['true_label'] == 0 else "purple"
    ax.set_title(f"{tag}\nP(AI)={row['pred_prob_ai']:.2f}\nTrue: {row['true_class_name']}", fontsize=10, color=color, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
save_path = 'screenshots/T2_4_failure_cases.png'
plt.savefig(save_path, dpi=200)
plt.close()
print(f"\nSaved failure cases figure to: {save_path}")
print("Task 2.4 analysis complete.")
