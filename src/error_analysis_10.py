import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

from pipeline import parse_image_and_label, LABEL_MAP

test_csv = 'data/processed/splits/test.csv'
model_path = 'models/mobilenetv2_finetuned.keras'
threshold = 0.4494  # Validation-derived optimal decision threshold

df_test = pd.read_csv(test_csv)
model = tf.keras.models.load_model(model_path)

records = []
for _, row in df_test.iterrows():
    img_tensor, label = parse_image_and_label(row['path'], LABEL_MAP[row['class']])
    prob = float(model.predict(tf.expand_dims(img_tensor, axis=0), verbose=0)[0][0])
    pred = 1 if prob >= threshold else 0
    true_label = int(label)
    
    if pred != true_label:
        error_type = 'False Positive (Real marked AI)' if true_label == 0 else 'False Negative (AI marked Real)'
        # Confidence score: how far the wrong prediction is from the threshold
        conf = prob if true_label == 0 else (1.0 - prob)
        records.append({
            'filename': row['filename'],
            'path': row['path'],
            'true_class': row['class'],
            'pred_prob_ai': prob,
            'error_type': error_type,
            'conf': conf
        })

err_df = pd.DataFrame(records).sort_values(by='conf', ascending=False)
top10 = err_df.head(10).reset_index(drop=True)

# Generate 2x5 Image Grid Deliverable
fig, axes = plt.subplots(2, 5, figsize=(18, 7.5))
fig.suptitle('VisionGuard (23L-0730_A2_Ops) - 10 Misclassified Test Images (Task 2.6)', fontsize=15, fontweight='bold')

for idx, row in top10.iterrows():
    ax = axes[idx // 5, idx % 5]
    img = Image.open(row['path'])
    ax.imshow(img)
    tag = "FP" if "Positive" in row['error_type'] else "FN"
    color = "crimson" if tag == "FP" else "indigo"
    ax.set_title(f"#{idx+1}: {tag} | P(AI)={row['pred_prob_ai']:.2f}\nTrue: {row['true_class']}", fontsize=9, color=color, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
grid_path = 'screenshots/T2_6_error_grid_10.png'
plt.savefig(grid_path, dpi=200)
plt.close()

top10.to_csv('data/processed/task2_6_top10_errors.csv', index=False)
print(f"Saved 10-image grid to: {grid_path}")
print(f"Top 10 misclassified files:")
for i, r in top10.iterrows():
    print(f"Image {i+1}: {r['filename']} | True: {r['true_class']} | P(AI): {r['pred_prob_ai']:.3f} | {r['error_type']}")
