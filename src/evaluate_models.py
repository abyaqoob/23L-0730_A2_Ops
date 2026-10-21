import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)

from pipeline import build_dataset_from_csv

os.makedirs('screenshots', exist_ok=True)

val_ds = build_dataset_from_csv('data/processed/splits/val.csv', is_training=False)
test_ds = build_dataset_from_csv('data/processed/splits/test.csv', is_training=False)

# Extract ground-truth labels once
y_val = []
for _, labels in val_ds:
    y_val.extend(labels.numpy())
y_val = np.array(y_val).astype(int)

y_test = []
for _, labels in test_ds:
    y_test.extend(labels.numpy())
y_test = np.array(y_test).astype(int)

# Load both models
models = {
    'Baseline CNN': tf.keras.models.load_model('models/baseline_cnn.keras'),
    'MobileNetV2 (Transfer)': tf.keras.models.load_model('models/mobilenetv2_finetuned.keras')
}

def get_probs(model, ds):
    probs = []
    for images, _ in ds:
        p = model.predict(images, verbose=0)
        probs.extend(p.flatten())
    return np.array(probs)

# Step 1: Choose optimal decision threshold using validation data only
print("\n=== STEP 1: VALIDATION THRESHOLD OPTIMIZATION ===")
thresholds_chosen = {}
for name, model in models.items():
    val_probs = get_probs(model, val_ds)
    fpr, tpr, thrs = roc_curve(y_val, val_probs)
    # Youden's J statistic: maximize (Sensitivity + Specificity - 1)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thr = thrs[best_idx]
    # Bound threshold strictly between 0.1 and 0.9
    best_thr = float(np.clip(best_thr, 0.2, 0.8))
    thresholds_chosen[name] = best_thr
    print(f"Optimal threshold on Validation Data for {name}: {best_thr:.4f} (Max Youden J: {j_scores[best_idx]:.4f})")

# Step 2: Evaluate on the Test Set strictly ONCE
print("\n=== STEP 2: TEST SET EVALUATION (ONCE) ===")
metric_rows = []
test_probs_dict = {}
test_preds_dict = {}

for name, model in models.items():
    probs = get_probs(model, test_ds)
    thr = thresholds_chosen[name]
    preds = (probs >= thr).astype(int)
    
    test_probs_dict[name] = probs
    test_preds_dict[name] = preds

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc_val = roc_auc_score(y_test, probs)

    metric_rows.append({
        'Model': name,
        'Val Threshold': round(thr, 4),
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4),
        'ROC-AUC': round(auc_val, 4)
    })

metrics_df = pd.DataFrame(metric_rows)
print(metrics_df.to_string(index=False))
metrics_df.to_csv('data/processed/task2_5_metrics_summary.csv', index=False)

# Step 3: Plot Combined ROC Curves
fig, ax = plt.subplots(figsize=(7, 6))
colors = {'Baseline CNN': 'gray', 'MobileNetV2 (Transfer)': 'blue'}
for name in models:
    fpr, tpr, _ = roc_curve(y_test, test_probs_dict[name])
    ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, test_probs_dict[name]):.3f})", color=colors[name], linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax.set_title('Test Set ROC Curves (23L-0730_A2_Ops)', fontsize=12, fontweight='bold')
ax.set_xlabel('False Positive Rate (1 - Specificity)')
ax.set_ylabel('True Positive Rate (Sensitivity)')
ax.legend(loc='lower right')
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('screenshots/T2_5_roc_curves.png', dpi=200)
plt.close()

# Step 4: Plot Side-by-Side Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('VisionGuard (23L-0730_A2_Ops) - Test Set Confusion Matrices', fontsize=14, fontweight='bold')
target_names = ['RealArt', 'AiArtData']

for idx, (name, preds) in enumerate(test_preds_dict.items()):
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap=plt.cm.Blues, ax=axes[idx], values_format='d')
    axes[idx].set_title(f"{name}\n(Threshold = {thresholds_chosen[name]:.2f})", fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('screenshots/T2_5_confusion_matrices.png', dpi=200)
plt.close()
print("\nGenerated screenshots: screenshots/T2_5_roc_curves.png and screenshots/T2_5_confusion_matrices.png")
