import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# 1. Load clean manifest
manifest_path = 'data/processed/dataset_clean.csv'
if not os.path.exists(manifest_path):
    raise FileNotFoundError("Clean manifest not found. Run Task 1.2 first.")

df = pd.read_csv(manifest_path)
os.makedirs('screenshots', exist_ok=True)

# 2. Figure 1: Sample Grid per Class
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle('VisionGuard (23L-0730_A2_Ops) - Sample Image Grid per Class', fontsize=16, fontweight='bold')

for row_idx, cls_name in enumerate(['AiArtData', 'RealArt']):
    subset = df[df['class'] == cls_name].sample(4, random_state=42)
    for col_idx, (_, row) in enumerate(subset.iterrows()):
        ax = axes[row_idx, col_idx]
        img = Image.open(row['path'])
        ax.imshow(img)
        ax.set_title(f"{cls_name}\n{row['width']}x{row['height']} ({row['size_kb']:.1f} KB)", fontsize=10)
        ax.axis('off')

plt.tight_layout()
sample_grid_path = 'screenshots/T1_3_sample_grid.png'
plt.savefig(sample_grid_path, dpi=200)
plt.close()
print(f"Saved sample grid to: {sample_grid_path}")

# 3. Figure 2: Histograms of Width, Height, and File Size
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('VisionGuard (23L-0730_A2_Ops) - Distribution Metrics per Class', fontsize=16, fontweight='bold')

metrics = [
    ('width', 'Image Width (px)', axes[0]),
    ('height', 'Image Height (px)', axes[1]),
    ('size_kb', 'File Size (KB)', axes[2])
]

colors = {'AiArtData': '#e74c3c', 'RealArt': '#2ecc71'}

for col_name, label_name, ax in metrics:
    for cls_name in ['AiArtData', 'RealArt']:
        subset = df[df['class'] == cls_name]
        ax.hist(subset[col_name], bins=30, alpha=0.6, label=cls_name, color=colors[cls_name])
    ax.set_title(f'Distribution of {label_name}', fontsize=12, fontweight='bold')
    ax.set_xlabel(label_name)
    ax.set_ylabel('Image Count')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
dist_path = 'screenshots/T1_3_distribution_histograms.png'
plt.savefig(dist_path, dpi=200)
plt.close()
print(f"Saved distribution histograms to: {dist_path}")
print("EDA figures generated successfully!")
