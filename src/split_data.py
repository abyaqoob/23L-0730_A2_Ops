import os
import pandas as pd
from sklearn.model_selection import train_test_split

manifest_path = 'data/processed/dataset_clean.csv'
if not os.path.exists(manifest_path):
    raise FileNotFoundError("Clean dataset manifest not found. Run Task 1.2 first.")

df = pd.read_csv(manifest_path)
print(f"Loaded clean dataset: {len(df)} total samples.")

# 70% Train, 15% Validation, 15% Test
# Step 1: Split out test set (15%)
train_val_df, test_df = train_test_split(
    df,
    test_size=0.15,
    random_state=42,
    stratify=df['class']
)

# Step 2: Split remaining 85% into Train (70/85 ~ 82.35%) and Val (15/85 ~ 17.65%)
val_size_adjusted = 0.15 / 0.85
train_df, val_df = train_test_split(
    train_val_df,
    test_size=val_size_adjusted,
    random_state=42,
    stratify=train_val_df['class']
)

# Save split CSVs
os.makedirs('data/processed/splits', exist_ok=True)
train_df.to_csv('data/processed/splits/train.csv', index=False)
val_df.to_csv('data/processed/splits/val.csv', index=False)
test_df.to_csv('data/processed/splits/test.csv', index=False)

print("\n=== DATASET SPLIT SUMMARY (Seed = 42) ===")
print(f"Train Set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
print(train_df['class'].value_counts())

print(f"\nValidation Set: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
print(val_df['class'].value_counts())

print(f"\nTest Set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
print(test_df['class'].value_counts())
print("=========================================")
