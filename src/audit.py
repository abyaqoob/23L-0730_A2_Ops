import os
import hashlib
import pandas as pd
from PIL import Image

def compute_sha256(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def audit_dataset(base_dir):
    records = []
    corrupt_count = 0
    classes = ['AiArtData', 'RealArt']
    
    for cls in classes:
        cls_dir = os.path.join(base_dir, cls)
        if not os.path.exists(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            fpath = os.path.join(cls_dir, fname)
            if not os.path.isfile(fpath) or fname.startswith('.'):
                continue
            
            fsize_bytes = os.path.getsize(fpath)
            try:
                sha256_hash = compute_sha256(fpath)
                with Image.open(fpath) as img:
                    img.verify()
                
                with Image.open(fpath) as img:
                    width, height = img.size
                    mode = img.mode
                    img_format = img.format

                records.append({
                    'class': cls,
                    'filename': fname,
                    'path': fpath,
                    'size_bytes': fsize_bytes,
                    'size_kb': fsize_bytes / 1024.0,
                    'width': width,
                    'height': height,
                    'mode': mode,
                    'format': img_format,
                    'sha256': sha256_hash
                })
            except Exception as e:
                corrupt_count += 1
                print(f"[CORRUPT] {fpath}: {e}")

    df = pd.DataFrame(records)
    return df, corrupt_count

if __name__ == '__main__':
    data_dir = 'data/raw'
    print("Starting dataset audit...")
    df, corrupt_count = audit_dataset(data_dir)
    
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/dataset_audit.csv', index=False)
    
    print("\n================ DATASET AUDIT SUMMARY ================")
    print(f"Total valid images scanned: {len(df)}")
    print(f"Corrupt/unreadable images: {corrupt_count}")
    
    print("\nCounts per class:")
    print(df['class'].value_counts())
    
    exact_duplicate_groups = df.duplicated(subset=['sha256']).sum()
    print(f"\nExact duplicate images detected: {exact_duplicate_groups}")
    
    stats = df.groupby('class').agg({
        'width': ['min', 'median', 'max'],
        'height': ['min', 'median', 'max'],
        'size_kb': ['min', 'median', 'max']
    }).round(2)
    
    print("\nDimension and Size Statistics (Min, Median, Max):")
    print(stats)
    print("=======================================================")
