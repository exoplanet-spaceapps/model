"""
Quick analysis of the actual training data
"""
import pandas as pd
import numpy as np
import os

# Check if data file exists
if os.path.exists('tsfresh_features.csv'):
    print("Loading tsfresh_features.csv...")
    data = pd.read_csv('tsfresh_features.csv')

    print("\n=== DATA ANALYSIS ===")
    print(f"Total samples: {len(data)}")
    print(f"Total features: {data.shape[1]}")

    # Check if there's a label column
    if 'label' in data.columns:
        print(f"\nLabel distribution:")
        print(data['label'].value_counts())
        print(f"Class balance: {data['label'].mean():.2%} positive")

    # Check last column (might be label)
    last_col = data.iloc[:, -1]
    if last_col.dtype in [np.int64, np.float64] and last_col.nunique() <= 10:
        print(f"\nLast column (potential label) distribution:")
        print(last_col.value_counts())

    print(f"\nData split (as per koi_project_nn.py):")
    print(f"  Train: 1-{len(data)-600} = {len(data)-600} samples")
    print(f"  Val: {len(data)-600+1}-{len(data)-369} = 231 samples")
    print(f"  Test: {len(data)-369+1}-{len(data)} = 369 samples")
else:
    print("tsfresh_features.csv not found")

# Check for KOI catalog
if os.path.exists('q1_q17_dr25_koi.csv'):
    print("\n=== KOI CATALOG ===")
    koi = pd.read_csv('q1_q17_dr25_koi.csv', nrows=10)
    print(f"Columns: {list(koi.columns[:10])}...")
    if 'koi_disposition' in koi.columns:
        full_koi = pd.read_csv('q1_q17_dr25_koi.csv')
        print(f"Total KOIs: {len(full_koi)}")
        print(f"Disposition distribution:")
        print(full_koi['koi_disposition'].value_counts())