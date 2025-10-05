"""
Quick Complete Comparison - All Models
=====================================
Fast version using Genesis quick test (3 models, 30 epochs)
+ XGBoost + Random Forest comparison

Estimated time: 15-20 minutes

Author: NASA Kepler Project
Date: 2025-10-05
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.append('..')

print("="*80)
print("QUICK COMPLETE COMPARISON - ALL MODELS")
print("="*80)
print("Estimated time: 15-20 minutes")
print("="*80)

# ============================================
# IMPORTS
# ============================================

print("\n[STEP 1/6] Importing libraries...")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report
)

try:
    import xgboost as xgb
    xgb_available = True
except ImportError:
    xgb_available = False

# GPU Config
gpus_tf = tf.config.list_physical_devices('GPU')
if gpus_tf:
    for gpu in gpus_tf:
        tf.config.experimental.set_memory_growth(gpu, True)
    from tensorflow.keras import mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    tf.config.optimizer.set_jit(True)
    print(f"[OK] TensorFlow GPU: {len(gpus_tf)} device(s), Mixed Precision enabled")
else:
    print("[INFO] TensorFlow: Using CPU")

np.random.seed(42)
tf.random.set_seed(42)

print("[OK] All imports successful")

# ============================================
# GENESIS FUNCTIONS
# ============================================

def process_lightcurve(lightcurve_data):
    target_length = 2001
    if len(lightcurve_data) != target_length:
        original_indices = np.linspace(0, len(lightcurve_data) - 1, len(lightcurve_data))
        target_indices = np.linspace(0, len(lightcurve_data) - 1, target_length)
        processed_data = np.interp(target_indices, original_indices, lightcurve_data)
    else:
        processed_data = lightcurve_data.copy()

    mean = np.mean(processed_data)
    std = np.std(processed_data)
    if std > 0:
        processed_data = (processed_data - mean) / std

    return processed_data

def augment_data(X_train, y_train):
    X_list = [X_train]
    y_list = [y_train]

    # Flip
    X_list.append(np.flip(X_train, axis=1))
    y_list.append(y_train)

    # Noise (2 copies for speed)
    data_std = np.std(X_train)
    for i in range(2):
        noise = np.random.normal(0, data_std, X_train.shape)
        X_list.append(X_train + noise)
        y_list.append(y_train)

    return np.vstack(X_list), np.vstack(y_list)

def build_genesis_model():
    model = models.Sequential([
        layers.Input(shape=(2001, 1)),
        layers.Conv1D(64, 50, padding='same', activation='relu', kernel_initializer='glorot_uniform'),
        layers.Conv1D(64, 50, padding='same', activation='relu', kernel_initializer='glorot_uniform'),
        layers.MaxPooling1D(32, strides=32),
        layers.Conv1D(64, 12, padding='same', activation='relu', kernel_initializer='glorot_uniform'),
        layers.Conv1D(64, 12, padding='same', activation='relu', kernel_initializer='glorot_uniform'),
        layers.AveragePooling1D(8),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, activation='relu', kernel_initializer='glorot_uniform'),
        layers.Dense(256, activation='relu', kernel_initializer='glorot_uniform'),
        layers.Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ============================================
# STEP 2: GENESIS DATA & TRAINING
# ============================================

print("\n[STEP 2/6] Generating Genesis data and training (3 models)...")

start_total = time.time()

n_samples = 300  # Reduced for speed
X_genesis = []
y_genesis = []

for i in range(n_samples):
    has_planet = i % 2
    lightcurve = np.random.normal(1.0, 0.001, 2001)

    if has_planet:
        transit_depth = np.random.uniform(0.002, 0.01)
        transit_width = 80
        transit_center = 1000
        for j in range(transit_width):
            idx = transit_center - transit_width//2 + j
            if 0 <= idx < 2001:
                lightcurve[idx] -= transit_depth

    X_genesis.append(process_lightcurve(lightcurve))
    y_genesis.append([1, 0] if has_planet == 0 else [0, 1])

X_genesis = np.array(X_genesis)
y_genesis = np.array(y_genesis)

X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(
    X_genesis, y_genesis, test_size=0.2, random_state=42
)

print("[INFO] Augmenting data...")
X_train_aug, y_train_aug = augment_data(X_train_g, y_train_g)
X_train_aug = X_train_aug.reshape(-1, 2001, 1)
X_test_g = X_test_g.reshape(-1, 2001, 1)

print(f"[OK] Training shape: {X_train_aug.shape}, Test shape: {X_test_g.shape}")

# Train ensemble
print("\n[INFO] Training Genesis ensemble (3 models, 30 epochs each)...")
genesis_start = time.time()
genesis_models = []
num_models = 3

for i in range(num_models):
    print(f"   Training model {i+1}/{num_models}...", end='')
    model = build_genesis_model()

    early_stopping = EarlyStopping(
        monitor='loss',
        patience=10,
        min_delta=0.001,
        restore_best_weights=True,
        verbose=0
    )

    model.fit(
        X_train_aug, y_train_aug,
        epochs=30,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )

    genesis_models.append(model)
    print(f" [OK]")

genesis_time = time.time() - genesis_start

# Predict
predictions = []
for model in genesis_models:
    pred = model.predict(X_test_g, verbose=0)
    predictions.append(pred)

genesis_pred_probs = np.mean(predictions, axis=0)[:, 1]
genesis_pred = np.argmax(np.mean(predictions, axis=0), axis=1)
y_test_g_labels = np.argmax(y_test_g, axis=1)

genesis_results = {
    'accuracy': accuracy_score(y_test_g_labels, genesis_pred),
    'roc_auc': roc_auc_score(y_test_g_labels, genesis_pred_probs),
    'pr_auc': average_precision_score(y_test_g_labels, genesis_pred_probs),
    'precision': precision_score(y_test_g_labels, genesis_pred, zero_division=0),
    'recall': recall_score(y_test_g_labels, genesis_pred, zero_division=0),
    'f1': f1_score(y_test_g_labels, genesis_pred, zero_division=0),
    'training_time': genesis_time,
    'n_models': num_models,
    'device': 'TensorFlow GPU' if gpus_tf else 'TensorFlow CPU',
    'type': 'Ensemble CNN',
    'data_type': 'Synthetic Light Curves'
}

print(f"\n[OK] Genesis Complete!")
print(f"   Accuracy: {genesis_results['accuracy']:.4f}")
print(f"   ROC-AUC: {genesis_results['roc_auc']:.4f}")
print(f"   Time: {genesis_time:.1f}s ({genesis_time/60:.1f} min)")

results = {'Genesis_Ensemble': genesis_results}

# ============================================
# STEP 3: LOAD TSFRESH DATA
# ============================================

print("\n[STEP 3/6] Loading TSFresh features...")

data = pd.read_csv('data/tsfresh_features.csv')
data = data.replace([np.inf, -np.inf], np.nan).fillna(0)

unique_cols = data.columns[data.nunique() <= 1]
data = data.drop(unique_cols, axis=1)

train_data = data[:-600]
val_data = data[-600:-369]
test_data = data[-369:]

X_train = train_data.iloc[:, 1:-1].values
y_train = train_data.iloc[:, -1].values
X_val = val_data.iloc[:, 1:-1].values
y_val = val_data.iloc[:, -1].values
X_test = test_data.iloc[:, 1:-1].values
y_test = test_data.iloc[:, -1].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train_combined = np.vstack([X_train, X_val])
y_train_combined = np.hstack([y_train, y_val])

print(f"[OK] Train: {X_train.shape}, Test: {X_test.shape}")
print(f"[OK] Positive class: {y_test.mean():.2%}")

# ============================================
# STEP 4: TRAIN XGBOOST
# ============================================

print("\n[STEP 4/6] Training XGBoost models...")

if xgb_available:
    # CPU version (most reliable)
    print("[INFO] Training XGBoost (CPU)...")
    start_time = time.time()

    xgb_cpu = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        tree_method='hist',
        n_jobs=-1,
        eval_metric='logloss'
    )

    xgb_cpu.fit(X_train_combined, y_train_combined)
    xgb_cpu_time = time.time() - start_time

    xgb_cpu_pred_probs = xgb_cpu.predict_proba(X_test)[:, 1]
    xgb_cpu_pred = xgb_cpu.predict(X_test)

    results['XGBoost_CPU'] = {
        'accuracy': accuracy_score(y_test, xgb_cpu_pred),
        'roc_auc': roc_auc_score(y_test, xgb_cpu_pred_probs),
        'pr_auc': average_precision_score(y_test, xgb_cpu_pred_probs),
        'precision': precision_score(y_test, xgb_cpu_pred, zero_division=0),
        'recall': recall_score(y_test, xgb_cpu_pred, zero_division=0),
        'f1': f1_score(y_test, xgb_cpu_pred, zero_division=0),
        'training_time': xgb_cpu_time,
        'device': 'CPU',
        'type': 'Gradient Boosting',
        'data_type': 'TSFresh Features'
    }

    print(f"[OK] XGBoost CPU - ROC-AUC: {results['XGBoost_CPU']['roc_auc']:.4f}, Time: {xgb_cpu_time:.2f}s")

# ============================================
# STEP 5: TRAIN RANDOM FOREST
# ============================================

print("\n[STEP 5/6] Training Random Forest...")

start_time = time.time()

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=9,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=42
)

rf_model.fit(X_train_combined, y_train_combined)
rf_time = time.time() - start_time

rf_pred_probs = rf_model.predict_proba(X_test)[:, 1]
rf_pred = rf_model.predict(X_test)

results['Random_Forest'] = {
    'accuracy': accuracy_score(y_test, rf_pred),
    'roc_auc': roc_auc_score(y_test, rf_pred_probs),
    'pr_auc': average_precision_score(y_test, rf_pred_probs),
    'precision': precision_score(y_test, rf_pred, zero_division=0),
    'recall': recall_score(y_test, rf_pred, zero_division=0),
    'f1': f1_score(y_test, rf_pred, zero_division=0),
    'training_time': rf_time,
    'device': 'CPU',
    'type': 'Tree Ensemble',
    'data_type': 'TSFresh Features'
}

print(f"[OK] Random Forest - ROC-AUC: {results['Random_Forest']['roc_auc']:.4f}, Time: {rf_time:.2f}s")

# ============================================
# STEP 6: GENERATE REPORT
# ============================================

print("\n[STEP 6/6] Generating comparison report...")

total_time = time.time() - start_total

comparison_data = []
for model_name, metrics in results.items():
    row = {'Model': model_name}
    row.update(metrics)
    comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)

genesis_df = comparison_df[comparison_df['data_type'] == 'Synthetic Light Curves']
tsfresh_df = comparison_df[comparison_df['data_type'] == 'TSFresh Features']
tsfresh_df = tsfresh_df.sort_values('roc_auc', ascending=False) if 'roc_auc' in tsfresh_df.columns else tsfresh_df

print("\n" + "="*80)
print("QUICK COMPLETE COMPARISON RESULTS")
print("="*80)

print("\n--- GENESIS ENSEMBLE (Synthetic Light Curves) ---")
print(genesis_df[['Model', 'accuracy', 'roc_auc', 'f1', 'n_models', 'training_time']].to_string(index=False))

print("\n--- OTHER MODELS (TSFresh Features) ---")
print(tsfresh_df[['Model', 'accuracy', 'roc_auc', 'pr_auc', 'f1', 'training_time']].to_string(index=False))

# Save results
os.makedirs('reports/results', exist_ok=True)
output_file = 'reports/results/quick_complete_comparison.json'

with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n[OK] Results saved to {output_file}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nGenesis Ensemble (Quick):")
print(f"  Accuracy: {genesis_results['accuracy']:.4f} ({genesis_results['accuracy']*100:.2f}%)")
print(f"  ROC-AUC: {genesis_results['roc_auc']:.4f}")
print(f"  F1-Score: {genesis_results['f1']:.4f}")
print(f"  Models: {num_models}")
print(f"  Time: {genesis_time:.1f}s")

if not tsfresh_df.empty:
    best_idx = tsfresh_df['roc_auc'].idxmax()
    best_model = tsfresh_df.loc[best_idx, 'Model']
    best_roc = tsfresh_df.loc[best_idx, 'roc_auc']
    best_time = tsfresh_df.loc[best_idx, 'training_time']

    print(f"\nBest on TSFresh Features:")
    print(f"  Model: {best_model}")
    print(f"  ROC-AUC: {best_roc:.4f}")
    print(f"  Time: {best_time:.2f}s")

print(f"\nTotal execution time: {total_time:.1f}s ({total_time/60:.1f} min)")

print("\n" + "="*80)
print("[OK] QUICK COMPLETE COMPARISON FINISHED!")
print("="*80)
print("\nKey Findings:")
print("1. Genesis trained on synthetic light curves (controlled data)")
print("2. XGBoost/RF trained on TSFresh features (real Kepler data)")
print("3. Different data types, different model strengths")
print("4. All models trained fresh in one clean run")
