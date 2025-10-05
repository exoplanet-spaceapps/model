"""
Complete Fresh Training and Comparison
======================================
Train ALL models from scratch and compare:
- Genesis Ensemble CNN (10 models)
- XGBoost (GPU & CPU)
- Random Forest
- Neural Networks

This script ensures clean training without interference.

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

# Add project root to path
sys.path.append('..')

print("="*80)
print("COMPLETE FRESH TRAINING - ALL MODELS FROM SCRATCH")
print("="*80)
print("\n[IMPORTANT] Please do not use the computer during training!")
print("Estimated time: 40-60 minutes")
print("="*80)

# ============================================
# IMPORT ALL LIBRARIES
# ============================================

print("\n[STEP 1/7] Importing libraries...")

# TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# PyTorch
import torch
import torch.nn as nn

# Scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report
)

# XGBoost
try:
    import xgboost as xgb
    xgb_available = True
except ImportError:
    xgb_available = False
    print("[WARN] XGBoost not available")

# GPU Configuration
gpus_tf = tf.config.list_physical_devices('GPU')
if gpus_tf:
    for gpu in gpus_tf:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"[OK] TensorFlow GPU: {len(gpus_tf)} device(s)")
else:
    print("[INFO] TensorFlow: Using CPU")

device_pytorch = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[OK] PyTorch device: {device_pytorch}")

# Enable GPU optimizations
if gpus_tf:
    from tensorflow.keras import mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print("[OK] Mixed precision (FP16) enabled")

    tf.config.optimizer.set_jit(True)
    print("[OK] XLA JIT compilation enabled")

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)

print("[OK] All imports successful")

# ============================================
# GENESIS MODEL FUNCTIONS
# ============================================

def process_lightcurve(lightcurve_data):
    """Process light curve to 2001 points"""
    target_length = 2001

    if len(lightcurve_data) != target_length:
        original_indices = np.linspace(0, len(lightcurve_data) - 1, len(lightcurve_data))
        target_indices = np.linspace(0, len(lightcurve_data) - 1, target_length)
        processed_data = np.interp(target_indices, original_indices, lightcurve_data)
    else:
        processed_data = lightcurve_data.copy()

    # Standardization
    mean = np.mean(processed_data)
    std = np.std(processed_data)
    if std > 0:
        processed_data = (processed_data - mean) / std

    return processed_data


def augment_data(X_train, y_train):
    """Data augmentation: flip + 4x Gaussian noise"""
    X_list = [X_train]
    y_list = [y_train]

    # Horizontal flip
    X_list.append(np.flip(X_train, axis=1))
    y_list.append(y_train)

    # Gaussian noise (4 copies)
    data_std = np.std(X_train)
    for i in range(4):
        noise = np.random.normal(0, data_std, X_train.shape)
        X_list.append(X_train + noise)
        y_list.append(y_train)

    return np.vstack(X_list), np.vstack(y_list)


def build_genesis_model():
    """Build Genesis CNN architecture"""
    model = models.Sequential([
        layers.Input(shape=(2001, 1)),

        # Conv Block 1
        layers.Conv1D(64, 50, padding='same', activation='relu', kernel_initializer='glorot_uniform'),
        layers.Conv1D(64, 50, padding='same', activation='relu', kernel_initializer='glorot_uniform'),
        layers.MaxPooling1D(32, strides=32),

        # Conv Block 2
        layers.Conv1D(64, 12, padding='same', activation='relu', kernel_initializer='glorot_uniform'),
        layers.Conv1D(64, 12, padding='same', activation='relu', kernel_initializer='glorot_uniform'),
        layers.AveragePooling1D(8),

        # Dense Block
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
# STEP 2: GENERATE GENESIS DATA
# ============================================

print("\n[STEP 2/7] Generating synthetic light curves for Genesis...")

n_samples = 500
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

# Split data
X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(
    X_genesis, y_genesis, test_size=0.2, random_state=42
)

# Augment training data
print("[INFO] Augmenting data...")
X_train_aug, y_train_aug = augment_data(X_train_g, y_train_g)
X_train_aug = X_train_aug.reshape(-1, 2001, 1)
X_test_g = X_test_g.reshape(-1, 2001, 1)

print(f"[OK] Training shape: {X_train_aug.shape}")
print(f"[OK] Test shape: {X_test_g.shape}")

# ============================================
# STEP 3: TRAIN GENESIS ENSEMBLE
# ============================================

print("\n[STEP 3/7] Training Genesis Ensemble (10 models)...")
print("[INFO] This will take ~30-40 minutes")

genesis_start = time.time()
genesis_models = []
num_models = 10

for i in range(num_models):
    print(f"\n   Training Genesis model {i+1}/{num_models}...")
    model = build_genesis_model()

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=50,
        min_delta=0.0001,
        restore_best_weights=True,
        verbose=0
    )

    model.fit(
        X_train_aug, y_train_aug,
        validation_split=0.1,
        epochs=125,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )

    genesis_models.append(model)
    print(f"   [OK] Model {i+1}/{num_models} completed")

genesis_time = time.time() - genesis_start

# Genesis prediction
print("\n[INFO] Genesis ensemble prediction...")
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
    'type': 'Ensemble CNN'
}

print(f"\n[OK] Genesis Ensemble Training Complete!")
print(f"   Accuracy: {genesis_results['accuracy']:.4f} ({genesis_results['accuracy']*100:.2f}%)")
print(f"   ROC-AUC: {genesis_results['roc_auc']:.4f}")
print(f"   Training Time: {genesis_time/60:.1f} minutes")

# ============================================
# STEP 4: LOAD TSFRESH DATA
# ============================================

print("\n[STEP 4/7] Loading TSFresh features for comparison models...")

data = pd.read_csv('data/tsfresh_features.csv')
data = data.replace([np.inf, -np.inf], np.nan).fillna(0)

# Clean
unique_cols = data.columns[data.nunique() <= 1]
data = data.drop(unique_cols, axis=1)

# Split
train_data = data[:-600]
val_data = data[-600:-369]
test_data = data[-369:]

X_train = train_data.iloc[:, 1:-1].values
y_train = train_data.iloc[:, -1].values
X_val = val_data.iloc[:, 1:-1].values
y_val = val_data.iloc[:, -1].values
X_test = test_data.iloc[:, 1:-1].values
y_test = test_data.iloc[:, -1].values

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Combine train and val for tree models
X_train_combined = np.vstack([X_train, X_val])
y_train_combined = np.hstack([y_train, y_val])

print(f"[OK] Train shape: {X_train.shape}")
print(f"[OK] Test shape: {X_test.shape}")
print(f"[OK] Class balance: {y_test.mean():.2%} positive")

results = {'Genesis_Ensemble': genesis_results}

# ============================================
# STEP 5: TRAIN XGBOOST
# ============================================

print("\n[STEP 5/7] Training XGBoost models...")

if xgb_available:
    # XGBoost GPU
    if torch.cuda.is_available():
        print("[INFO] Training XGBoost (GPU)...")
        start_time = time.time()

        xgb_gpu = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            tree_method='gpu_hist',
            gpu_id=0,
            predictor='gpu_predictor',
            eval_metric='logloss'
        )

        xgb_gpu.fit(X_train_combined, y_train_combined)
        xgb_gpu_time = time.time() - start_time

        xgb_gpu_pred_probs = xgb_gpu.predict_proba(X_test)[:, 1]
        xgb_gpu_pred = xgb_gpu.predict(X_test)

        results['XGBoost_GPU'] = {
            'accuracy': accuracy_score(y_test, xgb_gpu_pred),
            'roc_auc': roc_auc_score(y_test, xgb_gpu_pred_probs),
            'pr_auc': average_precision_score(y_test, xgb_gpu_pred_probs),
            'precision': precision_score(y_test, xgb_gpu_pred, zero_division=0),
            'recall': recall_score(y_test, xgb_gpu_pred, zero_division=0),
            'f1': f1_score(y_test, xgb_gpu_pred, zero_division=0),
            'training_time': xgb_gpu_time,
            'device': 'GPU',
            'type': 'Gradient Boosting'
        }

        print(f"[OK] XGBoost GPU - Accuracy: {results['XGBoost_GPU']['accuracy']:.4f}, Time: {xgb_gpu_time:.2f}s")

    # XGBoost CPU
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
        'type': 'Gradient Boosting'
    }

    print(f"[OK] XGBoost CPU - Accuracy: {results['XGBoost_CPU']['accuracy']:.4f}, Time: {xgb_cpu_time:.2f}s")

# ============================================
# STEP 6: TRAIN RANDOM FOREST
# ============================================

print("\n[STEP 6/7] Training Random Forest...")

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
    'type': 'Tree Ensemble'
}

print(f"[OK] Random Forest - Accuracy: {results['Random_Forest']['accuracy']:.4f}, Time: {rf_time:.2f}s")

# ============================================
# STEP 7: GENERATE FINAL REPORT
# ============================================

print("\n[STEP 7/7] Generating comparison report...")

# Create comparison DataFrame
comparison_data = []
for model_name, metrics in results.items():
    row = {'Model': model_name}
    row.update(metrics)
    comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)

# Sort by data type
genesis_df = comparison_df[comparison_df['Model'] == 'Genesis_Ensemble']
tsfresh_df = comparison_df[comparison_df['Model'] != 'Genesis_Ensemble']
tsfresh_df = tsfresh_df.sort_values('roc_auc', ascending=False) if 'roc_auc' in tsfresh_df.columns else tsfresh_df

print("\n" + "="*80)
print("FINAL COMPARISON RESULTS")
print("="*80)

print("\n--- GENESIS (Synthetic Light Curves) ---")
print(genesis_df[['Model', 'accuracy', 'roc_auc', 'f1', 'training_time']].to_string(index=False))

print("\n--- OTHER MODELS (TSFresh Features) ---")
display_cols = ['Model', 'accuracy', 'roc_auc', 'pr_auc', 'f1', 'training_time', 'device']
available_cols = [col for col in display_cols if col in tsfresh_df.columns]
print(tsfresh_df[available_cols].to_string(index=False))

# Save results
os.makedirs('reports/results', exist_ok=True)
output_file = 'reports/results/fresh_complete_comparison.json'

with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n[OK] Results saved to {output_file}")

# Generate summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\nGenesis Ensemble:")
print(f"  Accuracy: {genesis_results['accuracy']:.4f} ({genesis_results['accuracy']*100:.2f}%)")
print(f"  ROC-AUC: {genesis_results['roc_auc']:.4f}")
print(f"  Training Time: {genesis_results['training_time']/60:.1f} min")

if not tsfresh_df.empty and 'roc_auc' in tsfresh_df.columns:
    best_model = tsfresh_df.iloc[0]['Model']
    best_roc = tsfresh_df.iloc[0]['roc_auc']
    best_time = tsfresh_df.iloc[0].get('training_time', 0)
    print(f"\nBest Model on TSFresh:")
    print(f"  Model: {best_model}")
    print(f"  ROC-AUC: {best_roc:.4f}")
    print(f"  Training Time: {best_time:.2f}s")

print("\n" + "="*80)
print("[OK] COMPLETE FRESH TRAINING FINISHED!")
print("="*80)
print("\nAll models trained from scratch in one clean run.")
print("No pre-trained models were used.")
