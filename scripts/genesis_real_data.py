"""
Genesis Model - Real Kepler Data Training
==========================================
Train Genesis ensemble on real Kepler light curves instead of synthetic data.

This script:
1. Loads real Kepler light curves using lightkurve
2. Preprocesses to 2001 points
3. Trains Genesis ensemble
4. Evaluates on real exoplanet detections

Author: NASA Kepler Project
Date: 2025-10-05
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("GENESIS MODEL - REAL KEPLER DATA TRAINING")
print("="*80)

# GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[OK] GPU Found: {len(gpus)} device(s)")

        from tensorflow.keras import mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        tf.config.optimizer.set_jit(True)
        print("[OK] Mixed precision + XLA JIT enabled")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("[WARN] No GPU found, using CPU")

np.random.seed(42)
tf.random.set_seed(42)

# ============================================
# STEP 1: LOAD REAL KEPLER DATA
# ============================================

print("\n[STEP 1/5] Loading real Kepler data...")

try:
    import lightkurve as lk
    lightkurve_available = True
    print("[OK] Lightkurve available - will download real data")
except ImportError:
    lightkurve_available = False
    print("[WARN] Lightkurve not available - will use preprocessed data")

# Load KOI catalog
koi_data = pd.read_csv('data/q1_q17_dr25_koi.csv')
print(f"[OK] Loaded {len(koi_data)} KOIs from catalog")

# Filter confirmed planets and false positives
confirmed = koi_data[koi_data['koi_disposition'] == 'CONFIRMED'].head(50)
false_pos = koi_data[koi_data['koi_disposition'] == 'FALSE POSITIVE'].head(50)

print(f"[OK] Selected {len(confirmed)} confirmed planets")
print(f"[OK] Selected {len(false_pos)} false positives")

# ============================================
# STEP 2: DOWNLOAD AND PROCESS LIGHT CURVES
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


def download_and_process_koi(kepid, has_planet):
    """Download and process a single KOI light curve"""
    try:
        if not lightkurve_available:
            return None

        # Search for light curves
        search_result = lk.search_lightcurve(f'KIC {kepid}', mission='Kepler', cadence='long')

        if len(search_result) == 0:
            return None

        # Download first available quarter
        lc = search_result[0].download()

        if lc is None:
            return None

        # Get flux and remove NaNs
        flux = lc.flux.value
        flux = flux[~np.isnan(flux)]

        if len(flux) < 100:
            return None

        # Normalize
        flux = flux / np.median(flux)

        # Process to 2001 points
        processed = process_lightcurve(flux)

        return processed

    except Exception as e:
        return None


print("\n[STEP 2/5] Downloading and processing light curves...")

X_data = []
y_labels = []
success_count = 0
fail_count = 0

if lightkurve_available:
    # Process confirmed planets (label = [0, 1])
    print("[INFO] Processing confirmed planets...")
    for idx, row in confirmed.iterrows():
        kepid = row['kepid']
        lc = download_and_process_koi(kepid, has_planet=True)
        if lc is not None:
            X_data.append(lc)
            y_labels.append([0, 1])  # Has planet
            success_count += 1
            if success_count % 10 == 0:
                print(f"   Processed {success_count} planets...")
        else:
            fail_count += 1

    # Process false positives (label = [1, 0])
    print("[INFO] Processing false positives...")
    for idx, row in false_pos.iterrows():
        kepid = row['kepid']
        lc = download_and_process_koi(kepid, has_planet=False)
        if lc is not None:
            X_data.append(lc)
            y_labels.append([1, 0])  # No planet
            success_count += 1
            if success_count % 10 == 0:
                print(f"   Processed {success_count} total objects...")
        else:
            fail_count += 1

    print(f"\n[OK] Successfully processed: {success_count}")
    print(f"[WARN] Failed: {fail_count}")

    if success_count < 20:
        print("\n[ERROR] Not enough data downloaded!")
        print("[INFO] This could be due to:")
        print("  1. No internet connection")
        print("  2. MAST archive is down")
        print("  3. Lightkurve installation issue")
        print("\n[INFO] Falling back to synthetic data for demonstration...")

        # Generate synthetic data as fallback
        for i in range(100):
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

            X_data.append(process_lightcurve(lightcurve))
            y_labels.append([1, 0] if has_planet == 0 else [0, 1])

        print(f"[OK] Generated {len(X_data)} synthetic light curves")

else:
    # No lightkurve - use synthetic data
    print("[INFO] Lightkurve not available, using synthetic data...")
    for i in range(100):
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

        X_data.append(process_lightcurve(lightcurve))
        y_labels.append([1, 0] if has_planet == 0 else [0, 1])

    print(f"[OK] Generated {len(X_data)} synthetic light curves")

X_data = np.array(X_data)
y_labels = np.array(y_labels)

print(f"\n[OK] Final data shape: {X_data.shape}")
print(f"[OK] Labels shape: {y_labels.shape}")

# ============================================
# STEP 3: DATA AUGMENTATION
# ============================================

def augment_data(X_train, y_train):
    """Data augmentation: flip + noise"""
    X_list = [X_train]
    y_list = [y_train]

    # Horizontal flip
    X_list.append(np.flip(X_train, axis=1))
    y_list.append(y_train)

    # Gaussian noise (2 copies for speed)
    data_std = np.std(X_train)
    for i in range(2):
        noise = np.random.normal(0, data_std, X_train.shape)
        X_list.append(X_train + noise)
        y_list.append(y_train)

    return np.vstack(X_list), np.vstack(y_list)


print("\n[STEP 3/5] Data preparation...")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_labels, test_size=0.2, random_state=42
)

print(f"[OK] Train: {X_train.shape}, Test: {X_test.shape}")

# Augment
print("[INFO] Augmenting training data...")
X_train_aug, y_train_aug = augment_data(X_train, y_train)
X_train_aug = X_train_aug.reshape(-1, 2001, 1)
X_test = X_test.reshape(-1, 2001, 1)

print(f"[OK] Augmented train: {X_train_aug.shape}")

# ============================================
# STEP 4: BUILD AND TRAIN GENESIS
# ============================================

def build_genesis_model():
    """Build Genesis CNN architecture"""
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


print("\n[STEP 4/5] Training Genesis ensemble...")

num_models = 3  # Quick ensemble
genesis_models = []

for i in range(num_models):
    print(f"\n   Training model {i+1}/{num_models}...")

    model = build_genesis_model()

    early_stopping = EarlyStopping(
        monitor='loss',
        patience=10,
        min_delta=0.001,
        restore_best_weights=True,
        verbose=0
    )

    history = model.fit(
        X_train_aug, y_train_aug,
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )

    genesis_models.append(model)
    final_acc = history.history['accuracy'][-1]
    print(f"   [OK] Model {i+1} trained - Final accuracy: {final_acc:.4f}")

# ============================================
# STEP 5: EVALUATE
# ============================================

print("\n[STEP 5/5] Ensemble evaluation...")

predictions = []
for model in genesis_models:
    pred = model.predict(X_test, verbose=0)
    predictions.append(pred)

ensemble_pred_probs = np.mean(predictions, axis=0)[:, 1]
ensemble_pred = np.argmax(np.mean(predictions, axis=0), axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Metrics
accuracy = accuracy_score(y_test_labels, ensemble_pred)
roc_auc = roc_auc_score(y_test_labels, ensemble_pred_probs)

print("\n" + "="*80)
print("RESULTS - GENESIS ON REAL KEPLER DATA")
print("="*80)

if lightkurve_available and success_count >= 20:
    print("\n[DATA SOURCE] Real Kepler Light Curves")
    print(f"  Confirmed planets: {sum(1 for y in y_test_labels if y == 1)}")
    print(f"  False positives: {sum(1 for y in y_test_labels if y == 0)}")
else:
    print("\n[DATA SOURCE] Synthetic Light Curves (fallback)")

print(f"\n[PERFORMANCE]")
print(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  ROC-AUC: {roc_auc:.4f}")
print(f"  Ensemble size: {num_models} models")

print("\n[CLASSIFICATION REPORT]")
print(classification_report(
    y_test_labels,
    ensemble_pred,
    target_names=['No Planet', 'Planet']
))

print("="*80)
print("[OK] Genesis Real Data Training Complete!")
print("="*80)

# Save results
import json
results = {
    'data_source': 'real_kepler' if (lightkurve_available and success_count >= 20) else 'synthetic',
    'n_samples': len(X_data),
    'n_train': len(X_train_aug),
    'n_test': len(X_test),
    'ensemble_size': num_models,
    'accuracy': float(accuracy),
    'roc_auc': float(roc_auc)
}

with open('reports/results/genesis_real_data_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n[OK] Results saved to reports/results/genesis_real_data_results.json")
