"""
Genesis Model - Final GPU-Optimized Report
==========================================
100 real Kepler light curves with full GPU optimization and PDF report.

Optimized for:
- Reasonable download time (~15-20 min)
- GPU-accelerated training (~20-30 min)
- Complete visualization and PDF generation

Author: NASA Kepler Project
Date: 2025-10-05
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

print("="*80)
print("GENESIS FINAL GPU-OPTIMIZED TRAINING WITH PDF REPORT")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

start_time_total = time.time()

# ============================================
# GPU SETUP
# ============================================

print("\n[GPU SETUP]")

gpus = tf.config.list_physical_devices('GPU')
gpu_available = False

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        print(f"[OK] GPU: {len(gpus)} device(s) - {gpus[0].name}")

        # Mixed Precision
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("[OK] Mixed Precision FP16 enabled")

        # XLA JIT
        tf.config.optimizer.set_jit(True)
        print("[OK] XLA JIT enabled")

        # TF32
        try:
            tf.config.experimental.enable_tensor_float_32_execution(True)
            print("[OK] TF32 enabled")
        except:
            pass

        gpu_available = True

    except Exception as e:
        print(f"[WARN] GPU setup failed: {e}")
        gpu_available = False
else:
    print("[WARN] No GPU - using CPU (will be slower)")

np.random.seed(42)
tf.random.set_seed(42)

# ============================================
# LOAD AND DOWNLOAD DATA
# ============================================

print("\n[STEP 1/5] Loading Kepler catalog...")

koi_data = pd.read_csv('data/q1_q17_dr25_koi.csv')
confirmed = koi_data[koi_data['koi_disposition'] == 'CONFIRMED'].head(50)
false_pos = koi_data[koi_data['koi_disposition'] == 'FALSE POSITIVE'].head(50)

print(f"[OK] Selected 50 confirmed + 50 false positives = 100 targets")

def process_lightcurve(lc_data):
    """Process to 2001 points"""
    target_length = 2001
    if len(lc_data) != target_length:
        orig_idx = np.linspace(0, len(lc_data) - 1, len(lc_data))
        target_idx = np.linspace(0, len(lc_data) - 1, target_length)
        processed = np.interp(target_idx, orig_idx, lc_data)
    else:
        processed = lc_data.copy()

    mean, std = np.mean(processed), np.std(processed)
    if std > 0:
        processed = (processed - mean) / std
    return processed

def download_koi(kepid):
    """Download light curve"""
    try:
        import lightkurve as lk
        search = lk.search_lightcurve(f'KIC {kepid}', mission='Kepler', cadence='long')
        if len(search) == 0:
            return None

        lc_coll = search.download_all()
        if lc_coll is None or len(lc_coll) == 0:
            return None

        lc = lc_coll.stitch()
        flux = lc.flux.value
        flux = flux[~np.isnan(flux)]

        if len(flux) < 100:
            return None

        flux = flux / np.median(flux)
        return process_lightcurve(flux)
    except:
        return None

print("\n[STEP 2/5] Downloading light curves (may take 10-15 min)...")

try:
    import lightkurve as lk
except ImportError:
    print("[ERROR] lightkurve not installed!")
    sys.exit(1)

X_data, y_labels, kepids = [], [], []
success, fail = 0, 0
dl_start = time.time()

for idx, row in pd.concat([confirmed, false_pos]).iterrows():
    kepid = row['kepid']
    is_planet = row['koi_disposition'] == 'CONFIRMED'

    lc = download_koi(kepid)
    if lc is not None:
        X_data.append(lc)
        y_labels.append([0, 1] if is_planet else [1, 0])
        kepids.append(kepid)
        success += 1

        if success % 10 == 0:
            elapsed = time.time() - dl_start
            print(f"   {success}/100 downloaded ({elapsed:.1f}s)")
    else:
        fail += 1

dl_time = time.time() - dl_start

print(f"\n[OK] Downloaded {success} light curves in {dl_time/60:.1f} min")
print(f"[WARN] Failed: {fail}")

if success < 30:
    print("[ERROR] Too few light curves!")
    sys.exit(1)

X_data = np.array(X_data)
y_labels = np.array(y_labels)

print(f"[OK] Data: {X_data.shape}")

# ============================================
# SPLIT AND AUGMENT
# ============================================

print("\n[STEP 3/5] Data preparation...")

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_labels, test_size=0.2, random_state=42, stratify=y_labels
)

print(f"[OK] Train: {X_train.shape}, Test: {X_test.shape}")

# Augmentation
def augment(X, y):
    X_list = [X, np.flip(X, axis=1)]
    y_list = [y, y]

    std = np.std(X)
    for _ in range(4):
        noise = np.random.normal(0, std * 0.1, X.shape)
        X_list.append(X + noise)
        y_list.append(y)

    return np.vstack(X_list), np.vstack(y_list)

X_train_aug, y_train_aug = augment(X_train, y_train)
X_train_aug = X_train_aug.reshape(-1, 2001, 1)
X_test = X_test.reshape(-1, 2001, 1)

print(f"[OK] Augmented: {X_train_aug.shape}")

# ============================================
# BUILD MODEL
# ============================================

def build_genesis():
    """Genesis CNN"""
    model = models.Sequential([
        layers.Input(shape=(2001, 1)),
        layers.Conv1D(64, 50, padding='same', activation='relu', name='conv1_1'),
        layers.Conv1D(64, 50, padding='same', activation='relu', name='conv1_2'),
        layers.MaxPooling1D(32, strides=32),
        layers.Conv1D(64, 12, padding='same', activation='relu', name='conv2_1'),
        layers.Conv1D(64, 12, padding='same', activation='relu', name='conv2_2'),
        layers.AveragePooling1D(8),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(2, activation='softmax', dtype='float32', name='output')
    ], name='Genesis')

    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ============================================
# TRAIN ENSEMBLE
# ============================================

print("\n[STEP 4/5] Training 5-model ensemble...")

num_models = 5
models_list = []
histories = []

train_start = time.time()

for i in range(num_models):
    print(f"\n--- Model {i+1}/{num_models} ---")

    model = build_genesis()

    if i == 0:
        model.summary()

    early_stop = EarlyStopping(
        monitor='loss', patience=15, min_delta=0.0001,
        restore_best_weights=True, verbose=0
    )

    lr_reduce = ReduceLROnPlateau(
        monitor='loss', factor=0.5, patience=5,
        min_lr=1e-6, verbose=1
    )

    hist = model.fit(
        X_train_aug, y_train_aug,
        epochs=80,
        batch_size=32,
        callbacks=[early_stop, lr_reduce],
        verbose=2
    )

    models_list.append(model)
    histories.append(hist.history)

    print(f"[OK] Model {i+1}: Acc={hist.history['accuracy'][-1]:.4f}")

train_time = time.time() - train_start

print(f"\n[OK] Training completed in {train_time/60:.1f} min")

# ============================================
# EVALUATE
# ============================================

print("\n[STEP 5/5] Evaluation...")

preds_individual = []
accs_individual = []

for i, m in enumerate(models_list):
    p = m.predict(X_test, verbose=0)
    preds_individual.append(p)

    p_labels = np.argmax(p, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    acc = accuracy_score(y_test_labels, p_labels)
    accs_individual.append(acc)
    print(f"   Model {i+1}: {acc:.4f}")

# Ensemble
ensemble_probs = np.mean(preds_individual, axis=0)
ensemble_pred = np.argmax(ensemble_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

acc = accuracy_score(y_true, ensemble_pred)
prec = precision_score(y_true, ensemble_pred, zero_division=0)
rec = recall_score(y_true, ensemble_pred, zero_division=0)
f1 = f1_score(y_true, ensemble_pred, zero_division=0)
auc = roc_auc_score(y_true, ensemble_probs[:, 1])

print(f"\n{'='*60}")
print("ENSEMBLE RESULTS")
print(f"{'='*60}")
print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {auc:.4f}")

print("\n" + classification_report(y_true, ensemble_pred,
                                   target_names=['No Planet', 'Planet']))

cm = confusion_matrix(y_true, ensemble_pred)

# ============================================
# VISUALIZATIONS
# ============================================

print("\n[GENERATING VISUALIZATIONS]")

os.makedirs('reports/figures', exist_ok=True)

# 1. Training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, h in enumerate(histories):
    axes[0].plot(h['loss'], alpha=0.6, label=f'M{i+1}')
    axes[1].plot(h['accuracy'], alpha=0.6, label=f'M{i+1}')

axes[0].set_title('Training Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_title('Training Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('reports/figures/genesis_final_training.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved training curves")

# 2. Confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Planet', 'Planet'],
            yticklabels=['No Planet', 'Planet'])
ax.set_title(f'Confusion Matrix (Acc: {acc:.2%})')
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
plt.tight_layout()
plt.savefig('reports/figures/genesis_final_cm.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved confusion matrix")

# 3. ROC curve
fpr, tpr, _ = roc_curve(y_true, ensemble_probs[:, 1])
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, 'b-', lw=2, label=f'AUC={auc:.4f}')
ax.plot([0,1], [0,1], 'k--', alpha=0.3)
ax.set_title('ROC Curve')
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('reports/figures/genesis_final_roc.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved ROC curve")

# 4. Model comparison
fig, ax = plt.subplots(figsize=(10, 6))
all_accs = accs_individual + [acc]
labels = [f'M{i+1}' for i in range(num_models)] + ['Ensemble']
colors = ['skyblue']*num_models + ['coral']
bars = ax.bar(range(len(all_accs)), all_accs, color=colors, edgecolor='black')
ax.set_ylabel('Accuracy')
ax.set_title('Model Performance Comparison')
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3, axis='y')

for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h, f'{h:.3f}',
            ha='center', va='bottom')

plt.tight_layout()
plt.savefig('reports/figures/genesis_final_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved model comparison")

# 5. Sample predictions
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i in range(min(6, len(X_test))):
    lc = X_test[i].squeeze()
    true = y_true[i]
    pred = ensemble_pred[i]
    conf = ensemble_probs[i, pred]

    axes[i].plot(lc, 'k-', lw=0.5)
    axes[i].set_title(
        f'True: {"Planet" if true==1 else "None"} | '
        f'Pred: {"Planet" if pred==1 else "None"} ({conf:.1%})',
        fontsize=9
    )
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Sample Predictions')
plt.tight_layout()
plt.savefig('reports/figures/genesis_final_samples.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] Saved samples")

# ============================================
# SAVE RESULTS
# ============================================

total_time = time.time() - start_time_total

results = {
    'metadata': {
        'timestamp': datetime.now().isoformat(),
        'total_time_min': total_time/60,
        'download_time_min': dl_time/60,
        'training_time_min': train_time/60,
        'gpu_used': gpu_available,
        'gpu_name': gpus[0].name if gpus else 'CPU'
    },
    'data': {
        'n_total': success,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_augmented': len(X_train_aug)
    },
    'model': {
        'ensemble_size': num_models,
        'parameters': int(models_list[0].count_params())
    },
    'performance': {
        'individual_accs': [float(a) for a in accs_individual],
        'ensemble_accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'roc_auc': float(auc)
    },
    'confusion_matrix': cm.tolist()
}

os.makedirs('reports/results', exist_ok=True)
with open('reports/results/genesis_final_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("[OK] Saved JSON results")

# ============================================
# PDF REPORT
# ============================================

print("\n[GENERATING PDF REPORT]")

from matplotlib.backends.backend_pdf import PdfPages

pdf_path = 'reports/Genesis_Final_Report.pdf'

with PdfPages(pdf_path) as pdf:
    # Page 1: Title
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.85, 'Genesis CNN Model',
             ha='center', fontsize=24, weight='bold')
    fig.text(0.5, 0.78, 'Exoplanet Detection from Real Kepler Light Curves',
             ha='center', fontsize=16)
    fig.text(0.5, 0.72, f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
             ha='center', fontsize=12, style='italic')

    summary = f"""
{'='*70}
EXECUTIVE SUMMARY
{'='*70}

Dataset:
  • Source: Real Kepler MAST archive
  • Total light curves: {success}
  • Training samples: {len(X_train)} → {len(X_train_aug)} (augmented)
  • Test samples: {len(X_test)}
  • Planet ratio: {y_labels[:, 1].mean():.1%}

Model Architecture:
  • Type: Genesis CNN (1D Convolutional Neural Network)
  • Ensemble: {num_models} independent models
  • Parameters: {models_list[0].count_params():,}
  • Layers: Conv1D (×4) + MaxPool + AvgPool + Dense (×2)

GPU Acceleration:
  • Device: {gpus[0].name if gpu_available else 'CPU'}
  • Mixed Precision: {'Enabled (FP16)' if gpu_available else 'N/A'}
  • XLA JIT: {'Enabled' if gpu_available else 'N/A'}

Performance Metrics:
  • Ensemble Accuracy: {acc:.4f} ({acc*100:.2f}%)
  • Precision: {prec:.4f}
  • Recall: {rec:.4f}
  • F1-Score: {f1:.4f}
  • ROC-AUC: {auc:.4f}

Individual Model Performance:
"""

    for i, a in enumerate(accs_individual):
        summary += f"  • Model {i+1}: {a:.4f}\n"

    summary += f"\n  • Ensemble: {acc:.4f} "
    summary += f"(+{acc - np.mean(accs_individual):.4f} improvement)\n"

    summary += f"""
Timing:
  • Data download: {dl_time/60:.1f} minutes
  • Model training: {train_time/60:.1f} minutes
  • Total runtime: {total_time/60:.1f} minutes

Confusion Matrix:
                Predicted
              No    Planet
  True No   {cm[0,0]:4d}   {cm[0,1]:4d}
       Yes  {cm[1,0]:4d}   {cm[1,1]:4d}

{'='*70}
"""

    fig.text(0.1, 0.05, summary, fontsize=9, family='monospace',
             verticalalignment='bottom')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # Page 2: Training
    img = plt.imread('reports/figures/genesis_final_training.png')
    fig = plt.figure(figsize=(11, 8.5))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Training Progress', fontsize=16, weight='bold', pad=20)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # Page 3: Performance
    fig = plt.figure(figsize=(11, 8.5))

    ax1 = plt.subplot(2, 2, 1)
    img1 = plt.imread('reports/figures/genesis_final_cm.png')
    ax1.imshow(img1)
    ax1.axis('off')

    ax2 = plt.subplot(2, 2, 2)
    img2 = plt.imread('reports/figures/genesis_final_roc.png')
    ax2.imshow(img2)
    ax2.axis('off')

    ax3 = plt.subplot(2, 1, 2)
    img3 = plt.imread('reports/figures/genesis_final_comparison.png')
    ax3.imshow(img3)
    ax3.axis('off')

    plt.suptitle('Performance Metrics', fontsize=16, weight='bold')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # Page 4: Samples
    img = plt.imread('reports/figures/genesis_final_samples.png')
    fig = plt.figure(figsize=(11, 8.5))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Sample Predictions', fontsize=16, weight='bold', pad=20)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

print(f"[OK] PDF saved: {pdf_path}")

# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "="*80)
print("GENESIS TRAINING COMPLETE!")
print("="*80)
print(f"\nExecution time: {total_time/60:.1f} minutes")
print(f"\nResults:")
print(f"  • {success} real Kepler light curves processed")
print(f"  • Ensemble accuracy: {acc:.2%}")
print(f"  • ROC-AUC: {auc:.4f}")
print(f"  • GPU: {gpu_available}")
print(f"\nOutput files:")
print(f"  • JSON: reports/results/genesis_final_results.json")
print(f"  • PDF:  {pdf_path}")
print(f"  • Figures: reports/figures/genesis_final_*.png")
print("="*80)
