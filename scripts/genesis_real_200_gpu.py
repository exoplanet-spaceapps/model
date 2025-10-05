"""
Genesis Model - 200 Real Kepler Light Curves with GPU Optimization
===================================================================
Complete training pipeline with visualization and PDF report generation.

Features:
- 200 real Kepler light curves
- Full GPU optimization (Mixed Precision, XLA JIT, TF32)
- Comprehensive training monitoring
- Detailed visualizations
- PDF report generation

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
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# Set style for better plots
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

print("="*80)
print("GENESIS MODEL - 200 REAL KEPLER LIGHT CURVES (GPU OPTIMIZED)")
print("="*80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ============================================
# GPU OPTIMIZATION CONFIGURATION
# ============================================

print("\n[GPU SETUP] Configuring GPU optimization...")

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # Get GPU details
        gpu_details = tf.config.experimental.get_device_details(gpus[0])
        print(f"[OK] GPU Found: {len(gpus)} device(s)")
        print(f"     Device: {gpus[0].name}")
        if gpu_details:
            print(f"     Compute Capability: {gpu_details.get('compute_capability', 'N/A')}")

        # Enable Mixed Precision (FP16) for faster training
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("[OK] Mixed Precision (FP16) enabled")

        # Enable XLA JIT compilation
        tf.config.optimizer.set_jit(True)
        print("[OK] XLA JIT compilation enabled")

        # Enable TF32 for Tensor Cores (Ampere GPUs)
        try:
            tf.config.experimental.enable_tensor_float_32_execution(True)
            print("[OK] TF32 Tensor Core execution enabled")
        except:
            pass

        # Set deterministic operations for reproducibility
        tf.config.experimental.enable_op_determinism()
        print("[OK] Deterministic operations enabled")

        gpu_available = True

    except RuntimeError as e:
        print(f"[ERROR] GPU configuration failed: {e}")
        gpu_available = False
else:
    print("[WARN] No GPU found - training will be MUCH slower on CPU!")
    print("[WARN] Consider using Google Colab or a machine with GPU")
    gpu_available = False

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Track start time
start_time_total = time.time()

# ============================================
# LOAD KEPLER CATALOG
# ============================================

print("\n[STEP 1/6] Loading Kepler catalog...")

koi_data = pd.read_csv('data/q1_q17_dr25_koi.csv')
print(f"[OK] Loaded {len(koi_data)} KOIs")

# Select 200 samples: 100 confirmed + 100 false positives
confirmed = koi_data[koi_data['koi_disposition'] == 'CONFIRMED'].head(100)
false_pos = koi_data[koi_data['koi_disposition'] == 'FALSE POSITIVE'].head(100)

print(f"[OK] Selected {len(confirmed)} confirmed planets")
print(f"[OK] Selected {len(false_pos)} false positives")
print(f"[OK] Total targets: {len(confirmed) + len(false_pos)}")

# ============================================
# DOWNLOAD LIGHT CURVES
# ============================================

def process_lightcurve(lightcurve_data):
    """Process light curve to 2001 points with standardization"""
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


def download_koi_lightcurve(kepid):
    """Download and process a single KOI light curve"""
    try:
        import lightkurve as lk

        # Search for light curves
        search_result = lk.search_lightcurve(f'KIC {kepid}', mission='Kepler', cadence='long')

        if len(search_result) == 0:
            return None

        # Download and stitch all quarters
        lc_collection = search_result.download_all()
        if lc_collection is None or len(lc_collection) == 0:
            return None

        # Stitch quarters together
        lc = lc_collection.stitch()

        # Get flux and remove NaNs
        flux = lc.flux.value
        time = lc.time.value

        # Remove NaNs
        mask = ~np.isnan(flux) & ~np.isnan(time)
        flux = flux[mask]

        if len(flux) < 100:
            return None

        # Normalize
        flux = flux / np.median(flux)

        # Process to 2001 points
        processed = process_lightcurve(flux)

        return processed

    except Exception as e:
        return None


print("\n[STEP 2/6] Downloading light curves from MAST...")
print("[INFO] This may take 5-10 minutes depending on internet speed...")

try:
    import lightkurve as lk
    lk_available = True
except ImportError:
    print("[ERROR] lightkurve not installed!")
    print("[INFO] Install with: pip install lightkurve")
    lk_available = False
    sys.exit(1)

X_data = []
y_labels = []
kepids_used = []
success_count = 0
fail_count = 0

download_start = time.time()

# Download confirmed planets
print("\n[INFO] Downloading confirmed planets...")
for idx, row in confirmed.iterrows():
    kepid = row['kepid']
    lc = download_koi_lightcurve(kepid)

    if lc is not None:
        X_data.append(lc)
        y_labels.append([0, 1])  # Has planet
        kepids_used.append(kepid)
        success_count += 1

        if success_count % 20 == 0:
            elapsed = time.time() - download_start
            print(f"   Progress: {success_count}/200 ({success_count/2:.0f}%) - {elapsed:.1f}s elapsed")
    else:
        fail_count += 1

# Download false positives
print("\n[INFO] Downloading false positives...")
for idx, row in false_pos.iterrows():
    kepid = row['kepid']
    lc = download_koi_lightcurve(kepid)

    if lc is not None:
        X_data.append(lc)
        y_labels.append([1, 0])  # No planet
        kepids_used.append(kepid)
        success_count += 1

        if success_count % 20 == 0:
            elapsed = time.time() - download_start
            print(f"   Progress: {success_count}/200 ({success_count/2:.0f}%) - {elapsed:.1f}s elapsed")
    else:
        fail_count += 1

download_time = time.time() - download_start

print(f"\n[OK] Download complete in {download_time:.1f}s ({download_time/60:.1f} min)")
print(f"[OK] Successfully downloaded: {success_count}")
print(f"[WARN] Failed: {fail_count}")

if success_count < 50:
    print("\n[ERROR] Too few light curves downloaded!")
    print("[INFO] Check internet connection or try again later")
    sys.exit(1)

X_data = np.array(X_data)
y_labels = np.array(y_labels)

print(f"\n[OK] Data shape: {X_data.shape}")
print(f"[OK] Labels shape: {y_labels.shape}")

# ============================================
# DATA AUGMENTATION
# ============================================

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
        noise = np.random.normal(0, data_std * 0.1, X_train.shape)
        X_list.append(X_train + noise)
        y_list.append(y_train)

    return np.vstack(X_list), np.vstack(y_list)


print("\n[STEP 3/6] Data preparation and augmentation...")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_labels, test_size=0.2, random_state=42, stratify=y_labels
)

print(f"[OK] Train: {X_train.shape}, Test: {X_test.shape}")
print(f"[OK] Train planet ratio: {y_train[:, 1].mean():.2%}")
print(f"[OK] Test planet ratio: {y_test[:, 1].mean():.2%}")

# Augment training data
print("[INFO] Augmenting training data...")
X_train_aug, y_train_aug = augment_data(X_train, y_train)
X_train_aug = X_train_aug.reshape(-1, 2001, 1)
X_test = X_test.reshape(-1, 2001, 1)

print(f"[OK] Augmented train: {X_train_aug.shape}")
print(f"[OK] Augmentation factor: {len(X_train_aug) / len(X_train):.1f}x")

# ============================================
# BUILD GENESIS MODEL
# ============================================

def build_genesis_model():
    """Build Genesis CNN architecture with GPU optimization"""
    model = models.Sequential([
        layers.Input(shape=(2001, 1)),

        # Conv Block 1
        layers.Conv1D(64, 50, padding='same', activation='relu',
                     kernel_initializer='glorot_uniform', name='conv1_1'),
        layers.Conv1D(64, 50, padding='same', activation='relu',
                     kernel_initializer='glorot_uniform', name='conv1_2'),
        layers.MaxPooling1D(32, strides=32, name='maxpool1'),

        # Conv Block 2
        layers.Conv1D(64, 12, padding='same', activation='relu',
                     kernel_initializer='glorot_uniform', name='conv2_1'),
        layers.Conv1D(64, 12, padding='same', activation='relu',
                     kernel_initializer='glorot_uniform', name='conv2_2'),
        layers.AveragePooling1D(8, name='avgpool1'),

        # Dense Block
        layers.Dropout(0.25, name='dropout1'),
        layers.Flatten(name='flatten'),
        layers.Dense(256, activation='relu', kernel_initializer='glorot_uniform', name='dense1'),
        layers.Dense(256, activation='relu', kernel_initializer='glorot_uniform', name='dense2'),
        layers.Dense(2, activation='softmax', dtype='float32', name='output')  # float32 for stability
    ], name='Genesis_CNN')

    # Optimizer with learning rate schedule
    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# Custom callback for training monitoring
class TrainingMonitor(Callback):
    def __init__(self, model_id):
        super().__init__()
        self.model_id = model_id
        self.history = {'loss': [], 'accuracy': []}

    def on_epoch_end(self, epoch, logs=None):
        self.history['loss'].append(logs.get('loss'))
        self.history['accuracy'].append(logs.get('accuracy'))


# ============================================
# TRAIN ENSEMBLE
# ============================================

print("\n[STEP 4/6] Training Genesis ensemble...")

num_models = 5  # 5-model ensemble for better performance
genesis_models = []
training_histories = []

ensemble_start = time.time()

for i in range(num_models):
    print(f"\n{'='*60}")
    print(f"Training Model {i+1}/{num_models}")
    print(f"{'='*60}")

    model = build_genesis_model()

    # Show architecture for first model
    if i == 0:
        print("\n[MODEL ARCHITECTURE]")
        model.summary()
        print(f"\nTotal parameters: {model.count_params():,}")

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='loss',
        patience=15,
        min_delta=0.0001,
        restore_best_weights=True,
        verbose=1
    )

    monitor = TrainingMonitor(i)

    # Train
    model_start = time.time()

    history = model.fit(
        X_train_aug, y_train_aug,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, monitor],
        verbose=1
    )

    model_time = time.time() - model_start

    genesis_models.append(model)
    training_histories.append(monitor.history)

    final_acc = history.history['accuracy'][-1]
    final_loss = history.history['loss'][-1]

    print(f"\n[OK] Model {i+1} completed in {model_time:.1f}s")
    print(f"     Final accuracy: {final_acc:.4f}")
    print(f"     Final loss: {final_loss:.4f}")

ensemble_time = time.time() - ensemble_start

print(f"\n{'='*60}")
print(f"[OK] Ensemble training completed in {ensemble_time:.1f}s ({ensemble_time/60:.1f} min)")
print(f"{'='*60}")

# ============================================
# EVALUATION
# ============================================

print("\n[STEP 5/6] Evaluating ensemble...")

# Individual model predictions
individual_preds = []
individual_accs = []

for i, model in enumerate(genesis_models):
    pred = model.predict(X_test, verbose=0)
    individual_preds.append(pred)

    pred_labels = np.argmax(pred, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    acc = accuracy_score(y_test_labels, pred_labels)
    individual_accs.append(acc)

    print(f"   Model {i+1} accuracy: {acc:.4f}")

# Ensemble prediction
ensemble_pred_probs = np.mean(individual_preds, axis=0)
ensemble_pred = np.argmax(ensemble_pred_probs, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Metrics
accuracy = accuracy_score(y_test_labels, ensemble_pred)
precision = precision_score(y_test_labels, ensemble_pred, zero_division=0)
recall = recall_score(y_test_labels, ensemble_pred, zero_division=0)
f1 = f1_score(y_test_labels, ensemble_pred, zero_division=0)
roc_auc = roc_auc_score(y_test_labels, ensemble_pred_probs[:, 1])

print(f"\n{'='*60}")
print(f"ENSEMBLE PERFORMANCE")
print(f"{'='*60}")
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print(f"\n[CLASSIFICATION REPORT]")
print(classification_report(y_test_labels, ensemble_pred,
                          target_names=['No Planet', 'Planet']))

# Confusion matrix
cm = confusion_matrix(y_test_labels, ensemble_pred)
print(f"\n[CONFUSION MATRIX]")
print(f"                 Predicted")
print(f"               No  Planet")
print(f"Actual No    {cm[0,0]:4d}  {cm[0,1]:4d}")
print(f"       Planet {cm[1,0]:4d}  {cm[1,1]:4d}")

# ============================================
# VISUALIZATION
# ============================================

print("\n[STEP 6/6] Generating visualizations...")

os.makedirs('reports/figures', exist_ok=True)

# 1. Training History
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, hist in enumerate(training_histories):
    axes[0].plot(hist['loss'], alpha=0.6, label=f'Model {i+1}')
    axes[1].plot(hist['accuracy'], alpha=0.6, label=f'Model {i+1}')

axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss per Model')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training Accuracy per Model')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('reports/figures/genesis_training_history.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: reports/figures/genesis_training_history.png")
plt.close()

# 2. Confusion Matrix Heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['No Planet', 'Planet'],
            yticklabels=['No Planet', 'Planet'])
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title(f'Confusion Matrix - Genesis Ensemble\n(Accuracy: {accuracy:.2%})')
plt.tight_layout()
plt.savefig('reports/figures/genesis_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: reports/figures/genesis_confusion_matrix.png")
plt.close()

# 3. ROC Curve
fpr, tpr, _ = roc_curve(y_test_labels, ensemble_pred_probs[:, 1])

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'Genesis Ensemble (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Classifier')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve - Genesis on Real Kepler Data')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('reports/figures/genesis_roc_curve.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: reports/figures/genesis_roc_curve.png")
plt.close()

# 4. Model Performance Comparison
fig, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(individual_accs) + 1)
accs = individual_accs + [accuracy]
labels = [f'Model {i+1}' for i in range(num_models)] + ['Ensemble']
colors = ['skyblue'] * num_models + ['coral']

bars = ax.bar(x_pos, accs, color=colors, alpha=0.7, edgecolor='black')
ax.set_xlabel('Model')
ax.set_ylabel('Accuracy')
ax.set_title('Individual Model vs Ensemble Performance')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}',
            ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('reports/figures/genesis_model_comparison.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: reports/figures/genesis_model_comparison.png")
plt.close()

# 5. Sample Light Curves
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i in range(6):
    if i < len(X_test):
        lc = X_test[i].squeeze()
        true_label = y_test_labels[i]
        pred_label = ensemble_pred[i]
        confidence = ensemble_pred_probs[i, pred_label]

        axes[i].plot(lc, 'k-', linewidth=0.5, alpha=0.7)
        axes[i].set_title(f'True: {"Planet" if true_label==1 else "No Planet"} | '
                         f'Pred: {"Planet" if pred_label==1 else "No Planet"} '
                         f'({confidence:.2%})', fontsize=9)
        axes[i].set_xlabel('Time Point')
        axes[i].set_ylabel('Normalized Flux')
        axes[i].grid(True, alpha=0.3)

plt.suptitle('Sample Light Curves with Predictions', fontsize=12, y=0.995)
plt.tight_layout()
plt.savefig('reports/figures/genesis_sample_lightcurves.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: reports/figures/genesis_sample_lightcurves.png")
plt.close()

# ============================================
# SAVE RESULTS
# ============================================

print("\n[SAVING RESULTS]")

total_time = time.time() - start_time_total

results = {
    'metadata': {
        'timestamp': datetime.now().isoformat(),
        'total_time_seconds': float(total_time),
        'download_time_seconds': float(download_time),
        'training_time_seconds': float(ensemble_time),
        'gpu_used': gpu_available,
        'gpu_name': gpus[0].name if gpus else 'CPU'
    },
    'data': {
        'n_total': int(success_count),
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'n_augmented': int(len(X_train_aug)),
        'augmentation_factor': float(len(X_train_aug) / len(X_train))
    },
    'model': {
        'architecture': 'Genesis CNN',
        'ensemble_size': int(num_models),
        'total_parameters': int(genesis_models[0].count_params()),
        'input_shape': [2001, 1],
        'output_classes': 2
    },
    'performance': {
        'individual_accuracies': [float(a) for a in individual_accs],
        'ensemble_accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc)
    },
    'confusion_matrix': cm.tolist()
}

os.makedirs('reports/results', exist_ok=True)

with open('reports/results/genesis_200_real_kepler.json', 'w') as f:
    json.dump(results, f, indent=2)

print("[OK] Saved: reports/results/genesis_200_real_kepler.json")

# ============================================
# GENERATE PDF REPORT
# ============================================

print("\n[GENERATING PDF REPORT]")

from matplotlib.backends.backend_pdf import PdfPages

pdf_path = 'reports/Genesis_Real_Kepler_200_Report.pdf'

with PdfPages(pdf_path) as pdf:
    # Page 1: Title and Summary
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.85, 'Genesis CNN Model Training Report',
             ha='center', fontsize=20, weight='bold')
    fig.text(0.5, 0.78, '200 Real Kepler Light Curves with GPU Optimization',
             ha='center', fontsize=14)
    fig.text(0.5, 0.72, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
             ha='center', fontsize=10, style='italic')

    # Summary statistics
    summary_text = f"""
SUMMARY STATISTICS
{'='*60}

Dataset:
  • Total light curves: {success_count}
  • Training samples: {len(X_train)} (augmented to {len(X_train_aug)})
  • Test samples: {len(X_test)}
  • Data source: Real Kepler MAST archive

Model:
  • Architecture: Genesis CNN (5-layer ensemble)
  • Ensemble size: {num_models} models
  • Total parameters: {genesis_models[0].count_params():,}
  • GPU acceleration: {'Yes (' + gpus[0].name + ')' if gpu_available else 'No (CPU)'}

Performance:
  • Ensemble accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)
  • Precision: {precision:.4f}
  • Recall: {recall:.4f}
  • F1-Score: {f1:.4f}
  • ROC-AUC: {roc_auc:.4f}

Training Time:
  • Data download: {download_time/60:.1f} minutes
  • Model training: {ensemble_time/60:.1f} minutes
  • Total time: {total_time/60:.1f} minutes

Individual Model Accuracies:
"""
    for i, acc in enumerate(individual_accs):
        summary_text += f"  • Model {i+1}: {acc:.4f}\n"

    summary_text += f"\n  • Ensemble: {accuracy:.4f} (improvement: {accuracy - np.mean(individual_accs):.4f})"

    fig.text(0.1, 0.05, summary_text, fontsize=9, family='monospace',
             verticalalignment='bottom')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # Page 2: Training History
    img = plt.imread('reports/figures/genesis_training_history.png')
    fig = plt.figure(figsize=(11, 8.5))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Training History', fontsize=16, weight='bold', pad=20)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # Page 3: Performance Metrics
    fig = plt.figure(figsize=(11, 8.5))

    # Confusion Matrix
    ax1 = plt.subplot(2, 2, 1)
    img1 = plt.imread('reports/figures/genesis_confusion_matrix.png')
    ax1.imshow(img1)
    ax1.axis('off')

    # ROC Curve
    ax2 = plt.subplot(2, 2, 2)
    img2 = plt.imread('reports/figures/genesis_roc_curve.png')
    ax2.imshow(img2)
    ax2.axis('off')

    # Model Comparison
    ax3 = plt.subplot(2, 1, 2)
    img3 = plt.imread('reports/figures/genesis_model_comparison.png')
    ax3.imshow(img3)
    ax3.axis('off')

    plt.suptitle('Performance Metrics', fontsize=16, weight='bold', y=0.98)
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # Page 4: Sample Predictions
    img = plt.imread('reports/figures/genesis_sample_lightcurves.png')
    fig = plt.figure(figsize=(11, 8.5))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Sample Light Curve Predictions', fontsize=16, weight='bold', pad=20)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

print(f"[OK] PDF report saved: {pdf_path}")

# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "="*80)
print("GENESIS TRAINING COMPLETE!")
print("="*80)
print(f"\nTotal execution time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
print(f"\nKey Results:")
print(f"  • Successfully trained on {success_count} real Kepler light curves")
print(f"  • Ensemble accuracy: {accuracy:.2%}")
print(f"  • ROC-AUC: {roc_auc:.4f}")
print(f"  • GPU accelerated: {gpu_available}")
print(f"\nFiles generated:")
print(f"  • JSON results: reports/results/genesis_200_real_kepler.json")
print(f"  • PDF report: {pdf_path}")
print(f"  • Figures: reports/figures/genesis_*.png")
print("\n" + "="*80)
