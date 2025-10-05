"""
Complete Model Training and Comparison - Kaggle Dataset
========================================================
Train Genesis CNN, XGBoost, and Random Forest on Kaggle Kepler data
Generate comprehensive comparison report with plots and PDF

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
from pathlib import Path
from datetime import datetime

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for Chinese display (if needed)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("="*80)
print("COMPLETE MODEL TRAINING & COMPARISON - KAGGLE DATASET")
print("="*80)
print()

# ============================================
# CONFIGURATION
# ============================================

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

REPORTS_DIR = Path('reports/kaggle_comparison')
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FIGURES_DIR = REPORTS_DIR / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

# ============================================
# STEP 1: LOAD KAGGLE DATA
# ============================================

print("[STEP 1/6] Loading Kaggle Kepler dataset...")

train_df = pd.read_csv('data/kaggle_kepler/exoTrain.csv')
test_df = pd.read_csv('data/kaggle_kepler/exoTest.csv')

# Extract features and labels
X_train_raw = train_df.iloc[:, 1:].values
y_train_raw = train_df.iloc[:, 0].values

X_test = test_df.iloc[:, 1:].values
y_test = test_df.iloc[:, 0].values

# Convert labels: 2 (planet) -> 1, 1 (non-planet) -> 0
y_train_raw = (y_train_raw == 2).astype(int)
y_test = (y_test == 2).astype(int)

print(f"  Train: {X_train_raw.shape}, Planets: {y_train_raw.sum()}, Non-planets: {(1-y_train_raw).sum()}")
print(f"  Test: {X_test.shape}, Planets: {y_test.sum()}, Non-planets: {(1-y_test).sum()}")
print(f"  Class imbalance: {y_train_raw.sum()/len(y_train_raw)*100:.2f}% planets")
print()

# ============================================
# STEP 2: HANDLE CLASS IMBALANCE
# ============================================

print("[STEP 2/6] Handling class imbalance with SMOTE...")

try:
    from imblearn.over_sampling import SMOTE

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train, y_train = smote.fit_resample(X_train_raw, y_train_raw)

    print(f"  After SMOTE - Train: {X_train.shape}")
    print(f"  Planets: {y_train.sum()}, Non-planets: {(1-y_train).sum()}")
    print(f"  Balance: {y_train.sum()/len(y_train)*100:.1f}% planets")
except ImportError:
    print("  [WARN] imbalanced-learn not installed, using original data")
    X_train, y_train = X_train_raw, y_train_raw

print()

# ============================================
# STEP 3: TRAIN GENESIS CNN
# ============================================

print("[STEP 3/6] Training Genesis CNN model...")
genesis_start = time.time()

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Dense, Flatten, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping

    # Enable GPU if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"  [INFO] GPU available: {len(gpus)} device(s)")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("  [INFO] No GPU found, using CPU")

    # Prepare data for CNN
    X_train_cnn = X_train.reshape(-1, X_train.shape[1], 1)
    X_test_cnn = X_test.reshape(-1, X_test.shape[1], 1)

    # Convert to categorical
    y_train_cat = tf.keras.utils.to_categorical(y_train, 2)
    y_test_cat = tf.keras.utils.to_categorical(y_test, 2)

    # Build Genesis model (adapted for 3197 time points)
    def build_genesis_adapted():
        model = Sequential([
            Conv1D(64, 50, padding='same', activation='relu', input_shape=(X_train.shape[1], 1)),
            Conv1D(64, 50, padding='same', activation='relu'),
            MaxPooling1D(pool_size=16, strides=16),  # Adapted pooling
            Conv1D(64, 12, padding='same', activation='relu'),
            Conv1D(64, 12, padding='same', activation='relu'),
            AveragePooling1D(pool_size=8),
            Dropout(0.25),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(256, activation='relu'),
            Dense(2, activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    # Build and train
    genesis_model = build_genesis_adapted()

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    print("  Training Genesis CNN (10 epochs on CPU)...")
    history = genesis_model.fit(
        X_train_cnn, y_train_cat,
        validation_data=(X_test_cnn, y_test_cat),
        epochs=10,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    # Evaluate
    genesis_loss, genesis_acc = genesis_model.evaluate(X_test_cnn, y_test_cat, verbose=0)
    genesis_time = time.time() - genesis_start

    # Predictions for metrics
    y_pred_genesis_proba = genesis_model.predict(X_test_cnn, verbose=0)
    y_pred_genesis = np.argmax(y_pred_genesis_proba, axis=1)

    print(f"  [OK] Genesis trained in {genesis_time:.1f}s")
    print(f"  Accuracy: {genesis_acc:.4f}")

    genesis_success = True

except Exception as e:
    print(f"  [ERROR] Genesis training failed: {e}")
    genesis_success = False
    y_pred_genesis = None

print()

# ============================================
# STEP 4: TRAIN XGBOOST
# ============================================

print("[STEP 4/6] Training XGBoost model...")
xgb_start = time.time()

try:
    import xgboost as xgb

    # Calculate scale_pos_weight
    scale_pos_weight = (1 - y_train).sum() / y_train.sum()

    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    xgb_model.fit(X_train, y_train, verbose=False)
    xgb_time = time.time() - xgb_start

    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

    from sklearn.metrics import accuracy_score
    xgb_acc = accuracy_score(y_test, y_pred_xgb)

    print(f"  [OK] XGBoost trained in {xgb_time:.2f}s")
    print(f"  Accuracy: {xgb_acc:.4f}")

    xgb_success = True

except Exception as e:
    print(f"  [ERROR] XGBoost training failed: {e}")
    xgb_success = False
    y_pred_xgb = None

print()

# ============================================
# STEP 5: TRAIN RANDOM FOREST
# ============================================

print("[STEP 5/6] Training Random Forest model...")
rf_start = time.time()

try:
    from sklearn.ensemble import RandomForestClassifier

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_train)
    rf_time = time.time() - rf_start

    y_pred_rf = rf_model.predict(X_test)
    y_pred_rf_proba = rf_model.predict_proba(X_test)[:, 1]

    rf_acc = accuracy_score(y_test, y_pred_rf)

    print(f"  [OK] Random Forest trained in {rf_time:.2f}s")
    print(f"  Accuracy: {rf_acc:.4f}")

    rf_success = True

except Exception as e:
    print(f"  [ERROR] Random Forest training failed: {e}")
    rf_success = False
    y_pred_rf = None

print()

# ============================================
# STEP 6: GENERATE COMPARISON REPORT
# ============================================

print("[STEP 6/6] Generating comparison report and visualizations...")

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Calculate metrics for all models
results = {}

if genesis_success:
    results['Genesis CNN'] = {
        'accuracy': accuracy_score(y_test, y_pred_genesis),
        'precision': precision_score(y_test, y_pred_genesis),
        'recall': recall_score(y_test, y_pred_genesis),
        'f1': f1_score(y_test, y_pred_genesis),
        'roc_auc': roc_auc_score(y_test, y_pred_genesis_proba[:, 1]),
        'training_time': genesis_time,
        'model_type': 'Deep Learning (CNN)',
        'data_type': 'Kaggle Time Series'
    }

if xgb_success:
    results['XGBoost'] = {
        'accuracy': accuracy_score(y_test, y_pred_xgb),
        'precision': precision_score(y_test, y_pred_xgb),
        'recall': recall_score(y_test, y_pred_xgb),
        'f1': f1_score(y_test, y_pred_xgb),
        'roc_auc': roc_auc_score(y_test, y_pred_xgb_proba),
        'training_time': xgb_time,
        'model_type': 'Gradient Boosting',
        'data_type': 'Kaggle Time Series'
    }

if rf_success:
    results['Random Forest'] = {
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'precision': precision_score(y_test, y_pred_rf),
        'recall': recall_score(y_test, y_pred_rf),
        'f1': f1_score(y_test, y_pred_rf),
        'roc_auc': roc_auc_score(y_test, y_pred_rf_proba),
        'training_time': rf_time,
        'model_type': 'Tree Ensemble',
        'data_type': 'Kaggle Time Series'
    }

# Save results JSON
with open(REPORTS_DIR / 'kaggle_comparison_results.json', 'w') as f:
    json.dump({
        'timestamp': datetime.now().isoformat(),
        'models': results,
        'dataset': {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'features': X_train.shape[1],
            'class_balance': f"{y_train.sum()/len(y_train)*100:.1f}% planets (after SMOTE)"
        }
    }, f, indent=2)

print(f"  [OK] Results saved to {REPORTS_DIR / 'kaggle_comparison_results.json'}")

# ============================================
# GENERATE VISUALIZATIONS
# ============================================

# 1. Performance Comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

metrics = ['accuracy', 'precision', 'recall', 'f1']
for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]

    models = list(results.keys())
    values = [results[m][metric] for m in models]
    colors = ['#2ecc71', '#3498db', '#e74c3c'][:len(models)]

    bars = ax.bar(models, values, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(f'{metric.capitalize()} Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'performance_comparison.png', dpi=300, bbox_inches='tight')
print(f"  [OK] Saved: {FIGURES_DIR / 'performance_comparison.png'}")
plt.close()

# 2. ROC AUC and Training Time
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

models = list(results.keys())
roc_scores = [results[m]['roc_auc'] for m in models]
times = [results[m]['training_time'] for m in models]

# ROC AUC
bars1 = ax1.barh(models, roc_scores, color='#9b59b6', alpha=0.8, edgecolor='black')
ax1.set_xlabel('ROC-AUC Score', fontsize=12)
ax1.set_title('ROC-AUC Comparison', fontsize=14, fontweight='bold')
ax1.set_xlim([0, 1.1])
ax1.grid(axis='x', alpha=0.3)

for bar, val in zip(bars1, roc_scores):
    ax1.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', ha='left', va='center', fontsize=10)

# Training Time
bars2 = ax2.barh(models, times, color='#f39c12', alpha=0.8, edgecolor='black')
ax2.set_xlabel('Training Time (seconds)', fontsize=12)
ax2.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

for bar, val in zip(bars2, times):
    ax2.text(bar.get_width() + max(times)*0.02, bar.get_y() + bar.get_height()/2,
             f'{val:.2f}s', ha='left', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'roc_time_comparison.png', dpi=300, bbox_inches='tight')
print(f"  [OK] Saved: {FIGURES_DIR / 'roc_time_comparison.png'}")
plt.close()

# 3. Confusion Matrices
fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4))
if len(results) == 1:
    axes = [axes]

predictions = {
    'Genesis CNN': y_pred_genesis if genesis_success else None,
    'XGBoost': y_pred_xgb if xgb_success else None,
    'Random Forest': y_pred_rf if rf_success else None
}

for idx, (model_name, y_pred) in enumerate(predictions.items()):
    if y_pred is not None:
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Non-Planet', 'Planet'],
                   yticklabels=['Non-Planet', 'Planet'])
        axes[idx].set_title(f'{model_name}\nConfusion Matrix', fontweight='bold')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
print(f"  [OK] Saved: {FIGURES_DIR / 'confusion_matrices.png'}")
plt.close()

# ============================================
# GENERATE PDF REPORT
# ============================================

print()
print("  Generating PDF report...")

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    pdf_path = REPORTS_DIR / 'KAGGLE_MODEL_COMPARISON_REPORT.pdf'
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30,
        alignment=TA_CENTER
    )

    story.append(Paragraph("Kaggle Kepler Dataset", title_style))
    story.append(Paragraph("Model Comparison Report", title_style))
    story.append(Spacer(1, 0.3*inch))

    # Metadata
    metadata = [
        ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['Dataset:', 'Kaggle Kepler Labelled Time Series'],
        ['Train Samples:', f"{len(X_train):,}"],
        ['Test Samples:', f"{len(X_test):,}"],
        ['Features:', f"{X_train.shape[1]:,} time points"],
        ['Models Trained:', ', '.join(results.keys())]
    ]

    t = Table(metadata, colWidths=[2.5*inch, 4*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ecf0f1')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.white)
    ]))
    story.append(t)
    story.append(Spacer(1, 0.4*inch))

    # Results Table
    story.append(Paragraph("Model Performance Metrics", styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))

    results_data = [['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Time (s)']]
    for model, metrics in results.items():
        results_data.append([
            model,
            f"{metrics['accuracy']:.4f}",
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            f"{metrics['f1']:.4f}",
            f"{metrics['roc_auc']:.4f}",
            f"{metrics['training_time']:.2f}"
        ])

    t = Table(results_data, colWidths=[1.5*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.8*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(t)
    story.append(PageBreak())

    # Visualizations
    story.append(Paragraph("Performance Visualizations", styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))

    # Add images
    img1 = Image(str(FIGURES_DIR / 'performance_comparison.png'), width=6.5*inch, height=5.2*inch)
    story.append(img1)
    story.append(Spacer(1, 0.3*inch))

    story.append(PageBreak())

    img2 = Image(str(FIGURES_DIR / 'roc_time_comparison.png'), width=6.5*inch, height=2.6*inch)
    story.append(img2)
    story.append(Spacer(1, 0.3*inch))

    img3 = Image(str(FIGURES_DIR / 'confusion_matrices.png'), width=6.5*inch, height=2.6*inch)
    story.append(img3)

    # Build PDF
    doc.build(story)
    print(f"  [OK] PDF report saved: {pdf_path}")

except Exception as e:
    print(f"  [WARN] PDF generation failed: {e}")
    print("  [INFO] Results still available in JSON and PNG formats")

# ============================================
# SUMMARY
# ============================================

print()
print("="*80)
print("TRAINING COMPLETE!")
print("="*80)
print()

# Find best model
best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"[BEST MODEL] {best_model[0]}")
print(f"  Accuracy: {best_model[1]['accuracy']:.4f}")
print(f"  F1-Score: {best_model[1]['f1']:.4f}")
print(f"  ROC-AUC: {best_model[1]['roc_auc']:.4f}")
print()

print("[OUTPUT FILES]")
print(f"  JSON Results: {REPORTS_DIR / 'kaggle_comparison_results.json'}")
print(f"  Performance Plot: {FIGURES_DIR / 'performance_comparison.png'}")
print(f"  ROC/Time Plot: {FIGURES_DIR / 'roc_time_comparison.png'}")
print(f"  Confusion Matrices: {FIGURES_DIR / 'confusion_matrices.png'}")
print(f"  PDF Report: {REPORTS_DIR / 'KAGGLE_MODEL_COMPARISON_REPORT.pdf'}")
print()

print("[COMPARISON SUMMARY]")
for model, metrics in results.items():
    print(f"  {model}:")
    print(f"    Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f} | Time: {metrics['training_time']:.2f}s")

print()
print("="*80)
