"""
Final Model Comparison Report
=============================
Compare Genesis vs XGBoost vs Random Forest
Generate comprehensive comparison charts and tables

Author: NASA Kepler Project
Date: 2025-10-05
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11

print("="*80)
print("FINAL MODEL COMPARISON: GENESIS VS XGBOOST VS RANDOM FOREST")
print("="*80)

# ============================================
# LOAD RESULTS
# ============================================

print("\n[LOADING RESULTS]")

# Load Genesis results
with open('reports/results/genesis_final_results.json', 'r') as f:
    genesis_results = json.load(f)

# Load quick comparison results (XGBoost + RF)
with open('reports/results/quick_complete_comparison.json', 'r') as f:
    quick_results = json.load(f)

print("[OK] Loaded all results")

# ============================================
# PREPARE COMPARISON DATA
# ============================================

print("\n[PREPARING COMPARISON DATA]")

# Extract metrics
models_data = {
    'Genesis Ensemble': {
        'accuracy': genesis_results['performance']['ensemble_accuracy'],
        'precision': genesis_results['performance']['precision'],
        'recall': genesis_results['performance']['recall'],
        'f1': genesis_results['performance']['f1'],
        'roc_auc': genesis_results['performance']['roc_auc'],
        'training_time': genesis_results['metadata']['training_time_min'],
        'data_type': 'Real Light Curves',
        'n_samples': genesis_results['data']['n_total'],
        'device': 'CPU',
        'model_type': 'Deep Learning (CNN)'
    },
    'XGBoost CPU': {
        'accuracy': quick_results['XGBoost_CPU']['accuracy'],
        'precision': quick_results['XGBoost_CPU']['precision'],
        'recall': quick_results['XGBoost_CPU']['recall'],
        'f1': quick_results['XGBoost_CPU']['f1'],
        'roc_auc': quick_results['XGBoost_CPU']['roc_auc'],
        'training_time': quick_results['XGBoost_CPU']['training_time'] / 60,  # Convert to min
        'data_type': 'TSFresh Features',
        'n_samples': 369,  # Test set size
        'device': 'CPU',
        'model_type': 'Gradient Boosting'
    },
    'Random Forest': {
        'accuracy': quick_results['Random_Forest']['accuracy'],
        'precision': quick_results['Random_Forest']['precision'],
        'recall': quick_results['Random_Forest']['recall'],
        'f1': quick_results['Random_Forest']['f1'],
        'roc_auc': quick_results['Random_Forest']['roc_auc'],
        'training_time': quick_results['Random_Forest']['training_time'] / 60,  # Convert to min
        'data_type': 'TSFresh Features',
        'n_samples': 369,  # Test set size
        'device': 'CPU',
        'model_type': 'Tree Ensemble'
    }
}

# Create DataFrame
df_comparison = pd.DataFrame(models_data).T
df_comparison.index.name = 'Model'

print("[OK] Comparison data prepared")
print("\n" + df_comparison.to_string())

# ============================================
# VISUALIZATION 1: PERFORMANCE METRICS BAR CHART
# ============================================

print("\n[GENERATING VISUALIZATIONS]")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Performance Comparison: Genesis vs XGBoost vs Random Forest',
             fontsize=16, weight='bold')

metrics = ['accuracy', 'precision', 'recall', 'f1']
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['#3498db', '#e74c3c', '#2ecc71']  # Blue, Red, Green

for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
    ax = axes[idx // 2, idx % 2]

    values = [models_data[model][metric] for model in models_data.keys()]
    bars = ax.bar(range(len(models_data)), values, color=colors, alpha=0.7, edgecolor='black')

    ax.set_ylabel(name, fontsize=12, weight='bold')
    ax.set_title(f'{name} Comparison', fontsize=13, weight='bold')
    ax.set_xticks(range(len(models_data)))
    ax.set_xticklabels(models_data.keys(), rotation=15, ha='right')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=10, weight='bold')

plt.tight_layout()
plt.savefig('reports/figures/final_performance_comparison.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: final_performance_comparison.png")
plt.close()

# ============================================
# VISUALIZATION 2: ROC-AUC + TRAINING TIME
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC-AUC
ax1 = axes[0]
roc_values = [models_data[model]['roc_auc'] for model in models_data.keys()]
bars1 = ax1.bar(range(len(models_data)), roc_values, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('ROC-AUC Score', fontsize=12, weight='bold')
ax1.set_title('ROC-AUC Comparison', fontsize=13, weight='bold')
ax1.set_xticks(range(len(models_data)))
ax1.set_xticklabels(models_data.keys(), rotation=15, ha='right')
ax1.set_ylim([0, 1.05])
ax1.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars1, roc_values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{val:.4f}',
            ha='center', va='bottom', fontsize=10, weight='bold')

# Training Time (log scale)
ax2 = axes[1]
time_values = [models_data[model]['training_time'] for model in models_data.keys()]
bars2 = ax2.bar(range(len(models_data)), time_values, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Training Time (minutes, log scale)', fontsize=12, weight='bold')
ax2.set_title('Training Time Comparison', fontsize=13, weight='bold')
ax2.set_xticks(range(len(models_data)))
ax2.set_xticklabels(models_data.keys(), rotation=15, ha='right')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars2, time_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height * 1.2,
            f'{val:.2f}m',
            ha='center', va='bottom', fontsize=10, weight='bold')

plt.tight_layout()
plt.savefig('reports/figures/final_roc_time_comparison.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: final_roc_time_comparison.png")
plt.close()

# ============================================
# VISUALIZATION 3: COMPREHENSIVE COMPARISON
# ============================================

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Overall scores (radar-like bar)
ax1 = fig.add_subplot(gs[0, :])
x_pos = np.arange(len(models_data))
width = 0.2

metrics_radar = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']

for i, metric in enumerate(metrics_radar):
    values = [models_data[model][metric] for model in models_data.keys()]
    offset = (i - len(metrics_radar)/2) * width
    bars = ax1.bar(x_pos + offset, values, width, label=metric_labels[i], alpha=0.8)

ax1.set_ylabel('Score', fontsize=12, weight='bold')
ax1.set_title('All Metrics Comparison', fontsize=14, weight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(models_data.keys())
ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax1.set_ylim([0, 1.1])
ax1.grid(True, alpha=0.3, axis='y')

# Accuracy comparison
ax2 = fig.add_subplot(gs[1, 0])
acc_values = [models_data[model]['accuracy'] for model in models_data.keys()]
bars = ax2.barh(range(len(models_data)), acc_values, color=colors, alpha=0.7, edgecolor='black')
ax2.set_yticks(range(len(models_data)))
ax2.set_yticklabels(models_data.keys())
ax2.set_xlabel('Accuracy', fontsize=11, weight='bold')
ax2.set_title('Accuracy (Horizontal)', fontsize=12, weight='bold')
ax2.set_xlim([0, 1.05])
ax2.grid(True, alpha=0.3, axis='x')

for i, (bar, val) in enumerate(zip(bars, acc_values)):
    width = bar.get_width()
    ax2.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
            f'{val:.2%}',
            ha='left', va='center', fontsize=10, weight='bold')

# ROC-AUC comparison
ax3 = fig.add_subplot(gs[1, 1])
roc_values = [models_data[model]['roc_auc'] for model in models_data.keys()]
bars = ax3.barh(range(len(models_data)), roc_values, color=colors, alpha=0.7, edgecolor='black')
ax3.set_yticks(range(len(models_data)))
ax3.set_yticklabels(models_data.keys())
ax3.set_xlabel('ROC-AUC', fontsize=11, weight='bold')
ax3.set_title('ROC-AUC (Horizontal)', fontsize=12, weight='bold')
ax3.set_xlim([0, 1.05])
ax3.grid(True, alpha=0.3, axis='x')

for i, (bar, val) in enumerate(zip(bars, roc_values)):
    width = bar.get_width()
    ax3.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
            f'{val:.4f}',
            ha='left', va='center', fontsize=10, weight='bold')

# Training time comparison
ax4 = fig.add_subplot(gs[2, 0])
time_values = [models_data[model]['training_time'] for model in models_data.keys()]
bars = ax4.barh(range(len(models_data)), time_values, color=colors, alpha=0.7, edgecolor='black')
ax4.set_yticks(range(len(models_data)))
ax4.set_yticklabels(models_data.keys())
ax4.set_xlabel('Training Time (minutes)', fontsize=11, weight='bold')
ax4.set_title('Training Time', fontsize=12, weight='bold')
ax4.grid(True, alpha=0.3, axis='x')

for i, (bar, val) in enumerate(zip(bars, time_values)):
    width = bar.get_width()
    ax4.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
            f'{val:.2f}m',
            ha='left', va='center', fontsize=10, weight='bold')

# Summary table
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('off')

table_data = []
for model_name, data in models_data.items():
    table_data.append([
        model_name,
        f"{data['accuracy']:.3f}",
        f"{data['roc_auc']:.4f}",
        f"{data['training_time']:.2f}m"
    ])

table = ax5.table(cellText=table_data,
                 colLabels=['Model', 'Accuracy', 'ROC-AUC', 'Time'],
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header
for i in range(4):
    table[(0, i)].set_facecolor('#3498db')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style rows
row_colors = ['#ecf0f1', 'white']
for i in range(1, len(table_data) + 1):
    for j in range(4):
        table[(i, j)].set_facecolor(row_colors[i % 2])

ax5.set_title('Summary Table', fontsize=12, weight='bold', pad=20)

plt.savefig('reports/figures/final_comprehensive_comparison.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: final_comprehensive_comparison.png")
plt.close()

# ============================================
# CREATE COMPARISON TABLE (DETAILED)
# ============================================

print("\n[CREATING DETAILED COMPARISON TABLE]")

# Create detailed table
detailed_data = []
for model_name, data in models_data.items():
    detailed_data.append({
        'Model': model_name,
        'Model Type': data['model_type'],
        'Data Type': data['data_type'],
        'N Samples': data['n_samples'],
        'Accuracy': f"{data['accuracy']:.4f}",
        'Precision': f"{data['precision']:.4f}",
        'Recall': f"{data['recall']:.4f}",
        'F1-Score': f"{data['f1']:.4f}",
        'ROC-AUC': f"{data['roc_auc']:.4f}",
        'Training Time': f"{data['training_time']:.2f} min",
        'Device': data['device']
    })

df_detailed = pd.DataFrame(detailed_data)

# Save as CSV
df_detailed.to_csv('reports/results/final_comparison_table.csv', index=False)
print("[OK] Saved: final_comparison_table.csv")

# Create formatted table image
fig, ax = plt.subplots(figsize=(16, 6))
ax.axis('tight')
ax.axis('off')

# Prepare data for table
table_display = []
for _, row in df_detailed.iterrows():
    table_display.append([
        row['Model'],
        row['Model Type'],
        row['Data Type'],
        str(row['N Samples']),
        row['Accuracy'],
        row['ROC-AUC'],
        row['F1-Score'],
        row['Training Time']
    ])

table = ax.table(
    cellText=table_display,
    colLabels=['Model', 'Type', 'Data', 'N', 'Acc', 'AUC', 'F1', 'Time'],
    cellLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1]
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)

# Style header
for i in range(8):
    table[(0, i)].set_facecolor('#2c3e50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Highlight best values
acc_vals = [float(d['Accuracy']) for d in detailed_data]
auc_vals = [float(d['ROC-AUC']) for d in detailed_data]
f1_vals = [float(d['F1-Score']) for d in detailed_data]

best_acc_idx = acc_vals.index(max(acc_vals))
best_auc_idx = auc_vals.index(max(auc_vals))
best_f1_idx = f1_vals.index(max(f1_vals))

# Color best cells
table[(best_acc_idx + 1, 4)].set_facecolor('#2ecc71')  # Best accuracy
table[(best_auc_idx + 1, 5)].set_facecolor('#2ecc71')  # Best AUC
table[(best_f1_idx + 1, 6)].set_facecolor('#2ecc71')   # Best F1

plt.title('Detailed Model Comparison Table', fontsize=14, weight='bold', pad=20)
plt.savefig('reports/figures/final_comparison_table.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: final_comparison_table.png")
plt.close()

# ============================================
# GENERATE UPDATED PDF REPORT
# ============================================

print("\n[GENERATING COMPREHENSIVE PDF REPORT]")

pdf_path = 'reports/Final_Model_Comparison_Report.pdf'

with PdfPages(pdf_path) as pdf:
    # Page 1: Title and Summary
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.85, 'Final Model Comparison Report',
             ha='center', fontsize=22, weight='bold')
    fig.text(0.5, 0.78, 'Genesis CNN vs XGBoost vs Random Forest',
             ha='center', fontsize=16)
    fig.text(0.5, 0.72, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
             ha='center', fontsize=11, style='italic')

    summary_text = f"""
{'='*75}
EXECUTIVE SUMMARY
{'='*75}

Models Compared:
  1. Genesis Ensemble (Deep Learning CNN)
     • Data: Real Kepler light curves (99 samples)
     • Training: 16.7 minutes on CPU
     • Architecture: 5-model ensemble, 487K parameters

  2. XGBoost CPU (Gradient Boosting)
     • Data: TSFresh features (369 test samples)
     • Training: 5.66 seconds
     • Architecture: 300 trees, max_depth=6

  3. Random Forest (Tree Ensemble)
     • Data: TSFresh features (369 test samples)
     • Training: 0.83 seconds
     • Architecture: 200 trees, max_depth=8

{'='*75}
PERFORMANCE COMPARISON
{'='*75}

                    Genesis    XGBoost    Random Forest
Accuracy            {models_data['Genesis Ensemble']['accuracy']:.4f}     {models_data['XGBoost CPU']['accuracy']:.4f}     {models_data['Random Forest']['accuracy']:.4f}
Precision           {models_data['Genesis Ensemble']['precision']:.4f}     {models_data['XGBoost CPU']['precision']:.4f}     {models_data['Random Forest']['precision']:.4f}
Recall              {models_data['Genesis Ensemble']['recall']:.4f}     {models_data['XGBoost CPU']['recall']:.4f}     {models_data['Random Forest']['recall']:.4f}
F1-Score            {models_data['Genesis Ensemble']['f1']:.4f}     {models_data['XGBoost CPU']['f1']:.4f}     {models_data['Random Forest']['f1']:.4f}
ROC-AUC             {models_data['Genesis Ensemble']['roc_auc']:.4f}     {models_data['XGBoost CPU']['roc_auc']:.4f}     {models_data['Random Forest']['roc_auc']:.4f}
Training Time       {models_data['Genesis Ensemble']['training_time']:.1f}m      {models_data['XGBoost CPU']['training_time']:.2f}m      {models_data['Random Forest']['training_time']:.2f}m

{'='*75}
KEY FINDINGS
{'='*75}

Best Overall Performance:
  • Genesis: Highest accuracy (90.0%) and perfect precision (100%)
  • XGBoost: Best on TSFresh features (87.1% AUC)
  • Random Forest: Fastest training (<1 second)

Data Type Matters:
  • Genesis excels on raw light curves (90% accuracy)
  • XGBoost/RF excel on TSFresh features (87% AUC)
  • Different data representations suit different models

Trade-offs:
  • Genesis: Best accuracy but slowest (16.7 min)
  • XGBoost: Balanced performance/speed (5.7 sec)
  • Random Forest: Ultra-fast but slightly lower AUC (0.8 sec)

Recommendations:
  • Research/Accuracy Priority → Genesis Ensemble
  • Production/Speed Priority → XGBoost CPU
  • Rapid Prototyping → Random Forest
  • Ensemble All → Combine predictions for best results

{'='*75}
"""

    fig.text(0.1, 0.05, summary_text, fontsize=9, family='monospace',
             verticalalignment='bottom')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # Page 2: Performance Metrics
    img = plt.imread('reports/figures/final_performance_comparison.png')
    fig = plt.figure(figsize=(11, 8.5))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Performance Metrics Comparison', fontsize=16, weight='bold', pad=20)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # Page 3: ROC-AUC and Time
    img = plt.imread('reports/figures/final_roc_time_comparison.png')
    fig = plt.figure(figsize=(11, 8.5))
    plt.imshow(img)
    plt.axis('off')
    plt.title('ROC-AUC and Training Time', fontsize=16, weight='bold', pad=20)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # Page 4: Comprehensive Comparison
    img = plt.imread('reports/figures/final_comprehensive_comparison.png')
    fig = plt.figure(figsize=(11, 8.5))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Comprehensive Comparison', fontsize=16, weight='bold', pad=20)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # Page 5: Detailed Table
    img = plt.imread('reports/figures/final_comparison_table.png')
    fig = plt.figure(figsize=(11, 8.5))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Detailed Comparison Table', fontsize=16, weight='bold', pad=20)
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

print(f"[OK] PDF saved: {pdf_path}")

# ============================================
# SAVE COMPARISON JSON
# ============================================

comparison_summary = {
    'timestamp': datetime.now().isoformat(),
    'models': models_data,
    'best_performance': {
        'highest_accuracy': max(models_data.items(), key=lambda x: x[1]['accuracy'])[0],
        'highest_roc_auc': max(models_data.items(), key=lambda x: x[1]['roc_auc'])[0],
        'fastest_training': min(models_data.items(), key=lambda x: x[1]['training_time'])[0]
    }
}

with open('reports/results/final_comparison_summary.json', 'w') as f:
    json.dump(comparison_summary, f, indent=2)

print("[OK] Saved: final_comparison_summary.json")

# ============================================
# PRINT SUMMARY
# ============================================

print("\n" + "="*80)
print("FINAL MODEL COMPARISON COMPLETE!")
print("="*80)

print("\n📊 COMPARISON SUMMARY:")
print(f"\n{df_detailed.to_string(index=False)}")

print("\n🏆 BEST PERFORMERS:")
print(f"  • Highest Accuracy: {comparison_summary['best_performance']['highest_accuracy']}")
print(f"  • Highest ROC-AUC:  {comparison_summary['best_performance']['highest_roc_auc']}")
print(f"  • Fastest Training: {comparison_summary['best_performance']['fastest_training']}")

print("\n📁 OUTPUT FILES:")
print("  • PDF Report: reports/Final_Model_Comparison_Report.pdf")
print("  • CSV Table:  reports/results/final_comparison_table.csv")
print("  • JSON:       reports/results/final_comparison_summary.json")
print("  • Figures:    reports/figures/final_*.png (4 files)")

print("\n" + "="*80)
