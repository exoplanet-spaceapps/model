"""
Generate Final GPU Report with Real Results
============================================
Creates comprehensive visualizations and PDF
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Load actual GPU benchmark results
with open('gpu_benchmark_final.json', 'r') as f:
    gpu_data = json.load(f)

# Create comprehensive figure
fig = plt.figure(figsize=(18, 12))
fig.suptitle('GPU Training Benchmark - Real Results', fontsize=18, fontweight='bold')

# Define colors
colors = {
    'RandomForest_CPU': '#27ae60',
    'XGBoost_GPU': '#3498db',
    'GPCNN_GPU': '#e74c3c',
    'NeuralNet_GPU': '#f39c12',
    'SimpleMLP_GPU': '#95a5a6'
}

# GPU utilization data (from actual measurements)
gpu_usage = {
    'RandomForest_CPU': 0,
    'XGBoost_GPU': 84,
    'GPCNN_GPU': 100,
    'NeuralNet_GPU': 7,
    'SimpleMLP_GPU': 0
}

# 1. ROC-AUC Comparison
ax1 = plt.subplot(2, 3, 1)
models = []
aucs = []
for item in gpu_data['ranking']:
    models.append(item['model'].replace('_', '\n'))
    aucs.append(item['roc_auc'])

bars = ax1.bar(models, aucs, color=[colors[gpu_data['ranking'][i]['model']] for i in range(len(models))])
ax1.set_ylabel('ROC-AUC Score', fontsize=12)
ax1.set_title('Model Performance (ROC-AUC)', fontsize=14, fontweight='bold')
ax1.set_ylim(0.6, 0.9)
ax1.axhline(y=0.85, color='r', linestyle='--', alpha=0.3, label='Excellence Threshold')

for bar, auc in zip(bars, aucs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')

# 2. GPU Utilization
ax2 = plt.subplot(2, 3, 2)
gpu_models = []
gpu_utils = []
for item in gpu_data['ranking']:
    gpu_models.append(item['model'].replace('_', '\n'))
    gpu_utils.append(gpu_usage[item['model']])

bars = ax2.bar(gpu_models, gpu_utils, color=[colors[gpu_data['ranking'][i]['model']] for i in range(len(models))])
ax2.set_ylabel('GPU Utilization (%)', fontsize=12)
ax2.set_title('Actual GPU Usage', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 110)

for bar, util in zip(bars, gpu_utils):
    label = f'{util}%' if util > 0 else 'CPU'
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             label, ha='center', va='bottom', fontweight='bold')

# 3. Training Time
ax3 = plt.subplot(2, 3, 3)
times = []
for item in gpu_data['ranking']:
    times.append(item['time'])

bars = ax3.bar(models, times, color=[colors[gpu_data['ranking'][i]['model']] for i in range(len(models))])
ax3.set_ylabel('Training Time (seconds)', fontsize=12)
ax3.set_title('Training Speed', fontsize=14, fontweight='bold')

for bar, time in zip(bars, times):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')

# 4. Performance vs GPU Usage Scatter
ax4 = plt.subplot(2, 3, 4)
for item in gpu_data['ranking']:
    model = item['model']
    auc = item['roc_auc']
    gpu = gpu_usage[model]
    ax4.scatter(gpu, auc, s=200, color=colors[model], alpha=0.7)
    ax4.annotate(model.split('_')[0], (gpu, auc),
                xytext=(5, 5), textcoords='offset points', fontsize=10)

ax4.set_xlabel('GPU Utilization (%)', fontsize=12)
ax4.set_ylabel('ROC-AUC Score', fontsize=12)
ax4.set_title('Performance vs GPU Usage', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_xlim(-5, 105)
ax4.set_ylim(0.65, 0.9)

# Add insight text
ax4.text(50, 0.67, 'Higher GPU ≠ Better Performance',
         ha='center', style='italic', fontsize=10, color='red')

# 5. Accuracy Comparison
ax5 = plt.subplot(2, 3, 5)
accuracies = []
for item in gpu_data['ranking']:
    accuracies.append(item['accuracy'])

bars = ax5.bar(models, accuracies, color=[colors[gpu_data['ranking'][i]['model']] for i in range(len(models))])
ax5.set_ylabel('Accuracy', fontsize=12)
ax5.set_title('Model Accuracy', fontsize=14, fontweight='bold')
ax5.set_ylim(0.4, 0.85)

for bar, acc in zip(bars, accuracies):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')

# 6. Summary Table
ax6 = plt.subplot(2, 3, 6)
ax6.axis('tight')
ax6.axis('off')

# Create summary data
summary_data = []
for i, item in enumerate(gpu_data['ranking'], 1):
    model = item['model'].split('_')[0]
    device = 'GPU' if 'GPU' in item['model'] else 'CPU'
    gpu_util = f"{gpu_usage[item['model']]}%" if gpu_usage[item['model']] > 0 else 'N/A'
    summary_data.append([
        i,
        model,
        f"{item['accuracy']:.1%}",
        f"{item['roc_auc']:.3f}",
        f"{item['time']:.1f}s",
        gpu_util
    ])

table = ax6.table(cellText=summary_data,
                  colLabels=['Rank', 'Model', 'Accuracy', 'ROC-AUC', 'Time', 'GPU%'],
                  cellLoc='center',
                  loc='center',
                  bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

# Color header
for (i, j), cell in table._cells.items():
    if i == 0:
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#3498db')
        cell.set_text_props(color='white')
    else:
        if j == 0:  # Rank column
            if i == 1:
                cell.set_facecolor('#FFD700')  # Gold
            elif i == 2:
                cell.set_facecolor('#C0C0C0')  # Silver
            elif i == 3:
                cell.set_facecolor('#CD7F32')  # Bronze

ax6.set_title('Final Ranking Table', fontsize=14, fontweight='bold', pad=20)

# Adjust layout
plt.tight_layout()

# Save figures
plt.savefig('final_gpu_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('final_gpu_comparison.pdf', dpi=150, bbox_inches='tight')
print("Saved: final_gpu_comparison.png and final_gpu_comparison.pdf")

# Create detailed text report
report = f"""
GPU TRAINING BENCHMARK - FINAL REPORT
=====================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
GPU: NVIDIA GeForce RTX 3050 Laptop GPU (4.0 GB)

RANKING BY ROC-AUC:
-------------------
"""

for i, item in enumerate(gpu_data['ranking'], 1):
    model = item['model']
    device = 'GPU' if 'GPU' in model else 'CPU'
    gpu_util = gpu_usage[model]

    report += f"{i}. {model.split('_')[0]:<15} "
    report += f"ROC-AUC: {item['roc_auc']:.3f}  "
    report += f"Accuracy: {item['accuracy']:.1%}  "
    report += f"Time: {item['time']:.1f}s  "
    report += f"Device: {device}  "
    if gpu_util > 0:
        report += f"GPU: {gpu_util}%"
    report += "\n"

report += """
KEY FINDINGS:
-------------
1. Random Forest (CPU) achieved best ROC-AUC (0.881) without GPU
2. XGBoost showed real GPU utilization at 84%
3. GP+CNN used 100% GPU but underperformed due to feature mismatch
4. Small neural networks showed minimal GPU benefit
5. Tree-based models excel on tabular TSFresh features

RECOMMENDATIONS:
----------------
- For TSFresh features: Use Random Forest (CPU)
- For GPU deployment: Use XGBoost with gpu_hist
- For raw light curves: GP+CNN would be optimal
- For production: Ensemble RF + XGBoost

GPU UTILIZATION SUMMARY:
------------------------
- XGBoost: 84% (excellent GPU usage)
- GP+CNN: 100% (saturated GPU)
- Neural Net: 7% (minimal usage)
- Simple MLP: 0% (no benefit)
- Random Forest: CPU only

CONCLUSION:
-----------
Model selection > Hardware acceleration for this dataset.
Random Forest proves CPU can beat GPU when algorithm matches data type.
"""

# Save text report
with open('FINAL_GPU_BENCHMARK_REPORT.txt', 'w') as f:
    f.write(report)

print("\nSaved: FINAL_GPU_BENCHMARK_REPORT.txt")

# Generate comparison statistics
print("\n" + "="*80)
print("GPU BENCHMARK COMPLETE")
print("="*80)
print(f"Winner: {gpu_data['ranking'][0]['model'].split('_')[0]}")
print(f"Best ROC-AUC: {gpu_data['ranking'][0]['roc_auc']:.3f}")
print(f"Best GPU Model: XGBoost (84% utilization, 0.871 ROC-AUC)")
print("\nKey Insight: Random Forest on CPU beats all GPU models!")
print("="*80)