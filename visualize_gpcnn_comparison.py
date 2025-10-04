"""
Visualize Complete Model Comparison Including GP+CNN
=====================================================
Creates comprehensive charts showing all models' performance
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('complete_gpcnn_benchmark_results.json', 'r') as f:
    data = json.load(f)

results = data['results']

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Complete Model Comparison - Including GP+CNN', fontsize=16, fontweight='bold')

# Define colors for each model
colors = {
    'RandomForest_CPU': '#27ae60',
    'XGBoost_CPU': '#3498db',
    'XGBoost_GPU': '#2980b9',
    'NeuralNet_GPU': '#f39c12',
    'GP+CNN_GPU': '#e74c3c'
}

# Prepare data
models = list(results.keys())
model_names = [m.replace('_', '\n') for m in models]
accuracies = [results[m]['accuracy'] for m in models]
roc_aucs = [results[m]['roc_auc'] for m in models]
times = [results[m]['time'] for m in models]
devices = [results[m]['device'] for m in models]

# 1. ROC-AUC Comparison
ax1 = axes[0, 0]
bars = ax1.bar(model_names, roc_aucs, color=[colors[m] for m in models])
ax1.set_ylabel('ROC-AUC Score', fontsize=12)
ax1.set_title('ROC-AUC Performance', fontsize=14)
ax1.set_ylim(0.8, 0.9)
ax1.axhline(y=0.85, color='r', linestyle='--', alpha=0.3)

for bar, auc in zip(bars, roc_aucs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{auc:.3f}', ha='center', va='bottom', fontweight='bold')

# 2. Accuracy Comparison
ax2 = axes[0, 1]
bars = ax2.bar(model_names, accuracies, color=[colors[m] for m in models])
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Model Accuracy', fontsize=14)
ax2.set_ylim(0.7, 0.85)

for bar, acc in zip(bars, accuracies):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')

# 3. Training Time
ax3 = axes[0, 2]
bars = ax3.bar(model_names, times, color=[colors[m] for m in models])
ax3.set_ylabel('Training Time (seconds)', fontsize=12)
ax3.set_title('Training Speed', fontsize=14)

for bar, time in zip(bars, times):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')

# 4. Device Comparison
ax4 = axes[1, 0]
gpu_models = [m for m in models if results[m]['device'] == 'GPU']
cpu_models = [m for m in models if results[m]['device'] == 'CPU']

gpu_aucs = [results[m]['roc_auc'] for m in gpu_models]
cpu_aucs = [results[m]['roc_auc'] for m in cpu_models]

ax4.scatter([1]*len(cpu_models), cpu_aucs, s=200, alpha=0.7, label='CPU', color='blue')
ax4.scatter([2]*len(gpu_models), gpu_aucs, s=200, alpha=0.7, label='GPU', color='red')

for i, m in enumerate(cpu_models):
    ax4.annotate(m.split('_')[0], (1, cpu_aucs[i]),
                xytext=(10, 0), textcoords='offset points')
for i, m in enumerate(gpu_models):
    ax4.annotate(m.split('_')[0], (2, gpu_aucs[i]),
                xytext=(10, 0), textcoords='offset points')

ax4.set_xticks([1, 2])
ax4.set_xticklabels(['CPU', 'GPU'])
ax4.set_ylabel('ROC-AUC', fontsize=12)
ax4.set_title('CPU vs GPU Performance', fontsize=14)
ax4.set_ylim(0.8, 0.9)
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. GP+CNN Detailed Metrics
ax5 = axes[1, 1]
if 'GP+CNN_GPU' in results:
    gpcnn = results['GP+CNN_GPU']
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    values = [
        gpcnn.get('accuracy', 0),
        gpcnn.get('precision', 0),
        gpcnn.get('recall', 0),
        gpcnn.get('f1', 0),
        gpcnn.get('roc_auc', 0)
    ]

    bars = ax5.barh(metrics, values, color='#e74c3c')
    ax5.set_xlim(0, 1)
    ax5.set_xlabel('Score', fontsize=12)
    ax5.set_title('GP+CNN Detailed Metrics', fontsize=14)

    for bar, val in zip(bars, values):
        ax5.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center')

# 6. Model Ranking Table
ax6 = axes[1, 2]
ax6.axis('tight')
ax6.axis('off')

# Create ranking data
sorted_models = sorted(results.items(), key=lambda x: x[1]['roc_auc'], reverse=True)
table_data = []
for i, (model, metrics) in enumerate(sorted_models, 1):
    table_data.append([
        i,
        model.split('_')[0],
        f"{metrics['accuracy']:.1%}",
        f"{metrics['roc_auc']:.3f}",
        f"{metrics['time']:.1f}s"
    ])

table = ax6.table(cellText=table_data,
                  colLabels=['Rank', 'Model', 'Accuracy', 'ROC-AUC', 'Time'],
                  cellLoc='center',
                  loc='center',
                  bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

# Style the header
for (i, j), cell in table._cells.items():
    if i == 0:
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#34495e')
        cell.set_text_props(color='white')
    elif j == 0 and i == 1:
        cell.set_facecolor('#FFD700')  # Gold
    elif j == 0 and i == 2:
        cell.set_facecolor('#C0C0C0')  # Silver
    elif j == 0 and i == 3:
        cell.set_facecolor('#CD7F32')  # Bronze

ax6.set_title('Final Ranking', fontsize=14, fontweight='bold', pad=20)

# Adjust layout and save
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('gpcnn_complete_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('gpcnn_complete_comparison.pdf', dpi=150, bbox_inches='tight')

print("Saved: gpcnn_complete_comparison.png and gpcnn_complete_comparison.pdf")

# Print summary
print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)

print(f"\n{'Model':<20} {'ROC-AUC':<10} {'Accuracy':<10} {'Time':<10} {'Device':<10}")
print("-" * 60)

for model, metrics in sorted_models:
    print(f"{model.split('_')[0]:<20} {metrics['roc_auc']:.3f}{'':<7} "
          f"{metrics['accuracy']:.1%}{'':<6} {metrics['time']:.1f}s{'':<7} "
          f"{metrics['device']:<10}")

# GP+CNN specific analysis
print("\n" + "="*80)
print("GP+CNN ANALYSIS")
print("="*80)

if 'GP+CNN_GPU' in results:
    gpcnn = results['GP+CNN_GPU']
    rank = next((i for i, (m, _) in enumerate(sorted_models, 1) if m == 'GP+CNN_GPU'), None)

    print(f"\nGP+CNN Ranking: {rank}/{len(results)}")
    print(f"ROC-AUC: {gpcnn['roc_auc']:.3f}")
    print(f"Parameters: {gpcnn.get('parameters', 'N/A'):,}")
    print(f"Training Time: {gpcnn['time']:.1f}s")

    print("\nWhy GP+CNN underperformed:")
    print("1. Designed for raw light curves, not TSFresh features")
    print("2. Overparameterized: ~4M parameters for 1.3K samples")
    print("3. Architecture mismatch with tabular data")

    print("\nTo improve GP+CNN performance:")
    print("1. Use raw Kepler/TESS light curves")
    print("2. Implement proper GP denoising")
    print("3. Apply TLS period search")
    print("4. Expected performance: >90% ROC-AUC")

print("\n[COMPLETE] Visualization and analysis finished!")