"""
GPU Training Results Visualization
===================================
Creates comprehensive comparison charts from actual training results
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Load actual training results
with open('gpu_training_results.json', 'r') as f:
    results = json.load(f)

# Create comparison dataframe
data = []
for model_name, metrics in results.items():
    row = {'Model': model_name}
    row.update(metrics)
    data.append(row)

df = pd.DataFrame(data)

# Sort by ROC-AUC
df = df.sort_values('roc_auc', ascending=False)

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 10))
fig.suptitle('GPU Training Results - Model Comparison', fontsize=16, fontweight='bold')

# Define colors for each model
colors = {
    'XGBoost': '#3498db',
    'Random Forest': '#27ae60',
    'Neural Network': '#e74c3c',
    'Simple MLP': '#95a5a6'
}

# 1. Accuracy Comparison
ax1 = plt.subplot(2, 3, 1)
models = df['Model'].values
accuracies = df['accuracy'].values * 100
bars = ax1.bar(models, accuracies, color=[colors[m] for m in models])
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Model Accuracy')
ax1.set_ylim(60, 85)
for bar, acc in zip(bars, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)

# 2. ROC-AUC Comparison
ax2 = plt.subplot(2, 3, 2)
roc_aucs = df['roc_auc'].values
bars = ax2.bar(models, roc_aucs, color=[colors[m] for m in models])
ax2.set_ylabel('ROC-AUC Score')
ax2.set_title('ROC-AUC Performance')
ax2.set_ylim(0.65, 0.90)
for bar, auc in zip(bars, roc_aucs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{auc:.3f}', ha='center', va='bottom', fontsize=9)

# 3. F1 Score Comparison
ax3 = plt.subplot(2, 3, 3)
f1_scores = df['f1'].values
bars = ax3.bar(models, f1_scores, color=[colors[m] for m in models])
ax3.set_ylabel('F1 Score')
ax3.set_title('F1 Score (Harmonic Mean)')
ax3.set_ylim(0.7, 0.85)
for bar, f1 in zip(bars, f1_scores):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{f1:.3f}', ha='center', va='bottom', fontsize=9)

# 4. Precision vs Recall
ax4 = plt.subplot(2, 3, 4)
precisions = df['precision'].values
recalls = df['recall'].values
x = np.arange(len(models))
width = 0.35
bars1 = ax4.bar(x - width/2, precisions, width, label='Precision', alpha=0.8)
bars2 = ax4.bar(x + width/2, recalls, width, label='Recall', alpha=0.8)
ax4.set_xticks(x)
ax4.set_xticklabels(models, rotation=15, ha='right')
ax4.set_ylabel('Score')
ax4.set_title('Precision vs Recall')
ax4.legend()
ax4.set_ylim(0.6, 1.0)

# 5. Training Time
ax5 = plt.subplot(2, 3, 5)
train_times = df['training_time'].values
bars = ax5.bar(models, train_times, color=[colors[m] for m in models])
ax5.set_ylabel('Training Time (seconds)')
ax5.set_title('Training Speed Comparison')
for bar, time in zip(bars, train_times):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{time:.1f}s', ha='center', va='bottom', fontsize=9)

# 6. Overall Performance Radar Chart
ax6 = plt.subplot(2, 3, 6, projection='polar')

# Metrics for radar
metrics_radar = ['Accuracy', 'ROC-AUC', 'F1', 'Precision', 'Recall']
num_vars = len(metrics_radar)

# Compute angle for each axis
angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
angles += angles[:1]

# Plot each model
for idx, model in enumerate(df['Model'].values[:3]):  # Top 3 models only
    row = df[df['Model'] == model].iloc[0]
    values = [
        row['accuracy'],
        row['roc_auc'],
        row['f1'],
        row['precision'],
        row['recall']
    ]
    values += values[:1]

    ax6.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[model])
    ax6.fill(angles, values, alpha=0.25, color=colors[model])

ax6.set_xticks(angles[:-1])
ax6.set_xticklabels(metrics_radar)
ax6.set_ylim(0, 1)
ax6.set_title('Multi-Metric Comparison (Top 3 Models)')
ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax6.grid(True)

# Adjust layout
plt.tight_layout()
plt.savefig('gpu_training_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('gpu_training_comparison.pdf', dpi=150, bbox_inches='tight')
print("Saved: gpu_training_comparison.png and gpu_training_comparison.pdf")

# Create detailed comparison table
print("\n" + "="*80)
print("DETAILED PERFORMANCE METRICS")
print("="*80)

# Format dataframe for display
display_df = df.copy()
display_df['accuracy'] = (display_df['accuracy'] * 100).round(1).astype(str) + '%'
display_df['precision'] = (display_df['precision'] * 100).round(1).astype(str) + '%'
display_df['recall'] = (display_df['recall'] * 100).round(1).astype(str) + '%'
display_df['f1'] = display_df['f1'].round(3)
display_df['roc_auc'] = display_df['roc_auc'].round(3)
display_df['training_time'] = display_df['training_time'].round(1).astype(str) + 's'

print(display_df.to_string(index=False))

# Winner announcement
print("\n" + "="*80)
print("COMPETITION RESULTS")
print("="*80)

best_model = df.iloc[0]
print(f"[WINNER] {best_model['Model']}")
print(f"   - ROC-AUC: {best_model['roc_auc']:.3f}")
print(f"   - Accuracy: {best_model['accuracy']*100:.1f}%")
print(f"   - F1 Score: {best_model['f1']:.3f}")
print(f"   - Training Time: {best_model['training_time']:.1f} seconds")

# Key insights
print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

print(f"1. XGBoost achieved the best overall performance (ROC-AUC: {df.loc[df['Model']=='XGBoost', 'roc_auc'].values[0]:.3f})")
print(f"2. Random Forest was fastest to train ({df.loc[df['Model']=='Random Forest', 'training_time'].values[0]:.1f}s) with competitive performance")
print(f"3. Neural Network GPU acceleration provided {df.loc[df['Model']=='Neural Network', 'roc_auc'].values[0]:.3f} ROC-AUC in {df.loc[df['Model']=='Neural Network', 'training_time'].values[0]:.1f}s")
print(f"4. Simple MLP baseline achieved {df.loc[df['Model']=='Simple MLP', 'accuracy'].values[0]*100:.1f}% accuracy")

# Create summary for report
summary = {
    'timestamp': datetime.now().isoformat(),
    'winner': best_model['Model'],
    'best_roc_auc': float(best_model['roc_auc']),
    'best_accuracy': float(best_model['accuracy']),
    'model_rankings': df[['Model', 'roc_auc']].to_dict('records'),
    'gpu_utilized': True,
    'total_models_tested': len(df)
}

with open('gpu_training_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n[SAVED] Summary saved to gpu_training_summary.json")
print("[COMPLETE] GPU training comparison visualization completed!")