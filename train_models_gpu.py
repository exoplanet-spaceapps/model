
"""
GPU-Accelerated Model Training Script
Run this to train all models with GPU acceleration
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import time
import sys
sys.path.append('.')

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else
                     'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
print("Loading data...")
data = pd.read_csv('tsfresh_features.csv')

# Prepare features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

results = {}

# 1. Train Neural Network
print("\n[1/3] Training Neural Network on GPU...")
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Convert to tensors
X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.FloatTensor(y_train).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)

# Create model
nn_model = SimpleNN(X_train.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = optim.AdamW(nn_model.parameters(), lr=3e-5)

# Train
start_time = time.time()
epochs = 50
batch_size = 64

for epoch in range(epochs):
    nn_model.train()
    for i in range(0, len(X_train_t), batch_size):
        batch_X = X_train_t[i:i+batch_size]
        batch_y = y_train_t[i:i+batch_size]

        optimizer.zero_grad()
        outputs = nn_model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

nn_time = time.time() - start_time

# Evaluate
nn_model.eval()
with torch.no_grad():
    nn_pred = nn_model(X_test_t).squeeze().cpu().numpy()
    nn_acc = accuracy_score(y_test, nn_pred > 0.5)
    nn_auc = roc_auc_score(y_test, nn_pred)

results['Neural Network'] = {
    'accuracy': nn_acc,
    'roc_auc': nn_auc,
    'training_time': nn_time
}

print(f"  NN Results: Accuracy={nn_acc:.3f}, ROC-AUC={nn_auc:.3f}, Time={nn_time:.1f}s")

# 2. Train XGBoost (with GPU if available)
print("\n[2/3] Training XGBoost...")
try:
    import xgboost as xgb

    start_time = time.time()
    if device.type == 'cuda':
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            tree_method='gpu_hist',
            predictor='gpu_predictor',
            gpu_id=0
        )
    else:
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5
        )

    xgb_model.fit(X_train, y_train)
    xgb_time = time.time() - start_time

    xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_acc = xgb_model.score(X_test, y_test)
    xgb_auc = roc_auc_score(y_test, xgb_pred_proba)

    results['XGBoost'] = {
        'accuracy': xgb_acc,
        'roc_auc': xgb_auc,
        'training_time': xgb_time
    }

    print(f"  XGB Results: Accuracy={xgb_acc:.3f}, ROC-AUC={xgb_auc:.3f}, Time={xgb_time:.1f}s")

except ImportError:
    print("  XGBoost not installed. Skipping...")

# 3. Train Random Forest (CPU only)
print("\n[3/3] Training Random Forest...")
start_time = time.time()
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=9,
    n_jobs=-1  # Use all CPU cores
)

rf_model.fit(X_train, y_train)
rf_time = time.time() - start_time

rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
rf_acc = rf_model.score(X_test, y_test)
rf_auc = roc_auc_score(y_test, rf_pred_proba)

results['Random Forest'] = {
    'accuracy': rf_acc,
    'roc_auc': rf_auc,
    'training_time': rf_time
}

print(f"  RF Results: Accuracy={rf_acc:.3f}, ROC-AUC={rf_auc:.3f}, Time={rf_time:.1f}s")

# Summary
print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"  Training Time: {metrics['training_time']:.1f} seconds")

# Save results
import json
with open('gpu_training_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to gpu_training_results.json")
