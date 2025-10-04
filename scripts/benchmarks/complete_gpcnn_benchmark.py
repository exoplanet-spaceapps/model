"""
Complete Benchmark with GP+CNN Included
========================================
Ensures GP+CNN runs and compares with all other models
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import multiprocessing
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COMPLETE BENCHMARK INCLUDING GP+CNN")
print("="*80)

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # GPU Optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("[OK] GPU optimizations enabled")

# CPU Configuration
physical_cores = multiprocessing.cpu_count() // 2
print(f"\nCPU Cores: {physical_cores} physical, {multiprocessing.cpu_count()} logical")

# CPU Optimizations
os.environ['MKL_NUM_THREADS'] = str(physical_cores)
os.environ['OMP_NUM_THREADS'] = str(physical_cores)
print("[OK] CPU threading optimized")

# ============================================
# LOAD DATA
# ============================================
print("\n[LOADING DATA]")
data = pd.read_csv('tsfresh_features.csv')
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

print(f"Data shape: {X_train.shape}")
print(f"Class balance: {y_train.mean():.2%} positive")

# Combine train and val for tree models
X_train_combined = np.vstack([X_train, X_val])
y_train_combined = np.hstack([y_train, y_val])

results = {}

# ============================================
# 1. GP+CNN MODEL (PRIORITY)
# ============================================
print("\n" + "="*80)
print("1. GP+CNN PIPELINE")
print("="*80)

class GPCNN(nn.Module):
    """
    GP+CNN Pipeline Architecture
    - Simulates GP denoising through learned transformations
    - CNN for pattern recognition
    - Optimized for GPU
    """
    def __init__(self, input_dim):
        super().__init__()

        # Phase 1: Feature to pseudo light curve (simulates GP preprocessing)
        self.gp_simulator = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.GELU()
        )

        # Phase 2: CNN for pattern extraction
        self.cnn_layers = nn.ModuleList([
            # First CNN block
            nn.Conv1d(1, 32, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),

            # Second CNN block
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),

            # Third CNN block
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(32)
        ])

        # Phase 3: Classification
        self.classifier = nn.Sequential(
            nn.Linear(128 * 32, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # GP simulation
        x = self.gp_simulator(x)

        # Reshape for CNN (add channel dimension)
        x = x.unsqueeze(1)  # [batch, 1, 2048]

        # CNN processing
        for i, layer in enumerate(self.cnn_layers):
            if isinstance(layer, (nn.Conv1d, nn.MaxPool1d, nn.AdaptiveAvgPool1d)):
                x = layer(x)
            elif isinstance(layer, nn.BatchNorm1d):
                x = F.gelu(layer(x))

        # Flatten and classify
        x = x.flatten(1)
        x = self.classifier(x)

        return x.squeeze()

# Train GP+CNN
if device.type == 'cuda':
    print("Training GP+CNN with GPU optimizations...")

    # Create data loader
    batch_size = 64  # Smaller batch size for stability
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0
    )

    # Initialize model
    gpcnn_model = GPCNN(X_train.shape[1]).to(device)
    print(f"Model parameters: {sum(p.numel() for p in gpcnn_model.parameters()):,}")

    # Optimizer and loss
    optimizer = torch.optim.AdamW(gpcnn_model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # Mixed precision
    scaler = GradScaler()

    # Training
    start_time = time.time()
    epochs = 30
    best_loss = float('inf')

    for epoch in range(epochs):
        gpcnn_model.train()
        epoch_loss = 0
        batch_count = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Mixed precision training
            with autocast():
                outputs = gpcnn_model(batch_X)
                loss = criterion(outputs, batch_y)

            # Check for NaN
            if not torch.isnan(loss):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
                batch_count += 1

        scheduler.step()

        if batch_count > 0:
            avg_loss = epoch_loss / batch_count
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = gpcnn_model.state_dict()

            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Loss={avg_loss:.4f}")

    gpcnn_time = time.time() - start_time

    # Load best model and evaluate
    gpcnn_model.load_state_dict(best_state)
    gpcnn_model.eval()

    X_test_t = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        with autocast():
            test_logits = gpcnn_model(X_test_t)
            test_pred = torch.sigmoid(test_logits).cpu().numpy()

    # Handle NaN predictions
    test_pred = np.nan_to_num(test_pred, nan=0.5)
    test_pred_binary = (test_pred > 0.5).astype(int)

    results['GP+CNN_GPU'] = {
        'accuracy': accuracy_score(y_test, test_pred_binary),
        'precision': precision_score(y_test, test_pred_binary),
        'recall': recall_score(y_test, test_pred_binary),
        'f1': f1_score(y_test, test_pred_binary),
        'roc_auc': roc_auc_score(y_test, test_pred),
        'time': gpcnn_time,
        'device': 'GPU',
        'parameters': sum(p.numel() for p in gpcnn_model.parameters())
    }

    print(f"\nGP+CNN Results:")
    print(f"  Accuracy: {results['GP+CNN_GPU']['accuracy']:.3f}")
    print(f"  Precision: {results['GP+CNN_GPU']['precision']:.3f}")
    print(f"  Recall: {results['GP+CNN_GPU']['recall']:.3f}")
    print(f"  F1 Score: {results['GP+CNN_GPU']['f1']:.3f}")
    print(f"  ROC-AUC: {results['GP+CNN_GPU']['roc_auc']:.3f}")
    print(f"  Training Time: {gpcnn_time:.1f}s")

# ============================================
# 2. NEURAL NETWORK (SIMPLE)
# ============================================
print("\n" + "="*80)
print("2. NEURAL NETWORK")
print("="*80)

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        return self.fc4(x).squeeze()

if device.type == 'cuda':
    print("Training Neural Network...")

    nn_model = SimpleNN(X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3)

    start_time = time.time()

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)

    for epoch in range(50):
        nn_model.train()
        optimizer.zero_grad()
        outputs = nn_model(X_train_t)
        loss = criterion(outputs, y_train_t)

        if not torch.isnan(loss):
            loss.backward()
            optimizer.step()

    nn_time = time.time() - start_time

    nn_model.eval()
    with torch.no_grad():
        test_logits = nn_model(X_test_t)
        test_pred = torch.sigmoid(test_logits).cpu().numpy()

    test_pred = np.nan_to_num(test_pred, nan=0.5)
    test_pred_binary = (test_pred > 0.5).astype(int)

    results['NeuralNet_GPU'] = {
        'accuracy': accuracy_score(y_test, test_pred_binary),
        'roc_auc': roc_auc_score(y_test, test_pred),
        'time': nn_time,
        'device': 'GPU'
    }

    print(f"Results: Acc={results['NeuralNet_GPU']['accuracy']:.3f}, "
          f"AUC={results['NeuralNet_GPU']['roc_auc']:.3f}, "
          f"Time={nn_time:.1f}s")

# ============================================
# 3. XGBOOST
# ============================================
print("\n" + "="*80)
print("3. XGBOOST")
print("="*80)

# XGBoost GPU
if device.type == 'cuda':
    print("Training XGBoost with GPU...")
    start = time.time()

    xgb_gpu = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        tree_method='gpu_hist',
        predictor='gpu_predictor',
        gpu_id=0,
        random_state=42
    )

    xgb_gpu.fit(X_train_combined, y_train_combined)
    xgb_gpu_time = time.time() - start

    xgb_gpu_pred = xgb_gpu.predict(X_test)
    xgb_gpu_proba = xgb_gpu.predict_proba(X_test)[:, 1]

    results['XGBoost_GPU'] = {
        'accuracy': accuracy_score(y_test, xgb_gpu_pred),
        'roc_auc': roc_auc_score(y_test, xgb_gpu_proba),
        'time': xgb_gpu_time,
        'device': 'GPU'
    }

    print(f"Results: Acc={results['XGBoost_GPU']['accuracy']:.3f}, "
          f"AUC={results['XGBoost_GPU']['roc_auc']:.3f}, "
          f"Time={xgb_gpu_time:.1f}s")

# XGBoost CPU
print("\nTraining XGBoost with CPU...")
start = time.time()

xgb_cpu = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    tree_method='hist',
    n_jobs=physical_cores - 1,
    random_state=42
)

xgb_cpu.fit(X_train_combined, y_train_combined)
xgb_cpu_time = time.time() - start

xgb_cpu_pred = xgb_cpu.predict(X_test)
xgb_cpu_proba = xgb_cpu.predict_proba(X_test)[:, 1]

results['XGBoost_CPU'] = {
    'accuracy': accuracy_score(y_test, xgb_cpu_pred),
    'roc_auc': roc_auc_score(y_test, xgb_cpu_proba),
    'time': xgb_cpu_time,
    'device': 'CPU'
}

print(f"Results: Acc={results['XGBoost_CPU']['accuracy']:.3f}, "
      f"AUC={results['XGBoost_CPU']['roc_auc']:.3f}, "
      f"Time={xgb_cpu_time:.1f}s")

# ============================================
# 4. RANDOM FOREST
# ============================================
print("\n" + "="*80)
print("4. RANDOM FOREST")
print("="*80)

print("Training Random Forest...")
start = time.time()

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    n_jobs=physical_cores - 1,
    random_state=42
)

rf.fit(X_train_combined, y_train_combined)
rf_time = time.time() - start

rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

results['RandomForest_CPU'] = {
    'accuracy': accuracy_score(y_test, rf_pred),
    'roc_auc': roc_auc_score(y_test, rf_proba),
    'time': rf_time,
    'device': 'CPU'
}

print(f"Results: Acc={results['RandomForest_CPU']['accuracy']:.3f}, "
      f"AUC={results['RandomForest_CPU']['roc_auc']:.3f}, "
      f"Time={rf_time:.1f}s")

# ============================================
# FINAL COMPARISON
# ============================================
print("\n" + "="*80)
print("FINAL COMPARISON - ALL MODELS INCLUDING GP+CNN")
print("="*80)

# Sort by ROC-AUC
sorted_results = sorted(results.items(), key=lambda x: x[1]['roc_auc'], reverse=True)

print(f"\n{'Rank':<5} {'Model':<20} {'Device':<8} {'Accuracy':<10} {'ROC-AUC':<10} {'Time (s)':<10}")
print("-" * 75)

for i, (name, metrics) in enumerate(sorted_results, 1):
    print(f"{i:<5} {name:<20} {metrics['device']:<8} "
          f"{metrics['accuracy']:.3f}{'':<7} "
          f"{metrics['roc_auc']:.3f}{'':<7} "
          f"{metrics['time']:.1f}")

# Additional metrics for GP+CNN
if 'GP+CNN_GPU' in results:
    print("\n" + "="*80)
    print("GP+CNN DETAILED METRICS")
    print("="*80)
    gpcnn_metrics = results['GP+CNN_GPU']
    print(f"Accuracy: {gpcnn_metrics['accuracy']:.3f}")
    print(f"Precision: {gpcnn_metrics['precision']:.3f}")
    print(f"Recall: {gpcnn_metrics['recall']:.3f}")
    print(f"F1 Score: {gpcnn_metrics['f1']:.3f}")
    print(f"ROC-AUC: {gpcnn_metrics['roc_auc']:.3f}")
    print(f"Parameters: {gpcnn_metrics['parameters']:,}")
    print(f"Training Time: {gpcnn_metrics['time']:.1f}s")

# Performance analysis
print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

best_model = sorted_results[0]
print(f"\nBest Model: {best_model[0]}")
print(f"  ROC-AUC: {best_model[1]['roc_auc']:.3f}")
print(f"  Device: {best_model[1]['device']}")

if 'GP+CNN_GPU' in results:
    gpcnn_rank = next((i for i, (name, _) in enumerate(sorted_results, 1) if name == 'GP+CNN_GPU'), None)
    print(f"\nGP+CNN Performance:")
    print(f"  Rank: {gpcnn_rank}/{len(results)}")
    print(f"  ROC-AUC: {results['GP+CNN_GPU']['roc_auc']:.3f}")
    print(f"  Note: GP+CNN is designed for raw light curves, not TSFresh features")

# Save results
with open('complete_gpcnn_benchmark_results.json', 'w') as f:
    json.dump({
        'results': results,
        'system': {
            'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'None',
            'cpu_cores': physical_cores
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }, f, indent=2)

print("\n[SAVED] Results to complete_gpcnn_benchmark_results.json")
print("[COMPLETE] All models including GP+CNN benchmarked!")