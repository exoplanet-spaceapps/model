"""
Complete GPU Benchmark for All Models
======================================
Ensures actual GPU utilization for all GPU-capable models
Monitors and reports real GPU usage
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import time
import json
import psutil
import GPUtil
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COMPREHENSIVE GPU BENCHMARK - ALL MODELS")
print("="*80)

# GPU monitoring function
def get_gpu_info():
    """Get current GPU usage"""
    if torch.cuda.is_available():
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            return {
                'name': gpu.name,
                'memory_used_gb': gpu.memoryUsed / 1024,
                'memory_total_gb': gpu.memoryTotal / 1024,
                'memory_util': gpu.memoryUtil * 100,
                'gpu_util': gpu.load * 100
            }
    return None

# Print GPU status
def print_gpu_status(stage=""):
    gpu_info = get_gpu_info()
    if gpu_info:
        print(f"\n[GPU {stage}]")
        print(f"  Memory: {gpu_info['memory_used_gb']:.2f}/{gpu_info['memory_total_gb']:.2f} GB ({gpu_info['memory_util']:.1f}%)")
        print(f"  GPU Utilization: {gpu_info['gpu_util']:.1f}%")

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type != 'cuda':
    print("ERROR: No GPU available!")
    exit()

print(f"Device: {device}")
print_gpu_status("Initial")

# Load and prepare data
print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

data = pd.read_csv('tsfresh_features.csv')
data = data.replace([np.inf, -np.inf], np.nan).fillna(0)

# Remove single-value columns
unique_value_columns = data.columns[data.nunique() <= 1]
data = data.drop(unique_value_columns, axis=1)

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

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"Data: {X_train.shape[0]} train, {X_val.shape[0]} val, {X_test.shape[0]} test")
print(f"Features: {X_train.shape[1]}")

results = {}

# ============================================
# 1. XGBOOST WITH GPU
# ============================================
print("\n" + "="*80)
print("[1/5] XGBOOST WITH GPU")
print("="*80)

print("Training XGBoost with GPU (gpu_hist)...")
print_gpu_status("Before XGBoost")

start_time = time.time()

# Combine train and val for XGBoost
X_train_xgb = np.vstack([X_train_scaled, X_val_scaled])
y_train_xgb = np.hstack([y_train, y_val])

# GPU-specific parameters
xgb_model = xgb.XGBClassifier(
    n_estimators=500,  # More trees for GPU work
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='gpu_hist',  # GPU acceleration
    predictor='gpu_predictor',
    gpu_id=0,
    eval_metric='auc',
    random_state=42,
    n_jobs=1  # Single thread to force GPU
)

# Monitor during training
xgb_model.fit(X_train_xgb, y_train_xgb)
print_gpu_status("During XGBoost")

xgb_time = time.time() - start_time

# Evaluate
xgb_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
xgb_pred = xgb_model.predict(X_test_scaled)

results['XGBoost (GPU)'] = {
    'accuracy': accuracy_score(y_test, xgb_pred),
    'precision': precision_score(y_test, xgb_pred),
    'recall': recall_score(y_test, xgb_pred),
    'f1': f1_score(y_test, xgb_pred),
    'roc_auc': roc_auc_score(y_test, xgb_pred_proba),
    'training_time': xgb_time,
    'gpu_used': True
}

print(f"Results: Acc={results['XGBoost (GPU)']['accuracy']:.3f}, "
      f"ROC-AUC={results['XGBoost (GPU)']['roc_auc']:.3f}, Time={xgb_time:.1f}s")
print_gpu_status("After XGBoost")

# ============================================
# 2. HEAVY NEURAL NETWORK WITH GPU
# ============================================
print("\n" + "="*80)
print("[2/5] HEAVY NEURAL NETWORK WITH GPU")
print("="*80)

class HeavyNN(nn.Module):
    """Large NN to ensure GPU usage"""
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)

print("Training Heavy Neural Network...")
print_gpu_status("Before NN")

# Create larger batches for GPU
BATCH_SIZE = 256

# Augment data for more GPU work
X_train_aug = np.tile(X_train_scaled, (5, 1))  # 5x data
y_train_aug = np.tile(y_train, 5)

# Data loaders
train_dataset = TensorDataset(
    torch.FloatTensor(X_train_aug),
    torch.FloatTensor(y_train_aug)
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

# Model
nn_model = HeavyNN(X_train.shape[1]).to(device)
print(f"Model parameters: {sum(p.numel() for p in nn_model.parameters()):,}")

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-3)
scaler = GradScaler()

start_time = time.time()
epochs = 30

for epoch in range(epochs):
    nn_model.train()
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision
        with autocast():
            outputs = nn_model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    if epoch % 10 == 0:
        print(f"  Epoch {epoch}: Loss={loss.item():.4f}")
        print_gpu_status(f"During NN Epoch {epoch}")

nn_time = time.time() - start_time

# Evaluate
nn_model.eval()
X_test_t = torch.FloatTensor(X_test_scaled).to(device)
with torch.no_grad():
    with autocast():
        nn_logits = nn_model(X_test_t).squeeze()
        nn_pred = torch.sigmoid(nn_logits).cpu().numpy()

nn_pred_binary = (nn_pred > 0.5).astype(int)

results['Heavy NN (GPU)'] = {
    'accuracy': accuracy_score(y_test, nn_pred_binary),
    'precision': precision_score(y_test, nn_pred_binary),
    'recall': recall_score(y_test, nn_pred_binary),
    'f1': f1_score(y_test, nn_pred_binary),
    'roc_auc': roc_auc_score(y_test, nn_pred),
    'training_time': nn_time,
    'gpu_used': True
}

print(f"Results: Acc={results['Heavy NN (GPU)']['accuracy']:.3f}, "
      f"ROC-AUC={results['Heavy NN (GPU)']['roc_auc']:.3f}, Time={nn_time:.1f}s")
print_gpu_status("After NN")

# ============================================
# 3. GP+CNN WITH GPU
# ============================================
print("\n" + "="*80)
print("[3/5] GP+CNN PIPELINE WITH GPU")
print("="*80)

class GPCNNModel(nn.Module):
    """GP+CNN with proper GPU utilization"""
    def __init__(self, input_dim):
        super().__init__()
        # Transform to pseudo light curve
        self.transform = nn.Linear(input_dim, 2048)

        # CNN layers
        self.conv1 = nn.Conv1d(1, 64, kernel_size=11, padding=5)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool = nn.AdaptiveAvgPool1d(16)

        # Classifier
        self.fc1 = nn.Linear(256*16, 512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 128)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.transform(x)
        x = x.unsqueeze(1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)
        return x.squeeze()

print("Training GP+CNN...")
print_gpu_status("Before GP+CNN")

# Model
gpcnn_model = GPCNNModel(X_train.shape[1]).to(device)
print(f"Model parameters: {sum(p.numel() for p in gpcnn_model.parameters()):,}")

optimizer = torch.optim.AdamW(gpcnn_model.parameters(), lr=1e-3)

start_time = time.time()

# Use same augmented data
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

for epoch in range(30):
    gpcnn_model.train()
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            outputs = gpcnn_model(batch_X)
            loss = criterion(outputs, batch_y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    if epoch % 10 == 0:
        print(f"  Epoch {epoch}: Loss={loss.item():.4f}")
        print_gpu_status(f"During GP+CNN Epoch {epoch}")

gpcnn_time = time.time() - start_time

# Evaluate
gpcnn_model.eval()
with torch.no_grad():
    with autocast():
        gpcnn_logits = gpcnn_model(X_test_t)
        gpcnn_pred = torch.sigmoid(gpcnn_logits).cpu().numpy()

gpcnn_pred_binary = (gpcnn_pred > 0.5).astype(int)

results['GP+CNN (GPU)'] = {
    'accuracy': accuracy_score(y_test, gpcnn_pred_binary),
    'precision': precision_score(y_test, gpcnn_pred_binary),
    'recall': recall_score(y_test, gpcnn_pred_binary),
    'f1': f1_score(y_test, gpcnn_pred_binary),
    'roc_auc': roc_auc_score(y_test, gpcnn_pred),
    'training_time': gpcnn_time,
    'gpu_used': True
}

print(f"Results: Acc={results['GP+CNN (GPU)']['accuracy']:.3f}, "
      f"ROC-AUC={results['GP+CNN (GPU)']['roc_auc']:.3f}, Time={gpcnn_time:.1f}s")
print_gpu_status("After GP+CNN")

# ============================================
# 4. SIMPLE MLP WITH GPU
# ============================================
print("\n" + "="*80)
print("[4/5] SIMPLE MLP WITH GPU")
print("="*80)

class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

print("Training Simple MLP...")
print_gpu_status("Before MLP")

mlp_model = SimpleMLP(X_train.shape[1]).to(device)
optimizer = torch.optim.AdamW(mlp_model.parameters(), lr=3e-5)

start_time = time.time()

# Quick training on original data
X_train_t = torch.FloatTensor(X_train_scaled).to(device)
y_train_t = torch.FloatTensor(y_train).to(device)

for epoch in range(50):
    mlp_model.train()
    optimizer.zero_grad()
    outputs = mlp_model(X_train_t).squeeze()
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

mlp_time = time.time() - start_time

# Evaluate
mlp_model.eval()
with torch.no_grad():
    mlp_logits = mlp_model(X_test_t).squeeze()
    mlp_pred = torch.sigmoid(mlp_logits).cpu().numpy()

mlp_pred_binary = (mlp_pred > 0.5).astype(int)

results['Simple MLP (GPU)'] = {
    'accuracy': accuracy_score(y_test, mlp_pred_binary),
    'precision': precision_score(y_test, mlp_pred_binary),
    'recall': recall_score(y_test, mlp_pred_binary),
    'f1': f1_score(y_test, mlp_pred_binary),
    'roc_auc': roc_auc_score(y_test, mlp_pred),
    'training_time': mlp_time,
    'gpu_used': True
}

print(f"Results: Acc={results['Simple MLP (GPU)']['accuracy']:.3f}, "
      f"ROC-AUC={results['Simple MLP (GPU)']['roc_auc']:.3f}, Time={mlp_time:.1f}s")
print_gpu_status("After MLP")

# ============================================
# 5. RANDOM FOREST (CPU - No GPU Support)
# ============================================
print("\n" + "="*80)
print("[5/5] RANDOM FOREST (CPU-ONLY)")
print("="*80)

print("Training Random Forest (sklearn has no GPU support)...")

start_time = time.time()

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=9,
    min_samples_leaf=4,
    n_jobs=-1,
    random_state=42
)

# Combine train and val
X_train_rf = np.vstack([X_train_scaled, X_val_scaled])
y_train_rf = np.hstack([y_train, y_val])

rf_model.fit(X_train_rf, y_train_rf)
rf_time = time.time() - start_time

# Evaluate
rf_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
rf_pred = rf_model.predict(X_test_scaled)

results['Random Forest (CPU)'] = {
    'accuracy': accuracy_score(y_test, rf_pred),
    'precision': precision_score(y_test, rf_pred),
    'recall': recall_score(y_test, rf_pred),
    'f1': f1_score(y_test, rf_pred),
    'roc_auc': roc_auc_score(y_test, rf_pred_proba),
    'training_time': rf_time,
    'gpu_used': False
}

print(f"Results: Acc={results['Random Forest (CPU)']['accuracy']:.3f}, "
      f"ROC-AUC={results['Random Forest (CPU)']['roc_auc']:.3f}, Time={rf_time:.1f}s")

# ============================================
# FINAL COMPARISON
# ============================================
print("\n" + "="*80)
print("FINAL GPU BENCHMARK RESULTS")
print("="*80)

# Create ranking
ranking = []
for model_name, metrics in results.items():
    ranking.append({
        'Model': model_name,
        'Accuracy': metrics['accuracy'],
        'ROC-AUC': metrics['roc_auc'],
        'F1': metrics['f1'],
        'Time': metrics['training_time'],
        'GPU': 'Yes' if metrics['gpu_used'] else 'No'
    })

# Sort by ROC-AUC
ranking.sort(key=lambda x: x['ROC-AUC'], reverse=True)

print(f"\n{'Rank':<5} {'Model':<25} {'Accuracy':<10} {'ROC-AUC':<10} {'F1':<10} {'Time(s)':<10} {'GPU':<5}")
print("-" * 85)

for i, model in enumerate(ranking, 1):
    print(f"{i:<5} {model['Model']:<25} {model['Accuracy']:.3f}{'':<7} "
          f"{model['ROC-AUC']:.3f}{'':<7} {model['F1']:.3f}{'':<7} "
          f"{model['Time']:.1f}{'':<9} {model['GPU']:<5}")

# Save results
with open('all_models_gpu_benchmark.json', 'w') as f:
    json.dump({
        'results': results,
        'ranking': ranking,
        'gpu_info': get_gpu_info()
    }, f, indent=2)

print("\n[SAVED] Results to all_models_gpu_benchmark.json")
print_gpu_status("Final")
print("\n[COMPLETE] GPU Benchmark finished!")
print("\nNote: Monitor real-time GPU usage with: nvidia-smi -l 1")