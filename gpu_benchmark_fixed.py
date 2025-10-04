"""
Fixed GPU Benchmark - Monitors Real GPU Usage
==============================================
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import time
import json
import subprocess
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("GPU BENCHMARK WITH REAL MONITORING")
print("="*80)

# Function to get GPU stats using nvidia-smi
def get_gpu_stats():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                               '--format=csv,noheader,nounits'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            stats = result.stdout.strip().split(', ')
            return {
                'gpu_util': float(stats[0]),
                'mem_used': float(stats[1]) / 1024,  # Convert to GB
                'mem_total': float(stats[2]) / 1024
            }
    except:
        pass
    return None

def print_gpu_stats(label=""):
    stats = get_gpu_stats()
    if stats:
        print(f"[{label}] GPU: {stats['gpu_util']:.0f}%, Memory: {stats['mem_used']:.2f}/{stats['mem_total']:.2f} GB")

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print_gpu_stats("Initial")

# Load data
print("\nLoading data...")
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
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print(f"Data: {X_train.shape}")

results = {}

# ============================================
# 1. XGBOOST GPU
# ============================================
print("\n" + "="*80)
print("1. XGBOOST WITH GPU")
print("="*80)

print_gpu_stats("Before XGBoost")

start = time.time()
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    gpu_id=0,
    random_state=42
)

X_train_xgb = np.vstack([X_train, X_val])
y_train_xgb = np.hstack([y_train, y_val])

print("Training XGBoost...")
xgb_model.fit(X_train_xgb, y_train_xgb)
print_gpu_stats("After XGBoost training")

xgb_time = time.time() - start

# Predict
xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
xgb_pred = xgb_model.predict(X_test)

results['XGBoost_GPU'] = {
    'accuracy': accuracy_score(y_test, xgb_pred),
    'roc_auc': roc_auc_score(y_test, xgb_pred_proba),
    'time': xgb_time
}

print(f"XGBoost: Acc={results['XGBoost_GPU']['accuracy']:.3f}, "
      f"AUC={results['XGBoost_GPU']['roc_auc']:.3f}, Time={xgb_time:.1f}s")

# ============================================
# 2. NEURAL NETWORK GPU
# ============================================
print("\n" + "="*80)
print("2. NEURAL NETWORK WITH GPU")
print("="*80)

class NeuralNet(nn.Module):
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
        x = torch.sigmoid(self.fc4(x))
        return x.squeeze()

print_gpu_stats("Before NN")

# Prepare tensors
X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.FloatTensor(y_train).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)

model = NeuralNet(X_train.shape[1]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

print("Training Neural Network...")
start = time.time()

for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"  Epoch {epoch}: Loss={loss.item():.4f}")
        print_gpu_stats(f"NN Epoch {epoch}")

nn_time = time.time() - start

# Predict
model.eval()
with torch.no_grad():
    nn_pred = model(X_test_t).cpu().numpy()

nn_pred_binary = (nn_pred > 0.5).astype(int)

results['NeuralNet_GPU'] = {
    'accuracy': accuracy_score(y_test, nn_pred_binary),
    'roc_auc': roc_auc_score(y_test, nn_pred),
    'time': nn_time
}

print(f"Neural Net: Acc={results['NeuralNet_GPU']['accuracy']:.3f}, "
      f"AUC={results['NeuralNet_GPU']['roc_auc']:.3f}, Time={nn_time:.1f}s")

# ============================================
# 3. GP+CNN GPU
# ============================================
print("\n" + "="*80)
print("3. GP+CNN WITH GPU")
print("="*80)

class GPCNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.transform = nn.Linear(input_dim, 1024)
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(32)
        self.fc1 = nn.Linear(64*32, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.transform(x).unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze()

print_gpu_stats("Before GP+CNN")

gpcnn = GPCNN(X_train.shape[1]).to(device)
optimizer = torch.optim.AdamW(gpcnn.parameters(), lr=1e-3)

print("Training GP+CNN...")
start = time.time()

for epoch in range(50):
    gpcnn.train()
    optimizer.zero_grad()
    outputs = gpcnn(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"  Epoch {epoch}: Loss={loss.item():.4f}")
        print_gpu_stats(f"GP+CNN Epoch {epoch}")

gpcnn_time = time.time() - start

# Predict
gpcnn.eval()
with torch.no_grad():
    gpcnn_pred = gpcnn(X_test_t).cpu().numpy()

gpcnn_pred_binary = (gpcnn_pred > 0.5).astype(int)

results['GPCNN_GPU'] = {
    'accuracy': accuracy_score(y_test, gpcnn_pred_binary),
    'roc_auc': roc_auc_score(y_test, gpcnn_pred),
    'time': gpcnn_time
}

print(f"GP+CNN: Acc={results['GPCNN_GPU']['accuracy']:.3f}, "
      f"AUC={results['GPCNN_GPU']['roc_auc']:.3f}, Time={gpcnn_time:.1f}s")

# ============================================
# 4. RANDOM FOREST (CPU)
# ============================================
print("\n" + "="*80)
print("4. RANDOM FOREST (CPU ONLY)")
print("="*80)

print("Training Random Forest...")
start = time.time()

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train_xgb, y_train_xgb)
rf_time = time.time() - start

rf_pred_proba = rf.predict_proba(X_test)[:, 1]
rf_pred = rf.predict(X_test)

results['RandomForest_CPU'] = {
    'accuracy': accuracy_score(y_test, rf_pred),
    'roc_auc': roc_auc_score(y_test, rf_pred_proba),
    'time': rf_time
}

print(f"Random Forest: Acc={results['RandomForest_CPU']['accuracy']:.3f}, "
      f"AUC={results['RandomForest_CPU']['roc_auc']:.3f}, Time={rf_time:.1f}s")

# ============================================
# 5. SIMPLE MLP GPU
# ============================================
print("\n" + "="*80)
print("5. SIMPLE MLP WITH GPU")
print("="*80)

class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x.squeeze()

print_gpu_stats("Before MLP")

mlp = SimpleMLP(X_train.shape[1]).to(device)
optimizer = torch.optim.AdamW(mlp.parameters(), lr=3e-5)

print("Training Simple MLP...")
start = time.time()

for epoch in range(30):
    mlp.train()
    optimizer.zero_grad()
    outputs = mlp(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

mlp_time = time.time() - start

mlp.eval()
with torch.no_grad():
    mlp_pred = mlp(X_test_t).cpu().numpy()

mlp_pred_binary = (mlp_pred > 0.5).astype(int)

results['SimpleMLP_GPU'] = {
    'accuracy': accuracy_score(y_test, mlp_pred_binary),
    'roc_auc': roc_auc_score(y_test, mlp_pred),
    'time': mlp_time
}

print(f"Simple MLP: Acc={results['SimpleMLP_GPU']['accuracy']:.3f}, "
      f"AUC={results['SimpleMLP_GPU']['roc_auc']:.3f}, Time={mlp_time:.1f}s")

print_gpu_stats("After MLP")

# ============================================
# FINAL RANKING
# ============================================
print("\n" + "="*80)
print("FINAL RANKING (WITH REAL GPU USAGE)")
print("="*80)

# Sort by ROC-AUC
sorted_results = sorted(results.items(), key=lambda x: x[1]['roc_auc'], reverse=True)

print(f"\n{'Rank':<5} {'Model':<20} {'Accuracy':<12} {'ROC-AUC':<12} {'Time (s)':<12} {'Device'}")
print("-" * 70)

for i, (name, metrics) in enumerate(sorted_results, 1):
    device_type = "GPU" if "GPU" in name else "CPU"
    print(f"{i:<5} {name:<20} {metrics['accuracy']:.3f}{'':<9} "
          f"{metrics['roc_auc']:.3f}{'':<9} {metrics['time']:.1f}{'':<11} {device_type}")

# Save results
with open('gpu_benchmark_final.json', 'w') as f:
    json.dump({
        'results': results,
        'ranking': [{'rank': i, 'model': name, **metrics}
                   for i, (name, metrics) in enumerate(sorted_results, 1)]
    }, f, indent=2)

print("\n[SAVED] Results to gpu_benchmark_final.json")
print("\nTo monitor GPU in real-time, run: nvidia-smi -l 1")
print("[COMPLETE] GPU benchmark finished!")