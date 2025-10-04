"""
GPU-Accelerated Training and Model Comparison
==============================================
Using actual data from tsfresh_features.csv
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import time
import json
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("GPU-ACCELERATED MODEL TRAINING AND COMPARISON")
print("=" * 80)

# Check GPU availability
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"[GPU FOUND] {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    device = torch.device('cpu')
    print("[WARNING] No GPU detected, using CPU")

# Load actual data
print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)
data = pd.read_csv('tsfresh_features.csv')
print(f"Total samples: {len(data)}")
print(f"Features: {data.shape[1] - 1}")

# Remove columns with only one unique value
unique_value_columns = data.columns[data.nunique() <= 1]
data = data.drop(unique_value_columns, axis=1)

# Handle infinity and NaN values
data = data.replace([np.inf, -np.inf], np.nan)
data = data.fillna(0)

# Split data (following koi_project_nn.py approach)
train_data = data[:-600]
val_data = data[-600:-369]
test_data = data[-369:]

# Separate features and labels
X_train = train_data.iloc[:, 1:-1].values
y_train = train_data.iloc[:, -1].values
X_val = val_data.iloc[:, 1:-1].values
y_val = val_data.iloc[:, -1].values
X_test = test_data.iloc[:, 1:-1].values
y_test = test_data.iloc[:, -1].values

# Scale features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print(f"\nData split:")
print(f"  Train: {X_train.shape[0]} samples")
print(f"  Val: {X_val.shape[0]} samples")
print(f"  Test: {X_test.shape[0]} samples")
print(f"  Features: {X_train.shape[1]}")
print(f"  Class balance: {y_train.mean():.2%} positive")

results = {}

# ============================================
# 1. NEURAL NETWORK (GPU)
# ============================================
print("\n" + "=" * 80)
print("[1/4] NEURAL NETWORK (GPU)")
print("=" * 80)

class ImprovedNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Convert to tensors
X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.FloatTensor(y_train).to(device)
X_val_t = torch.FloatTensor(X_val).to(device)
y_val_t = torch.FloatTensor(y_val).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)

# Create data loaders for batch processing
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize model
nn_model = ImprovedNN(X_train.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = optim.AdamW(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

# Training
print("Training Neural Network...")
start_time = time.time()
epochs = 100
best_val_loss = float('inf')
patience = 20
patience_counter = 0

for epoch in range(epochs):
    # Train
    nn_model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = nn_model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validate
    nn_model.eval()
    with torch.no_grad():
        val_outputs = nn_model(X_val_t).squeeze()
        val_loss = criterion(val_outputs, y_val_t).item()

    scheduler.step(val_loss)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = nn_model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    if epoch % 10 == 0:
        print(f"  Epoch {epoch}: Train Loss={train_loss/len(train_loader):.4f}, Val Loss={val_loss:.4f}")

nn_model.load_state_dict(best_model_state)
nn_time = time.time() - start_time

# Evaluate
nn_model.eval()
with torch.no_grad():
    nn_pred = nn_model(X_test_t).squeeze().cpu().numpy()

nn_pred_binary = (nn_pred > 0.5).astype(int)
nn_acc = accuracy_score(y_test, nn_pred_binary)
nn_prec = precision_score(y_test, nn_pred_binary)
nn_rec = recall_score(y_test, nn_pred_binary)
nn_f1 = f1_score(y_test, nn_pred_binary)
nn_auc = roc_auc_score(y_test, nn_pred)

results['Neural Network'] = {
    'accuracy': nn_acc,
    'precision': nn_prec,
    'recall': nn_rec,
    'f1': nn_f1,
    'roc_auc': nn_auc,
    'training_time': nn_time
}

print(f"Results: Acc={nn_acc:.3f}, ROC-AUC={nn_auc:.3f}, Time={nn_time:.1f}s")

# ============================================
# 2. XGBOOST (GPU)
# ============================================
print("\n" + "=" * 80)
print("[2/4] XGBOOST (GPU)")
print("=" * 80)

try:
    import xgboost as xgb

    print("Training XGBoost with GPU acceleration...")
    start_time = time.time()

    # Combine train and validation for XGBoost
    X_train_xgb = np.vstack([X_train, X_val])
    y_train_xgb = np.hstack([y_train, y_val])

    if device.type == 'cuda':
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='gpu_hist',
            predictor='gpu_predictor',
            gpu_id=0,
            eval_metric='auc',
            random_state=42
        )
        print("  Using GPU acceleration (gpu_hist)")
    else:
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='auc',
            random_state=42
        )
        print("  Using CPU")

    xgb_model.fit(X_train_xgb, y_train_xgb)
    xgb_time = time.time() - start_time

    # Evaluate
    xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_pred = xgb_model.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_prec = precision_score(y_test, xgb_pred)
    xgb_rec = recall_score(y_test, xgb_pred)
    xgb_f1 = f1_score(y_test, xgb_pred)
    xgb_auc = roc_auc_score(y_test, xgb_pred_proba)

    results['XGBoost'] = {
        'accuracy': xgb_acc,
        'precision': xgb_prec,
        'recall': xgb_rec,
        'f1': xgb_f1,
        'roc_auc': xgb_auc,
        'training_time': xgb_time
    }

    print(f"Results: Acc={xgb_acc:.3f}, ROC-AUC={xgb_auc:.3f}, Time={xgb_time:.1f}s")

except ImportError:
    print("XGBoost not installed. Install with: pip install xgboost")
    results['XGBoost'] = {'error': 'Not installed'}

# ============================================
# 3. RANDOM FOREST (CPU)
# ============================================
print("\n" + "=" * 80)
print("[3/4] RANDOM FOREST (CPU)")
print("=" * 80)

print("Training Random Forest...")
start_time = time.time()

# Combine train and validation
X_train_rf = np.vstack([X_train, X_val])
y_train_rf = np.hstack([y_train, y_val])

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=9,
    min_samples_leaf=4,
    n_jobs=-1,  # Use all CPU cores
    random_state=42
)

rf_model.fit(X_train_rf, y_train_rf)
rf_time = time.time() - start_time

# Evaluate
rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
rf_prec = precision_score(y_test, rf_pred)
rf_rec = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_pred_proba)

results['Random Forest'] = {
    'accuracy': rf_acc,
    'precision': rf_prec,
    'recall': rf_rec,
    'f1': rf_f1,
    'roc_auc': rf_auc,
    'training_time': rf_time
}

print(f"Results: Acc={rf_acc:.3f}, ROC-AUC={rf_auc:.3f}, Time={rf_time:.1f}s")

# ============================================
# 4. SIMPLE MLP (following koi_project_nn.py)
# ============================================
print("\n" + "=" * 80)
print("[4/4] SIMPLE MLP (Original Architecture)")
print("=" * 80)

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

            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Train simple MLP
print("Training Simple MLP...")
start_time = time.time()

simple_model = SimpleMLP(X_train.shape[1]).to(device)
optimizer = optim.AdamW(simple_model.parameters(), lr=3e-5)

epochs = 50
for epoch in range(epochs):
    simple_model.train()
    optimizer.zero_grad()
    outputs = simple_model(X_train_t).squeeze()
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"  Epoch {epoch}: Loss={loss.item():.4f}")

simple_time = time.time() - start_time

# Evaluate
simple_model.eval()
with torch.no_grad():
    simple_pred = simple_model(X_test_t).squeeze().cpu().numpy()

simple_pred_binary = (simple_pred > 0.5).astype(int)
simple_acc = accuracy_score(y_test, simple_pred_binary)
simple_prec = precision_score(y_test, simple_pred_binary)
simple_rec = recall_score(y_test, simple_pred_binary)
simple_f1 = f1_score(y_test, simple_pred_binary)
simple_auc = roc_auc_score(y_test, simple_pred)

results['Simple MLP'] = {
    'accuracy': simple_acc,
    'precision': simple_prec,
    'recall': simple_rec,
    'f1': simple_f1,
    'roc_auc': simple_auc,
    'training_time': simple_time
}

print(f"Results: Acc={simple_acc:.3f}, ROC-AUC={simple_auc:.3f}, Time={simple_time:.1f}s")

# ============================================
# FINAL COMPARISON
# ============================================
print("\n" + "=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)

# Create comparison table
comparison_data = []
for model_name, metrics in results.items():
    if 'error' not in metrics:
        comparison_data.append({
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']:.3f}",
            'Precision': f"{metrics['precision']:.3f}",
            'Recall': f"{metrics['recall']:.3f}",
            'F1': f"{metrics['f1']:.3f}",
            'ROC-AUC': f"{metrics['roc_auc']:.3f}",
            'Time (s)': f"{metrics['training_time']:.1f}"
        })

df = pd.DataFrame(comparison_data)
print("\n" + df.to_string(index=False))

# Find best model
best_model = max(results.items(),
                 key=lambda x: x[1]['roc_auc'] if 'roc_auc' in x[1] else 0)

print("\n" + "=" * 80)
print("WINNER")
print("=" * 80)
print(f"Best Model: {best_model[0]}")
print(f"ROC-AUC: {best_model[1]['roc_auc']:.3f}")
print(f"Training Time: {best_model[1]['training_time']:.1f} seconds")

# Save results
with open('gpu_training_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n[SAVED] Results saved to gpu_training_results.json")

# Create detailed comparison report
report = f"""
GPU TRAINING RESULTS REPORT
===========================
Date: {pd.Timestamp.now()}
Device: {device}
Data: tsfresh_features.csv
Samples: {len(data)} total ({X_train.shape[0]} train, {X_val.shape[0]} val, {X_test.shape[0]} test)
Features: {X_train.shape[1]}

PERFORMANCE COMPARISON
----------------------
{df.to_string(index=False)}

KEY FINDINGS
------------
1. Best Model: {best_model[0]} (ROC-AUC: {best_model[1]['roc_auc']:.3f})
2. GPU Acceleration:
   - Neural Network: {results.get('Neural Network', {}).get('training_time', 0):.1f}s
   - XGBoost: {results.get('XGBoost', {}).get('training_time', 0):.1f}s
3. CPU Performance:
   - Random Forest: {results.get('Random Forest', {}).get('training_time', 0):.1f}s

RECOMMENDATIONS
---------------
{'- XGBoost achieved the best ROC-AUC score' if best_model[0] == 'XGBoost' else ''}
{'- Neural Network showed competitive performance with GPU acceleration' if 'Neural Network' in results else ''}
{'- Random Forest provides good interpretability' if 'Random Forest' in results else ''}
"""

with open('training_report.txt', 'w') as f:
    f.write(report)

print(f"[SAVED] Detailed report saved to training_report.txt")
print("\n[COMPLETE] All models trained and compared successfully!")