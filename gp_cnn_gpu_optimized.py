"""
GPU-Optimized GP+CNN Pipeline with Real GPU Utilization
=========================================================
Larger model, batch processing, mixed precision for actual GPU usage
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import time
import json
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("GPU-OPTIMIZED GP+CNN PIPELINE")
print("="*80)

# Force GPU and check
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type != 'cuda':
    print("ERROR: No GPU available!")
    exit()

print(f"Device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Monitor GPU usage
def print_gpu_usage():
    if torch.cuda.is_available():
        print(f"GPU Memory Used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU Utilization: Check nvidia-smi for real-time usage")

# Load data
print("\nLoading data...")
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
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print(f"Data shape: {X_train.shape}")

# ============================================
# LARGE GPU-INTENSIVE MODEL
# ============================================
class HeavyGPCNN(nn.Module):
    """Large model to ensure GPU utilization"""
    def __init__(self, input_dim):
        super().__init__()

        # Feature expansion for GPU processing
        self.feature_expand = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU()
        )

        # Reshape for CNN processing
        self.reshape_size = 64
        self.reshape_channels = 64

        # Heavy CNN layers (Global Branch)
        self.global_branch = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8)
        )

        # Heavy CNN layers (Local Branch)
        self.local_branch = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8)
        )

        # Heavy classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024*8*2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 1)
        )

    def forward(self, x):
        # Expand features
        x = self.feature_expand(x)

        # Reshape for CNN
        batch_size = x.size(0)
        x = x.view(batch_size, self.reshape_channels, -1)

        # Process through both branches
        global_features = self.global_branch(x)
        local_features = self.local_branch(x)

        # Concatenate
        combined = torch.cat([global_features.flatten(1), local_features.flatten(1)], dim=1)

        # Classify
        output = self.classifier(combined)
        return torch.sigmoid(output).squeeze()

# ============================================
# DATA AUGMENTATION FOR MORE GPU WORK
# ============================================
def augment_batch(X, y, augment_factor=4):
    """Create augmented batches for more GPU processing"""
    X_aug = []
    y_aug = []

    for _ in range(augment_factor):
        # Random noise augmentation
        noise = np.random.normal(0, 0.01, X.shape)
        X_noisy = X + noise
        X_aug.append(X_noisy)
        y_aug.append(y)

    X_aug = np.vstack(X_aug)
    y_aug = np.hstack(y_aug)
    return X_aug, y_aug

# Augment training data for more GPU work
print("\nAugmenting data for GPU processing...")
X_train_aug, y_train_aug = augment_batch(X_train, y_train, augment_factor=4)
print(f"Augmented training data: {X_train_aug.shape}")

# ============================================
# GPU-OPTIMIZED TRAINING
# ============================================
print("\nInitializing Heavy Model for GPU...")

# Create large model
model = HeavyGPCNN(X_train.shape[1]).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print_gpu_usage()

# Use larger batch size for GPU
BATCH_SIZE = 128

# Create data loaders
train_dataset = TensorDataset(
    torch.FloatTensor(X_train_aug),
    torch.FloatTensor(y_train_aug)
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

val_dataset = TensorDataset(
    torch.FloatTensor(X_val),
    torch.FloatTensor(y_val)
)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, pin_memory=True)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Mixed precision training for GPU
scaler = GradScaler()

print("\nStarting GPU Training with Mixed Precision...")
print("Monitor GPU usage with: nvidia-smi -l 1")
print_gpu_usage()

start_time = time.time()
epochs = 20  # Reduced epochs but heavier computation per epoch
best_val_auc = 0

for epoch in range(epochs):
    # Training
    model.train()
    train_loss = 0
    batch_count = 0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision training
        with autocast():
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

        # Backward pass with scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        batch_count += 1

        # Force GPU synchronization periodically
        if batch_count % 10 == 0:
            torch.cuda.synchronize()

    avg_train_loss = train_loss / batch_count

    # Validation
    model.eval()
    val_preds = []
    val_true = []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            with autocast():
                outputs = model(batch_X)

            val_preds.extend(outputs.cpu().numpy())
            val_true.extend(batch_y.cpu().numpy())

    val_auc = roc_auc_score(val_true, val_preds)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_model_state = model.state_dict().copy()

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}: Loss={avg_train_loss:.4f}, Val AUC={val_auc:.4f}")
        print_gpu_usage()

training_time = time.time() - start_time
model.load_state_dict(best_model_state)

print(f"\nTraining completed in {training_time:.1f}s")
print_gpu_usage()

# ============================================
# EVALUATION
# ============================================
print("\nEvaluating GPU-Optimized GP+CNN...")

model.eval()
test_preds = []

X_test_t = torch.FloatTensor(X_test).to(device)

# Process in batches
with torch.no_grad():
    for i in range(0, len(X_test), BATCH_SIZE):
        batch = X_test_t[i:i+BATCH_SIZE]
        with autocast():
            outputs = model(batch)
        test_preds.extend(outputs.cpu().numpy())

test_pred = np.array(test_preds)
test_pred_binary = (test_pred > 0.5).astype(int)

accuracy = accuracy_score(y_test, test_pred_binary)
precision = precision_score(y_test, test_pred_binary)
recall = recall_score(y_test, test_pred_binary)
f1 = f1_score(y_test, test_pred_binary)
roc_auc = roc_auc_score(y_test, test_pred)

print("\n" + "="*80)
print("GPU-OPTIMIZED GP+CNN RESULTS")
print("="*80)
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"ROC-AUC: {roc_auc:.3f}")
print(f"Training Time: {training_time:.1f}s")
print(f"Model Size: {sum(p.numel() for p in model.parameters()):,} parameters")
print_gpu_usage()

# Save results
results = {
    'model': 'GP+CNN Heavy (GPU-Optimized)',
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1': float(f1),
    'roc_auc': float(roc_auc),
    'training_time': float(training_time),
    'model_parameters': sum(p.numel() for p in model.parameters()),
    'batch_size': BATCH_SIZE,
    'gpu_used': torch.cuda.get_device_name(0),
    'mixed_precision': True
}

with open('gp_cnn_gpu_optimized_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n[SAVED] Results to gp_cnn_gpu_optimized_results.json")
print("\nTo monitor GPU usage, run in another terminal:")
print("nvidia-smi -l 1")
print("\n[COMPLETE] GPU-Optimized GP+CNN training finished!")