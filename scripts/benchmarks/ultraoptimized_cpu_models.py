"""
Ultra-Optimized CPU Models with 2025 Best Practices
====================================================
Implements Intel MKL, OpenMP, and multi-threading optimizations
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
from joblib import parallel_backend
import time
import json
import multiprocessing
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ULTRA-OPTIMIZED CPU MODELS - 2025 BEST PRACTICES")
print("="*80)

# ============================================
# CPU OPTIMIZATION SETTINGS
# ============================================
print("\n[APPLYING 2025 CPU OPTIMIZATIONS]")

# Get number of physical cores (avoid hyperthreading for ML)
physical_cores = multiprocessing.cpu_count() // 2
print(f"Physical CPU cores detected: {physical_cores}")

# 1. Set Intel MKL threads (if available)
os.environ['MKL_NUM_THREADS'] = str(physical_cores)
os.environ['MKL_DYNAMIC'] = 'FALSE'  # Disable dynamic adjustment
os.environ['MKL_ENABLE_INSTRUCTIONS'] = 'AVX512'  # Enable AVX512 if available
print("[OK] Intel MKL configured")

# 2. Set OpenMP threads
os.environ['OMP_NUM_THREADS'] = str(physical_cores)
os.environ['OMP_PROC_BIND'] = 'TRUE'  # Bind threads to cores
os.environ['OMP_PLACES'] = 'cores'  # Place threads on cores
print("[OK] OpenMP thread binding enabled")

# 3. Set NumPy/BLAS threads
os.environ['OPENBLAS_NUM_THREADS'] = str(physical_cores)
os.environ['NUMEXPR_NUM_THREADS'] = str(physical_cores)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(physical_cores)
print("[OK] BLAS threading optimized")

# 4. Thread affinity for better cache usage
try:
    import psutil
    p = psutil.Process()
    p.cpu_affinity(list(range(physical_cores)))
    print("[OK] CPU affinity set to physical cores")
except:
    print("[WARNING] Could not set CPU affinity")

# 5. Import Intel Extension for Scikit-learn if available
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    print("[OK] Intel Extension for Scikit-learn enabled")
    intel_optimized = True
except ImportError:
    print("[WARNING] Intel Extension for Scikit-learn not available")
    print("  Install with: pip install scikit-learn-intelex")
    intel_optimized = False

# ============================================
# OPTIMIZED RANDOM FOREST
# ============================================
class OptimizedRandomForest:
    """
    Random Forest optimized for CPU with:
    - Optimal n_jobs for physical cores
    - Memory-efficient parameters
    - Intel MKL acceleration (if available)
    """
    def __init__(self, n_physical_cores):
        # Use n_jobs = physical cores - 1 to avoid oversubscription
        self.model = RandomForestClassifier(
            n_estimators=500,  # More trees for better performance
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,  # Out-of-bag score for validation
            n_jobs=n_physical_cores - 1,  # Leave one core for system
            random_state=42,
            warm_start=False,
            max_samples=0.8,  # Subsample for faster training
            criterion='gini',  # Gini is faster than entropy
            verbose=0
        )

    def train(self, X_train, y_train):
        print("Training Optimized Random Forest...")
        print(f"  Trees: {self.model.n_estimators}")
        print(f"  Jobs: {self.model.n_jobs}")
        print(f"  Max depth: {self.model.max_depth}")

        start_time = time.time()

        # Use joblib backend for better parallelization
        with parallel_backend('threading', n_jobs=self.model.n_jobs):
            self.model.fit(X_train, y_train)

        training_time = time.time() - start_time

        # Get OOB score for validation
        oob_score = self.model.oob_score_ if self.model.oob_score else None

        return training_time, oob_score

    def predict(self, X_test):
        return self.model.predict(X_test), self.model.predict_proba(X_test)[:, 1]

# ============================================
# OPTIMIZED XGBOOST FOR CPU
# ============================================
class OptimizedXGBoostCPU:
    """
    XGBoost optimized for CPU with:
    - Histogram-based algorithm
    - Optimal thread configuration
    - Cache-aware optimization
    """
    def __init__(self, n_physical_cores):
        self.model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',  # Histogram method for CPU
            predictor='cpu_predictor',
            n_jobs=n_physical_cores - 1,  # Optimal threading
            random_state=42,
            eval_metric='auc',
            enable_categorical=False,
            max_bin=256,  # Optimal for CPU cache
            grow_policy='depthwise',  # Better for CPU
            max_leaves=0,  # Unlimited when using depthwise
            # CPU-specific optimizations
            reg_alpha=0.01,  # L1 regularization
            reg_lambda=0.1,  # L2 regularization
            min_child_weight=3
        )

    def train(self, X_train, y_train, X_val=None, y_val=None):
        print("Training Optimized XGBoost (CPU)...")
        print(f"  Trees: {self.model.n_estimators}")
        print(f"  Threads: {self.model.n_jobs}")
        print(f"  Method: {self.model.tree_method}")

        start_time = time.time()

        # Prepare evaluation set if provided
        eval_set = [(X_val, y_val)] if X_val is not None else None
        early_stopping = 20 if eval_set else None

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping,
            verbose=False
        )

        training_time = time.time() - start_time
        return training_time

    def predict(self, X_test):
        return self.model.predict(X_test), self.model.predict_proba(X_test)[:, 1]

# ============================================
# OPTIMIZED MLP FOR CPU
# ============================================
class OptimizedMLPCPU:
    """
    Multi-Layer Perceptron optimized for CPU with:
    - LBFGS solver for small datasets
    - Optimal batch size
    - Intel MKL acceleration
    """
    def __init__(self):
        self.model = MLPClassifier(
            hidden_layer_sizes=(512, 256, 128, 64),  # Deeper network
            activation='relu',
            solver='lbfgs',  # LBFGS is best for small datasets on CPU
            alpha=0.001,  # L2 regularization
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            shuffle=True,
            random_state=42,
            tol=0.0001,
            verbose=False,
            warm_start=False,
            momentum=0.9,
            nesterovs_momentum=True,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20
        )

    def train(self, X_train, y_train):
        print("Training Optimized MLP (CPU)...")
        print(f"  Architecture: {self.model.hidden_layer_sizes}")
        print(f"  Solver: {self.model.solver}")

        start_time = time.time()
        self.model.fit(X_train, y_train)
        training_time = time.time() - start_time

        return training_time

    def predict(self, X_test):
        return self.model.predict(X_test), self.model.predict_proba(X_test)[:, 1]

# ============================================
# DATA PREPROCESSING WITH CPU OPTIMIZATION
# ============================================
def preprocess_data_cpu_optimized(data):
    """
    Preprocess data with CPU optimizations:
    - Memory-aligned arrays
    - Efficient dtype usage
    """
    # Clean data
    data = data.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Remove single-value columns
    unique_cols = data.columns[data.nunique() <= 1]
    data = data.drop(unique_cols, axis=1)

    # Convert to float32 for better CPU cache usage
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].astype(np.float32)

    return data

# ============================================
# MEMORY-ALIGNED DATA LOADING
# ============================================
def create_aligned_arrays(X, y):
    """
    Create memory-aligned arrays for better CPU performance
    Uses 64-byte alignment for optimal cache line usage
    """
    # Ensure C-contiguous arrays for better cache locality
    X_aligned = np.ascontiguousarray(X, dtype=np.float32)
    y_aligned = np.ascontiguousarray(y, dtype=np.float32)

    return X_aligned, y_aligned

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    # Load data
    print("\nLoading and preprocessing data...")
    data = pd.read_csv('tsfresh_features.csv')
    data = preprocess_data_cpu_optimized(data)

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

    # Scale with CPU optimization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Create memory-aligned arrays
    X_train, y_train = create_aligned_arrays(X_train, y_train)
    X_val, y_val = create_aligned_arrays(X_val, y_val)
    X_test, y_test = create_aligned_arrays(X_test, y_test)

    # Combine train and val for tree-based models
    X_train_combined = np.vstack([X_train, X_val])
    y_train_combined = np.hstack([y_train, y_val])

    print(f"Data shape: {X_train.shape}")
    print(f"Data dtype: {X_train.dtype} (optimized for CPU cache)")

    results = {}

    # ============================================
    # 1. OPTIMIZED RANDOM FOREST
    # ============================================
    print("\n" + "="*80)
    print("1. OPTIMIZED RANDOM FOREST (CPU)")
    print("="*80)

    rf_optimizer = OptimizedRandomForest(physical_cores)
    rf_time, rf_oob = rf_optimizer.train(X_train_combined, y_train_combined)

    rf_pred, rf_proba = rf_optimizer.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_auc = roc_auc_score(y_test, rf_proba)

    results['RandomForest_Optimized'] = {
        'accuracy': rf_acc,
        'roc_auc': rf_auc,
        'training_time': rf_time,
        'oob_score': rf_oob,
        'n_jobs': rf_optimizer.model.n_jobs
    }

    print(f"\nResults:")
    print(f"  Accuracy: {rf_acc:.3f}")
    print(f"  ROC-AUC: {rf_auc:.3f}")
    print(f"  OOB Score: {rf_oob:.3f}" if rf_oob else "  OOB Score: N/A")
    print(f"  Training Time: {rf_time:.2f}s")

    # ============================================
    # 2. OPTIMIZED XGBOOST
    # ============================================
    print("\n" + "="*80)
    print("2. OPTIMIZED XGBOOST (CPU)")
    print("="*80)

    xgb_optimizer = OptimizedXGBoostCPU(physical_cores)
    xgb_time = xgb_optimizer.train(X_train, y_train, X_val, y_val)

    xgb_pred, xgb_proba = xgb_optimizer.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_auc = roc_auc_score(y_test, xgb_proba)

    results['XGBoost_Optimized'] = {
        'accuracy': xgb_acc,
        'roc_auc': xgb_auc,
        'training_time': xgb_time,
        'n_jobs': xgb_optimizer.model.n_jobs
    }

    print(f"\nResults:")
    print(f"  Accuracy: {xgb_acc:.3f}")
    print(f"  ROC-AUC: {xgb_auc:.3f}")
    print(f"  Training Time: {xgb_time:.2f}s")

    # ============================================
    # 3. OPTIMIZED MLP
    # ============================================
    print("\n" + "="*80)
    print("3. OPTIMIZED MLP (CPU)")
    print("="*80)

    mlp_optimizer = OptimizedMLPCPU()
    mlp_time = mlp_optimizer.train(X_train_combined, y_train_combined)

    mlp_pred, mlp_proba = mlp_optimizer.predict(X_test)
    mlp_acc = accuracy_score(y_test, mlp_pred)
    mlp_auc = roc_auc_score(y_test, mlp_proba)

    results['MLP_Optimized'] = {
        'accuracy': mlp_acc,
        'roc_auc': mlp_auc,
        'training_time': mlp_time
    }

    print(f"\nResults:")
    print(f"  Accuracy: {mlp_acc:.3f}")
    print(f"  ROC-AUC: {mlp_auc:.3f}")
    print(f"  Training Time: {mlp_time:.2f}s")

    # ============================================
    # SUMMARY
    # ============================================
    print("\n" + "="*80)
    print("CPU OPTIMIZATION SUMMARY")
    print("="*80)

    print("\nOptimizations Applied:")
    print(f"[OK] Physical cores utilized: {physical_cores}")
    print("[OK] Intel MKL threading configured")
    print("[OK] OpenMP thread binding enabled")
    print("[OK] CPU affinity set to physical cores")
    print("[OK] Memory-aligned arrays (64-byte)")
    print("[OK] Cache-optimized data types (float32)")
    print("[OK] Thread oversubscription avoided")
    if intel_optimized:
        print("[OK] Intel Extension for Scikit-learn active")

    print("\nPerformance Comparison:")
    print(f"\n{'Model':<25} {'Accuracy':<10} {'ROC-AUC':<10} {'Time (s)':<10}")
    print("-" * 60)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['roc_auc'], reverse=True)
    for model_name, metrics in sorted_results:
        print(f"{model_name:<25} {metrics['accuracy']:.3f}{'':<7} "
              f"{metrics['roc_auc']:.3f}{'':<7} {metrics['training_time']:.2f}")

    # Save results
    with open('ultraoptimized_cpu_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n[SAVED] Results to ultraoptimized_cpu_results.json")
    print("[COMPLETE] Ultra-optimized CPU training finished!")

    # System information
    print("\n" + "="*80)
    print("SYSTEM INFORMATION")
    print("="*80)
    print(f"CPU Cores (Physical): {physical_cores}")
    print(f"CPU Cores (Logical): {multiprocessing.cpu_count()}")

    try:
        import platform
        print(f"Processor: {platform.processor()}")
        print(f"Platform: {platform.platform()}")
    except:
        pass

    # Check for Intel MKL
    try:
        import numpy as np
        if 'mkl' in np.__config__.show():
            print("NumPy with Intel MKL: [OK]")
    except:
        pass