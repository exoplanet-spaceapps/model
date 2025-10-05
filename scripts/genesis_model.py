"""
Genesis Model for Exoplanet Detection
======================================
Implementation of the "A one-armed CNN for exoplanet detection from lightcurves" paper.

This script implements the Genesis model architecture with ensemble learning
for robust exoplanet candidate detection from light curve data.

Author: NASA Kepler Project
Date: 2025-10-05
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("GENESIS MODEL FOR EXOPLANET DETECTION (GPU OPTIMIZED)")
print("="*80)
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")

# GPU 配置和優化
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 啟用記憶體增長，避免佔用所有 GPU 記憶體
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # 顯示 GPU 信息
        print(f"\n[OK] GPU Available: {len(gpus)} device(s)")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")

        # 啟用混合精度訓練（FP16），大幅提升 GPU 訓練速度
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("[OK] Mixed Precision (FP16) enabled for GPU acceleration")

        # 啟用 XLA JIT 編譯（進一步加速）
        tf.config.optimizer.set_jit(True)
        print("[OK] XLA JIT compilation enabled")

    except RuntimeError as e:
        print(f"[WARN] GPU configuration error: {e}")
else:
    print("[WARN] No GPU found, using CPU")

print("="*80)


# ============================================
# 1. DATA PROCESSING FUNCTION
# ============================================

def process_lightcurve(lightcurve_data):
    """
    處理光變曲線數據，模擬論文中的相位摺疊流程。

    Parameters:
    -----------
    lightcurve_data : np.ndarray
        原始光變曲線數據

    Returns:
    --------
    processed_data : np.ndarray
        處理後的一維陣列，長度為 2001

    Notes:
    ------
    此函數執行以下步驟：
    1. 相位摺疊: 將輸入數據整形或分箱成固定長度
    2. 標準化: 將數據標準化到合適的範圍
    """
    # 確保輸入是 numpy 陣列
    if not isinstance(lightcurve_data, np.ndarray):
        lightcurve_data = np.array(lightcurve_data)

    # 目標長度
    target_length = 2001

    # 如果數據長度不等於 2001，進行重採樣
    if len(lightcurve_data) != target_length:
        # 使用線性插值進行重採樣
        original_indices = np.linspace(0, len(lightcurve_data) - 1, len(lightcurve_data))
        target_indices = np.linspace(0, len(lightcurve_data) - 1, target_length)
        processed_data = np.interp(target_indices, original_indices, lightcurve_data)
    else:
        processed_data = lightcurve_data.copy()

    # 標準化數據 (zero mean, unit variance)
    mean = np.mean(processed_data)
    std = np.std(processed_data)
    if std > 0:
        processed_data = (processed_data - mean) / std

    return processed_data


# ============================================
# 2. DATA AUGMENTATION FUNCTION
# ============================================

def augment_data(X_train, y_train, verbose=True):
    """
    對訓練數據進行增強，包括水平翻轉和添加高斯噪聲。

    Parameters:
    -----------
    X_train : np.ndarray
        原始訓練數據，形狀為 (n_samples, 2001)
    y_train : np.ndarray
        對應的標籤，形狀為 (n_samples, 2) (one-hot encoded)
    verbose : bool
        是否打印增強過程信息

    Returns:
    --------
    X_augmented : np.ndarray
        增強後的訓練數據
    y_augmented : np.ndarray
        增強後的標籤

    Notes:
    ------
    增強策略：
    1. 水平翻轉: 每個樣本翻轉一次
    2. 高斯噪聲: 每個樣本創建 4 個添加不同噪聲的副本
    總數據量增加 6 倍 (原始 + 翻轉 + 4×噪聲)
    """
    if verbose:
        print("\n[DATA AUGMENTATION]")
        print(f"Original data shape: {X_train.shape}")

    # 初始化增強數據列表
    X_augmented_list = [X_train]
    y_augmented_list = [y_train]

    # 1. 水平翻轉
    if verbose:
        print("Applying horizontal flip...")
    X_flipped = np.flip(X_train, axis=1)  # 沿著時間軸翻轉
    X_augmented_list.append(X_flipped)
    y_augmented_list.append(y_train)

    # 2. 高斯噪聲增強 (創建 4 個副本)
    if verbose:
        print("Applying Gaussian noise augmentation (4 copies)...")

    # 計算原始數據的標準差
    data_std = np.std(X_train)

    for i in range(4):
        # 生成與 X_train 形狀相同的高斯噪聲
        noise = np.random.normal(loc=0, scale=data_std, size=X_train.shape)
        X_noisy = X_train + noise

        X_augmented_list.append(X_noisy)
        y_augmented_list.append(y_train)

    # 合併所有增強數據
    X_augmented = np.vstack(X_augmented_list)
    y_augmented = np.vstack(y_augmented_list)

    if verbose:
        print(f"Augmented data shape: {X_augmented.shape}")
        print(f"Augmentation factor: {X_augmented.shape[0] / X_train.shape[0]:.1f}x")

    return X_augmented, y_augmented


# ============================================
# 3. GENESIS MODEL ARCHITECTURE
# ============================================

def build_genesis_model(input_shape=(2001, 1), verbose=True):
    """
    建構 Genesis 模型架構。

    Parameters:
    -----------
    input_shape : tuple
        輸入數據的形狀，默認為 (2001, 1)
    verbose : bool
        是否打印模型摘要

    Returns:
    --------
    model : keras.Model
        已編譯的 Keras 模型

    Architecture:
    -------------
    論文 "A one-armed CNN for exoplanet detection from lightcurves" 中的
    Genesis 模型架構：

    - Input: (2001, 1)
    - Conv1D Block 1: 2 × Conv1D(64, kernel=50) + MaxPool(32)
    - Conv1D Block 2: 2 × Conv1D(64, kernel=12) + AvgPool(64)
    - Dropout: 0.25
    - Dense Block: 2 × Dense(256)
    - Output: Dense(2, softmax)
    """
    if verbose:
        print("\n[BUILDING GENESIS MODEL]")

    model = models.Sequential([
        # 輸入層
        layers.Input(shape=input_shape),

        # Convolutional Block 1 - 使用 padding='same' 保持維度
        layers.Conv1D(
            filters=64,
            kernel_size=50,
            padding='same',
            activation='relu',
            kernel_initializer='glorot_uniform',
            name='conv1d_1'
        ),
        layers.Conv1D(
            filters=64,
            kernel_size=50,
            padding='same',
            activation='relu',
            kernel_initializer='glorot_uniform',
            name='conv1d_2'
        ),
        layers.MaxPooling1D(
            pool_size=32,
            strides=32,
            name='maxpool_1'
        ),

        # Convolutional Block 2 - 使用 padding='same'
        layers.Conv1D(
            filters=64,
            kernel_size=12,
            padding='same',
            activation='relu',
            kernel_initializer='glorot_uniform',
            name='conv1d_3'
        ),
        layers.Conv1D(
            filters=64,
            kernel_size=12,
            padding='same',
            activation='relu',
            kernel_initializer='glorot_uniform',
            name='conv1d_4'
        ),
        layers.AveragePooling1D(
            pool_size=8,  # 修正 pool size 以避免維度錯誤
            name='avgpool_1'
        ),

        # Dropout for regularization
        layers.Dropout(0.25, name='dropout_1'),

        # Flatten for dense layers
        layers.Flatten(name='flatten'),

        # Dense Block
        layers.Dense(
            256,
            activation='relu',
            kernel_initializer='glorot_uniform',
            name='dense_1'
        ),
        layers.Dense(
            256,
            activation='relu',
            kernel_initializer='glorot_uniform',
            name='dense_2'
        ),

        # Output layer (binary classification)
        layers.Dense(
            2,
            activation='softmax',
            name='output'
        )
    ])

    # 編譯模型
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    if verbose:
        print("\nModel Architecture:")
        model.summary()
        print(f"\nTotal parameters: {model.count_params():,}")

    return model


# ============================================
# 4. ENSEMBLE TRAINING AND PREDICTION
# ============================================

def train_ensemble_models(X_train, y_train, X_val, y_val,
                         num_models=10, epochs=125, patience=50,
                         verbose=True):
    """
    訓練集成學習模型（多個 Genesis 模型）。

    Parameters:
    -----------
    X_train, y_train : np.ndarray
        訓練數據和標籤
    X_val, y_val : np.ndarray
        驗證數據和標籤
    num_models : int
        要訓練的模型數量
    epochs : int
        最大訓練 epochs
    patience : int
        Early stopping 的耐心值
    verbose : bool
        是否打印訓練過程

    Returns:
    --------
    models : list
        訓練好的模型列表
    histories : list
        訓練歷史列表
    """
    if verbose:
        print("\n" + "="*80)
        print("ENSEMBLE TRAINING")
        print("="*80)
        print(f"Number of models: {num_models}")
        print(f"Max epochs: {epochs}")
        print(f"Early stopping patience: {patience}")

    models = []
    histories = []

    # 定義 Early Stopping 回調
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        min_delta=0.001,
        restore_best_weights=True,
        verbose=0
    )

    # 訓練多個模型
    for i in range(num_models):
        if verbose:
            print(f"\n{'='*80}")
            print(f"Training Model {i+1}/{num_models}")
            print(f"{'='*80}")

        # 建構新模型
        model = build_genesis_model(verbose=(i==0))  # 只在第一次打印架構

        # 訓練模型
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1 if verbose else 0
        )

        # 保存模型和歷史
        models.append(model)
        histories.append(history)

        if verbose:
            best_epoch = np.argmin(history.history['val_loss']) + 1
            best_val_loss = np.min(history.history['val_loss'])
            best_val_acc = history.history['val_accuracy'][best_epoch - 1]
            print(f"\nModel {i+1} Summary:")
            print(f"  Best epoch: {best_epoch}/{len(history.history['loss'])}")
            print(f"  Best val_loss: {best_val_loss:.4f}")
            print(f"  Best val_accuracy: {best_val_acc:.4f}")

    return models, histories


def ensemble_predict(models, X_test, verbose=True):
    """
    使用集成模型進行預測。

    Parameters:
    -----------
    models : list
        訓練好的模型列表
    X_test : np.ndarray
        測試數據
    verbose : bool
        是否打印預測信息

    Returns:
    --------
    ensemble_predictions : np.ndarray
        集成預測結果 (平均概率)
    ensemble_labels : np.ndarray
        集成預測標籤
    """
    if verbose:
        print("\n" + "="*80)
        print("ENSEMBLE PREDICTION")
        print("="*80)

    predictions = []

    # 收集每個模型的預測
    for i, model in enumerate(models):
        pred = model.predict(X_test, verbose=0)
        predictions.append(pred)
        if verbose:
            print(f"Model {i+1} predictions collected")

    # 平均所有模型的預測
    ensemble_predictions = np.mean(predictions, axis=0)

    # 轉換為類別標籤
    ensemble_labels = np.argmax(ensemble_predictions, axis=1)

    if verbose:
        print(f"\nEnsemble prediction shape: {ensemble_predictions.shape}")
        print(f"Ensemble labels shape: {ensemble_labels.shape}")

    return ensemble_predictions, ensemble_labels


# ============================================
# 5. MAIN EXECUTION
# ============================================

def main():
    """主執行流程"""

    print("\n" + "="*80)
    print("STEP 1: GENERATE SIMULATED DATA")
    print("="*80)

    # 設置隨機種子以確保可重現性
    np.random.seed(42)
    tf.random.set_seed(42)

    # 生成模擬數據
    n_samples = 500  # 總樣本數
    n_features = 2001  # 光變曲線長度

    # 生成模擬光變曲線數據
    # 類別 0: 無行星 (隨機噪聲)
    # 類別 1: 有行星 (添加週期性凹陷)
    X_data = []
    y_labels = []

    for i in range(n_samples):
        # 隨機選擇類別
        has_planet = i % 2  # 交替生成兩類樣本

        # 生成基礎光變曲線（隨機噪聲）
        lightcurve = np.random.normal(loc=1.0, scale=0.001, size=n_features)

        if has_planet:
            # 添加行星凌日信號（週期性凹陷）
            transit_depth = np.random.uniform(0.002, 0.01)
            transit_width = int(np.random.uniform(50, 150))
            transit_center = n_features // 2

            # 創建凹陷
            for j in range(transit_width):
                idx = transit_center - transit_width//2 + j
                if 0 <= idx < n_features:
                    lightcurve[idx] -= transit_depth

        # 處理光變曲線
        processed_lc = process_lightcurve(lightcurve)
        X_data.append(processed_lc)

        # One-hot 編碼標籤
        y_labels.append([1, 0] if has_planet == 0 else [0, 1])

    X_data = np.array(X_data)
    y_labels = np.array(y_labels)

    print(f"Generated data shape: {X_data.shape}")
    print(f"Labels shape: {y_labels.shape}")
    print(f"Class distribution: Class 0: {np.sum(y_labels[:, 0])}, Class 1: {np.sum(y_labels[:, 1])}")

    # 分割數據集
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_data, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )

    print(f"\nTrain set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")

    # ============================================
    print("\n" + "="*80)
    print("STEP 2: DATA AUGMENTATION")
    print("="*80)

    # 數據增強
    X_train_aug, y_train_aug = augment_data(X_train, y_train)

    # 添加通道維度 (samples, timesteps, channels)
    X_train_aug = X_train_aug.reshape(-1, 2001, 1)
    X_val = X_val.reshape(-1, 2001, 1)
    X_test = X_test.reshape(-1, 2001, 1)

    print(f"\nFinal training data shape: {X_train_aug.shape}")

    # ============================================
    print("\n" + "="*80)
    print("STEP 3: ENSEMBLE TRAINING")
    print("="*80)

    # 訓練集成模型
    num_models = 10
    models, histories = train_ensemble_models(
        X_train_aug, y_train_aug,
        X_val, y_val,
        num_models=num_models,
        epochs=125,
        patience=50,
        verbose=True
    )

    # ============================================
    print("\n" + "="*80)
    print("STEP 4: ENSEMBLE PREDICTION & EVALUATION")
    print("="*80)

    # 集成預測
    ensemble_probs, ensemble_preds = ensemble_predict(models, X_test)

    # 真實標籤
    y_test_labels = np.argmax(y_test, axis=1)

    # 計算準確率
    accuracy = accuracy_score(y_test_labels, ensemble_preds)

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"\nEnsemble Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # 詳細分類報告
    print("\nClassification Report:")
    print(classification_report(
        y_test_labels,
        ensemble_preds,
        target_names=['No Planet', 'Planet']
    ))

    # 混淆矩陣
    cm = confusion_matrix(y_test_labels, ensemble_preds)
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("                 No Planet  Planet")
    print(f"Actual No Planet    {cm[0,0]:5d}     {cm[0,1]:5d}")
    print(f"       Planet       {cm[1,0]:5d}     {cm[1,1]:5d}")

    # 個別模型性能
    print("\n" + "="*80)
    print("INDIVIDUAL MODEL PERFORMANCES")
    print("="*80)

    for i, model in enumerate(models):
        pred = model.predict(X_test, verbose=0)
        pred_labels = np.argmax(pred, axis=1)
        acc = accuracy_score(y_test_labels, pred_labels)
        print(f"Model {i+1:2d}: Accuracy = {acc:.4f} ({acc*100:.2f}%)")

    print("\n" + "="*80)
    print("GENESIS MODEL TRAINING COMPLETED!")
    print("="*80)

    return models, histories, accuracy


# ============================================
# SCRIPT ENTRY POINT
# ============================================

if __name__ == "__main__":
    models, histories, accuracy = main()

    print(f"\n[OK] Successfully trained {len(models)} Genesis models")
    print(f"[OK] Final ensemble accuracy: {accuracy:.4f}")
    print("\n[OK] Script execution completed successfully!")
