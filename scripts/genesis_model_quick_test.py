"""
Genesis Model - Quick Test Script
==================================
A simplified version for quick testing with fewer epochs and models.

Usage:
    python scripts/genesis_model_quick_test.py
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("Genesis Model - Quick Test (GPU Optimized)")
print("="*60)

# GPU 配置
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 啟用記憶體增長
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[OK] GPU Found: {len(gpus)} device(s)")
        print(f"  {gpus[0].name}")

        # 啟用混合精度訓練（GPU 加速）
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("[OK] Mixed precision (FP16) enabled for GPU acceleration")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("[WARN] No GPU found, using CPU")

# 設置隨機種子
np.random.seed(42)
tf.random.set_seed(42)

# ============================================
# 1. 簡化的數據處理函數
# ============================================

def process_lightcurve(lightcurve_data):
    """處理光變曲線數據到 2001 個點"""
    target_length = 2001

    if len(lightcurve_data) != target_length:
        original_indices = np.linspace(0, len(lightcurve_data) - 1, len(lightcurve_data))
        target_indices = np.linspace(0, len(lightcurve_data) - 1, target_length)
        processed_data = np.interp(target_indices, original_indices, lightcurve_data)
    else:
        processed_data = lightcurve_data.copy()

    # 標準化
    mean = np.mean(processed_data)
    std = np.std(processed_data)
    if std > 0:
        processed_data = (processed_data - mean) / std

    return processed_data


# ============================================
# 2. 簡化的數據增強函數
# ============================================

def augment_data(X_train, y_train):
    """數據增強：水平翻轉 + 高斯噪聲"""
    X_list = [X_train]
    y_list = [y_train]

    # 水平翻轉
    X_list.append(np.flip(X_train, axis=1))
    y_list.append(y_train)

    # 高斯噪聲（2個副本而不是4個，加快速度）
    data_std = np.std(X_train)
    for i in range(2):
        noise = np.random.normal(0, data_std, X_train.shape)
        X_list.append(X_train + noise)
        y_list.append(y_train)

    return np.vstack(X_list), np.vstack(y_list)


# ============================================
# 3. Genesis 模型建構
# ============================================

def build_genesis_model():
    """建構 Genesis 模型（GPU 優化版本）"""
    model = models.Sequential([
        layers.Input(shape=(2001, 1)),

        # Conv Block 1 - 使用 padding='same' 保持維度
        layers.Conv1D(64, 50, padding='same', activation='relu', kernel_initializer='glorot_uniform'),
        layers.Conv1D(64, 50, padding='same', activation='relu', kernel_initializer='glorot_uniform'),
        layers.MaxPooling1D(32, strides=32),

        # Conv Block 2 - 使用 padding='same'
        layers.Conv1D(64, 12, padding='same', activation='relu', kernel_initializer='glorot_uniform'),
        layers.Conv1D(64, 12, padding='same', activation='relu', kernel_initializer='glorot_uniform'),
        layers.AveragePooling1D(8),  # 修正 pool size

        # Dense Block
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, activation='relu', kernel_initializer='glorot_uniform'),
        layers.Dense(256, activation='relu', kernel_initializer='glorot_uniform'),
        layers.Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ============================================
# 4. 主函數
# ============================================

def main():
    # 生成模擬數據
    print("\n[1/4] Generating simulated data...")
    n_samples = 200  # 減少樣本數
    X_data = []
    y_labels = []

    for i in range(n_samples):
        has_planet = i % 2
        lightcurve = np.random.normal(1.0, 0.001, 2001)

        if has_planet:
            transit_depth = np.random.uniform(0.002, 0.01)
            transit_width = 80
            transit_center = 1000
            for j in range(transit_width):
                idx = transit_center - transit_width//2 + j
                if 0 <= idx < 2001:
                    lightcurve[idx] -= transit_depth

        X_data.append(process_lightcurve(lightcurve))
        y_labels.append([1, 0] if has_planet == 0 else [0, 1])

    X_data = np.array(X_data)
    y_labels = np.array(y_labels)

    print(f"   Data shape: {X_data.shape}")

    # 分割數據
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_labels, test_size=0.2, random_state=42
    )

    # 數據增強
    print("\n[2/4] Augmenting data...")
    X_train_aug, y_train_aug = augment_data(X_train, y_train)
    X_train_aug = X_train_aug.reshape(-1, 2001, 1)
    X_test = X_test.reshape(-1, 2001, 1)
    print(f"   Augmented shape: {X_train_aug.shape}")

    # 訓練集成模型（3個模型而不是10個）
    print("\n[3/4] Training ensemble (3 models)...")
    num_models = 3
    models = []

    for i in range(num_models):
        print(f"\n   Training model {i+1}/{num_models}...")
        model = build_genesis_model()

        early_stopping = EarlyStopping(
            monitor='loss',
            patience=10,
            min_delta=0.001,
            restore_best_weights=True
        )

        model.fit(
            X_train_aug, y_train_aug,
            epochs=30,  # 減少到 30 epochs
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )

        models.append(model)
        print(f"   Model {i+1} trained successfully")

    # 集成預測
    print("\n[4/4] Ensemble prediction...")
    predictions = []
    for model in models:
        pred = model.predict(X_test, verbose=0)
        predictions.append(pred)

    ensemble_pred = np.mean(predictions, axis=0)
    ensemble_labels = np.argmax(ensemble_pred, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)

    # 評估
    accuracy = accuracy_score(y_test_labels, ensemble_labels)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nEnsemble Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(
        y_test_labels,
        ensemble_labels,
        target_names=['No Planet', 'Planet']
    ))

    print("\n[OK] Quick test completed successfully!")

    return models, accuracy


if __name__ == "__main__":
    models, accuracy = main()
