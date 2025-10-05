# Kaggle Kepler 數據集下載完成報告

## ✅ 下載成功！

**下載時間**: 2025-10-05
**數據來源**: Kaggle - Kepler Labelled Time Series Data
**下載速度**: 55.9 MB / 數秒鐘

---

## 📊 數據集摘要

### **檔案位置**
```
data/kaggle_kepler/
├── exoTrain.csv  (251 MB)
└── exoTest.csv   (28 MB)
```

### **數據統計**

| 類別 | 訓練集 | 測試集 | 總計 | 比例 |
|------|--------|--------|------|------|
| **總樣本** | 5,087 | 570 | **5,657** | 100% |
| **確認行星** | 37 | 5 | **42** | 0.7% |
| **非行星** | 5,050 | 565 | **5,615** | 99.3% |

### **數據格式**

- **特徵數**: 3,197 個時間點（光變曲線）
- **標籤**:
  - `1` = 非系外行星
  - `2` = 確認系外行星
- **每筆數據**: 1 個標籤 + 3,197 個 flux 值

---

## 🚀 使用方式

### **載入數據（Python）**

```python
import pandas as pd
import numpy as np

# 載入訓練和測試數據
train = pd.read_csv('data/kaggle_kepler/exoTrain.csv')
test = pd.read_csv('data/kaggle_kepler/exoTest.csv')

# 分離特徵和標籤
X_train = train.iloc[:, 1:].values  # 3197 個 flux 值
y_train = train.iloc[:, 0].values   # 標籤 (1 或 2)

X_test = test.iloc[:, 1:].values
y_test = test.iloc[:, 0].values

# 轉換標籤 (2→1 行星, 1→0 非行星)
y_train_binary = (y_train == 2).astype(int)
y_test_binary = (y_test == 2).astype(int)

print(f"訓練集: {X_train.shape}")
print(f"測試集: {X_test.shape}")
print(f"行星數: {y_train_binary.sum()} (訓練) + {y_test_binary.sum()} (測試)")
```

### **預處理建議**

```python
from sklearn.preprocessing import StandardScaler

# 標準化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 或重塑為 CNN 輸入
X_train_cnn = X_train.reshape(-1, 3197, 1)
X_test_cnn = X_test.reshape(-1, 3197, 1)
```

---

## 🎯 與其他數據集比較

| 數據集 | 樣本數 | 時間點 | 行星比例 | 狀態 |
|--------|--------|--------|----------|------|
| **Kaggle** | 5,657 | 3,197 | 0.7% | ✅ 已下載 |
| AWS S3 下載 | 32/400 | 2,001 | ~50% | 🔄 進行中 |
| Mendeley 2024 | 5,302 | ~60,000 | - | ⏳ 待下載 |

---

## 📈 訓練建議

### **類別不平衡處理**

由於行星樣本僅佔 0.7%，建議：

1. **過採樣** (SMOTE)
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train_binary)
```

2. **類別權重**
```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced',
                                      classes=np.unique(y_train_binary),
                                      y=y_train_binary)
```

3. **焦點損失** (Focal Loss)
```python
# 適合深度學習模型
focal_loss = tfa.losses.SigmoidFocalCrossEntropy()
```

### **模型選擇**

**傳統機器學習**:
- XGBoost (scale_pos_weight 參數)
- Random Forest (class_weight='balanced')
- SVM (class_weight='balanced')

**深度學習**:
- 1D CNN (適合時間序列)
- LSTM/GRU (適合序列模式)
- Transformer (適合長序列)

---

## 🔬 下一步行動

### **立即可執行：**

1. **訓練 Genesis 模型** (使用 Kaggle 數據)
```bash
python scripts/genesis_train_kaggle_dataset.py
```

2. **模型比較**
- Kaggle 數據集 (5,657 筆)
- AWS S3 數據 (持續下載中)
- 結合兩者進行集成學習

3. **數據增強**
```python
# 水平翻轉
X_augmented = np.flip(X_train, axis=1)

# 高斯噪音
noise = np.random.normal(0, 0.1 * X_train.std(), X_train.shape)
X_noisy = X_train + noise
```

---

## 📝 引用

**數據集來源**:
```
Kepler Labelled Time Series Data
https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data
License: CC0 1.0 Universal (Public Domain)
```

---

## ✨ 成果總結

✅ **Kaggle 數據集**: 5,657 筆 (已完成)
🔄 **AWS S3 下載**: 32/400 筆 (背景運行中)
⏳ **Mendeley 2024**: 5,302 筆 (待下載)

**總可用數據**: 5,657+ 筆光變曲線
**足夠用於**: 深度學習模型訓練、模型比較、論文研究

---

**報告生成時間**: 2025-10-05
**狀態**: ✅ 數據集已就緒，可開始訓練
