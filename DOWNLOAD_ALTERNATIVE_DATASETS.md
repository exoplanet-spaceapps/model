# 替代數據集下載指南

兩個高品質的預處理 Kepler 數據集可立即使用：

---

## 🎯 方案 1: Kaggle 數據集（推薦 - 最簡單）

### **Kepler Labelled Time Series Data**
- **數量**: 5,657 筆光變曲線
- **大小**: 58.6 MB
- **來源**: https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data

### **下載步驟：**

#### 選項 A - 使用 Kaggle API（自動化）

```bash
# 1. 安裝 Kaggle CLI
pip install kaggle

# 2. 設定 Kaggle API Token
# 訪問 https://www.kaggle.com/settings
# 點擊 "Create New API Token" 下載 kaggle.json
# 將 kaggle.json 放到 ~/.kaggle/ (Linux/Mac) 或 %USERPROFILE%\.kaggle\ (Windows)

# 3. 下載數據集
kaggle datasets download -d keplersmachines/kepler-labelled-time-series-data

# 4. 解壓縮
unzip kepler-labelled-time-series-data.zip -d data/kaggle_kepler/
```

#### 選項 B - 手動下載（無需 API）

1. 訪問：https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data
2. 點擊 "Download" 按鈕（需要登入 Kaggle）
3. 下載 `kepler-labelled-time-series-data.zip`
4. 解壓縮到 `data/kaggle_kepler/`

### **數據格式：**
- **訓練集**: `exoTrain.csv` (5,087 rows × 3,198 columns)
- **測試集**: `exoTest.csv` (570 rows × 3,198 columns)
- **標籤**:
  - `2` = 確認系外行星 (42 筆)
  - `1` = 非系外行星 (5,615 筆)

---

## 🔬 方案 2: Mendeley 2024 數據集（最新、學術級）

### **Dataset_Machine_Learning_Exoplanets_2024**
- **數量**: 5,302 筆光變曲線
- **每條數據點**: ~60,000 點
- **發布**: 2024年7月
- **DOI**: 10.17632/wctcv34962.3
- **來源**: https://data.mendeley.com/datasets/wctcv34962/3

### **特點：**
- ✅ 使用 Lightkurve 提取
- ✅ PDCSAP flux（最適合系外行星偵測）
- ✅ 已標準化
- ✅ 線性插值填補缺失值
- ✅ 2 標準差離群值移除
- ✅ LightGBM 訓練達 82.92% 準確率

### **下載步驟：**

#### 選項 A - 使用 Mendeley API（需要帳號）

```bash
# 1. 訪問 Mendeley Data 並註冊帳號
# https://data.mendeley.com/

# 2. 訪問數據集頁面
# https://data.mendeley.com/datasets/wctcv34962/3

# 3. 點擊 "Download All" 按鈕

# 4. 解壓縮到專案目錄
unzip mendeley-dataset.zip -d data/mendeley_kepler/
```

#### 選項 B - 使用 wget/curl（如果有直接連結）

```bash
# Mendeley 需要認證，需先在網頁下載
# 下載後手動解壓縮
```

### **引用：**
```
Macedo, B. H. D., & Zalewski, W. (2024).
Dataset_Machine_Learning_Exoplanets_2024 (Version 3) [Data set].
Mendeley Data. https://doi.org/10.17632/wctcv34962.3
```

---

## 📦 方案 3: Kaggle - Kepler & TESS 數據（2025 最新）

### **來源**
https://www.kaggle.com/datasets/vijayveersingh/kepler-and-tess-exoplanet-data

### **下載：**
```bash
kaggle datasets download -d vijayveersingh/kepler-and-tess-exoplanet-data
unzip kepler-and-tess-exoplanet-data.zip -d data/kepler_tess/
```

---

## 🚀 快速開始（推薦流程）

### **步驟 1: 安裝 Kaggle CLI**
```bash
pip install kaggle
```

### **步驟 2: 設定 Kaggle API Token**
1. 訪問 https://www.kaggle.com/settings
2. 點擊 "Create New API Token"
3. 下載 `kaggle.json`
4. 移動到正確位置：
   ```bash
   # Windows
   mkdir %USERPROFILE%\.kaggle
   move kaggle.json %USERPROFILE%\.kaggle\

   # Linux/Mac
   mkdir -p ~/.kaggle
   mv kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

### **步驟 3: 下載數據集**
```bash
cd C:\Users\thc1006\Desktop\NASA\model

# 下載 Kepler 時間序列數據
kaggle datasets download -d keplersmachines/kepler-labelled-time-series-data

# 解壓縮
powershell Expand-Archive kepler-labelled-time-series-data.zip -DestinationPath data/kaggle_kepler/
```

### **步驟 4: 載入數據使用**
```python
import pandas as pd
import numpy as np

# 載入訓練數據
train_df = pd.read_csv('data/kaggle_kepler/exoTrain.csv')
test_df = pd.read_csv('data/kaggle_kepler/exoTest.csv')

# 分離特徵和標籤
X_train = train_df.iloc[:, 1:].values  # 3197 個時間點
y_train = train_df.iloc[:, 0].values   # 標籤 (1 或 2)

X_test = test_df.iloc[:, 1:].values
y_test = test_df.iloc[:, 0].values

print(f"訓練集: {X_train.shape}")
print(f"測試集: {X_test.shape}")
print(f"確認行星數: {np.sum(y_train == 2)}")
print(f"非行星數: {np.sum(y_train == 1)}")
```

---

## 📊 數據集比較

| 數據集 | 樣本數 | 大小 | 預處理 | 下載難度 | 推薦度 |
|--------|--------|------|--------|----------|--------|
| Kaggle 時間序列 | 5,657 | 58.6 MB | ✅ | 低 | ⭐⭐⭐⭐⭐ |
| Mendeley 2024 | 5,302 | 未知 | ✅ | 中 | ⭐⭐⭐⭐⭐ |
| Kaggle Kepler & TESS | 未知 | 未知 | ✅ | 低 | ⭐⭐⭐⭐ |
| AWS S3 下載（當前） | 400 | 實時 | ❌ | 高（慢） | ⭐⭐⭐ |

---

## ⚡ 立即執行腳本

已為您準備好下載腳本：

```bash
# 快速下載 Kaggle 數據集（需先設定 API token）
bash scripts/download_kaggle_dataset.sh
```

---

**生成時間**: 2025-10-05
**當前 AWS S3 下載**: 持續進行中（32/400）
**建議**: 使用 Kaggle 數據集作為主要數據來源
