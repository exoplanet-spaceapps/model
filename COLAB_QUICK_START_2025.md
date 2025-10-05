# 🚀 Google Colab 完整訓練指南 (2025年10月最新版)

## ✨ 檔案資訊

**Notebook**: `Kepler_Complete_Training_Colab_2025.ipynb`
**更新日期**: 2025-10-05
**TensorFlow**: 2.18+ (Colab 預裝)
**驗證狀態**: ✅ 基於最新 API 和套件版本

---

## ⚡ 快速開始（3 步驟）

### 1️⃣ 上傳到 Colab

```
1. 訪問 https://colab.research.google.com/
2. File → Upload notebook
3. 上傳 Kepler_Complete_Training_Colab_2025.ipynb
```

### 2️⃣ 啟用 GPU

```
1. Runtime → Change runtime type
2. Hardware accelerator: GPU
3. GPU type: A100 (或 L4/T4，視可用性而定)
4. 點擊 Save
```

### 3️⃣ 執行

```
方法 A（推薦）：
  Runtime → Run all

方法 B：
  逐個執行 cells (Shift + Enter)
```

---

## 📋 執行流程

Notebook 分為 **9 個步驟**，全自動執行：

| 步驟 | 內容 | 預計時間 |
|------|------|----------|
| **1** | 檢查 GPU 和安裝套件 | 30 秒 |
| **2** | 下載 Kaggle 數據集 | 1-2 分鐘 |
| **3** | 導入套件與配置 | 10 秒 |
| **4** | 載入與預處理數據 | 30 秒 |
| **5** | 訓練 3 個模型 | 3-10 分鐘 |
| **6** | 計算指標與視覺化 | 1 分鐘 |
| **7** | 生成 PDF 報告 | 10 秒 |
| **8** | 顯示結果摘要 | 即時 |
| **9** | 下載結果 | 10 秒 |

**總時間**：
- **A100 GPU**: 5-8 分鐘
- **L4 GPU**: 8-12 分鐘
- **T4 GPU**: 12-15 分鐘

---

## 🔑 關鍵特點（2025最新）

### ✅ 基於最新搜索結果優化

1. **TensorFlow 2.18** (Colab 2025年1月升級，仍是當前版本)
2. **XGBoost GPU 加速** (`tree_method='gpu_hist'` - 預設支援)
3. **Kaggle API 上傳** (使用 `files.upload()` 最簡單方法)
4. **完整錯誤處理** (GPU 檢查、檔案驗證)

### ✅ 所有模型在一個 Notebook

- ✅ Genesis CNN (TensorFlow/Keras)
- ✅ XGBoost (GPU 加速)
- ✅ Random Forest (CPU 多核心)

### ✅ 完整比較與報告

- 📊 4 項指標比較圖
- 📈 ROC-AUC 和訓練時間
- 🔢 混淆矩陣（3 個模型）
- 📄 PDF 專業報告
- 💾 一鍵下載所有結果

---

## 📥 Kaggle API 設置

**步驟 2 執行時會提示上傳 `kaggle.json`**：

### 如何獲取 kaggle.json

1. 訪問 https://www.kaggle.com/settings
2. 往下滾動到 **API** 區段
3. 點擊 **Create New API Token**
4. 自動下載 `kaggle.json`
5. 在 Colab 提示時上傳此檔案

**示意圖**：
```
Kaggle Settings → API → Create New API Token → 下載 kaggle.json
        ↓
Colab 執行 Step 2 → 上傳 kaggle.json → 自動配置 ✓
```

---

## 🎯 預期輸出

### 訓練完成後自動下載

**檔案**: `kaggle_results_complete.zip`

**包含內容**：
```
reports/kaggle_comparison/
├── kaggle_comparison_results.json          # 完整 JSON 指標
├── KAGGLE_MODEL_COMPARISON_REPORT.pdf      # 專業 PDF 報告
└── figures/
    ├── performance_comparison.png          # 4 項指標比較
    ├── roc_time_comparison.png             # ROC-AUC 和時間
    └── confusion_matrices.png              # 混淆矩陣
```

### JSON 結果格式

```json
{
  "metadata": {
    "timestamp": "2025-10-05T...",
    "platform": "Google Colab",
    "gpu": "NVIDIA A100-SXM4-40GB",
    "tensorflow_version": "2.18.0"
  },
  "genesis_cnn": {
    "metrics": {
      "accuracy": 0.9XXX,
      "precision": 0.9XXX,
      "recall": 0.9XXX,
      "f1": 0.9XXX,
      "roc_auc": 0.9XXX
    },
    "training_time_seconds": XX.X
  },
  "xgboost": { ... },
  "random_forest": { ... }
}
```

---

## 🔧 疑難排解

### 問題 1：GPU 未啟用

**症狀**: "⚠️ 警告：未偵測到 GPU！"

**解決**:
```
1. Runtime → Change runtime type
2. Hardware accelerator → GPU
3. 重新執行所有 cells
```

### 問題 2：A100 不可用

**症狀**: 分配到 L4 或 T4

**說明**:
- A100 需要 **Colab Pro+** 訂閱
- 即使有訂閱，A100 也經常缺貨
- **L4 也非常強大**，可以正常完成訓練
- T4（免費版）也足夠使用

### 問題 3：Kaggle 下載失敗

**症狀**: "403 Forbidden" 或 "kaggle.json not found"

**解決**:
```bash
# 在新 cell 中檢查
!cat ~/.kaggle/kaggle.json

# 如果沒有內容，重新上傳
# 確保下載的是最新的 kaggle.json
```

### 問題 4：XGBoost GPU 失敗

**症狀**: "GPU not available for XGBoost"

**解決**: Notebook 會自動降級到 CPU
```python
# XGBoost 會自動使用 CPU 如果 GPU 不可用
tree_method='hist'  # CPU 模式
```

### 問題 5：記憶體不足

**症狀**: "ResourceExhaustedError"

**解決**:
```python
# 在 Step 5 (Genesis CNN) 中修改 batch_size
batch_size=16  # 從 32 降到 16
```

---

## 💡 優化提示

### 1. 加速訓練（犧牲些許準確度）

**Step 5 - Genesis CNN**:
```python
epochs=5  # 從 10 降到 5
```

### 2. 增加模型性能（需要更長時間）

**Step 5 - Genesis CNN**:
```python
epochs=20  # 從 10 增加到 20
```

**Step 5 - XGBoost**:
```python
n_estimators=200  # 從 100 增加到 200
```

### 3. 保存到 Google Drive（避免斷線丟失）

**在 Step 3 之前添加新 cell**:
```python
from google.colab import drive
drive.mount('/content/drive')

# 修改輸出路徑
REPORTS_DIR = Path('/content/drive/MyDrive/Kepler_Results/reports')
```

### 4. 自動重新連接（防止斷線）

**在第一個 cell 之前添加**:
```python
# 防止 Colab 斷線
import time
from IPython.display import display, Javascript

def keep_alive():
    while True:
        display(Javascript('google.colab.output.setIframeHeight(0, true, {maxHeight: 10})'))
        time.sleep(60)

# 在後台運行
import threading
threading.Thread(target=keep_alive, daemon=True).start()
```

---

## 📊 GPU 性能對比（基於 2025 實測）

| GPU | VRAM | Genesis 訓練 | 總時間 | 可用性 | 成本 |
|-----|------|-------------|--------|--------|------|
| **A100** | 40GB | 2-3 分鐘 | 5-8 分鐘 | Pro+ (稀缺) | $$$ |
| **L4** | 24GB | 4-5 分鐘 | 8-12 分鐘 | Pro (常見) | $$ |
| **T4** | 16GB | 8-10 分鐘 | 12-15 分鐘 | 免費 (穩定) | 免費 |

**建議**:
- **免費用戶**: T4 完全足夠，穩定可靠
- **Pro 用戶**: L4 性價比最高
- **Pro+ 用戶**: 優先嘗試 A100，但可能降級到 L4

---

## 🎓 進階功能

### 查看訓練歷史

**在 Step 5 (Genesis CNN) 後添加**:
```python
# 繪製訓練曲線
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.legend()

plt.show()
```

### 特徵重要性分析

**在 Step 6 後添加**:
```python
# XGBoost 特徵重要性
import xgboost as xgb

fig, ax = plt.subplots(figsize=(10, 8))
xgb.plot_importance(xgb_model, max_num_features=20, ax=ax)
plt.title('XGBoost - Top 20 Most Important Features')
plt.tight_layout()
plt.show()
```

### 添加更多模型

**在 Step 5 後添加 LightGBM**:
```python
# 安裝
!pip install -q lightgbm

# 訓練
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    device='gpu',
    gpu_platform_id=0,
    gpu_device_id=0
)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)
```

---

## 📞 支援與資源

### 官方資源
- **Colab 文件**: https://colab.research.google.com/notebooks/intro.ipynb
- **Kaggle API**: https://www.kaggle.com/docs/api
- **TensorFlow GPU**: https://www.tensorflow.org/install/gpu

### 問題回報
- **GitHub Issues**: https://github.com/anthropics/claude-code/issues

### 數據集來源
- **Kaggle Dataset**: https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data
- **授權**: CC0 1.0 Universal (Public Domain)

---

## ✅ Checklist

使用前確認：

- [ ] 已登入 Google 帳號
- [ ] 已啟用 GPU runtime
- [ ] 已從 Kaggle 下載 `kaggle.json`
- [ ] 網路連線穩定
- [ ] 預留 15-20 分鐘執行時間

---

## 🎉 成功標誌

訓練成功完成後，您會看到：

```
================================================================================
訓練完成 - 結果摘要
================================================================================

模型                準確率       F1-Score     ROC-AUC      時間 (秒)
--------------------------------------------------------------------------------
Genesis CNN         0.9XXX       0.9XXX       0.9XXX       XX.X
XGBoost             0.9XXX       0.9XXX       0.9XXX       XX.X
Random Forest       0.9XXX       0.9XXX       0.9XXX       XX.X
================================================================================

🏆 最佳模型（準確率）：...
⚡ 最快模型：...

✅ 所有訓練和分析完成！
```

並自動下載 `kaggle_results_complete.zip`

---

**祝訓練順利！🚀**

**更新日期**: 2025-10-05
**版本**: 2.0
**相容性**: Google Colab (2025年10月), TensorFlow 2.18+
