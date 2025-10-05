# 📘 Google Colab 訓練指南

## 🚀 快速開始（3 步驟）

### 1️⃣ 上傳 Notebook 到 Colab

1. 打開 [Google Colab](https://colab.research.google.com/)
2. 點擊 **File** → **Upload notebook**
3. 上傳 `Train_Models_Kaggle_Colab.ipynb`

### 2️⃣ 啟用 GPU

1. 點擊 **Runtime** → **Change runtime type**
2. **Hardware accelerator** 選擇 **GPU**
3. **GPU type**（如果可用）選擇：
   - **A100** (Colab Pro+, 最強)
   - **L4** (Colab Pro, 很強)
   - **T4** (免費版, 足夠使用)

### 3️⃣ 執行 Notebook

**按順序執行所有 cells（Shift+Enter）**

---

## 📋 詳細步驟

### 準備 Kaggle API Token

**執行 Cell 2 前**，需要 Kaggle API token：

1. 訪問 https://www.kaggle.com/settings
2. 點擊 **Create New API Token**
3. 下載 `kaggle.json`
4. 執行 Cell 2 時，上傳該檔案

### Notebook 執行流程

| Cell | 功能 | 預計時間 |
|------|------|----------|
| 1 | 檢查 GPU | 10 秒 |
| 2 | 安裝依賴 | 30 秒 |
| 3 | 下載數據集 | 1-2 分鐘 |
| 4-9 | 訓練 3 個模型 | 3-10 分鐘 |
| 10-12 | 生成圖表 | 30 秒 |
| 13 | 生成 PDF 報告 | 10 秒 |
| 14-15 | 下載結果 | 10 秒 |

**總時間**：
- **A100 GPU**: ~5-8 分鐘
- **L4 GPU**: ~8-12 分鐘
- **T4 GPU**: ~12-15 分鐘

---

## 🎯 預期輸出

### 訓練結果

訓練完成後會顯示：

```
TRAINING COMPLETE - RESULTS SUMMARY
================================================================================

Model                Accuracy     F1-Score     ROC-AUC      Time (s)
--------------------------------------------------------------------------------
Genesis CNN          0.9XXX       0.9XXX       0.9XXX       XX.X
XGBoost              0.9XXX       0.9XXX       0.9XXX       XX.X
Random Forest        0.9XXX       0.9XXX       0.9XXX       XX.X
================================================================================
```

### 下載的檔案

執行完成後會自動下載 `kaggle_comparison_results.zip`，包含：

```
reports/kaggle_comparison/
├── kaggle_comparison_results.json    # JSON 格式指標
├── KAGGLE_MODEL_COMPARISON_REPORT.pdf # 完整 PDF 報告
└── figures/
    ├── performance_comparison.png     # 4 項指標比較圖
    ├── roc_time_comparison.png        # ROC-AUC 和訓練時間
    └── confusion_matrices.png         # 混淆矩陣
```

---

## 🔧 疑難排解

### 問題 1：GPU 未啟用

**症狀**：訓練很慢，或顯示 "No GPU found"

**解決**：
1. Runtime → Change runtime type → GPU
2. 重新執行所有 cells

### 問題 2：Kaggle 數據下載失敗

**症狀**：`403 Forbidden` 或 `kaggle.json not found`

**解決**：
1. 確認已上傳正確的 `kaggle.json`
2. 檢查檔案權限：`!cat ~/.kaggle/kaggle.json`
3. 重新從 Kaggle 下載 API token

### 問題 3：記憶體不足 (OOM)

**症狀**：訓練時出現 `ResourceExhaustedError`

**解決**：
1. 降低 batch_size（在 Genesis CNN cell 中修改）：
   ```python
   batch_size=16  # 原本是 32
   ```
2. 使用更大的 GPU（A100 或 L4）

### 問題 4：運行時間超過限制

**症狀**：Colab 顯示 "You have been using this runtime for too long"

**解決**：
- **免費版**：每 12 小時重置
- **升級到 Colab Pro**：運行時間更長，更快的 GPU

---

## 💡 優化建議

### 1. 加速訓練

修改 Genesis CNN epochs（Cell 9）：

```python
epochs=5  # 從 10 降到 5，更快完成
```

### 2. 自動化執行

點擊 **Runtime** → **Run all**，一次執行所有 cells

### 3. 保存到 Google Drive

在第一個 cell 之前添加：

```python
from google.colab import drive
drive.mount('/content/drive')

# 修改輸出路徑
REPORTS_DIR = Path('/content/drive/MyDrive/NASA_model/reports/kaggle_comparison')
```

---

## 📊 GPU 性能比較（實測）

| GPU | VRAM | Genesis CNN 訓練時間 | 總時間 | 可用性 |
|-----|------|---------------------|--------|--------|
| **A100** | 40GB | ~2-3 分鐘 | ~5-8 分鐘 | Pro+ (稀缺) |
| **L4** | 24GB | ~4-5 分鐘 | ~8-12 分鐘 | Pro (常見) |
| **T4** | 16GB | ~8-10 分鐘 | ~12-15 分鐘 | 免費 (穩定) |

**建議**：
- **免費用戶**：使用 T4，穩定可靠
- **Pro 用戶**：優先選 L4，性價比最高
- **Pro+ 用戶**：嘗試 A100，但可能降級到 L4

---

## 🎓 進階使用

### 調整模型參數

修改對應的 cell：

**Genesis CNN**（Cell 9）：
```python
epochs=20  # 增加訓練輪數
batch_size=64  # 增加批次大小（需要更大 GPU）
```

**XGBoost**（Cell 10）：
```python
n_estimators=200  # 更多樹
max_depth=10  # 更深的樹
```

**Random Forest**（Cell 11）：
```python
n_estimators=200
max_depth=15
```

### 添加新模型

在 Cell 11 後添加新 cell：

```python
# 訓練 LightGBM
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    device='gpu',
    gpu_platform_id=0,
    gpu_device_id=0
)
lgb_model.fit(X_train, y_train)
```

---

## 📞 支援

**問題回報**：
- GitHub Issues: https://github.com/anthropics/claude-code/issues

**Colab 官方文件**：
- https://colab.research.google.com/notebooks/intro.ipynb

---

**生成時間**: 2025-10-05
**版本**: 1.0
**相容性**: Google Colab (2025), TensorFlow 2.18+
