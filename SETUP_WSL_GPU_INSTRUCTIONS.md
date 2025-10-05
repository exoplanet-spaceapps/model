# WSL2 GPU 訓練設置指南

## 方法 1：自動化腳本（推薦）

### 步驟：

1. **打開 WSL Ubuntu 終端**
   - 按 `Win + R`，輸入 `wsl -d Ubuntu`，按 Enter
   - 或從開始菜單搜索 "Ubuntu"

2. **運行設置腳本**
   ```bash
   cd /mnt/c/Users/thc1006/Desktop/NASA/model
   chmod +x setup_wsl_gpu.sh
   ./setup_wsl_gpu.sh
   ```

3. **啟動訓練**（設置完成後）
   ```bash
   source wsl_venv/bin/activate
   python3 -u scripts/train_all_models_kaggle.py > training_log_wsl.txt 2>&1 &
   tail -f training_log_wsl.txt
   ```

---

## 方法 2：手動逐步設置

### 1. 打開 WSL Ubuntu 終端
```bash
wsl -d Ubuntu
```

### 2. 安裝必要套件
```bash
sudo apt update
sudo apt install -y python3.12-venv python3-pip
```

### 3. 切換到項目目錄
```bash
cd /mnt/c/Users/thc1006/Desktop/NASA/model
```

### 4. 創建虛擬環境
```bash
python3 -m venv wsl_venv
source wsl_venv/bin/activate
```

### 5. 安裝依賴
```bash
pip install --upgrade pip
pip install tensorflow pandas scikit-learn xgboost imbalanced-learn reportlab seaborn matplotlib
```

### 6. 驗證 GPU
```bash
python3 -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

應該看到：
```
GPU: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### 7. 啟動訓練
```bash
python3 -u scripts/train_all_models_kaggle.py > training_log_wsl.txt 2>&1 &
```

### 8. 監控進度
```bash
tail -f training_log_wsl.txt
```

按 `Ctrl+C` 停止監控（訓練會繼續在背景運行）

---

## 預期訓練時間

- **GPU (RTX 3050)**: ~5-10 分鐘
- **CPU**: ~25-30 分鐘

---

## 疑難排解

### 問題：GPU 未偵測到
```bash
# 檢查 NVIDIA 驅動
nvidia-smi

# 重新安裝 TensorFlow
pip install --upgrade --force-reinstall tensorflow
```

### 問題：虛擬環境創建失敗
```bash
# 安裝完整 Python 環境
sudo apt install -y python3-full python3-venv
```

---

## 訓練完成後

查看結果：
```bash
ls -lh reports/kaggle_comparison/
cat reports/kaggle_comparison/kaggle_comparison_results.json
```

檔案位置：
- **JSON 結果**: `reports/kaggle_comparison/kaggle_comparison_results.json`
- **比較圖表**: `reports/kaggle_comparison/figures/*.png`
- **PDF 報告**: `reports/kaggle_comparison/KAGGLE_MODEL_COMPARISON_REPORT.pdf`

---

**生成時間**: 2025-10-05
