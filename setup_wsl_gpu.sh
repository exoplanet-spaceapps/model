#!/bin/bash
# WSL2 GPU 訓練環境自動設置腳本

echo "========================================"
echo "WSL2 GPU 訓練環境設置"
echo "========================================"
echo ""

# 檢查 GPU
echo "[1/6] 檢查 GPU 可用性..."
nvidia-smi | grep "GeForce RTX"
if [ $? -eq 0 ]; then
    echo "✓ GPU 偵測成功"
else
    echo "✗ GPU 未偵測到"
    exit 1
fi
echo ""

# 安裝系統依賴
echo "[2/6] 安裝 Python 虛擬環境支援..."
sudo apt update
sudo apt install -y python3.12-venv python3-pip
echo ""

# 切換到項目目錄
echo "[3/6] 切換到項目目錄..."
cd /mnt/c/Users/thc1006/Desktop/NASA/model
pwd
echo ""

# 創建虛擬環境
echo "[4/6] 創建 Python 虛擬環境..."
rm -rf wsl_venv
python3 -m venv wsl_venv
source wsl_venv/bin/activate
echo "✓ 虛擬環境已啟用"
echo ""

# 安裝依賴
echo "[5/6] 安裝 TensorFlow GPU 及相關套件..."
pip install --upgrade pip
pip install tensorflow pandas scikit-learn xgboost imbalanced-learn reportlab seaborn matplotlib
echo ""

# 驗證 GPU
echo "[6/6] 驗證 TensorFlow GPU..."
python3 << 'PYEOF'
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print("GPU devices:", gpus)
if len(gpus) > 0:
    print("✓ GPU 可用！")
else:
    print("✗ GPU 未偵測到")
PYEOF
echo ""

echo "========================================"
echo "設置完成！"
echo "========================================"
echo ""
echo "啟動訓練："
echo "  source wsl_venv/bin/activate"
echo "  python3 -u scripts/train_all_models_kaggle.py > training_log_wsl.txt 2>&1 &"
echo "  tail -f training_log_wsl.txt"
echo ""
