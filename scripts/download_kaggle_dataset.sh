#!/bin/bash
# Kaggle Kepler 數據集自動下載腳本

echo "======================================================================"
echo "Kaggle Kepler 數據集下載腳本"
echo "======================================================================"
echo ""

# 檢查 Kaggle CLI
if ! command -v kaggle &> /dev/null; then
    echo "[錯誤] Kaggle CLI 未安裝"
    echo ""
    echo "請執行以下指令安裝："
    echo "  pip install kaggle"
    echo ""
    exit 1
fi

# 檢查 API token
if [ ! -f ~/.kaggle/kaggle.json ] && [ ! -f %USERPROFILE%\.kaggle\kaggle.json ]; then
    echo "[錯誤] Kaggle API token 未設定"
    echo ""
    echo "設定步驟："
    echo "1. 訪問 https://www.kaggle.com/settings"
    echo "2. 點擊 'Create New API Token'"
    echo "3. 下載 kaggle.json"
    echo "4. 移動到："
    echo "   - Windows: %USERPROFILE%\.kaggle\"
    echo "   - Linux/Mac: ~/.kaggle/"
    echo ""
    exit 1
fi

echo "[步驟 1/3] 下載數據集..."
kaggle datasets download -d keplersmachines/kepler-labelled-time-series-data

if [ $? -ne 0 ]; then
    echo "[錯誤] 下載失敗"
    exit 1
fi

echo ""
echo "[步驟 2/3] 創建目錄..."
mkdir -p data/kaggle_kepler

echo ""
echo "[步驟 3/3] 解壓縮..."
unzip -o kepler-labelled-time-series-data.zip -d data/kaggle_kepler/

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✅ 下載完成！"
    echo "======================================================================"
    echo ""
    echo "數據位置: data/kaggle_kepler/"
    echo ""
    ls -lh data/kaggle_kepler/
    echo ""
    echo "使用範例："
    echo "  python"
    echo "  >>> import pandas as pd"
    echo "  >>> train = pd.read_csv('data/kaggle_kepler/exoTrain.csv')"
    echo "  >>> print(train.shape)"
    echo ""

    # 清理 zip 檔案
    rm kepler-labelled-time-series-data.zip
else
    echo "[錯誤] 解壓縮失敗"
    exit 1
fi
