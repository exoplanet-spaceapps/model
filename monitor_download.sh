#!/bin/bash
# 下載監控腳本
# 用法: bash monitor_download.sh

echo "========================================"
echo "Kepler 光變曲線下載監控"
echo "========================================"
echo ""

while true; do
    # 檢查進程
    if ps aux | grep -E "fast_download.*auto-confirm" | grep -v grep > /dev/null; then
        PID=$(ps aux | grep -E "fast_download.*auto-confirm" | grep -v grep | awk '{print $1}')
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 進程運行中 (PID: $PID)"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 進程已停止"
        break
    fi

    # 計算進度
    CACHED=$(ls data/cached_lightcurves/*.npy 2>/dev/null | wc -l)
    PERCENT=$(echo "scale=1; $CACHED * 100 / 400" | bc)

    echo "  已快取: $CACHED / 400 ($PERCENT%)"

    # 顯示最新日誌
    if [ -f download_log.txt ]; then
        LAST_LINE=$(tail -1 download_log.txt)
        echo "  最新日誌: $LAST_LINE"
    fi

    echo ""

    # 每 5 分鐘檢查一次
    sleep 300
done

echo "監控結束"
