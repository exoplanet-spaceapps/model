# Kepler 光變曲線替代下載來源完整報告

## 🚀 最佳方案：AWS S3 公開數據桶（推薦）

### **優勢：最快、免費、穩定**

**S3 Bucket 資訊：**
- Bucket: `s3://stpubdata/kepler/public`
- 區域: AWS US-East
- 費用: 免費（從 US-East 區域存取）
- 包含: 所有 17 季 Kepler 觀測數據

**存取方式（透過 astroquery）：**
```python
from astroquery.mast import Observations

# 啟用 AWS S3 公開數據存取
Observations.enable_cloud_dataset()

# 之後的下載會自動從 S3 取得（更快）
import lightkurve as lk
search = lk.search_lightcurve('KIC 11502867', mission='Kepler')
lc = search.download()
```

**優點：**
- ✅ 比 MAST 直接 HTTP 下載快很多
- ✅ 無需 AWS 帳號或認證
- ✅ 與 lightkurve 完全相容
- ✅ 穩定性高

---

## 📊 方案 2：Kaggle 數據集（快速、已處理）

### **2.1 Kepler 標記時間序列數據**
- **來源**: https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data
- **大小**: 58.6 MB (ZIP)
- **樣本數**: 5,657 筆（訓練 5,087 + 測試 570）
- **格式**: CSV，3,198 個時間點
- **標籤**:
  - 2 = 確認系外行星
  - 1 = 非系外行星
- **分布**:
  - 確認行星: 42 筆
  - 非行星: 5,615 筆

**下載方式：**
```bash
# 需要 Kaggle API
kaggle datasets download -d keplersmachines/kepler-labelled-time-series-data
```

### **2.2 Kepler & TESS 系外行星數據（2025 最新）**
- **來源**: https://www.kaggle.com/datasets/vijayveersingh/kepler-and-tess-exoplanet-data
- **更新**: 2025 年 2 月
- **包含**: 光變曲線、候選行星、確認行星

### **2.3 Kepler 系外行星搜尋結果**
- **來源**: https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results
- **樣本數**: 10,000 筆候選行星
- **更新**: 2017 年（較舊）

---

## 🔬 方案 3：Mendeley 學術數據集（2024 最新，已處理）

**來源**: https://data.mendeley.com/datasets/wctcv34962/3

**數據集資訊：**
- **名稱**: Dataset_Machine_Learning_Exoplanets_2024
- **光變曲線數**: 5,302 筆
- **每條數據點**: ~60,000 點
- **來源**: NASA Exoplanet Archive (Kepler telescope)
- **授權**: CC BY 4.0

**包含欄位：**
- kepid
- koi_disposition
- koi_period
- koi_time0bk
- koi_duration
- koi_quarters

**預處理：**
- ✅ 使用 Lightkurve 提取
- ✅ PDCSAP flux（最適合系外行星偵測）
- ✅ 已標準化
- ✅ 線性插值填補缺失值
- ✅ 2 標準差離群值移除

**論文參考：**
"Automated Light Curve Processing for Exoplanet Detection Using Machine Learning Algorithms"
(Macedo & Zalewski, 2024)

---

## 🌐 方案 4：NASA Exoplanet Archive API/TAP（官方）

### **4.1 TAP Service（推薦用於 KOI 表格）**

**端點**: https://exoplanetarchive.ipac.caltech.edu/TAP/

**支援的 KOI 表格：**
- cumulative
- q1_q6_koi
- q1_q8_koi
- q1_q12_koi
- q1_q16_koi
- q1_q17_dr24_koi
- q1_q17_dr25_koi
- q1_q17_dr25_sup_koi

**使用 PyVO 存取：**
```python
from pyvo import dal

# 連接 TAP 服務
tap = dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP/")

# 查詢 KOI 表格
query = """
SELECT kepid, koi_disposition, koi_period, koi_prad
FROM q1_q17_dr25_koi
WHERE koi_disposition IN ('CONFIRMED', 'FALSE POSITIVE')
LIMIT 400
"""

result = tap.search(query)
df = result.to_table().to_pandas()
```

### **4.2 傳統 API**

**範例 URL：**
```
https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=cumulative&select=kepid,koi_disposition&format=csv
```

### **4.3 批量下載（Wget 腳本）**

**來源**: https://exoplanetarchive.ipac.caltech.edu/bulk_data_download/

**選項：**
- Kepler_Quarterly_wget.tar.gz (36.72 MB 腳本，實際數據 ~3 TB)
- 單季下載腳本 Q0-Q17（每季 ~175 GB）
- KOI 時間序列專用腳本
- TCE 時間序列專用腳本
- 確認行星時間序列專用腳本

**注意事項：**
- ⚠️ 建議不超過 4 個 wget 並行
- ⚠️ 數據量極大（3 TB）

---

## 📦 方案 5：直接使用現有處理好的數據集

### **優勢：**
- 無需下載原始 FITS 檔
- 已標準化處理
- 立即可用於機器學習
- 大小合理（<100 MB）

### **推薦數據集：**

1. **Kaggle - Kepler 標記時間序列** (58.6 MB)
   - 適合：快速原型開發
   - 5,657 個樣本

2. **Mendeley 2024** (需查詢大小)
   - 適合：學術研究
   - 5,302 個樣本，已完整預處理

---

## 🔧 實作建議

### **最快速方案（推薦）：**

**步驟 1: 啟用 AWS S3 存取**
```python
from astroquery.mast import Observations
Observations.enable_cloud_dataset()
```

**步驟 2: 修改下載腳本**
在 `scripts/fast_download_kepler_data.py` 開頭加入：
```python
# 啟用 AWS S3 加速
from astroquery.mast import Observations
Observations.enable_cloud_dataset()
print("[INFO] AWS S3 cloud access enabled - downloads will be faster!")
```

### **中等速度方案：**

直接從 Kaggle 下載已處理數據：
```bash
pip install kaggle
kaggle datasets download -d keplersmachines/kepler-labelled-time-series-data
unzip kepler-labelled-time-series-data.zip -d data/
```

### **學術研究方案：**

從 Mendeley 下載 2024 數據集（5,302 筆，已處理）

---

## 📊 各方案比較

| 方案 | 速度 | 數據量 | 預處理 | 難度 | 推薦度 |
|------|------|--------|--------|------|--------|
| AWS S3 + lightkurve | ⭐⭐⭐⭐⭐ | 可控 | ❌ | 低 | ⭐⭐⭐⭐⭐ |
| Kaggle 數據集 | ⭐⭐⭐⭐⭐ | 小 (58 MB) | ✅ | 極低 | ⭐⭐⭐⭐ |
| Mendeley 2024 | ⭐⭐⭐⭐ | 中 | ✅ | 低 | ⭐⭐⭐⭐⭐ |
| NASA API/TAP | ⭐⭐⭐ | 可控 | ❌ | 中 | ⭐⭐⭐ |
| MAST HTTP (目前) | ⭐ | 可控 | ❌ | 低 | ⭐ |
| 批量 Wget | ⭐⭐ | 極大 (3 TB) | ❌ | 高 | ⭐⭐ |

---

## 🎯 立即可執行方案

### **方案 A：修改現有腳本使用 AWS S3（最推薦）**
1. 在 `fast_download_kepler_data.py` 加入 S3 啟用代碼
2. 重新運行下載（速度應大幅提升）

### **方案 B：使用 Kaggle 數據集（最快速）**
1. 安裝 Kaggle CLI
2. 下載預處理數據
3. 直接用於訓練

### **方案 C：使用 Mendeley 2024（學術級）**
1. 訪問 Mendeley 連結
2. 下載 5,302 筆已處理數據
3. 引用相關論文

---

## 📝 後續步驟

**建議優先嘗試：**
1. ✅ 修改腳本啟用 AWS S3（代碼改動最小）
2. ✅ 如果仍慢，直接使用 Kaggle 或 Mendeley 數據集
3. ✅ 完成訓練後，可再用 TAP API 補充更多數據

---

**報告生成時間**: 2025-10-05
**數據來源**: NASA, Kaggle, Mendeley, AWS
**整理者**: Claude Code
