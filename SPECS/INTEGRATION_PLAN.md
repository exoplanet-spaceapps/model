# INTEGRATION_PLAN — 從 XGBoost 過渡
- 新增 denoise/tls 模組；新增 02/03b/06 Notebook。
- 04 推論：若發現 cnn1d.pt+calibrator.joblib → 走 CNN；否則回 XGBoost。
- 05 Dashboard：CNN vs XGB 指標並排。
- 風險：依賴缺失→Notebook pip 安裝；資料不足→合成注入/交叉驗證。
