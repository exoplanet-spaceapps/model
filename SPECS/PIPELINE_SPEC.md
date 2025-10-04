# PIPELINE_SPEC — GP → TLS → CNN（校準）
A. 去噪：優先 celerite2 或 starry_process；缺依賴回退到 Savitzky–Golay。
B. 週期搜尋：transitleastsquares（TLS）。Astropy BLS 為基線/交叉驗證。
C. 分級：Two‑Branch 1D‑CNN（global/local 摺疊視圖）。
D. 校準：Isotonic或Platt；輸出 calibrator.joblib + 可靠度圖。
E. I/O：維持 `outputs/candidates_*.csv`、`provenance.yaml` 與 `docs/*.html`。
F. 指標：PR‑AUC/ROC‑AUC/Precision@K/Recall@Known/ECE/Brier/Latency。
