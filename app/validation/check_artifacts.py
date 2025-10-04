import sys, pandas as pd
from pathlib import Path
REQ = ["target_id","mission","bls_period_d","bls_duration_hr","bls_depth_ppm","snr","model_score","run_id","model_version"]
def main(out_dir="outputs"):
    out = Path(out_dir); csvs = sorted(out.glob("candidates_*.csv")); ok=True
    if not csvs: print("No candidates_*.csv", file=sys.stderr); sys.exit(2)
    for p in csvs:
        df=pd.read_csv(p); miss=[c for c in REQ if c not in df.columns]
        if miss: print(f"[FAIL] {p.name} missing: {miss}", file=sys.stderr); ok=False
        else: print(f"[OK] {p.name}: {len(df)} rows")
    if not (out/'provenance.yaml').exists(): print("[FAIL] provenance.yaml missing", file=sys.stderr); ok=False
    sys.exit(0 if ok else 1)
if __name__=="__main__": main()
