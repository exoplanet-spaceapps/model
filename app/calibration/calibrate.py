import os, numpy as np, joblib, matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
def fit_isotonic(y_true, y_prob): m=IsotonicRegression(out_of_bounds='clip'); m.fit(y_prob, y_true); return m
def fit_platt(y_true, y_prob): m=LogisticRegression(solver='lbfgs'); m.fit(y_prob.reshape(-1,1), y_true); return m
def apply_calibrator(cal, y_prob):
    return cal.predict(y_prob) if hasattr(cal,'predict') and not hasattr(cal,'predict_proba') else cal.predict_proba(y_prob.reshape(-1,1))[:,1]
def reliability_plot(y_true, y_prob, path):
    import matplotlib; matplotlib.use("Agg")
    fracs, mean_pred = calibration_curve(y_true, y_prob, n_bins=15, strategy='quantile')
    plt.figure(figsize=(4,4)); plt.plot([0,1],[0,1],'--'); plt.plot(mean_pred, fracs, marker='o'); plt.xlabel('Pred'); plt.ylabel('Obs'); plt.tight_layout(); plt.savefig(path,dpi=140); plt.close()
def run_and_save(y_true, y_prob, out_dir='artifacts', method='isotonic'):
    os.makedirs(out_dir, exist_ok=True)
    cal = fit_isotonic(y_true, y_prob) if method=='isotonic' else fit_platt(y_true, y_prob)
    y_cal = apply_calibrator(cal, y_prob); from sklearn.metrics import brier_score_loss as B; brier=B(y_true, y_cal)
    joblib.dump(cal, os.path.join(out_dir,'calibrator.joblib')); reliability_plot(y_true, y_cal, os.path.join(out_dir,'calibration_cnn.png'))
    return {"method":method,"brier":float(brier)}
