from __future__ import annotations
import numpy as np
def detrend_savgol(time, flux, window=101, poly=2):
    try:
        from scipy.signal import savgol_filter
        trend = savgol_filter(flux, window_length=max(5, window//2*2+1), polyorder=poly)
        return flux - trend, trend
    except Exception:
        return flux - np.median(flux), np.full_like(flux, np.median(flux))
def detrend_celerite2(time, flux):
    try:
        import celerite2
        from celerite2 import terms
        kernel = terms.SHOTerm(S0=1.0, w0=1.0, Q=1/2**0.5)
        gp = celerite2.GaussianProcess(kernel, mean=float(np.nanmedian(flux)))
        gp.compute(time, diag=np.full_like(time, 1e-5))
        mu,_ = gp.predict(flux, time)
        return flux - mu, mu
    except Exception:
        return detrend_savgol(time, flux)
def denoise(time, flux, backend="auto"):
    if backend=="celerite2":
        return detrend_celerite2(time, flux)
    try:
        return detrend_celerite2(time, flux)
    except Exception:
        return detrend_savgol(time, flux)
