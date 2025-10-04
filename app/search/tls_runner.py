from __future__ import annotations
def run_tls(time, flux, Rstar=1.0, Mstar=1.0):
    from transitleastsquares import transitleastsquares
    model = transitleastsquares(time, flux)
    results = model.power(use_threads=0, R_star=Rstar, M_star=Mstar)
    return {
        "period": float(results.period),
        "duration": float(results.duration),
        "t0": float(results.T0),
        "SDE": float(results.SDE),
        "SR": float(results.SR),
        "depth": float(results.depth),
        "periods": [float(x) for x in results.periods[:500]],
        "power": [float(x) for x in results.power[:500]]
    }
