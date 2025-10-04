# app/data/fold.py
"""
Phase folding & view construction for 1D-CNN.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List

def phase_fold(t: np.ndarray, period: float, t0: float) -> np.ndarray:
    return ((t - t0) / period) % 1.0

def resample_equal(phase: np.ndarray, y: np.ndarray, length: int) -> np.ndarray:
    idx = np.argsort(phase)
    p, v = phase[idx], y[idx]
    p_ext = np.concatenate(([0.0], p, [1.0]))
    v_ext = np.concatenate(([v[0]], v, [v[-1]]))
    grid  = np.linspace(0.0, 1.0, num=length, endpoint=False)
    return np.interp(grid, p_ext, v_ext).astype(np.float32)

def robust_norm(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + 1e-8
    return ((x - med) / (1.4826*mad)).astype(np.float32)

def make_views(time: np.ndarray, flux: np.ndarray, period: float, t0: float, duration: float,
               g_len:int=2000, l_len:int=512, k:float=3.0):
    phase = phase_fold(time, period, t0)
    y = robust_norm(flux.astype(np.float32))
    g = resample_equal(phase, y, g_len)
    w = (k * duration) / period
    mask = (phase <= w) | (phase >= 1.0 - w)
    p_local = phase[mask].copy(); y_local = y[mask].copy()
    p_local[p_local > 0.5] -= 1.0
    p_local = p_local - p_local.min(); p_local = p_local / (p_local.max() + 1e-8)
    l = resample_equal(p_local, y_local, l_len)
    return g[None,:], l[None,:]

@dataclass
class Item:
    time: np.ndarray; flux: np.ndarray; period: float; t0: float; duration: float; label: int

class LightCurveViewsDataset:
    def __init__(self, items: List[Item], g_len:int=2000, l_len:int=512, k:float=3.0):
        self.items = items; self.g_len=g_len; self.l_len=l_len; self.k=k
    def __len__(self): return len(self.items)
    def __getitem__(self, i:int):
        it = self.items[i]
        g,l = make_views(it.time, it.flux, it.period, it.t0, it.duration, self.g_len, self.l_len, self.k)
        y = np.array(it.label, dtype=np.int64)
        return g,l,y
