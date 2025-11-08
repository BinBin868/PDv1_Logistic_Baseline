import numpy as np

def _clip(p, eps=1e-6): return np.clip(p, eps, 1 - eps)
def _logit(p): p=_clip(p); return np.log(p/(1-p))
def _sigmoid(z): return 1/(1+np.exp(-z))

def platt_calibrate(p_raw: np.ndarray, s: float, b: float) -> np.ndarray:
    return _clip(_sigmoid(s*_logit(p_raw) + b))

def mean_match(p: np.ndarray, target_mean: float, max_shift=2.0) -> np.ndarray:
    p=_clip(p); cur=float(p.mean())
    target=float(np.clip(target_mean,1e-6,1-1e-6))
    if abs(cur-target)<1e-9: return p
    z=_logit(p); c=0.0
    for _ in range(30):
        pc=_sigmoid(z+c); f=pc.mean()-target
        if abs(f)<1e-7: break
        g=(pc*(1-pc)).mean() or 1e-6
        c = float(np.clip(c - f/g, -max_shift, max_shift))
    return _clip(_sigmoid(z+c))
