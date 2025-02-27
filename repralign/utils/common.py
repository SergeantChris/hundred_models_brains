from typing import Optional
import numpy as np
from sklearn.linear_model import LinearRegression


def raw2rdm(raw: np.ndarray, dim: Optional[int] = None) -> np.ndarray:
    if dim is not None:
        raw = (raw - np.mean(raw, axis=dim, keepdims=True)) / (np.std(raw, axis=dim, keepdims=True) + 1e-7)
    rdm = 1 - np.corrcoef(raw)
    return rdm


def residuals(a, b, c):
    reg_a = LinearRegression().fit(c, a)
    res_a = a - reg_a.predict(c)
    reg_b = LinearRegression().fit(c, b)
    res_b = b - reg_b.predict(c)
    return res_a, res_b
