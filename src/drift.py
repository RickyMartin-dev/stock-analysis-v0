import numpy as np
import pandas as pd
# Population Stability Index for numeric vectors
def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    ex, ax = np.histogram(expected, bins=bins)[0], np.histogram(actual,
    bins=bins)[0]
    ex = ex / (ex.sum()+1e-12); ax = ax / (ax.sum()+1e-12)
    ex = np.clip(ex, 1e-6, 1); ax = np.clip(ax, 1e-6, 1)
    return float(((ax - ex) * np.log(ax / ex)).sum())