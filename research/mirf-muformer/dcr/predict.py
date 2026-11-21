"""Combine a restored μFormer score with ``η r`` outside ``forward``.

``η`` is chosen on the outer validation rows only. The loss that trains ``r``
does not contain ``η``. ``η = 0`` leaves the μFormer score unchanged.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

ETA_GRID = tuple(index / 20 for index in range(21))


def fuse(score, eta: float, residual) -> np.ndarray:
    """Return ``s_μF + η r``. This is not a ``Muformer.forward`` output."""
    base = np.asarray(score, dtype=np.float64)
    correction = np.asarray(residual, dtype=np.float64)
    if base.shape != correction.shape:
        raise ValueError("the μFormer score and the residual have different shapes")
    return base + float(eta) * correction


def choose_eta(y, score, residual) -> float:
    """Pick ``η`` by validation Spearman, then RMSE, then the smaller value."""
    target = np.asarray(y, dtype=np.float64)
    base = np.asarray(score, dtype=np.float64)
    correction = np.asarray(residual, dtype=np.float64)
    if target.shape != base.shape or target.shape != correction.shape:
        raise ValueError("eta selection received mismatched rows")
    if target.ndim != 1 or target.shape[0] < 2:
        raise ValueError("eta selection needs at least two validation rows")
    best = None
    best_key = None
    for eta in ETA_GRID:
        prediction = fuse(base, eta, correction)
        if np.unique(prediction).size < 2:
            rho = -np.inf
        else:
            rho = spearmanr(target, prediction).correlation
            if rho is None or not np.isfinite(rho):
                rho = -np.inf
        rmse = float(np.sqrt(np.mean((target - prediction) ** 2)))
        key = (-float(rho), rmse, float(eta))
        if best_key is None or key < best_key:
            best_key = key
            best = float(eta)
    return best


def residual_vector(model, batch) -> np.ndarray:
    """Evaluate ``r`` on one restored representation batch. No score is added."""
    import torch

    model.eval()
    with torch.no_grad():
        values = model(batch["site"], batch["source"], batch["other"], batch["values"], batch["mask"])
    return values.detach().cpu().numpy().astype(np.float64)
