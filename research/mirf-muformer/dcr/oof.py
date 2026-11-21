"""Read the sealed precheck indices and run the residual schedule on synthetic rows.

This module does not train μFormer, does not regenerate ``dcr/precheck/``, and
does not compute the ten-seed test metrics. ``mode="formal"`` is refused.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

import predict
import residual

OUTER_SEEDS = (
    1975576344,
    94706340,
    1220745215,
    508376101,
    1910795809,
    141521756,
    1119615152,
    929659347,
    120967996,
    1571307030,
)
SCALES = ("n1000", "full")
EXPECTED_ROWS = {
    "n1000": {
        "n": 1000,
        "val_counts": (200, 200, 200, 200, 200),
        "train_counts": (800, 800, 800, 800, 800),
        "residual_train": 900,
        "residual_val": 100,
    },
    "full": {
        "n": 3497,
        "val_counts": (700, 700, 699, 699, 699),
        "train_counts": (2797, 2797, 2798, 2798, 2798),
        "residual_train": 3147,
        "residual_val": 350,
    },
}
RESIDUAL_DEFAULTS = {
    "max_epochs": 300,
    "patience": 30,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_index(path: Path) -> np.ndarray:
    lines = Path(path).read_text().splitlines()
    values = [int(line) for line in lines if line != ""]
    return np.asarray(values, dtype=np.int64)


def verify_precheck_hashes(precheck: Path) -> None:
    """Recompute ``SHA256SUMS`` without writing the index files."""
    lines = (Path(precheck) / "SHA256SUMS").read_text().splitlines()
    if len(lines) != 27:
        raise ValueError("SHA256SUMS does not have the sealed 27 entries")
    for line in lines:
        digest, name = line.split()
        if file_sha256(Path(precheck) / name) != digest:
            raise ValueError(f"{name} does not match the sealed precheck hash")


def _as_set(values: np.ndarray) -> set[int]:
    return set(int(value) for value in values.tolist())


def load_scale(precheck: Path, scale: str) -> dict:
    """Load one scale's saved indices. n=1000 and the full set stay separate."""
    if scale not in EXPECTED_ROWS:
        raise ValueError("scale must be n1000 or full")
    directory = Path(precheck)
    expected = EXPECTED_ROWS[scale]
    folds = []
    covered = np.zeros(expected["n"], dtype=bool)
    for fold in range(5):
        train_idx = read_index(directory / f"{scale}_oof_fold{fold}_train.txt")
        val_idx = read_index(directory / f"{scale}_oof_fold{fold}_val.txt")
        if int(train_idx.shape[0]) != expected["train_counts"][fold]:
            raise ValueError(f"{scale} fold {fold} train count is not the sealed count")
        if int(val_idx.shape[0]) != expected["val_counts"][fold]:
            raise ValueError(f"{scale} fold {fold} validation count is not the sealed count")
        train_set = _as_set(train_idx)
        val_set = _as_set(val_idx)
        universe = set(range(expected["n"]))
        if train_set & val_set or train_set | val_set != universe:
            raise ValueError(f"{scale} fold {fold} is not a partition of the outer training rows")
        if covered[val_idx].any():
            raise ValueError(f"{scale} assigns one row to two OOF folds")
        covered[val_idx] = True
        folds.append({"fold": fold, "train_idx": train_idx, "val_idx": val_idx})
    if not bool(covered.all()):
        raise ValueError(f"{scale} OOF folds do not cover every outer training row")
    fold_id = read_index(directory / f"{scale}_oof_fold_id.txt")
    if fold_id.shape != (expected["n"],):
        raise ValueError(f"{scale} fold ids do not have one entry per outer training row")
    for fold in folds:
        if not np.array_equal(fold_id[fold["val_idx"]], np.full(fold["val_idx"].shape, fold["fold"])):
            raise ValueError(f"{scale} fold ids do not match the saved validation rows")
    residual_train = read_index(directory / f"{scale}_residual_train.txt")
    residual_val = read_index(directory / f"{scale}_residual_val.txt")
    if residual_train.shape != (expected["residual_train"],) or residual_val.shape != (expected["residual_val"],):
        raise ValueError(f"{scale} internal 10 percent split does not have the sealed sizes")
    if _as_set(residual_train) & _as_set(residual_val) or _as_set(residual_train) | _as_set(residual_val) != set(range(expected["n"])):
        raise ValueError(f"{scale} internal split is not a partition of the outer training rows")
    return {
        "scale": scale,
        "n": expected["n"],
        "folds": folds,
        "fold_id": fold_id,
        "residual_train_idx": residual_train,
        "residual_val_idx": residual_val,
    }


def load_precheck(precheck: Path) -> dict[str, dict]:
    """Read both sealed scales after checking hashes. Files are not rewritten."""
    verify_precheck_hashes(precheck)
    loaded = {scale: load_scale(precheck, scale) for scale in SCALES}
    if loaded["n1000"]["n"] == loaded["full"]["n"]:
        raise ValueError("n=1000 and the full training set were not kept separate")
    if np.array_equal(loaded["n1000"]["fold_id"], loaded["full"]["fold_id"][: loaded["n1000"]["n"]]):
        raise ValueError("n=1000 fold ids were read from the full-set file")
    return loaded


def folds_for_seed(loaded: dict, e_s: int) -> list[dict]:
    """Give every fold the same ``E_s``. The five folds do not use residual patience."""
    if int(e_s) < 1:
        raise ValueError("E_s must be a positive epoch count")
    scheduled = []
    for fold in loaded["folds"]:
        scheduled.append(
            {
                "fold": fold["fold"],
                "train_idx": fold["train_idx"],
                "val_idx": fold["val_idx"],
                "epochs": int(e_s),
            }
        )
    return scheduled


def assemble_oof(n: int, folds: list[dict], e_s: int, fit_fold) -> np.ndarray:
    """One OOF prediction per training row, from a fold model trained for ``E_s``."""
    prediction = np.full(n, np.nan, dtype=np.float64)
    occupied = np.zeros(n, dtype=bool)
    for fold in folds:
        if int(fold["epochs"]) != int(e_s):
            raise ValueError("a fold model does not use the shared E_s")
        val_idx = np.asarray(fold["val_idx"], dtype=np.int64)
        train_idx = np.asarray(fold["train_idx"], dtype=np.int64)
        if np.intersect1d(train_idx, val_idx).size:
            raise ValueError("a fold trains on rows it also predicts")
        produced = np.asarray(fit_fold(train_idx, val_idx, int(e_s)), dtype=np.float64)
        if produced.shape != val_idx.shape:
            raise ValueError("a fold prediction does not match its validation rows")
        if occupied[val_idx].any():
            raise ValueError("a training row received more than one OOF prediction")
        prediction[val_idx] = produced
        occupied[val_idx] = True
    if not bool(occupied.all()) or not np.isfinite(prediction).all():
        raise ValueError("OOF predictions do not cover each training row once")
    return prediction


def _arm_batch(batch: dict[str, torch.Tensor], mask: torch.Tensor) -> dict[str, torch.Tensor]:
    packed = dict(batch)
    packed["mask"] = mask
    return packed


def run_seed_protocol(
    *,
    scale: str,
    outer_seed: int,
    e_s: int,
    loaded_scale: dict,
    y_oof: np.ndarray,
    train_batch: dict[str, torch.Tensor],
    y_valid: np.ndarray,
    s_valid: np.ndarray,
    valid_batch: dict[str, torch.Tensor],
    fit_fold,
    mode: str = "synthetic",
    max_epochs: int = RESIDUAL_DEFAULTS["max_epochs"],
    patience: int = RESIDUAL_DEFAULTS["patience"],
    batch_size: int = RESIDUAL_DEFAULTS["batch_size"],
    learning_rate: float = RESIDUAL_DEFAULTS["learning_rate"],
    weight_decay: float = RESIDUAL_DEFAULTS["weight_decay"],
    seed: int = 0,
    embed_dim: int,
) -> dict:
    """Run OOF assembly and three residual arms on arrays already in memory.

    ``s_valid`` and ``e_s`` are shared by the baseline and the three residual
    arms. Each arm has its own predictor and its own ``η``. Formal test
    metrics are not computed.
    """
    if mode != "synthetic":
        raise RuntimeError("formal training and formal fusion metrics are not started in this step")
    if scale not in SCALES:
        raise ValueError("scale must be n1000 or full")
    if int(outer_seed) not in OUTER_SEEDS:
        raise ValueError("the outer seed is not in the locked list of ten")
    if loaded_scale["scale"] != scale:
        raise ValueError("the loaded indices belong to a different scale")
    y_rows = np.asarray(y_oof, dtype=np.float64)
    if y_rows.shape != (loaded_scale["n"],):
        raise ValueError("OOF labels do not match this scale's outer training rows")
    if np.asarray(s_valid).shape != np.asarray(y_valid).shape:
        raise ValueError("the shared μFormer score does not match the validation labels")
    folds = folds_for_seed(loaded_scale, e_s)
    oof_score = assemble_oof(loaded_scale["n"], folds, int(e_s), fit_fold)
    target = torch.as_tensor(y_rows - oof_score, dtype=torch.float32)
    observed = train_batch["mask"]
    arms = {}
    for offset, arm in enumerate(residual.ARMS):
        mask = residual.arm_mask(observed, arm)
        fitted = residual.fit_predictor(
            _arm_batch(train_batch, mask),
            target,
            loaded_scale["residual_train_idx"],
            loaded_scale["residual_val_idx"],
            embed_dim=embed_dim,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=int(seed) + offset,
        )
        valid_mask = residual.arm_mask(valid_batch["mask"], arm)
        correction = predict.residual_vector(fitted["model"], _arm_batch(valid_batch, valid_mask))
        eta = predict.choose_eta(y_valid, s_valid, correction)
        train_r = predict.residual_vector(fitted["model"], _arm_batch(train_batch, mask))
        oof_residual = y_rows - oof_score
        arms[arm] = {
            "model": fitted["model"],
            "eta": eta,
            "best_epoch": fitted["best_epoch"],
            "selection_epochs_run": fitted["selection_epochs_run"],
            "retrain_epochs_run": fitted["retrain_epochs_run"],
            "retrain_rows": fitted["retrain_rows"],
            "prediction": predict.fuse(s_valid, eta, correction),
            "observed_slots": int(torch.as_tensor(mask)[0].sum().item()) if mask.ndim == 2 else int(torch.as_tensor(mask).sum().item()),
            "oof_residual_mean": float(np.mean(oof_residual)),
            "retrained_residual_mean": float(np.mean(train_r)),
        }
    return {
        "scale": scale,
        "outer_seed": int(outer_seed),
        "e_s": int(e_s),
        "s_mu_f": s_valid,
        "baseline": s_valid,
        "oof_score": oof_score,
        "fold_epochs": tuple(int(fold["epochs"]) for fold in folds),
        "arms": arms,
        "formal_metrics_computed": False,
    }
