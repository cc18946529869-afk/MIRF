"""Read the sealed precheck indices and run the residual schedule on synthetic rows.

This module does not train μFormer, does not regenerate ``dcr/precheck/``, and
does not compute the ten-seed test metrics. ``mode="formal"`` is refused.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

import metrics
import predict
import residual
import sealed_config

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
_SEALED = sealed_config.require()
RESIDUAL_DEFAULTS = {
    "max_epochs": _SEALED.max_epochs,
    "patience": _SEALED.patience,
    "batch_size": _SEALED.batch_size,
    "learning_rate": _SEALED.learning_rate,
    "weight_decay": _SEALED.weight_decay,
    "optimizer": _SEALED.optimizer,
    "scheduler": _SEALED.scheduler,
    "early_stop_metric": _SEALED.early_stop_metric,
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
    """Recompute ``SHA256SUMS`` without writing the index files.

    The manifest file has its own sealed SHA256. Rewriting that file to match
    a changed index does not make the index acceptable.
    """
    manifest = Path(precheck) / "SHA256SUMS"
    if file_sha256(manifest) != sealed_config.PRECHECK_MANIFEST_SHA256:
        raise ValueError("SHA256SUMS does not match the sealed manifest hash")
    lines = manifest.read_text().splitlines()
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
    directory = sealed_config.assert_precheck_path(precheck)
    before = sealed_config.precheck_snapshot(directory)
    verify_precheck_hashes(directory)
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
    sealed_config.assert_precheck_unchanged(before, sealed_config.precheck_snapshot(directory))
    return {
        "scale": scale,
        "n": expected["n"],
        "folds": folds,
        "fold_id": fold_id,
        "residual_train_idx": residual_train,
        "residual_val_idx": residual_val,
        "precheck_dir_name": directory.name,
        "precheck_parent_name": directory.parent.name,
        "indices_regenerated": False,
        "kfold_seed": _SEALED.kfold_seed,
        "residual_holdout_seed": _SEALED.residual_holdout_seed,
    }


def load_precheck(precheck: Path) -> dict[str, dict]:
    """Read both sealed scales after checking hashes. Files are not rewritten."""
    directory = sealed_config.assert_precheck_path(precheck)
    before = sealed_config.precheck_snapshot(directory)
    verify_precheck_hashes(directory)
    loaded = {scale: load_scale(directory, scale) for scale in SCALES}
    sealed_config.assert_precheck_unchanged(before, sealed_config.precheck_snapshot(directory))
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


def _sequence_tuple(values) -> tuple:
    """One sequence per row, in the given order. Identity is not a set."""
    column = np.asarray(values)
    if column.ndim != 1 or int(column.shape[0]) < 1:
        raise ValueError("sample identity is sequence; the column is empty or not one-dimensional")
    return tuple(str(value) for value in column.tolist())


def require_same_sequences(*groups) -> np.ndarray:
    """Reject columns that differ in length, sequence identity, or order.

    Equal length is not alignment. The same sequences in another order are
    not the same rows.
    """
    if len(groups) < 2:
        raise ValueError("sequence alignment needs two sequence columns")
    columns = [_sequence_tuple(group) for group in groups]
    first = columns[0]
    for column in columns[1:]:
        if column != first:
            raise ValueError(
                "sample identity is sequence; the columns differ in length, identity, or order"
            )
    return np.asarray(first)


def _batch_rows(batch: dict) -> int:
    rows = None
    for key in ("site", "source", "other", "values", "mask"):
        count = int(batch[key].shape[0])
        if rows is None:
            rows = count
        elif count != rows:
            raise ValueError("the validation residual batch rows do not match each other")
    if rows is None:
        raise ValueError("the validation residual batch has no rows")
    return rows


def select_eta(y_valid, s_valid, residual, label_sequences, score_sequences, residual_sequences) -> float:
    """Choose ``η`` from the outer validation rows only.

    The arguments are the validation labels, the μFormer scores on those
    rows, and the residual predictions on those rows. Test labels are not
    an argument.
    """
    identity = require_same_sequences(label_sequences, score_sequences, residual_sequences)
    labels = np.asarray(y_valid, dtype=np.float64)
    score = np.asarray(s_valid, dtype=np.float64)
    correction = np.asarray(residual, dtype=np.float64)
    if labels.shape != identity.shape or score.shape != identity.shape or correction.shape != identity.shape:
        raise ValueError("validation labels, μFormer scores, and residuals are not aligned to the sequences")
    return predict.choose_eta(labels, score, correction)


def fuse_test_after_eta(
    *,
    eta_locked: bool,
    eta,
    y_test,
    s_test,
    residual_test,
    label_sequences,
    score_sequences,
    residual_sequences,
    mode: str = "synthetic",
) -> dict:
    """Fusion metrics on the test rows, only after ``η`` is already locked.

    ``mode`` other than ``synthetic`` is refused. This does not read an
    experiment directory and does not decide that a model beat the baseline.
    """
    if mode != "synthetic":
        raise RuntimeError("formal fusion metrics are not started")
    if eta_locked is not True:
        raise RuntimeError("test fusion metrics require eta to be locked on the validation set")
    identity = require_same_sequences(label_sequences, score_sequences, residual_sequences)
    labels = np.asarray(y_test, dtype=np.float64)
    score = np.asarray(s_test, dtype=np.float64)
    correction = np.asarray(residual_test, dtype=np.float64)
    if labels.shape != identity.shape or score.shape != identity.shape or correction.shape != identity.shape:
        raise ValueError("test labels, μFormer scores, and residuals are not aligned to the sequences")
    prediction = predict.fuse(score, float(eta), correction)
    k = metrics.top_fraction_k(int(labels.shape[0]))
    y_min = float(np.min(labels))
    y_max = float(np.max(labels))
    point = {
        name: metrics.metric_value(name, labels, prediction, identity, y_min, y_max, k)
        for name in metrics.METRIC_NAMES
    }
    return {
        "prediction": prediction,
        "eta": float(eta),
        "synthetic_point": point,
        "formal_metrics_computed": False,
        "formal_experiment": False,
    }


def lock_eta_and_fuse_test(
    *,
    y_valid,
    s_valid,
    residual_valid,
    valid_label_sequences,
    valid_score_sequences,
    valid_residual_sequences,
    y_test,
    s_test,
    residual_test,
    test_label_sequences,
    test_score_sequences,
    test_residual_sequences,
    mode: str = "synthetic",
) -> dict:
    """Lock ``η`` on the validation rows, then score fusion on the test rows."""
    eta = select_eta(
        y_valid,
        s_valid,
        residual_valid,
        valid_label_sequences,
        valid_score_sequences,
        valid_residual_sequences,
    )
    fused = fuse_test_after_eta(
        eta_locked=True,
        eta=eta,
        y_test=y_test,
        s_test=s_test,
        residual_test=residual_test,
        label_sequences=test_label_sequences,
        score_sequences=test_score_sequences,
        residual_sequences=test_residual_sequences,
        mode=mode,
    )
    return fused


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
    valid_sequences=None,
    score_sequences=None,
    batch_sequences=None,
    fit_fold,
    mode: str = "synthetic",
    seed: int = 0,
    embed_dim: int,
    baseline_best_epoch: int | None = None,
    e_s_source: str | None = None,
    residual_fit=None,
    config: sealed_config.SealedExecution | None = None,
) -> dict:
    """Run OOF assembly and three residual arms on arrays already in memory.

    ``s_valid`` and ``e_s`` are shared by the baseline and the three residual
    arms. ``e_s`` must be this outer seed's own baseline best epoch. Each arm
    has its own predictor and its own ``η``. The three arms share one initial
    state and the shuffle stream of ``outer_seed``. ``seed`` is not an arm
    offset. Residual hyperparameters come from the sealed config. Formal test
    metrics are not computed. ``residual_fit`` records a call in a short test;
    the default is ``residual.fit_predictor``.
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
    if valid_sequences is None or score_sequences is None or batch_sequences is None:
        raise ValueError("sample identity is sequence; validation rows were not identified")
    identity = require_same_sequences(valid_sequences, score_sequences, batch_sequences)
    if np.asarray(y_valid).shape != identity.shape or np.asarray(s_valid).shape != identity.shape:
        raise ValueError("the shared μFormer score does not match the validation labels")
    if _batch_rows(valid_batch) != int(identity.shape[0]):
        raise ValueError("the validation residual batch is not aligned to the validation sequences")
    cfg = sealed_config.require(config)
    sealed_config.assert_saved_soft_weight_audit()
    if baseline_best_epoch is None or e_s_source is None:
        sealed_config.assert_e_s(e_s, e_s, "", cfg)
    e_s = sealed_config.assert_e_s(e_s, baseline_best_epoch, e_s_source, cfg)
    folds = folds_for_seed(loaded_scale, e_s)
    sealed_config.assert_folds_share_e_s(
        e_s,
        [fold["epochs"] for fold in folds],
        held_out_selection=False,
    )
    oof_score = assemble_oof(loaded_scale["n"], folds, int(e_s), fit_fold)
    target = torch.as_tensor(y_rows - oof_score, dtype=torch.float32)
    observed = train_batch["mask"]
    initial_state = residual.shared_initial_state(embed_dim, int(outer_seed))
    del seed
    fit = residual.fit_predictor if residual_fit is None else residual_fit
    arms = {}
    for arm in residual.ARMS:
        mask = residual.arm_mask(observed, arm)
        fitted = fit(
            _arm_batch(train_batch, mask),
            target,
            loaded_scale["residual_train_idx"],
            loaded_scale["residual_val_idx"],
            embed_dim=embed_dim,
            outer_seed=int(outer_seed),
            initial_state=initial_state,
            config=cfg,
        )
        valid_mask = residual.arm_mask(valid_batch["mask"], arm)
        correction = predict.residual_vector(fitted["model"], _arm_batch(valid_batch, valid_mask))
        eta = select_eta(y_valid, s_valid, correction, valid_sequences, score_sequences, batch_sequences)
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
            "initial_state": fitted["initial_state"],
            "selection_initial": fitted["selection_initial"],
            "retrain_initial": fitted["retrain_initial"],
            "selection_orders": fitted["selection_orders"],
            "retrain_orders": fitted["retrain_orders"],
            "selection_index_source": fitted["selection_index_source"],
            "retrain_index_source": fitted["retrain_index_source"],
            "selection_first_dropout_mask": fitted["selection_first_dropout_mask"],
            "retrain_first_dropout_mask": fitted["retrain_first_dropout_mask"],
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


def record_residual_call(records: list):
    """Return a fit stand-in that records the sealed call and does not train.

    The recorded hyperparameters are the sealed config. This does not replace
    them with a shorter schedule. Row-order checks call ``residual.fit_predictor``.
    """

    def fit(batch, target, train_idx, val_idx, *, embed_dim, outer_seed, initial_state, config):
        del batch
        cfg = sealed_config.require(config)
        train_rows = np.asarray(train_idx)
        val_rows = np.asarray(val_idx)
        records.append(
            {
                "optimizer": cfg.optimizer,
                "learning_rate": cfg.learning_rate,
                "weight_decay": cfg.weight_decay,
                "batch_size": cfg.batch_size,
                "max_epochs": cfg.max_epochs,
                "patience": cfg.patience,
                "scheduler": cfg.scheduler,
                "early_stop_metric": cfg.early_stop_metric,
                "train_rows": int(train_rows.shape[0]),
                "val_rows": int(val_rows.shape[0]),
                "outer_seed": int(outer_seed),
            }
        )
        state = {key: value.detach().cpu().clone() for key, value in initial_state.items()}
        model = residual.ResidualPredictor(embed_dim)
        model.load_state_dict(state)
        model.eval()
        selection = torch.arange(int(train_rows.shape[0]), dtype=torch.long)
        retrain = torch.arange(int(target.shape[0]), dtype=torch.long)
        mask = torch.ones(1, 1)
        return {
            "model": model,
            "best_epoch": 1,
            "best_mae": 0.0,
            "selection_epochs_run": 1,
            "retrain_epochs_run": 1,
            "retrain_rows": int(target.shape[0]),
            "selection_train_rows": int(train_rows.shape[0]),
            "outer_seed": int(outer_seed),
            "initial_state": state,
            "selection_initial": {key: value.clone() for key, value in state.items()},
            "retrain_initial": {key: value.clone() for key, value in state.items()},
            "selection_orders": [selection],
            "retrain_orders": [retrain],
            "selection_index_source": torch.as_tensor(train_rows, dtype=torch.long),
            "retrain_index_source": retrain.clone(),
            "selection_first_dropout_mask": mask,
            "retrain_first_dropout_mask": mask.clone(),
        }

    return fit
