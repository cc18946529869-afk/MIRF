"""Prepare one outer seed without starting μFormer training.

``run_ten_outer_seeds`` stays refused. This module reads saved indices,
training scores, and the soft-weight table. It does not call ``Trainer``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import feature_align
import oof
import outer_train
import residual
import sealed_config

TRAIN_FILES = {
    "n1000": feature_align.N1000_TRAIN,
    "full": feature_align.FULL_TRAIN,
}


def oof_residual(scores, predictions, label_sequences, prediction_sequences) -> np.ndarray:
    """``e = y - yhat`` after the sequences match in identity and order."""
    oof.require_same_sequences(label_sequences, prediction_sequences)
    labels = np.asarray(scores, dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    if labels.shape != predicted.shape:
        raise ValueError("OOF labels and predictions do not have the same rows")
    return labels - predicted


def run_single_outer_seed(scale: str, outer_seed: int, *, start_training: bool = False, trainer_factory=None, loaded_scale: dict | None = None) -> dict:
    """One outer seed. Training runs only when a factory is passed explicitly.

    The default call does not construct ``Trainer``. ``run_ten_outer_seeds``
    is not used. A missing factory still refuses, so this round does not
    start the real μFormer.
    """
    if not start_training:
        if trainer_factory is not None:
            raise RuntimeError("a trainer factory is not called before training is explicitly started")
        return prepare_single_outer_seed(scale, outer_seed)
    if trainer_factory is None or loaded_scale is None:
        raise RuntimeError("real single-seed μFormer training is not started")
    return outer_train.train_one_outer_seed(scale, int(outer_seed), loaded_scale, trainer_factory)


def prepare_single_outer_seed(scale: str, outer_seed: int, *, start_training: bool = False) -> dict:
    """Check one seed's data path. ``start_training=True`` is refused."""
    if start_training:
        raise RuntimeError("real single-seed μFormer training is not started")
    if scale not in TRAIN_FILES:
        raise ValueError("scale must be n1000 or full")
    if int(outer_seed) not in oof.OUTER_SEEDS:
        raise ValueError("the outer seed is not in the locked list of ten")
    precheck = Path(__file__).resolve().parent / "precheck"
    loaded = oof.load_scale(precheck, scale)
    sequences, scores = feature_align.read_scored_tsv(TRAIN_FILES[scale])
    if len(sequences) != oof.EXPECTED_ROWS[scale]["n"]:
        raise ValueError("the training file row count is not the count for this scale")
    index = feature_align.FeatureIndex(feature_align.table_path())
    aligned = feature_align.align_scored_sequences(index, sequences, scores)
    covered = np.zeros(len(sequences), dtype=bool)
    for fold in loaded["folds"]:
        val_idx = np.asarray(fold["val_idx"], dtype=np.int64)
        train_idx = np.asarray(fold["train_idx"], dtype=np.int64)
        if np.intersect1d(train_idx, val_idx).size:
            raise ValueError("a fold trains on a row it holds out")
        if covered[val_idx].any():
            raise ValueError("a training row is held out by two folds")
        covered[val_idx] = True
    if not bool(covered.all()):
        raise ValueError("the five folds do not cover every training row once")
    values = torch.as_tensor(aligned["values"], dtype=torch.float32)
    mask = torch.as_tensor(aligned["mask"], dtype=torch.float32)
    if values.shape != (len(sequences), 14) or mask.shape != values.shape:
        raise ValueError("the residual batch does not have one 14-column row per training sequence")
    full_request = {
        "kind": "full",
        "epochs": outer_train.BASELINE_EPOCHS,
        "patience": outer_train.BASELINE_PATIENCE,
        "restore_best": True,
        "train_idx": np.arange(len(sequences), dtype=np.int64),
        "muformer_config": sealed_config.muformer_config(),
    }
    fold_requests = []
    for fold in loaded["folds"]:
        fold_requests.append({
            "kind": "fold",
            "fold": int(fold["fold"]),
            "epochs": None,
            "restore_best": False,
            "train_idx": np.asarray(fold["train_idx"], dtype=np.int64),
            "val_idx": np.asarray(fold["val_idx"], dtype=np.int64),
            "muformer_config": sealed_config.muformer_config(),
        })
    return {
        "scale": scale,
        "outer_seed": int(outer_seed),
        "training_started": False,
        "muformer_called": False,
        "oof_predictions_computed": False,
        "arms_trained": False,
        "arms": list(residual.ARMS),
        "residual_config": sealed_config.residual_config(),
        "ten_seed_entry_called": False,
        "rows": len(sequences),
        "feature_rows": int(values.shape[0]),
        "values": values,
        "mask": mask,
        "score": aligned["score"],
        "sequence": aligned["sequence"],
        "folds": loaded["folds"],
        "full_request": full_request,
        "fold_requests": fold_requests,
        "sealed_batch_size": sealed_config.SEALED.batch_size,
        "comparison_batch_size": sealed_config.muformer_config()["batch_size"],
    }
