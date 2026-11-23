"""Wire real rows through a small stand-in. This does not train μFormer.

The 14 feature columns are the saved soft-weight table. The fold predictions
are a linear fit on those columns. They are not μFormer OOF predictions.
Site and lobe vectors come from each row's own token indices applied to a
position table, not from ``pmlm_650m.pt``. ``run_single_outer_seed`` refuses
``start_training=True``. The executable entry is ``designated_seed``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import feature_align
import frozen_pmlm
import metrics
import oof
import pmlm_coords
import predict
import residual
import single_seed

N1000_VALID = feature_align.N1000_TRAIN.parent / "BLAT_ECOLX_Stiffler_2015_n1000_valid.tsv"
N1000_TEST = feature_align.N1000_TRAIN.parent / "BLAT_ECOLX_Stiffler_2015_n1000_test.tsv"
EMBED_DIM = 8
ASSEMBLER_EPOCH = 1


class _PositionTable(nn.Module):
    """Hidden state whose time axis is the token position. Not a PMLM."""

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        time = int(tokens.shape[1])
        batch = int(tokens.shape[0])
        position = torch.arange(time, dtype=torch.float32, device=tokens.device).view(time, 1, 1)
        return position.expand(time, batch, EMBED_DIM).clone()


def choose_eta_from_validation(*, y_valid, s_valid, residual_valid, sequences, y_test=None) -> float:
    """Select η from validation rows. A test-label argument is refused."""
    if y_test is not None:
        raise RuntimeError("test labels cannot enter eta selection")
    return oof.select_eta(y_valid, s_valid, residual_valid, sequences, sequences, sequences)


def _linear_predict(values: np.ndarray, scores: np.ndarray, train_idx, val_idx) -> np.ndarray:
    train_rows = np.asarray(train_idx, dtype=np.int64)
    val_rows = np.asarray(val_idx, dtype=np.int64)
    if np.intersect1d(train_rows, val_rows).size:
        raise ValueError("the linear fold model saw a row it is asked to predict")
    design = np.concatenate([values[train_rows], np.ones((train_rows.shape[0], 1))], axis=1)
    coef, *_unused = np.linalg.lstsq(design, scores[train_rows], rcond=None)
    query = np.concatenate([values[val_rows], np.ones((val_rows.shape[0], 1))], axis=1)
    return query @ coef


def _embedding_batch(index: feature_align.FeatureIndex, sequences: list[str], values: np.ndarray, mask: np.ndarray) -> dict:
    site_index = []
    source_index = []
    other_index = []
    dictionary = pmlm_coords.load_dictionary()
    for sequence in sequences:
        ref_pos = index.row(sequence)["ref_pos"]
        site_index.append(pmlm_coords.token_index_for_reference(dictionary, sequence, ref_pos))
        source, other = pmlm_coords.per_row_lobe_indices(pmlm_coords.DOMAIN_MAP, ref_pos)
        source_index.append(torch.as_tensor(source, dtype=torch.long))
        other_index.append(torch.as_tensor(other, dtype=torch.long))
    tokens = torch.zeros(len(sequences), pmlm_coords.FULL_LENGTH + 2, dtype=torch.long)
    encoder = frozen_pmlm.FrozenPmlm(_PositionTable())
    site, source, other = encoder.representations(tokens, torch.tensor(site_index), source_index, other_index)
    return {
        "site": site,
        "source": source,
        "other": other,
        "values": torch.as_tensor(values, dtype=torch.float32),
        "mask": torch.as_tensor(mask, dtype=torch.float32),
        "site_index": site_index,
        "source_lengths": sorted({int(item.numel()) for item in source_index}),
    }


def run_n1000_stand_in(precheck: Path | None = None) -> dict:
    """Real n=1000 rows, linear OOF, three residual arms, validation η, test metrics."""
    index = feature_align.FeatureIndex(feature_align.table_path())
    train_sequences, train_scores = feature_align.read_scored_tsv(feature_align.N1000_TRAIN)
    valid_sequences, valid_scores = feature_align.read_scored_tsv(N1000_VALID)
    test_sequences, test_scores = feature_align.read_scored_tsv(N1000_TEST)
    train = feature_align.align_scored_sequences(index, train_sequences, train_scores)
    valid = feature_align.align_scored_sequences(index, valid_sequences, valid_scores)
    test = feature_align.align_scored_sequences(index, test_sequences, test_scores)
    for sequence in train["sequence"]:
        if index.row(sequence)["split"] != "train":
            raise ValueError("an n=1000 training sequence is not in the table's train split")
    for sequence in valid["sequence"]:
        if index.row(sequence)["split"] != "valid":
            raise ValueError("a validation sequence is not in the table's valid split")
    for sequence in test["sequence"]:
        if index.row(sequence)["split"] != "test":
            raise ValueError("a test sequence is not in the table's test split")
    if set(train["sequence"]) & set(valid["sequence"]) or set(train["sequence"]) & set(test["sequence"]) or set(valid["sequence"]) & set(test["sequence"]):
        raise ValueError("train, validation, and test sequences overlap")
    directory = Path(__file__).resolve().parent / "precheck" if precheck is None else Path(precheck)
    loaded = oof.load_scale(directory, "n1000")
    train_batch = _embedding_batch(index, train["sequence"], train["values"], train["mask"])
    folds = []
    for fold in loaded["folds"]:
        folds.append({
            "fold": fold["fold"],
            "train_idx": fold["train_idx"],
            "val_idx": fold["val_idx"],
            "epochs": ASSEMBLER_EPOCH,
        })

    def fit_fold(train_idx, val_idx, epochs):
        if int(epochs) != ASSEMBLER_EPOCH:
            raise ValueError("the linear stand-in was given a μFormer epoch count")
        return _linear_predict(train["values"], train["score"], train_idx, val_idx)

    oof_score = oof.assemble_oof(len(train["sequence"]), folds, ASSEMBLER_EPOCH, fit_fold)
    oof_residual = single_seed.oof_residual(train["score"], oof_score, train["sequence"], train["sequence"])
    initial = residual.shared_initial_state(EMBED_DIM, oof.OUTER_SEEDS[0])
    target = torch.as_tensor(oof_residual, dtype=torch.float32)
    arms = {}
    for arm in residual.ARMS:
        mask = residual.arm_mask(train_batch["mask"], arm)
        fitted = residual.fit_predictor(
            {**train_batch, "mask": mask},
            target,
            loaded["residual_train_idx"],
            loaded["residual_val_idx"],
            embed_dim=EMBED_DIM,
            outer_seed=oof.OUTER_SEEDS[0],
            initial_state=initial,
        )
        arms[arm] = fitted
    valid_batch = _embedding_batch(index, valid["sequence"], valid["values"], valid["mask"])
    test_batch = _embedding_batch(index, test["sequence"], test["values"], test["mask"])
    design = np.concatenate([train["values"], np.ones((len(train["sequence"]), 1))], axis=1)
    coef, *_unused = np.linalg.lstsq(design, train["score"], rcond=None)
    base_valid = np.concatenate([valid["values"], np.ones((len(valid["sequence"]), 1))], axis=1) @ coef
    base_test = np.concatenate([test["values"], np.ones((len(test["sequence"]), 1))], axis=1) @ coef
    full = arms[residual.ARM_FULL]
    valid_mask = residual.arm_mask(valid_batch["mask"], residual.ARM_FULL)
    test_mask = residual.arm_mask(test_batch["mask"], residual.ARM_FULL)
    valid_r = predict.residual_vector(full["model"], {**valid_batch, "mask": valid_mask})
    eta = choose_eta_from_validation(
        y_valid=valid["score"],
        s_valid=base_valid,
        residual_valid=valid_r,
        sequences=valid["sequence"],
    )
    test_r = predict.residual_vector(full["model"], {**test_batch, "mask": test_mask})
    fused = predict.fuse(base_test, eta, test_r)
    y_min = float(np.min(test["score"]))
    y_max = float(np.max(test["score"]))
    k = metrics.top_fraction_k(len(test["sequence"]))
    point = {
        name: metrics.metric_value(name, test["score"], fused, test["sequence"], y_min, y_max, k)
        for name in metrics.METRIC_NAMES
    }
    first_orders = [arms[arm]["selection_orders"][0] for arm in residual.ARMS]
    return {
        "muformer_trained": False,
        "formal_experiment": False,
        "oof_model": "linear_soft_weight_features",
        "pmlm_weights_used": False,
        "rows": len(train["sequence"]),
        "valid_rows": len(valid["sequence"]),
        "test_rows": len(test["sequence"]),
        "feature_shape": list(train["values"].shape),
        "source_lengths": train_batch["source_lengths"],
        "oof_coverage": int(np.isfinite(oof_score).sum()),
        "arms": list(residual.ARMS),
        "shared_init": all(
            torch.equal(arms[arm]["initial_state"][key], initial[key])
            for arm in residual.ARMS
            for key in initial
        ),
        "shared_first_order": all(torch.equal(order, first_orders[0]) for order in first_orders),
        "best_epochs": {arm: int(arms[arm]["best_epoch"]) for arm in residual.ARMS},
        "eta": float(eta),
        "test_point": point,
        "values": train["values"],
        "received_values": train_batch["values"].detach().cpu().numpy(),
    }


def write_report(payload: dict, destination: Path) -> None:
    public = {key: value for key, value in payload.items() if key not in {"values", "received_values"}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(public, indent=2) + "\n")
