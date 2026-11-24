"""Sealed values that enter outer training, OOF, and residual training.

Callers do not substitute these numbers. A different config is refused.
``E_s`` is not a stored epoch: it is the baseline best epoch of the outer
seed being trained. Seed 42's recorded epoch is not a default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SealedExecution:
    kfold_splits: int = 5
    kfold_shuffle: bool = True
    kfold_seed: int = 20261121
    e_s_source: str = "this_outer_seed_baseline"
    seed42_recorded_epoch: int = 229
    residual_holdout_seed: int = 20261122
    residual_holdout_fraction: float = 0.1
    early_stop_metric: str = "mae"
    eval_every_epoch: bool = True
    patience: int = 30
    optimizer: str = "AdamW"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 300
    scheduler: str | None = None
    d_s: int = 64
    d_l: int = 32
    d_r: int = 32
    concat_width: int = 160
    hidden: int = 64
    dropout: float = 0.1
    soft_weight_strategy: str = "soft_weight_keeps_82_and_182"
    soft_weight_positions: tuple = (82, 182)


SEALED = SealedExecution()

# SHA256 of ``dcr/precheck/SHA256SUMS`` itself. Matching the lines inside that
# file is not enough: replacing an index and the manifest together still fails.
PRECHECK_MANIFEST_SHA256 = "eea8056329a822a51a8702759fe685c969b3a18f0cab8622668cc69a2ce3c7c6"

MUFORMER_TRAIN_KEYS = frozenset({
    "epochs",
    "eval_freq",
    "patience",
    "save_checkpoint",
    "resume_path",
    "restore_best",
    "train_row_index",
    "drop_last",
})
MUFORMER_EXCLUDED_KEYS = frozenset({
    "batch_size",
    "encoder_lr",
    "decoder_lr",
    "weight_decay",
    "hidden",
    "hidden_size",
    "dropout",
    "siamese",
    "decoder_name",
    "warmup",
    "warmup_epochs",
})
MUFORMER_CONFIG = {
    "batch_size": 16,
    "encoder_lr": 1e-5,
    "decoder_lr": 1e-4,
    "weight_decay": 0.0,
    "hidden": 256,
    "dropout": 0.2,
    "decoder_name": "siamese",
    "warmup_epochs": 10,
}

SOFT_WEIGHT_AUDIT = "soft_weight_coords_audit.json"
SOFT_WEIGHT_TABLE = "blat_ecolx_stiffler_2015_soft_weight_coords.tsv"
V01_TABLE = "blat_ecolx_stiffler_2015_v0.1.tsv"


def assert_sealed_values(config: SealedExecution) -> None:
    """Refuse a config whose sealed numbers have been replaced."""
    if config.kfold_splits != 5 or config.kfold_shuffle is not True or config.kfold_seed != 20261121:
        raise RuntimeError("OOF does not use KFold(5, shuffle=True, random_state=20261121)")
    if config.e_s_source != "this_outer_seed_baseline" or config.seed42_recorded_epoch != 229:
        raise RuntimeError("E_s must be this outer seed's own baseline best epoch, not seed 42's 229")
    if config.residual_holdout_seed != 20261122 or config.residual_holdout_fraction != 0.1:
        raise RuntimeError("the residual holdout is not the sealed internal 10 percent split")
    if config.early_stop_metric != "mae" or config.patience != 30 or config.eval_every_epoch is not True:
        raise RuntimeError("residual early stopping is MAE every epoch with patience 30")
    if config.optimizer != "AdamW" or config.learning_rate != 1e-3 or config.weight_decay != 1e-4:
        raise RuntimeError("residual optimizer is not AdamW with the sealed learning rate and weight decay")
    if config.batch_size != 64 or config.max_epochs != 300 or config.scheduler is not None:
        raise RuntimeError("residual training is not batch size 64, 300 epochs, and no scheduler")
    if (config.d_s, config.d_l, config.d_r, config.concat_width) != (64, 32, 32, 160):
        raise RuntimeError("the readout widths are not d_s=64, d_l=32, d_r=32, concatenation 160")
    if config.hidden != 64 or config.dropout != 0.1:
        raise RuntimeError("the readout is not a 64-wide layer with dropout 0.1")
    if config.soft_weight_strategy != "soft_weight_keeps_82_and_182" or tuple(config.soft_weight_positions) != (82, 182):
        raise RuntimeError("positions 82 and 182 are not the sealed soft-weight coordinate set")


def require(config: SealedExecution | None = None) -> SealedExecution:
    """Return the sealed config. Any other config is refused."""
    selected = SEALED if config is None else config
    if selected != SEALED:
        raise RuntimeError("the execution config is not the sealed config")
    assert_sealed_values(selected)
    return selected


def assert_e_s(e_s, baseline_best_epoch, source: str, config: SealedExecution | None = None) -> int:
    """``E_s`` is this outer seed's baseline best epoch, shared later by five folds."""
    cfg = require(config)
    if source != cfg.e_s_source:
        raise RuntimeError("E_s must be this outer seed's own baseline best epoch, not seed 42's 229")
    if int(e_s) != int(baseline_best_epoch):
        raise RuntimeError("E_s must be this outer seed's own baseline best epoch, not seed 42's 229")
    if int(e_s) < 1:
        raise RuntimeError("E_s must be a positive epoch")
    return int(e_s)


def assert_folds_share_e_s(e_s: int, fold_epochs, *, held_out_selection: bool) -> None:
    """Five fold models train for ``E_s`` and do not pick an epoch on the held-out fold."""
    if held_out_selection:
        raise RuntimeError("a fold model does not select its epoch on the held-out fold")
    epochs = tuple(int(item) for item in fold_epochs)
    if len(epochs) != 5 or any(item != int(e_s) for item in epochs):
        raise RuntimeError("the five fold models must share this outer seed's baseline best epoch")


def assert_residual_optimizer(optimizer, config: SealedExecution | None = None) -> None:
    """The live residual optimizer is AdamW at the sealed learning rate and decay."""
    cfg = require(config)
    if optimizer.__class__.__name__ != cfg.optimizer or cfg.scheduler is not None:
        raise RuntimeError("residual optimizer must be AdamW with no learning-rate scheduler")
    groups = optimizer.param_groups
    if len(groups) != 1:
        raise RuntimeError("residual AdamW does not have one parameter group")
    if groups[0]["lr"] != cfg.learning_rate or groups[0]["weight_decay"] != cfg.weight_decay:
        raise RuntimeError("residual AdamW learning rate or weight decay does not match the sealed config")


def assert_no_scheduler(scheduler) -> None:
    if scheduler is not None:
        raise RuntimeError("residual training does not use a learning-rate scheduler")


def muformer_config() -> dict:
    """The locked μFormer comparison configuration, as its own namespace."""
    return dict(MUFORMER_CONFIG)


def assert_muformer_config(config: dict) -> dict:
    """Refuse a μFormer comparison config that is not the locked one."""
    expected = MUFORMER_CONFIG
    if (
        not isinstance(config, dict)
        or set(config) != set(expected)
        or any(config[key] != expected[key] for key in expected)
    ):
        raise RuntimeError("muformer_config does not match the comparison configuration")
    if (
        config["batch_size"] == SEALED.batch_size
        or config["hidden"] == SEALED.hidden
        or config["dropout"] == SEALED.dropout
    ):
        raise RuntimeError("muformer_config was mixed with the residual batch size or the residual readout")
    return dict(expected)


def residual_config(config: SealedExecution | None = None) -> dict:
    """Residual hyperparameters that are not the μFormer comparison configuration.

    The readout width and dropout stay on the sealed execution config.
    """
    cfg = require(config)
    return {
        "batch_size": int(cfg.batch_size),
        "optimizer": cfg.optimizer,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "max_epochs": int(cfg.max_epochs),
        "patience": int(cfg.patience),
    }


def assert_residual_config(config: dict) -> dict:
    """Refuse a residual namespace that is not the sealed residual hyperparameters."""
    expected = residual_config()
    if (
        not isinstance(config, dict)
        or set(config) != set(expected)
        or any(config[key] != expected[key] for key in expected)
    ):
        raise RuntimeError("residual_config does not match the sealed residual hyperparameters")
    return dict(expected)


def assert_muformer_train_call(call: dict, request: dict) -> None:
    """``train`` keeps its original arguments.

    The request top level still refuses ``batch_size`` and the comparison-configuration
    keys. A nested ``muformer_config`` may remain after it has been validated.
    """
    if set(call) != MUFORMER_TRAIN_KEYS:
        raise RuntimeError("the μFormer train call does not match the sealed argument set")
    top_level = set(request) - {"muformer_config", "residual_config"}
    if MUFORMER_EXCLUDED_KEYS & set(call) or MUFORMER_EXCLUDED_KEYS & top_level:
        raise RuntimeError("batch size and the comparison configuration are not top-level μFormer train arguments")
    if "batch_size" in call or "batch_size" in top_level:
        raise RuntimeError("batch size is not a top-level μFormer train argument")
    if "muformer_config" in request:
        assert_muformer_config(request["muformer_config"])
    if "residual_config" in request:
        assert_residual_config(request["residual_config"])


def assert_precheck_path(precheck: Path) -> Path:
    """The saved indices are read from ``dcr/precheck`` and are not regenerated."""
    path = Path(precheck).resolve()
    if path.name != "precheck" or path.parent.name != "dcr":
        raise RuntimeError("precheck indices are read only from dcr/precheck and are not regenerated")
    if not (path / "SHA256SUMS").is_file():
        raise RuntimeError("precheck indices are read only from dcr/precheck and are not regenerated")
    return path


def precheck_snapshot(precheck: Path) -> dict[str, int]:
    path = assert_precheck_path(precheck)
    return {item.name: item.stat().st_mtime_ns for item in path.iterdir() if item.is_file()}


def assert_precheck_unchanged(before: dict[str, int], after: dict[str, int]) -> None:
    if before != after:
        raise RuntimeError("precheck indices were regenerated")


def features_directory() -> Path:
    return Path(__file__).resolve().parent / "features"


def assert_saved_soft_weight_audit(features: Path | None = None) -> dict:
    """Read the saved soft-weight audit. Do not recompute features or rewrite tables."""
    cfg = require()
    directory = features_directory() if features is None else Path(features)
    audit_path = directory / SOFT_WEIGHT_AUDIT
    table_path = directory / SOFT_WEIGHT_TABLE
    v01_path = directory / V01_TABLE
    watched = (audit_path, table_path, v01_path)
    before = {path: path.stat().st_mtime_ns for path in watched}
    audit = json.loads(audit_path.read_text())
    if audit.get("coordinate_set") != cfg.soft_weight_strategy:
        raise RuntimeError("the saved soft-weight audit does not use soft_weight_keeps_82_and_182")
    if audit.get("positions_82_and_182_in_coordinate_set") is not True:
        raise RuntimeError("the saved soft-weight audit does not keep positions 82 and 182")
    if audit.get("used_in_coord_equals_not_mismatched") is not False:
        raise RuntimeError("the saved soft-weight audit records the mismatch complement as the coordinate set")
    recorded = {int(item["ref_pos"]): item for item in audit.get("mismatches", [])}
    for position in cfg.soft_weight_positions:
        item = recorded.get(int(position))
        if item is None or item.get("in_soft_weight_coordinate_set") is not True:
            raise RuntimeError("positions 82 and 182 are not both in the saved coordinate set")
    after = {path: path.stat().st_mtime_ns for path in watched}
    if before != after:
        raise RuntimeError("reading the soft-weight audit rewrote a feature file")
    return audit
