"""Schedule one outer μFormer and five fold models.

The default call refuses the ten-seed formal run. A synthetic test passes a
factory that records row indices and epoch counts. Fold models train for the
full model's ``E_s`` and do not reload an earlier held-out best state.
Completed epochs and restored-weight flags are read from the trainer record.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import oof
import sealed_config

BASELINE_EPOCHS = 300
BASELINE_PATIENCE = 300
BASELINE_EVAL_FREQ = 1


def _sequence_values(frame):
    columns = getattr(frame, "columns", None)
    if columns is not None and "sequence" not in columns:
        raise ValueError("sample identity is sequence; the split has no sequence column")
    if isinstance(frame, dict) and "sequence" not in frame:
        raise ValueError("sample identity is sequence; the split has no sequence column")
    try:
        return frame["sequence"]
    except Exception as exc:
        raise ValueError("sample identity is sequence; the split has no sequence column") from exc


def _rows_from_score(scored, frame):
    """Predictions and sequences for the split that was actually requested."""
    requested = _sequence_values(frame)
    if isinstance(scored, dict) and "df" in scored:
        oof.require_same_sequences(requested, _sequence_values(scored["df"]))
        predictions = np.asarray(scored["df"]["prediction"], dtype=np.float64)
    elif isinstance(scored, dict) and "predictions" in scored:
        if "sequence" not in scored:
            raise ValueError("sample identity is sequence; predictions were not paired with sequences")
        oof.require_same_sequences(requested, scored["sequence"])
        predictions = np.asarray(scored["predictions"], dtype=np.float64)
    else:
        raise ValueError("the restored score did not include predictions")
    identity = oof.require_same_sequences(requested, requested)
    if predictions.shape != identity.shape:
        raise ValueError("scored predictions do not match the requested split")
    return predictions, identity


def score_restored_test_split(trainer) -> dict:
    """Score the test split on the module that training already restored.

    This is not an input to ``choose_eta``.
    """
    test_loader = getattr(trainer, "test_loader", None)
    test_df = getattr(trainer, "test_df", None)
    if test_loader is None or test_df is None:
        raise RuntimeError("test scoring requires the test set to be passed explicitly")
    scored = trainer.score_test(
        test_loader,
        test_df,
        mode="ensemble",
        save_prediction=False,
    )
    score, sequences = _rows_from_score(scored, test_df)
    return {"score": score, "sequences": sequences, "split": "test"}


def run_ten_outer_seeds(*args, **kwargs):
    """The formal ten-seed entry. It does not construct a trainer or a loop."""
    raise RuntimeError("formal training of the ten outer seeds is not started")


def _listed(trainer, name: str):
    value = getattr(trainer, name, None)
    if value is None:
        return None
    return list(value)


def _check_model_record(record: dict) -> None:
    """The four recorded facts have to agree with each other."""
    completed = int(record["epochs_completed"])
    best_epoch = record["best_epoch"]
    restored_best = bool(record["restored_best_weights"])
    restored_earlier = bool(record["restored_earlier_weights"])
    if completed < 1:
        raise RuntimeError("the trainer did not record a completed epoch")
    if restored_earlier and not restored_best:
        raise RuntimeError("earlier weights were restored without restoring the best weights")
    if not restored_best:
        if best_epoch is not None:
            raise RuntimeError("a fold model selected a best epoch on the held-out fold")
        return
    if best_epoch is None:
        raise RuntimeError("E_s is recorded only when is_best is true")
    best_epoch = int(best_epoch)
    if best_epoch < 1 or best_epoch > completed:
        raise RuntimeError("the best epoch is outside the completed epochs")
    if restored_earlier != (best_epoch < completed):
        raise RuntimeError(
            "whether earlier weights were restored does not match the best epoch and the completed epochs"
        )


def _model_records(trainer) -> list[dict]:
    completed = _listed(trainer, "epochs_completed")
    best_epochs = _listed(trainer, "best_epoch_per_model")
    restored_best = _listed(trainer, "restored_best_weights")
    restored_earlier = _listed(trainer, "restored_earlier_weights")
    if completed is None or best_epochs is None or restored_best is None or restored_earlier is None:
        raise RuntimeError("the trainer did not record completed epochs, the best epoch, and restored weights")
    if not completed or not (len(completed) == len(best_epochs) == len(restored_best) == len(restored_earlier)):
        raise RuntimeError("the trainer record of completed epochs, best epoch, and restored weights is incomplete")
    records = []
    for done, best_epoch, did_restore, did_replace in zip(completed, best_epochs, restored_best, restored_earlier):
        records.append(
            {
                "epochs_completed": int(done),
                "best_epoch": None if best_epoch is None else int(best_epoch),
                "restored_best_weights": bool(did_restore),
                "restored_earlier_weights": bool(did_replace),
            }
        )
    return records


def _require_full_record(trainer) -> dict:
    records = _model_records(trainer)
    saved_best = [item for item in list(getattr(trainer, "best_epochs", []) or [])]
    if len(saved_best) != len(records):
        raise RuntimeError("the best epoch was not recorded once for each model")
    for saved, record in zip(saved_best, records):
        _check_model_record(record)
        if saved is None or record["best_epoch"] is None or int(saved) != int(record["best_epoch"]):
            raise RuntimeError("E_s is recorded only when is_best is true")
        if not record["restored_best_weights"]:
            raise RuntimeError("the full model did not restore its best weights")
    return records[-1]


def _require_fold_record(trainer, request: dict) -> dict:
    records = _model_records(trainer)
    if list(getattr(trainer, "best_epochs", []) or []):
        raise RuntimeError("a fold model kept a held-out best epoch")
    for record in records:
        _check_model_record(record)
        if record["best_epoch"] is not None or record["restored_best_weights"] or record["restored_earlier_weights"]:
            raise RuntimeError("a fold model restored or selected weights on the held-out fold")
        if int(record["epochs_completed"]) != int(request["epochs"]):
            raise RuntimeError("a fold model did not complete this seed's E_s epochs")
    return records[-1]


def _recorded_fold_rows(trainer, request: dict) -> np.ndarray:
    stored = getattr(trainer, "_train_row_index", None)
    if stored is None:
        stored = getattr(trainer, "train_row_index", None)
    if stored is None:
        raise RuntimeError("the trainer did not record the fold training rows")
    rows = np.asarray(stored, dtype=np.int64)
    requested = np.asarray(request["train_idx"], dtype=np.int64)
    if rows.shape != requested.shape or not np.array_equal(rows, requested):
        raise RuntimeError("the recorded fold training rows are not this fold's training rows")
    return rows


def muformer_standin_state(config: dict) -> dict:
    """Fields ``Trainer.__init__`` and ``init_model`` would write for this config.

    This does not call ``Trainer`` or ``init_model``, and it does not build Adam.
    """
    checked = sealed_config.assert_muformer_config(config)
    weight_decay = checked["weight_decay"]
    return {
        "batch_size": checked["batch_size"],
        "hidden_size": checked["hidden"],
        "dropout": checked["dropout"],
        "decoder_name": checked["decoder_name"],
        "encoder_lr": checked["encoder_lr"],
        "decoder_lr": checked["decoder_lr"],
        "weight_decay": weight_decay,
        "warmup_epochs": checked["warmup_epochs"],
        "adam_groups": (
            {"lr": checked["encoder_lr"], "weight_decay": weight_decay},
            {"lr": checked["decoder_lr"], "weight_decay": weight_decay},
        ),
    }


def apply_muformer_standin(trainer, config: dict) -> None:
    """Write the μFormer constructor fields onto a stand-in trainer."""
    state = muformer_standin_state(config)
    trainer.batch_size = state["batch_size"]
    trainer.hidden_size = state["hidden_size"]
    trainer.dropout = state["dropout"]
    trainer.decoder_name = state["decoder_name"]
    trainer.args = SimpleNamespace(
        encoder_lr=state["encoder_lr"],
        decoder_lr=state["decoder_lr"],
        weight_decay=state["weight_decay"],
        warmup_epochs=state["warmup_epochs"],
    )
    trainer.adam_groups = state["adam_groups"]


def _assert_trainer_carries_muformer_config(trainer, request: dict) -> None:
    """The comparison configuration has to be on the trainer, not only in the request."""
    if "muformer_config" not in request:
        return
    state = muformer_standin_state(request["muformer_config"])
    args = getattr(trainer, "args", None)
    if (
        getattr(trainer, "batch_size", None) != state["batch_size"]
        or getattr(trainer, "hidden_size", None) != state["hidden_size"]
        or getattr(trainer, "dropout", None) != state["dropout"]
        or getattr(trainer, "decoder_name", None) != state["decoder_name"]
        or args is None
        or getattr(args, "encoder_lr", None) != state["encoder_lr"]
        or getattr(args, "decoder_lr", None) != state["decoder_lr"]
        or getattr(args, "weight_decay", None) != state["weight_decay"]
        or getattr(args, "warmup_epochs", None) != state["warmup_epochs"]
        or getattr(trainer, "adam_groups", None) != state["adam_groups"]
    ):
        raise RuntimeError("μFormer constructor fields are not on the trainer before train")


def _comparison_request_fields() -> dict:
    """Nested comparison namespaces. The request top level has no ``batch_size``."""
    return {
        "muformer_config": sealed_config.assert_muformer_config(sealed_config.muformer_config()),
        "residual_config": sealed_config.assert_residual_config(sealed_config.residual_config()),
    }


def execute_muformer_request(trainer, request: dict) -> dict:
    """Call ``train`` on one request.

    The full model restores best weights and reads ``E_s`` from the best epoch
    recorded only when ``is_best`` is true. Completed epochs and the two
    restoration flags are read from the trainer, not copied from the request.
    A fold request keeps the weights of epoch ``E_s`` and does not score a
    split. The score returned here is the outer validation split. The test
    split is a separate call.
    """
    fold = not bool(request["restore_best"])
    if request.get("select_on_held_out_fold"):
        raise RuntimeError("a fold model does not select its epoch on the held-out fold")
    if fold:
        sealed_config.assert_e_s(
            request["epochs"],
            request["baseline_best_epoch"],
            request["e_s_source"],
        )
    train_call = {
        "epochs": int(request["epochs"]),
        "eval_freq": int(request["eval_freq"]),
        "patience": int(request["patience"]),
        "save_checkpoint": False,
        "resume_path": None,
        "restore_best": not fold,
        "train_row_index": None if not fold else np.asarray(request["train_idx"], dtype=np.int64),
        "drop_last": not fold,
    }
    sealed_config.assert_muformer_train_call(train_call, request)
    _assert_trainer_carries_muformer_config(trainer, request)
    trainer.train(**train_call)
    if request["restore_best"]:
        record = _require_full_record(trainer)
        valid_loader = getattr(trainer, "valid_loader", None)
        valid_df = getattr(trainer, "valid_df", None)
        if valid_loader is None or valid_df is None:
            raise RuntimeError(
                "validation scoring requires the outer validation set and does not fall back to the test set"
            )
        scored = trainer.score_validation(
            valid_loader,
            valid_df,
            mode="ensemble",
            save_prediction=False,
        )
        score, sequences = _rows_from_score(scored, valid_df)
        return {
            "e_s": int(record["best_epoch"]),
            "score": score,
            "sequences": sequences,
            "split": "validation",
            "epochs_run": int(record["epochs_completed"]),
            "restore_best": bool(record["restored_best_weights"]),
            "replaced_with_earlier_best": bool(record["restored_earlier_weights"]),
        }
    record = _require_fold_record(trainer, request)
    return {
        "epochs_run": int(record["epochs_completed"]),
        "restore_best": bool(record["restored_best_weights"]),
        "train_idx": _recorded_fold_rows(trainer, request),
        "replaced_with_earlier_best": bool(record["restored_earlier_weights"]),
    }


def _complement(n: int, val_idx: np.ndarray) -> set[int]:
    return set(range(int(n))) - set(int(value) for value in val_idx.tolist())


def train_outer_models(scale: str, outer_seed: int, loaded_scale: dict, trainer_factory, mode: str = "formal") -> dict:
    """Train one full outer model, then five fold models on the sealed indices.

    ``mode`` other than ``synthetic`` raises before the factory is called.
    n=1000 indices are not read from the full-file split.
    """
    if mode != "synthetic":
        raise RuntimeError("formal training of the ten outer seeds is not started")
    if trainer_factory is None:
        raise RuntimeError("formal training of the ten outer seeds is not started")
    if scale not in oof.SCALES:
        raise ValueError("scale must be n1000 or full")
    if loaded_scale.get("scale") != scale:
        raise ValueError("n=1000 and the full index were mixed")
    if int(outer_seed) not in oof.OUTER_SEEDS:
        raise ValueError("the outer seed is not in the locked list of ten")
    if loaded_scale.get("indices_regenerated") is not False:
        raise RuntimeError("precheck indices are read from dcr/precheck and are not regenerated")
    if loaded_scale.get("precheck_dir_name") != "precheck" or loaded_scale.get("precheck_parent_name") != "dcr":
        raise RuntimeError("precheck indices are read only from dcr/precheck and are not regenerated")
    cfg = sealed_config.require()
    if int(loaded_scale.get("kfold_seed", -1)) != cfg.kfold_seed or int(loaded_scale.get("residual_holdout_seed", -1)) != cfg.residual_holdout_seed:
        raise RuntimeError("the loaded precheck indices are not tied to the sealed split seeds")
    sealed_config.assert_saved_soft_weight_audit()
    n = int(loaded_scale["n"])
    if n != oof.EXPECTED_ROWS[scale]["n"]:
        raise ValueError("the loaded row count is not the count for this scale")
    full_request = {
        "kind": "full",
        "scale": scale,
        "outer_seed": int(outer_seed),
        "train_idx": np.arange(n, dtype=np.int64),
        "val_idx": None,
        "fold": None,
        "epochs": BASELINE_EPOCHS,
        "patience": BASELINE_PATIENCE,
        "eval_freq": BASELINE_EVAL_FREQ,
        "restore_best": True,
        "select_on_held_out_fold": False,
    }
    full_request.update(_comparison_request_fields())
    full_result = execute_muformer_request(trainer_factory(full_request), full_request)
    e_s = sealed_config.assert_e_s(full_result["e_s"], full_result["e_s"], cfg.e_s_source, cfg)
    score = np.asarray(full_result["score"], dtype=np.float64)
    sequences = np.asarray(full_result["sequences"])
    if full_result.get("split") != "validation":
        raise RuntimeError("the restored full model was not scored on the outer validation set")
    fold_results = []
    for fold in loaded_scale["folds"]:
        train_idx = np.asarray(fold["train_idx"], dtype=np.int64)
        val_idx = np.asarray(fold["val_idx"], dtype=np.int64)
        if set(int(value) for value in train_idx.tolist()) != _complement(n, val_idx):
            raise ValueError("fold training rows are not the complement of that fold's validation rows")
        if np.intersect1d(train_idx, val_idx).size:
            raise ValueError("a fold trains on a row it also holds out")
        request = {
            "kind": "fold",
            "scale": scale,
            "outer_seed": int(outer_seed),
            "train_idx": train_idx,
            "val_idx": val_idx,
            "fold": int(fold["fold"]),
            "epochs": e_s,
            "patience": BASELINE_PATIENCE,
            "eval_freq": BASELINE_EVAL_FREQ,
            "restore_best": False,
            "select_on_held_out_fold": False,
            "baseline_best_epoch": e_s,
            "e_s_source": cfg.e_s_source,
        }
        request.update(_comparison_request_fields())
        if request["muformer_config"] != full_request["muformer_config"]:
            raise RuntimeError("a fold model's muformer_config is not the full model's muformer_config")
        if request["residual_config"] != full_request["residual_config"]:
            raise RuntimeError("a fold model's residual_config is not the full model's residual_config")
        produced = execute_muformer_request(trainer_factory(request), request)
        if int(produced["epochs_run"]) != e_s or produced["restore_best"] or produced["replaced_with_earlier_best"]:
            raise RuntimeError("a fold model did not keep the weights from epoch E_s")
        fold_results.append(produced)
    sealed_config.assert_folds_share_e_s(
        e_s,
        [item["epochs_run"] for item in fold_results],
        held_out_selection=any(bool(item["restore_best"]) or bool(item["replaced_with_earlier_best"]) for item in fold_results),
    )
    return {
        "scale": scale,
        "outer_seed": int(outer_seed),
        "e_s": e_s,
        "s_mu_f": score,
        "s_mu_f_sequences": sequences,
        "s_mu_f_split": "validation",
        "baseline": score,
        "folds": fold_results,
        "formal_training_started": False,
        "arms_share_e_s": True,
        "arms_share_base_score": True,
    }
