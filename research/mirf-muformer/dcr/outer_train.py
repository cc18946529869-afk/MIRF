"""Schedule one outer μFormer and five fold models.

The default call refuses the ten-seed formal run. A synthetic test passes a
factory that records row indices and epoch counts. Fold models train for the
full model's ``E_s`` and do not reload an earlier held-out best state.
"""

from __future__ import annotations

import numpy as np

import oof

BASELINE_EPOCHS = 300
BASELINE_PATIENCE = 300
BASELINE_EVAL_FREQ = 1


def run_ten_outer_seeds(*args, **kwargs):
    """The formal ten-seed entry. It does not construct a trainer or a loop."""
    raise RuntimeError("formal training of the ten outer seeds is not started")


def execute_muformer_request(trainer, request: dict) -> dict:
    """Call ``train`` on one request.

    The full model restores best weights and reads ``E_s`` from ``best_epochs``,
    which ``Trainer`` fills only when ``is_best`` is true. A fold request keeps
    the weights of epoch ``E_s`` and does not call ``score_restored``.
    """
    fold = not bool(request["restore_best"])
    trainer.train(
        epochs=int(request["epochs"]),
        eval_freq=int(request["eval_freq"]),
        patience=int(request["patience"]),
        save_checkpoint=False,
        resume_path=None,
        restore_best=not fold,
        train_row_index=None if not fold else np.asarray(request["train_idx"], dtype=np.int64),
        drop_last=not fold,
    )
    if request["restore_best"]:
        recorded = list(getattr(trainer, "best_epochs", []) or [])
        if not recorded or recorded[-1] is None:
            raise RuntimeError("E_s is recorded only when is_best is true")
        scored = trainer.score_restored(mode="ensemble", save_prediction=False)
        if isinstance(scored, dict) and "predictions" in scored:
            score = np.asarray(scored["predictions"], dtype=np.float64)
        else:
            score = np.asarray(scored["df"]["prediction"], dtype=np.float64)
        return {
            "e_s": int(recorded[-1]),
            "score": score,
            "epochs_run": int(recorded[-1]),
            "restore_best": True,
        }
    return {
        "epochs_run": int(request["epochs"]),
        "restore_best": False,
        "train_idx": np.asarray(request["train_idx"], dtype=np.int64),
        "replaced_with_earlier_best": False,
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
    full_result = execute_muformer_request(trainer_factory(full_request), full_request)
    e_s = int(full_result["e_s"])
    if e_s < 1:
        raise RuntimeError("E_s must be a positive epoch")
    score = np.asarray(full_result["score"], dtype=np.float64)
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
        }
        produced = execute_muformer_request(trainer_factory(request), request)
        if int(produced["epochs_run"]) != e_s or produced["restore_best"] or produced["replaced_with_earlier_best"]:
            raise RuntimeError("a fold model did not keep the weights from epoch E_s")
        fold_results.append(produced)
    return {
        "scale": scale,
        "outer_seed": int(outer_seed),
        "e_s": e_s,
        "s_mu_f": score,
        "baseline": score,
        "folds": fold_results,
        "formal_training_started": False,
        "arms_share_e_s": True,
        "arms_share_base_score": True,
    }
