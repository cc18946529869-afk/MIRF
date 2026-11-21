"""Synthetic walk-through of the sealed split, fold calls, residual arms and metrics.

``mode`` other than ``synthetic`` is refused. This does not start the ten-seed
μFormer run and does not score a real experiment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import metrics
import oof
import outer_train
import predict
import residual


class _RecordingTrainer:
    def __init__(self, request, calls, e_s: int, score: np.ndarray):
        self.request = request
        self.calls = calls
        self.best_epochs = []
        self._e_s = int(e_s)
        self._score = np.asarray(score, dtype=np.float64)

    def train(self, epochs, eval_freq, patience, save_checkpoint, resume_path, restore_best=True,
            train_row_index=None, drop_last=True):
        if self.request["kind"] == "fold":
            recorded = np.asarray(train_row_index, dtype=np.int64).copy()
        else:
            recorded = np.asarray(self.request["train_idx"], dtype=np.int64).copy()
        self.calls.append(
            {
                "kind": self.request["kind"],
                "fold": self.request["fold"],
                "epochs": int(epochs),
                "restore_best": bool(restore_best),
                "train_idx": recorded,
                "drop_last": bool(drop_last),
                "select_on_held_out_fold": bool(self.request["select_on_held_out_fold"]),
            }
        )
        if restore_best:
            self.best_epochs.append(self._e_s)
        elif self.best_epochs:
            raise RuntimeError("fold training kept a held-out best epoch")

    def score_restored(self, **kwargs):
        if "checkpoint_dir" in kwargs and kwargs["checkpoint_dir"] is not None:
            raise RuntimeError("score_restored received a checkpoint directory")
        self.calls.append({"kind": "score", "fold": self.request["fold"]})
        return {"predictions": self._score.copy()}


def _states_equal(left, right) -> bool:
    if set(left) != set(right):
        return False
    return all(torch.equal(left[key], right[key]) for key in left)


def rehearse(precheck: Path, *, outer_seed: int | None = None, mode: str = "synthetic") -> dict:
    """Run one synthetic outer seed through the sealed indices and the metric functions."""
    if mode != "synthetic":
        raise RuntimeError("formal training of the ten outer seeds is not started")
    try:
        outer_train.run_ten_outer_seeds()
        formal_refused = False
    except RuntimeError as exc:
        formal_refused = "not started" in str(exc)
    if not formal_refused:
        raise RuntimeError("the ten-seed entry did not refuse")
    loaded = oof.load_precheck(precheck)
    scale = loaded["n1000"]
    seed = int(oof.OUTER_SEEDS[0] if outer_seed is None else outer_seed)
    calls = []
    score = np.array([0.1, 0.4, 0.2, 0.8], dtype=np.float64)

    def factory(request):
        return _RecordingTrainer(request, calls, e_s=4, score=score)

    scheduled = outer_train.train_outer_models("n1000", seed, scale, factory, mode="synthetic")
    if scheduled["formal_training_started"] or scheduled["e_s"] != 4:
        raise RuntimeError("the synthetic schedule did not keep the injected E_s")
    fold_calls = [item for item in calls if item.get("kind") == "fold"]
    if len(fold_calls) != 5:
        raise RuntimeError("the synthetic schedule did not record five fold calls")
    for fold, call in zip(scale["folds"], fold_calls):
        if call["epochs"] != 4 or call["restore_best"] or call["select_on_held_out_fold"] or call["drop_last"]:
            raise RuntimeError("a fold call did not use the fixed E_s without held-out selection")
        if not np.array_equal(call["train_idx"], fold["train_idx"]):
            raise RuntimeError("a fold call did not receive that file's training rows")
    if sum(item.get("kind") == "score" for item in calls) != 1:
        raise RuntimeError("the base score was not taken once from the restored full model")
    generator = np.random.default_rng(0)
    y_oof = generator.normal(size=scale["n"])
    features = generator.normal(size=(scale["n"], 14)).astype(np.float32)
    representation = generator.normal(size=(scale["n"], 4)).astype(np.float32)
    train_batch = {
        "site": torch.tensor(representation),
        "source": torch.tensor(representation),
        "other": torch.tensor(representation),
        "values": torch.tensor(features),
        "mask": torch.ones(scale["n"], 14),
    }
    y_valid = np.array([0.0, 1.0, 2.0, 3.0])
    valid_features = np.zeros((4, 14), dtype=np.float32)
    valid_rep = np.zeros((4, 4), dtype=np.float32)
    valid_batch = {
        "site": torch.tensor(valid_rep),
        "source": torch.tensor(valid_rep),
        "other": torch.tensor(valid_rep),
        "values": torch.tensor(valid_features),
        "mask": torch.ones(4, 14),
    }

    def fit_fold(train_idx, val_idx, epochs):
        if int(epochs) != scheduled["e_s"]:
            raise RuntimeError("OOF fit was not given the shared E_s")
        return np.zeros(len(val_idx), dtype=np.float64)

    protocol = oof.run_seed_protocol(
        scale="n1000",
        outer_seed=seed,
        e_s=scheduled["e_s"],
        loaded_scale=scale,
        y_oof=y_oof,
        train_batch=train_batch,
        y_valid=y_valid,
        s_valid=scheduled["s_mu_f"],
        valid_batch=valid_batch,
        fit_fold=fit_fold,
        mode="synthetic",
        max_epochs=2,
        patience=1,
        batch_size=250,
        embed_dim=4,
    )
    reference_state = residual.shared_initial_state(4, seed)
    first = protocol["arms"][residual.ARMS[0]]
    for arm in residual.ARMS:
        item = protocol["arms"][arm]
        if not _states_equal(item["initial_state"], reference_state):
            raise RuntimeError("an arm did not start from the shared initial state")
        if not _states_equal(item["selection_initial"], item["retrain_initial"]):
            raise RuntimeError("selection and retraining did not start from the same state")
        if not torch.equal(item["selection_orders"][0], first["selection_orders"][0]):
            raise RuntimeError("arms did not share the selection row order")
        if not torch.equal(item["selection_first_dropout_mask"], first["selection_first_dropout_mask"]):
            raise RuntimeError("arms did not share the selection dropout mask")
        fused = predict.fuse(scheduled["s_mu_f"], item["eta"], predict.residual_vector(item["model"], {**valid_batch, "mask": residual.arm_mask(valid_batch["mask"], arm)}))
        if not np.allclose(item["prediction"], fused):
            raise RuntimeError("the external sum was not s_μF + η r")
    if protocol["e_s"] != scheduled["e_s"] or not np.array_equal(protocol["s_mu_f"], scheduled["s_mu_f"]):
        raise RuntimeError("the residual arms did not share E_s and the base score")
    y_test = np.arange(20, dtype=np.float64)
    sequences = np.array([f"s{index:02d}" for index in range(20)])
    baseline = np.stack([-y_test for _ in range(10)])
    dcr = np.stack([y_test for _ in range(10)])
    synthetic_metrics = metrics.evaluate_paired(
        y_test,
        sequences,
        baseline,
        dcr,
        n_bootstrap=20,
        bootstrap_seed=0,
    )
    return {
        "mode": "synthetic",
        "precheck_scales": sorted(loaded),
        "fold_calls": len(fold_calls),
        "e_s": scheduled["e_s"],
        "shared_base_score": True,
        "shared_initial_state": True,
        "external_sum": True,
        "k_for_999": metrics.top_fraction_k(metrics.N_TEST_STIFLER),
        "formal_bootstrap_default": metrics.FORMAL_BOOTSTRAP,
        "formal_training_started": False,
        "formal_metrics_computed": False,
        "ten_seed_entry_refused": True,
        "synthetic_spearman_passes_on_constructed_arrays": synthetic_metrics["spearman_passes"],
    }
