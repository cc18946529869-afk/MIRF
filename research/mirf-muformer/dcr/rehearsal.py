"""Synthetic walk-through of the sealed split, fold calls, residual arms and metrics.

``mode`` other than ``synthetic`` is refused. This does not start the ten-seed
μFormer run and does not score a real experiment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

import metrics
import oof
import outer_train
import predict
import residual
import sealed_config


class _RecordingTrainer:
    def __init__(self, request, calls, e_s: int, valid_score: np.ndarray, valid_sequences: np.ndarray,
            test_score: np.ndarray, test_sequences: np.ndarray):
        self.request = request
        self.calls = calls
        self.best_epochs = []
        self.epochs_completed = []
        self.best_epoch_per_model = []
        self.restored_best_weights = []
        self.restored_earlier_weights = []
        self._train_row_index = None
        self._e_s = int(e_s)
        self._valid_score = np.asarray(valid_score, dtype=np.float64)
        self._test_score = np.asarray(test_score, dtype=np.float64)
        self.valid_loader = ("validation", id(self))
        self.test_loader = ("test", id(self))
        self.valid_df = pd.DataFrame({
            "sequence": np.asarray(valid_sequences),
            "score": self._valid_score,
        })
        self.test_df = pd.DataFrame({
            "sequence": np.asarray(test_sequences),
            "score": np.arange(len(test_sequences), dtype=np.float64),
        })
        outer_train.apply_muformer_standin(self, request["muformer_config"])

    def train(self, epochs, eval_freq, patience, save_checkpoint, resume_path, restore_best=True,
            train_row_index=None, drop_last=True):
        if self.request["kind"] == "fold":
            recorded = np.asarray(train_row_index, dtype=np.int64).copy()
        else:
            recorded = np.asarray(self.request["train_idx"], dtype=np.int64).copy()
        train_arguments = {
            "epochs": int(epochs),
            "eval_freq": int(eval_freq),
            "patience": int(patience),
            "save_checkpoint": save_checkpoint,
            "resume_path": resume_path,
            "restore_best": bool(restore_best),
            "train_row_index": None if train_row_index is None else np.asarray(train_row_index, dtype=np.int64).copy(),
            "drop_last": bool(drop_last),
        }
        self.calls.append(
            {
                "kind": self.request["kind"],
                "fold": self.request["fold"],
                "epochs": int(epochs),
                "patience": int(patience),
                "restore_best": bool(restore_best),
                "train_idx": recorded,
                "drop_last": bool(drop_last),
                "select_on_held_out_fold": bool(self.request["select_on_held_out_fold"]),
                "train_arguments": train_arguments,
            }
        )
        if train_row_index is None:
            self._train_row_index = None
        else:
            self._train_row_index = np.asarray(train_row_index, dtype=np.int64).copy()
        if restore_best:
            self.best_epochs.append(self._e_s)
            self.epochs_completed.append(int(self._e_s))
            self.best_epoch_per_model.append(int(self._e_s))
            self.restored_best_weights.append(True)
            self.restored_earlier_weights.append(False)
        elif self.best_epochs:
            raise RuntimeError("fold training kept a held-out best epoch")
        else:
            self.epochs_completed.append(int(epochs))
            self.best_epoch_per_model.append(None)
            self.restored_best_weights.append(False)
            self.restored_earlier_weights.append(False)

    def score_validation(self, valid_loader, valid_df, save_prediction=False, mode="ensemble"):
        if valid_loader is None or valid_df is None:
            raise RuntimeError("validation scoring requires the outer validation set")
        if not np.array_equal(np.asarray(valid_df["sequence"]), np.asarray(self.valid_df["sequence"])):
            raise RuntimeError("validation scoring received a different split")
        self.calls.append({"kind": "score", "fold": self.request["fold"], "split": "validation"})
        frame = valid_df.copy()
        frame["prediction"] = self._valid_score
        return {"df": frame}

    def score_test(self, test_loader, test_df, save_prediction=False, mode="ensemble"):
        if test_loader is None or test_df is None:
            raise RuntimeError("test scoring requires the test set to be passed explicitly")
        if not np.array_equal(np.asarray(test_df["sequence"]), np.asarray(self.test_df["sequence"])):
            raise RuntimeError("test scoring received a different split")
        self.calls.append({"kind": "score_test", "fold": self.request["fold"], "split": "test"})
        frame = test_df.copy()
        frame["prediction"] = self._test_score
        return {"df": frame}


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
    trainers = []
    valid_sequences = np.array(["v0", "v1", "v2", "v3"])
    valid_score = np.array([0.1, 0.4, 0.2, 0.8], dtype=np.float64)
    test_sequences = np.array(["t0", "t1", "t2", "t3", "t4", "t5"])
    test_score = np.arange(test_sequences.shape[0], dtype=np.float64)

    def factory(request):
        trainer = _RecordingTrainer(
            request,
            calls,
            e_s=4,
            valid_score=valid_score,
            valid_sequences=valid_sequences,
            test_score=test_score,
            test_sequences=test_sequences,
        )
        trainers.append(trainer)
        return trainer

    scheduled = outer_train.train_outer_models("n1000", seed, scale, factory, mode="synthetic")
    if scheduled["formal_training_started"] or scheduled["e_s"] != 4:
        raise RuntimeError("the synthetic schedule did not keep the injected E_s")
    fold_calls = [item for item in calls if item.get("kind") == "fold"]
    full_calls = [item for item in calls if item.get("kind") == "full"]
    if len(fold_calls) != 5:
        raise RuntimeError("the synthetic schedule did not record five fold calls")
    if (
        len(full_calls) != 1
        or full_calls[0]["epochs"] != outer_train.BASELINE_EPOCHS
        or full_calls[0]["patience"] != outer_train.BASELINE_PATIENCE
        or not full_calls[0]["restore_best"]
    ):
        raise RuntimeError("the full model train call is not epochs 300, patience 300, and restore_best")
    for item in full_calls + fold_calls:
        if set(item["train_arguments"]) != sealed_config.MUFORMER_TRAIN_KEYS:
            raise RuntimeError("train received a parameter outside the original set")
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
    if scheduled.get("s_mu_f_split") != "validation":
        raise RuntimeError("the scheduled score is not the outer validation score")
    if not np.array_equal(scheduled["s_mu_f_sequences"], valid_sequences):
        raise RuntimeError("the scheduled score is not aligned to the validation sequences")
    if scheduled["s_mu_f"].shape == test_score.shape:
        raise RuntimeError("the validation score was not distinguished from the test score")
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

    recorded_residual = []
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
        valid_sequences=valid_sequences,
        score_sequences=scheduled["s_mu_f_sequences"],
        batch_sequences=valid_sequences,
        fit_fold=fit_fold,
        mode="synthetic",
        embed_dim=4,
        baseline_best_epoch=scheduled["e_s"],
        e_s_source=sealed_config.SEALED.e_s_source,
        residual_fit=oof.record_residual_call(recorded_residual),
    )
    cfg = sealed_config.require()
    if scheduled["e_s"] == cfg.seed42_recorded_epoch:
        raise RuntimeError("the synthetic baseline reused seed 42's recorded epoch")
    if len(recorded_residual) != 3:
        raise RuntimeError("the three residual arms were not called with the sealed config")
    for item in recorded_residual:
        if (
            item["optimizer"] != cfg.optimizer
            or item["learning_rate"] != cfg.learning_rate
            or item["weight_decay"] != cfg.weight_decay
            or item["batch_size"] != cfg.batch_size
            or item["max_epochs"] != cfg.max_epochs
            or item["patience"] != cfg.patience
            or item["scheduler"] is not None
            or item["early_stop_metric"] != cfg.early_stop_metric
        ):
            raise RuntimeError("the rehearsal substituted the sealed residual hyperparameters")
        if item["patience"] == 1 or item["batch_size"] == 250 or item["max_epochs"] == 2:
            raise RuntimeError("the rehearsal substituted the sealed residual hyperparameters")
        if item["train_rows"] != 900 or item["val_rows"] != 100:
            raise RuntimeError("the residual call did not receive the saved internal split")
    muformer_configs = []
    for trainer in trainers:
        muformer = sealed_config.assert_muformer_config(trainer.request["muformer_config"])
        residual_cfg = sealed_config.assert_residual_config(trainer.request["residual_config"])
        muformer_configs.append(muformer)
        if muformer["batch_size"] != 16 or residual_cfg["batch_size"] != 64:
            raise RuntimeError("the μFormer batch size 16 and the residual batch size 64 were mixed")
        if (
            trainer.batch_size != muformer["batch_size"]
            or trainer.hidden_size != muformer["hidden"]
            or trainer.dropout != muformer["dropout"]
            or trainer.decoder_name != muformer["decoder_name"]
            or trainer.args.encoder_lr != muformer["encoder_lr"]
            or trainer.args.decoder_lr != muformer["decoder_lr"]
            or trainer.args.weight_decay != muformer["weight_decay"]
            or trainer.args.warmup_epochs != muformer["warmup_epochs"]
            or trainer.batch_size == residual_cfg["batch_size"]
            or trainer.hidden_size == cfg.hidden
            or trainer.dropout == cfg.dropout
            or "hidden" in residual_cfg
            or "dropout" in residual_cfg
        ):
            raise RuntimeError("the stand-in did not keep the μFormer constructor fields apart from the residual readout")
    if any(item != muformer_configs[0] for item in muformer_configs):
        raise RuntimeError("the full model and the fold models do not share one muformer_config")
    reference_state = residual.shared_initial_state(4, seed)
    for arm in residual.ARMS:
        item = protocol["arms"][arm]
        if not _states_equal(item["initial_state"], reference_state):
            raise RuntimeError("an arm did not start from the shared initial state")
        if not _states_equal(item["selection_initial"], item["retrain_initial"]):
            raise RuntimeError("selection and retraining did not start from the same state")
        fused = predict.fuse(scheduled["s_mu_f"], item["eta"], predict.residual_vector(item["model"], {**valid_batch, "mask": residual.arm_mask(valid_batch["mask"], arm)}))
        if not np.allclose(item["prediction"], fused):
            raise RuntimeError("the external sum was not s_μF + η r")
    if protocol["e_s"] != scheduled["e_s"] or not np.array_equal(protocol["s_mu_f"], scheduled["s_mu_f"]):
        raise RuntimeError("the residual arms did not share E_s and the base score")
    if sum(item.get("kind") == "score_test" for item in calls) != 0:
        raise RuntimeError("the test split was scored before eta was locked")
    test_rows = outer_train.score_restored_test_split(trainers[0])
    if test_rows["split"] != "test" or test_rows["score"].shape == scheduled["s_mu_f"].shape:
        raise RuntimeError("test scoring was not separated from the validation score")
    if np.array_equal(test_rows["sequences"], valid_sequences):
        raise RuntimeError("test sequences were not distinguished from validation sequences")
    y_test_rows = np.arange(test_rows["score"].shape[0], dtype=np.float64)
    for arm in residual.ARMS:
        fused_test = oof.fuse_test_after_eta(
            eta_locked=True,
            eta=protocol["arms"][arm]["eta"],
            y_test=y_test_rows,
            s_test=test_rows["score"],
            residual_test=np.zeros(test_rows["score"].shape, dtype=np.float64),
            label_sequences=test_rows["sequences"],
            score_sequences=test_rows["sequences"],
            residual_sequences=test_rows["sequences"],
            mode="synthetic",
        )
        if fused_test["formal_metrics_computed"] or fused_test["formal_experiment"]:
            raise RuntimeError("synthetic test fusion was reported as a formal result")
        if fused_test["prediction"].shape != test_rows["score"].shape:
            raise RuntimeError("test fusion did not stay on the test rows")
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
        "sealed_residual_calls": len(recorded_residual),
        "sealed_patience": cfg.patience,
        "sealed_batch_size": cfg.batch_size,
        "sealed_max_epochs": cfg.max_epochs,
        "muformer_batch_size": int(trainers[0].batch_size),
        "residual_batch_size": int(trainers[0].request["residual_config"]["batch_size"]),
        "muformer_hidden_size": int(trainers[0].hidden_size),
        "muformer_dropout": trainers[0].dropout,
        "muformer_decoder_name": trainers[0].decoder_name,
        "muformer_encoder_lr": trainers[0].args.encoder_lr,
        "muformer_decoder_lr": trainers[0].args.decoder_lr,
        "muformer_weight_decay": trainers[0].args.weight_decay,
        "muformer_warmup_epochs": int(trainers[0].args.warmup_epochs),
        "k_for_999": metrics.top_fraction_k(metrics.N_TEST_STIFLER),
        "formal_bootstrap_default": metrics.FORMAL_BOOTSTRAP,
        "formal_training_started": False,
        "formal_metrics_computed": False,
        "ten_seed_entry_refused": True,
        "synthetic_spearman_passes_on_constructed_arrays": synthetic_metrics["spearman_passes"],
    }
