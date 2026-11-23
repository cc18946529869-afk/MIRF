"""The one designated n=1000 run. It is not the ten-seed entry.

``execute=False`` only checks the locked files and indices. ``execute=True``
calls the supplied factory. The default factory constructs the real Trainer
and would train; this module's tests pass a stand-in factory instead.
``run_ten_outer_seeds`` is not called. ``run_seed_protocol`` is not called.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

import feature_align
import frozen_pmlm
import metrics
import oof
import outer_train
import potts_features
import predict
import residual
import sealed_config

DESIGNATED_SCALE = "n1000"
DESIGNATED_SEED = 1975576344
HIDDEN = 1280
REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = REPO_ROOT / "Mu-Protein" / "pretrained" / "pmlm_650m.pt"
TRAIN_TSV = feature_align.N1000_TRAIN
VALID_TSV = potts_features.MAIN_SPLITS / "BLAT_ECOLX_Stiffler_2015_valid.tsv"
TEST_TSV = potts_features.MAIN_SPLITS / "BLAT_ECOLX_Stiffler_2015_test.tsv"
FASTA = TRAIN_TSV.parent / "BLAT_ECOLX_Stiffler_2015_n1000.fasta"
MARKER_NAME = "designated_run.json"


def read_fasta_sequence(path: Path) -> str:
    lines = Path(path).read_text().splitlines()
    sequence = "".join(line.strip() for line in lines if line and not line.startswith(">"))
    if not sequence:
        raise ValueError(f"{path} does not contain a sequence")
    return sequence


def _require_designated(scale: str, outer_seed: int) -> None:
    if scale != DESIGNATED_SCALE or int(outer_seed) != DESIGNATED_SEED:
        raise RuntimeError("this entry only runs n=1000 and outer seed 1975576344")


def production_trainer_kwargs(request: dict, output_dir: Path) -> dict:
    """Arguments for the real Trainer. This does not construct it."""
    config = sealed_config.assert_muformer_config(dict(request["muformer_config"]))
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"frozen checkpoint is missing: {CHECKPOINT}")
    for path in (TRAIN_TSV, VALID_TSV, TEST_TSV, FASTA):
        if not Path(path).is_file():
            raise FileNotFoundError(f"designated data file is missing: {path}")
    wild = read_fasta_sequence(FASTA)
    if len(wild) != 286:
        raise ValueError("the designated wild-type sequence is not length 286")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Both validation and test files are passed, so PairedDataset keeps those
    # files. split_ratio is not applied to the 1000 training rows.
    return {
        "output_dir": str(output),
        "train_tsv": str(TRAIN_TSV),
        "valid_tsv": str(VALID_TSV),
        "test_tsv": str(TEST_TSV),
        "fasta": str(FASTA),
        "split_ratio": [0.7, 0.1, 0.2],
        "random_seed": DESIGNATED_SEED,
        "batch_size": int(config["batch_size"]),
        "hidden_size": int(config["hidden"]),
        "dropout": float(config["dropout"]),
        "save_log": False,
        "num_workers": 0,
        "aug_ratio": -1,
        "encoder_name": "pmlm",
        "pretrained_model": str(CHECKPOINT),
        "loss": "mae",
        "lr": None,
        "label_oper": "minus",
        "decoder_name": config["decoder_name"],
        "freeze_lm": False,
        "args": argparse.Namespace(
            encoder_lr=float(config["encoder_lr"]),
            decoder_lr=float(config["decoder_lr"]),
            weight_decay=float(config["weight_decay"]),
            warmup_epochs=int(config["warmup_epochs"]),
            skip_bad_lines=False,
        ),
    }


def load_trainer_class():
    """Import the real Trainer. Importing it does not call ``train``."""
    source = REPO_ROOT / "Mu-Protein" / "mu-former" / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    importlib.import_module("protein")
    return importlib.import_module("trainer").Trainer


def note_built_adam_groups(trainer) -> None:
    """Record the learning rate and weight decay of the Adam groups already built.

    ``execute_muformer_request`` checks this attribute before ``train``. The
    two dictionaries are not added to the ``train`` arguments.
    """
    optimizers = getattr(trainer, "optimizers", None)
    if not optimizers or len(optimizers) != 1:
        raise RuntimeError("the real Trainer does not have one Adam optimizer")
    groups = optimizers[0].param_groups
    if len(groups) != 2:
        raise RuntimeError("the real Trainer Adam does not have two parameter groups")
    trainer.adam_groups = (
        {"lr": groups[0]["lr"], "weight_decay": groups[0]["weight_decay"]},
        {"lr": groups[1]["lr"], "weight_decay": groups[1]["weight_decay"]},
    )


def production_trainer_factory(request: dict, *, trainer_cls=None, output_dir: Path):
    """Construct one Trainer for a request. The caller decides whether to train."""
    kwargs = production_trainer_kwargs(request, output_dir)
    cls = load_trainer_class() if trainer_cls is None else trainer_cls
    trainer = cls(**kwargs)
    note_built_adam_groups(trainer)
    return trainer


def _claim_once(output_dir: Path) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / MARKER_NAME
    if marker.exists():
        raise RuntimeError("the designated single-seed run was already started")
    marker.write_text(json.dumps({
        "scale": DESIGNATED_SCALE,
        "outer_seed": DESIGNATED_SEED,
        "started": True,
    }) + "\n")


def _batch_from_rows(values, mask, embedded) -> dict:
    site = embedded["site"]
    source = embedded["source"]
    other = embedded["other"]
    rows = int(values.shape[0])
    if (
        tuple(site.shape) != (rows, HIDDEN)
        or tuple(source.shape) != (rows, HIDDEN)
        or tuple(other.shape) != (rows, HIDDEN)
        or tuple(values.shape) != (rows, 14)
        or tuple(mask.shape) != (rows, 14)
    ):
        raise ValueError("the residual batch is not aligned 14-d features plus 1280-d frozen states")
    if embedded["metadata"]["hidden"] != HIDDEN or embedded["metadata"]["fc_reduce_called"] or embedded["metadata"]["lm_head_called"]:
        raise RuntimeError("the frozen representation metadata is not the 1280-d readout")
    return {
        "site": site,
        "source": source,
        "other": other,
        "values": torch.as_tensor(values, dtype=torch.float32),
        "mask": torch.as_tensor(mask, dtype=torch.float32),
    }


def fit_three_arms(
    *,
    train_batch,
    train_scores,
    train_sequences,
    oof_predictions,
    oof_sequences,
    valid_batch,
    valid_scores,
    valid_base,
    valid_sequences,
    test_batch,
    test_scores,
    test_sequences,
    test_score_fn,
    train_idx,
    val_idx,
    outer_seed: int,
) -> dict:
    """Train the three sealed arms, lock each η, then score the test rows.

    ``test_score_fn`` is called only after every arm has chosen η. It returns
    the μFormer scores and sequences for the test rows. Test labels are not
    an argument of η selection. ``run_seed_protocol`` is not used.
    """
    target = torch.as_tensor(
        _residual_target(train_scores, oof_predictions, train_sequences, oof_sequences),
        dtype=torch.float32,
    )
    initial = residual.shared_initial_state(HIDDEN, int(outer_seed))
    reference = {key: value.detach().cpu().clone() for key, value in initial.items()}
    chosen = []
    for arm in residual.ARMS:
        train_mask = residual.arm_mask(train_batch["mask"], arm)
        fitted = residual.fit_predictor(
            {**train_batch, "mask": train_mask},
            target,
            train_idx,
            val_idx,
            embed_dim=HIDDEN,
            outer_seed=int(outer_seed),
            initial_state=initial,
        )
        valid_mask = residual.arm_mask(valid_batch["mask"], arm)
        valid_r = predict.residual_vector(fitted["model"], {**valid_batch, "mask": valid_mask})
        eta = oof.select_eta(valid_scores, valid_base, valid_r, valid_sequences, valid_sequences, valid_sequences)
        chosen.append((arm, fitted, float(eta)))
    test_base, scored_sequences = test_score_fn()
    test_base = np.asarray(test_base, dtype=np.float64)
    if [str(item) for item in scored_sequences] != [str(item) for item in test_sequences]:
        raise RuntimeError("the test score is not on the designated test sequences")
    if test_base.shape != (len(test_sequences),):
        raise ValueError("the test μFormer score does not match the test sequences")
    arms = {}
    for arm, fitted, eta in chosen:
        test_mask = residual.arm_mask(test_batch["mask"], arm)
        test_r = predict.residual_vector(fitted["model"], {**test_batch, "mask": test_mask})
        fused = predict.fuse(test_base, eta, test_r)
        scored = metrics.evaluate_paired(
            test_scores,
            test_sequences,
            test_base.reshape(1, -1),
            np.asarray(fused, dtype=np.float64).reshape(1, -1),
        )
        arms[arm] = {
            "eta": float(eta),
            "best_epoch": int(fitted["best_epoch"]),
            "initial_state": fitted["initial_state"],
            "selection_orders": fitted["selection_orders"],
            "retrain_orders": fitted["retrain_orders"],
            "n_bootstrap": int(scored["n_bootstrap"]),
            "formal_experiment": bool(scored["formal_experiment"]),
            "spearman_passes": bool(scored["spearman_passes"]),
        }
    first = [arms[arm]["selection_orders"][0] for arm in residual.ARMS]
    return {
        "arms": list(residual.ARMS),
        "shared_init": all(
            torch.equal(arms[arm]["initial_state"][key], reference[key])
            for arm in residual.ARMS
            for key in reference
        ),
        "shared_first_order": all(torch.equal(order, first[0]) for order in first),
        "best_epochs": {arm: arms[arm]["best_epoch"] for arm in residual.ARMS},
        "etas": {arm: arms[arm]["eta"] for arm in residual.ARMS},
        "n_bootstrap": {arm: arms[arm]["n_bootstrap"] for arm in residual.ARMS},
        "spearman_passes": {arm: bool(arms[arm]["spearman_passes"]) for arm in residual.ARMS},
        "formal_experiment": False,
        "ten_seed_conclusion": False,
    }


def _residual_target(scores, predictions, label_sequences, prediction_sequences) -> np.ndarray:
    import single_seed

    return single_seed.oof_residual(scores, predictions, label_sequences, prediction_sequences)


def run_designated_single_seed(
    *,
    execute: bool = False,
    scale: str = DESIGNATED_SCALE,
    outer_seed: int = DESIGNATED_SEED,
    trainer_factory=None,
    output_dir: Path | None = None,
    compute_embeddings: bool = False,
) -> dict:
    """Check or run the one designated seed.

    The default does not construct Trainer and does not train. Passing
    ``execute=True`` without a factory uses the real Trainer and would train;
    callers in this round pass a stand-in factory. Other scales and seeds are
    refused. A second ``execute=True`` in the same output directory is refused.
    """
    _require_designated(scale, outer_seed)
    import single_seed

    prepared = single_seed.prepare_single_outer_seed(scale, outer_seed)
    if prepared["outer_seed"] != DESIGNATED_SEED or prepared["rows"] != 1000:
        raise RuntimeError("the designated preflight is not n=1000 seed 1975576344")
    if not execute:
        return {
            "scale": DESIGNATED_SCALE,
            "outer_seed": DESIGNATED_SEED,
            "training_started": False,
            "muformer_train_called": False,
            "oof_predictions_computed": False,
            "embeddings_computed": False,
            "arms_trained": False,
            "formal_ten_seed_started": False,
            "ten_seed_entry_called": False,
            "rows": 1000,
            "executable": True,
            "executed": False,
        }
    if output_dir is None:
        raise RuntimeError("an output directory is required before the designated run starts")
    _claim_once(output_dir)
    loaded = oof.load_scale(Path(__file__).resolve().parent / "precheck", DESIGNATED_SCALE)
    keepers = {}
    upstream = trainer_factory

    def factory(request):
        if upstream is None:
            created = production_trainer_factory(request, output_dir=output_dir)
        else:
            created = upstream(request)
        if request.get("kind") == "full":
            keepers["full"] = created
        return created

    trained = outer_train.train_one_outer_seed(
        DESIGNATED_SCALE,
        DESIGNATED_SEED,
        loaded,
        factory,
        training_sequences=prepared["sequence"],
    )
    if trained["training_started"] is not True or trained["muformer_train_called"] is not True:
        raise RuntimeError("train was called but the result still says training did not start")
    if trained["formal_training_started"]:
        raise RuntimeError("the designated run reported the ten-seed formal experiment")
    result = {
        "scale": DESIGNATED_SCALE,
        "outer_seed": DESIGNATED_SEED,
        "training_started": True,
        "muformer_train_called": True,
        "formal_training_started": False,
        "formal_ten_seed_started": False,
        "ten_seed_entry_called": False,
        "e_s": trained["e_s"],
        "oof_predictions_computed": bool(trained["oof_predictions_computed"]),
        "oof_rows": None if trained["oof_predictions"] is None else int(np.asarray(trained["oof_predictions"]).shape[0]),
        "embeddings_computed": False,
        "arms_trained": False,
        "executed": True,
        "executable": True,
    }
    if not compute_embeddings:
        return result
    if trained["oof_predictions"] is None:
        raise RuntimeError("residual arms require one OOF prediction per training row")
    index = feature_align.FeatureIndex(feature_align.table_path())
    model = frozen_pmlm.load_independent_pretrained(str(CHECKPOINT))
    if model.training or any(bool(parameter.requires_grad) for parameter in model.parameters()):
        raise RuntimeError("the PMLM used by the residual arms is not frozen")
    train_embedded = frozen_pmlm.embed_sequences(
        model, prepared["sequence"], index, checkpoint=CHECKPOINT, batch_size=2,
    )
    valid_sequences, valid_scores = feature_align.read_scored_tsv(VALID_TSV)
    test_sequences, test_scores = feature_align.read_scored_tsv(TEST_TSV)
    valid_aligned = feature_align.align_scored_sequences(index, valid_sequences, valid_scores)
    test_aligned = feature_align.align_scored_sequences(index, test_sequences, test_scores)
    valid_embedded = frozen_pmlm.embed_sequences(
        model, valid_aligned["sequence"], index, checkpoint=CHECKPOINT, batch_size=2,
    )
    test_embedded = frozen_pmlm.embed_sequences(
        model, test_aligned["sequence"], index, checkpoint=CHECKPOINT, batch_size=2,
    )
    valid_base = np.asarray(trained["s_mu_f"], dtype=np.float64)
    valid_names = [str(item) for item in np.asarray(trained["s_mu_f_sequences"]).tolist()]
    if valid_names != valid_aligned["sequence"] or valid_base.shape != (len(valid_names),):
        raise RuntimeError("the restored full model was not scored on the designated validation sequences")
    full_trainer = keepers.get("full")
    if full_trainer is None or not hasattr(full_trainer, "score_test"):
        raise RuntimeError("the full model cannot score the test split")

    def test_score_fn():
        test_scored = full_trainer.score_test(
            full_trainer.test_loader,
            full_trainer.test_df,
            save_prediction=False,
            mode="ensemble",
        )
        names = [str(item) for item in test_scored["df"]["sequence"].tolist()]
        values = np.asarray(test_scored["df"]["prediction"], dtype=np.float64)
        return values, names

    arms = fit_three_arms(
        train_batch=_batch_from_rows(prepared["values"], prepared["mask"], train_embedded),
        train_scores=prepared["score"],
        train_sequences=prepared["sequence"],
        oof_predictions=trained["oof_predictions"],
        oof_sequences=prepared["sequence"],
        valid_batch=_batch_from_rows(valid_aligned["values"], valid_aligned["mask"], valid_embedded),
        valid_scores=valid_aligned["score"],
        valid_base=valid_base,
        valid_sequences=valid_aligned["sequence"],
        test_batch=_batch_from_rows(test_aligned["values"], test_aligned["mask"], test_embedded),
        test_scores=test_aligned["score"],
        test_sequences=test_aligned["sequence"],
        test_score_fn=test_score_fn,
        train_idx=loaded["residual_train_idx"],
        val_idx=loaded["residual_val_idx"],
        outer_seed=DESIGNATED_SEED,
    )
    result["embeddings_computed"] = True
    result["arms_trained"] = True
    result["arms"] = arms
    return result
