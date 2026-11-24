"""Saved stages can be reloaded, and a saved μFormer is not trained again."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

import experiment_store
import oof
import outer_train


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)


class _Holder:
    def __init__(self, factory) -> None:
        self.factory = factory
        self.models = [_Net()]

    def get_bare_model(self, net):
        return net

    def train(self, *args, **kwargs):
        self.factory.trains += 1


class _Factory:
    def __init__(self) -> None:
        self.builds = 0
        self.trains = 0

    def __call__(self, request):
        self.builds += 1
        return _Holder(self)


class ExperimentStoreTests(unittest.TestCase):
    def test_incomplete_directory_requires_resume_and_a_finished_one_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment_store.open_run(root, resume=False, schedule_only=False)
            experiment_store.mark_failed(root, "full_model", RuntimeError("stopped"))
            with self.assertRaises(RuntimeError):
                experiment_store.open_run(root, resume=False, schedule_only=False)
            resumed = experiment_store.open_run(root, resume=True, schedule_only=False)
            self.assertEqual(resumed["status"], "running")
            self.assertIsNone(resumed["error"])
            experiment_store.mark_complete(root)
            with self.assertRaises(RuntimeError):
                experiment_store.open_run(root, resume=True, schedule_only=False)

    def test_saved_muformer_stages_are_not_trained_again(self) -> None:
        loaded = oof.load_scale(Path(__file__).resolve().parent / "precheck", "n1000")
        sequences = [f"seq{index}" for index in range(int(loaded["n"]))]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment_store.open_run(root, resume=False, schedule_only=False)
            holder = _Holder(_Factory())
            experiment_store.write_full(root, holder, {
                "e_s": 4,
                "epochs_run": 4,
                "restore_best": True,
                "replaced_with_earlier_best": False,
                "split": "validation",
                "score": np.array([1.0, 2.0]),
                "sequences": np.array(["valid-a", "valid-b"]),
            })
            for fold in loaded["folds"]:
                val_idx = np.asarray(fold["val_idx"], dtype=np.int64)
                names = [sequences[int(item)] for item in val_idx.tolist()]
                experiment_store.write_fold(root, holder, int(fold["fold"]), {
                    "epochs_run": 4,
                    "restore_best": False,
                    "replaced_with_earlier_best": False,
                    "held_out_prediction": val_idx.astype(np.float64),
                    "held_out_sequence": names,
                })
            factory = _Factory()
            trained = outer_train.train_one_outer_seed(
                "n1000",
                oof.OUTER_SEEDS[0],
                loaded,
                factory,
                training_sequences=sequences,
                artifact_dir=root,
            )
            self.assertEqual(factory.trains, 0)
            self.assertEqual(factory.builds, 1)
            self.assertFalse(trained["training_started"])
            self.assertFalse(trained["muformer_train_called"])
            self.assertTrue(trained["loaded_saved_muformer"])
            self.assertEqual(int(trained["e_s"]), 4)
            np.testing.assert_array_equal(trained["oof_predictions"], np.arange(int(loaded["n"])))
            saved = experiment_store.read_oof(root, sequences)
            self.assertEqual(saved["sequence_sha256"], experiment_store.sequence_sha256(sequences))
            self.assertTrue(saved["weights_saved"])
            self.assertFalse(saved["schedule_only"])

    def test_report_keeps_the_five_metric_intervals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment_store.open_run(root, resume=False, schedule_only=False)
            metrics = {
                name: {"mean_delta": 0.1, "ci_low": -0.2, "ci_high": 0.4, "std_delta": 0.0, "n_improved": 1, "deltas": [0.1]}
                for name in ("spearman", "pearson", "ndcg", "recall", "rmse")
            }
            point = {name: 0.5 for name in metrics}
            report = experiment_store.write_report(root, {
                "scale": "n1000",
                "outer_seed": 1975576344,
                "formal_experiment": False,
                "ten_seed_conclusion": False,
                "e_s": 4,
                "oof_rows": 1000,
                "by_arm": {
                    "full": {"eta": 0.0, "best_epoch": 3, "metrics": metrics, "point": {"baseline": point, "fused": point}},
                },
            })
            text = (root / "single_seed_report.md").read_text()
            self.assertIn("spearman", text)
            self.assertIn("rmse", text)
            self.assertEqual(report["by_arm"]["full"]["metrics"]["spearman"]["ci_low"], -0.2)
            self.assertFalse(report["formal_experiment"])
            self.assertFalse(report["ten_seed_conclusion"])
