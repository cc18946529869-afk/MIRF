"""Saved stages can be reloaded, and a saved μFormer is not trained again."""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

import experiment_store
import oof
import outer_train
import sealed_config
import test_outer_train


def _hold_directory(path: str) -> None:
    experiment_store.open_run(path, resume=False, schedule_only=False)
    time.sleep(30)


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


class _Loadable:
    def __init__(self, factory, request) -> None:
        self.factory = factory
        self.models = [_Net()]
        outer_train.apply_muformer_standin(self, request["muformer_config"])

    def get_bare_model(self, net):
        return net

    def train(self, **kwargs):
        self.factory.trains += 1
        self.epochs_completed = [int(kwargs["epochs"])]
        self.train_row_index = kwargs.get("train_row_index")
        self.best_epochs = []
        self.best_epoch_per_model = [None]
        self.restored_best_weights = [False]
        self.restored_earlier_weights = [False]

    def predict_rows(self, sequences, row_index):
        index = np.asarray(row_index, dtype=np.int64)
        return {
            "prediction": index.astype(np.float64),
            "sequence": [str(item) for item in sequences],
            "row_index": index,
        }


class _LoadableFactory:
    def __init__(self) -> None:
        self.builds = 0
        self.trains = 0

    def __call__(self, request):
        self.builds += 1
        return _Loadable(self, request)


_FactoryBase = test_outer_train._Factory


def _consistent_sequences() -> list[str]:
    return [f"seq{index}" for index in range(1000)]


def write_consistent_run(root: Path, sequences, e_s: int = 4) -> None:
    """Write one finished directory whose report matches the saved artifacts."""
    names = [str(item) for item in sequences]
    if len(names) != 1000:
        raise ValueError("a consistent run has 1000 training sequences")
    experiment_store.open_run(root, resume=False, schedule_only=False)
    experiment_store.seal_pmlm_checkpoint(root, experiment_store._checkpoint_path())
    holder = _Holder(_Factory())
    experiment_store.write_full(root, holder, {
        "e_s": int(e_s),
        "epochs_run": int(e_s),
        "restore_best": True,
        "replaced_with_earlier_best": False,
        "split": "validation",
        "score": np.array([1.0, 2.0]),
        "sequences": np.array(["valid-a", "valid-b"]),
    })
    loaded = oof.load_scale(Path(__file__).resolve().parent / "precheck", "n1000")
    assembled = np.full(len(names), np.nan, dtype=np.float64)
    for fold in loaded["folds"]:
        val_idx = np.asarray(fold["val_idx"], dtype=np.int64)
        prediction = val_idx.astype(np.float64)
        held = [names[int(item)] for item in val_idx.tolist()]
        assembled[val_idx] = prediction
        experiment_store.write_fold(root, holder, int(fold["fold"]), {
            "epochs_run": int(e_s),
            "restore_best": False,
            "replaced_with_earlier_best": False,
            "held_out_prediction": prediction,
            "held_out_sequence": held,
        })
    experiment_store.write_oof(
        root,
        assembled,
        names,
        int(e_s),
        schedule_only=False,
        weights_saved=True,
    )
    experiment_store.write_checked_pmlm_identity(root, names)
    names = ("spearman", "pearson", "ndcg", "recall", "rmse")
    scored = {name: {"mean_delta": 0.1, "ci_low": -0.2, "ci_high": 0.4} for name in names}
    point = {name: 0.5 for name in names}
    arms = ("full", "sum", "no_potts")
    experiment_store.write_arms(root, {
        "arms": list(arms),
        "etas": {arm: 0.0 for arm in arms},
        "best_epochs": {arm: 1 for arm in arms},
        "n_bootstrap": {arm: 10000 for arm in arms},
        "spearman_passes": {arm: False for arm in arms},
        "metrics": {arm: scored for arm in arms},
        "point": {arm: {"baseline": point, "fused": point} for arm in arms},
        "state_dict": {arm: {"weight": torch.zeros(1)} for arm in arms},
        "shared_init": True,
        "shared_first_order": True,
    })
    experiment_store.write_report(root, {
        "scale": "n1000",
        "outer_seed": 1975576344,
        "formal_experiment": False,
        "ten_seed_conclusion": False,
        "training_started": True,
        "muformer_train_called": True,
        "e_s": int(e_s),
        "oof_rows": len(sequences),
        "by_arm": {
            arm: {
                "eta": 0.0,
                "best_epoch": 1,
                "metrics": scored,
                "point": {"baseline": point, "fused": point},
                "n_bootstrap": 10000,
                "spearman_passes": False,
            }
            for arm in arms
        },
    })


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
            self.assertTrue(experiment_store.recorded_training(root)["muformer_train_called"])
            self.assertTrue(experiment_store.recorded_training(root)["training_started"])
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

    def test_changed_precheck_file_fails_sha256(self) -> None:
        source = Path(__file__).resolve().parent / "precheck"
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "dcr" / "precheck"
            shutil.copytree(source, dest)
            target = dest / "n1000_oof_fold0_val.txt"
            target.write_text(target.read_text() + "9\n")
            with self.assertRaises(ValueError) as caught:
                oof.load_scale(dest, "n1000")
            self.assertIn("n1000_oof_fold0_val.txt", str(caught.exception))

    def test_precheck_manifest_and_index_cannot_be_replaced_together(self) -> None:
        source = Path(__file__).resolve().parent / "precheck"
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory) / "dcr" / "precheck"
            shutil.copytree(source, dest)
            target = dest / "n1000_oof_fold0_val.txt"
            target.write_text(target.read_text() + "9\n")
            rewritten = []
            for line in (dest / "SHA256SUMS").read_text().splitlines():
                if line.strip() == "":
                    continue
                _digest, name = line.split()
                rewritten.append(f"{oof.file_sha256(dest / name)}  {name}")
            (dest / "SHA256SUMS").write_text("\n".join(rewritten) + "\n")
            self.assertNotEqual(oof.file_sha256(dest / "SHA256SUMS"), sealed_config.PRECHECK_MANIFEST_SHA256)
            self.assertNotEqual(target.read_text(), (source / "n1000_oof_fold0_val.txt").read_text())
            with self.assertRaises(ValueError) as caught:
                oof.load_scale(dest, "n1000")
            self.assertIn("sealed manifest", str(caught.exception))
            self.assertNotEqual(target.read_text(), (source / "n1000_oof_fold0_val.txt").read_text())

    def test_checkpoint_cache_rehashes_a_same_size_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.bin"
            path.write_bytes(b"aaaa")
            slot: dict = {}
            first = experiment_store.sha256_cached(path, slot)
            stamp = path.stat()
            handle = os.open(path, os.O_WRONLY)
            try:
                os.write(handle, b"bbbb")
            finally:
                os.close(handle)
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            second = experiment_store.sha256_cached(path, slot)
            self.assertEqual(second, experiment_store.file_sha256(path))
            self.assertNotEqual(first, second)
            self.assertEqual(experiment_store.sha256_cached(path, slot), second)

    def test_full_training_record_is_refused_and_not_deleted(self) -> None:
        sequences = _consistent_sequences()

        def set_epochs_below_e_s(root: Path) -> None:
            path = root / "muformer" / "full.json"
            record = json.loads(path.read_text())
            record["epochs_run"] = int(record["e_s"]) - 1
            path.write_text(json.dumps(record))

        def clear_restore_best(root: Path) -> None:
            path = root / "muformer" / "full.json"
            record = json.loads(path.read_text())
            record["restore_best"] = False
            path.write_text(json.dumps(record))

        def mismatch_replaced_flag(root: Path) -> None:
            path = root / "muformer" / "full.json"
            record = json.loads(path.read_text())
            record["replaced_with_earlier_best"] = not bool(record["replaced_with_earlier_best"])
            path.write_text(json.dumps(record))

        def bool_impersonates_epoch(root: Path) -> None:
            path = root / "muformer" / "full.json"
            record = json.loads(path.read_text())
            record["e_s"] = True
            path.write_text(json.dumps(record))

        for mutate in (set_epochs_below_e_s, clear_restore_best, mismatch_replaced_flag, bool_impersonates_epoch):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_consistent_run(root, sequences)
                weights = experiment_store.weights_path(root, "full")
                record_path = root / "muformer" / "full.json"
                mutate(root)
                self.assertFalse(experiment_store.report_ready(root, sequences))
                with self.assertRaises(RuntimeError):
                    experiment_store.require_consistent_report(root, sequences)
                with self.assertRaises(RuntimeError):
                    experiment_store.read_full(root)
                self.assertTrue(weights.is_file())
                self.assertTrue(record_path.is_file())

    def test_write_full_rejects_untyped_fields_before_storage(self) -> None:
        valid = {
            "e_s": 4,
            "epochs_run": 4,
            "restore_best": True,
            "replaced_with_earlier_best": False,
            "split": "validation",
            "score": np.array([1.0, 2.0]),
            "sequences": np.array(["valid-a", "valid-b"]),
        }
        illegal = (
            {"e_s": True},
            {"epochs_run": True},
            {"restore_best": 1},
            {"restore_best": "true"},
            {"replaced_with_earlier_best": 0},
        )

        class _Unused:
            pass

        for override in illegal:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                result = dict(valid)
                result.update(override)
                with self.assertRaises(RuntimeError):
                    experiment_store.write_full(root, _Unused(), result)
                self.assertFalse((root / "muformer" / "full.json").exists())
                self.assertFalse((root / "muformer" / "full.pt").exists())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment_store.open_run(root, resume=False, schedule_only=False)
            holder = _Holder(_Factory())
            experiment_store.write_full(root, holder, valid)
            kept = (root / "muformer" / "full.json").read_text()
            rejected = dict(valid)
            rejected["e_s"] = True
            with self.assertRaises(RuntimeError):
                experiment_store.write_full(root, holder, rejected)
            self.assertEqual((root / "muformer" / "full.json").read_text(), kept)
            self.assertIsNotNone(experiment_store.read_full(root))

    def test_model_records_reject_raw_types_before_coercion(self) -> None:
        class _Trainer:
            def __init__(self, **fields):
                self.epochs_completed = fields["epochs_completed"]
                self.best_epoch_per_model = fields["best_epoch"]
                self.restored_best_weights = fields["restored_best"]
                self.restored_earlier_weights = fields["restored_earlier"]
                self.best_epochs = fields["saved_best"]

        illegal = (
            {"epochs_completed": [True], "best_epoch": [1], "restored_best": [True], "restored_earlier": [False], "saved_best": [1]},
            {"epochs_completed": [4], "best_epoch": [True], "restored_best": [True], "restored_earlier": [False], "saved_best": [1]},
            {"epochs_completed": [4], "best_epoch": [4], "restored_best": [1], "restored_earlier": [False], "saved_best": [4]},
            {"epochs_completed": [4], "best_epoch": [4], "restored_best": ["true"], "restored_earlier": [False], "saved_best": [4]},
            {"epochs_completed": [4], "best_epoch": [4], "restored_best": [True], "restored_earlier": [0], "saved_best": [4]},
        )
        for fields in illegal:
            trainer = _Trainer(**fields)
            with self.assertRaises(RuntimeError):
                outer_train._model_records(trainer)
            with self.assertRaises(RuntimeError):
                outer_train._require_full_record(trainer)

    def test_repository_directory_and_old_report_are_refused(self) -> None:
        repo = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory(dir=repo) as directory:
            with self.assertRaises(RuntimeError) as caught:
                experiment_store.open_run(directory, resume=False, schedule_only=False)
            self.assertIn("outside the repository", str(caught.exception))
            self.assertFalse((Path(directory) / "designated_run.json").exists())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "single_seed_report.json").write_text("{}")
            with self.assertRaises(RuntimeError) as caught:
                experiment_store.open_run(root, resume=False, schedule_only=False)
            self.assertIn("empty directory", str(caught.exception))
            self.assertFalse((root / "designated_run.json").exists())

    def test_resume_rejects_a_marker_for_another_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment_store.open_run(root, resume=False, schedule_only=False)
            path = root / "designated_run.json"
            record = json.loads(path.read_text())
            record["outer_seed"] = 1
            path.write_text(json.dumps(record))
            with self.assertRaises(RuntimeError) as caught:
                experiment_store.open_run(root, resume=True, schedule_only=False)
            self.assertIn("different scale or outer seed", str(caught.exception))

    def test_one_process_holds_the_directory(self) -> None:
        context = multiprocessing.get_context("fork")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = context.Process(target=_hold_directory, args=(str(root),))
            process.start()
            deadline = time.time() + 10
            while time.time() < deadline and not (root / "designated_run.json").exists():
                time.sleep(0.05)
            self.assertTrue((root / "designated_run.json").exists())
            try:
                with self.assertRaises(RuntimeError) as caught:
                    experiment_store.open_run(root, resume=True, schedule_only=False)
                self.assertIn("already holds", str(caught.exception))
            finally:
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)

    def test_interrupted_full_checkpoint_is_retrained(self) -> None:
        loaded = oof.load_scale(Path(__file__).resolve().parent / "precheck", "n1000")
        sequences = [f"seq{index}" for index in range(int(loaded["n"]))]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment_store.open_run(root, resume=False, schedule_only=False)
            weights = experiment_store.weights_path(root, "full")
            weights.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"orphan": torch.zeros(1)}, weights)
            weights.with_name(weights.name + ".partial").write_bytes(b"partial")
            self.assertIsNone(experiment_store.read_full(root))
            self.assertFalse(weights.exists())
            self.assertFalse(weights.with_name("full.pt.partial").exists())

            class _Retrain(_FactoryBase):
                def __call__(self, request):
                    trainer = super().__call__(request)
                    trainer.models = [_Net()]
                    trainer.get_bare_model = lambda net: net

                    def predict_rows(rows, row_index):
                        index = np.asarray(row_index, dtype=np.int64)
                        return {
                            "prediction": index.astype(np.float64),
                            "sequence": [str(item) for item in rows],
                            "row_index": index,
                        }

                    trainer.predict_rows = predict_rows
                    return trainer

            factory = _Retrain(4)
            trained = outer_train.train_one_outer_seed(
                "n1000",
                oof.OUTER_SEEDS[0],
                loaded,
                factory,
                training_sequences=sequences,
                artifact_dir=root,
            )
            self.assertEqual(len(factory.trainers), 6)
            self.assertEqual(int(trained["e_s"]), 4)
            self.assertIsNotNone(experiment_store.read_full(root))
            self.assertTrue(experiment_store.recorded_training(root)["muformer_train_called"])

    def test_interrupted_fold_checkpoint_retrains_only_that_fold(self) -> None:
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
                if int(fold["fold"]) == 0:
                    continue
                val_idx = np.asarray(fold["val_idx"], dtype=np.int64)
                names = [sequences[int(item)] for item in val_idx.tolist()]
                experiment_store.write_fold(root, holder, int(fold["fold"]), {
                    "epochs_run": 4,
                    "restore_best": False,
                    "replaced_with_earlier_best": False,
                    "held_out_prediction": val_idx.astype(np.float64),
                    "held_out_sequence": names,
                })
            orphan = experiment_store.weights_path(root, "fold_0")
            torch.save({"orphan": torch.zeros(1)}, orphan)
            kept = experiment_store.weights_path(root, "fold_1")
            self.assertTrue(kept.is_file())
            factory = _LoadableFactory()
            trained = outer_train.train_one_outer_seed(
                "n1000",
                oof.OUTER_SEEDS[0],
                loaded,
                factory,
                training_sequences=sequences,
                artifact_dir=root,
            )
            self.assertEqual(factory.trains, 1)
            self.assertEqual(factory.builds, 2)
            self.assertTrue(kept.is_file())
            self.assertIsNotNone(experiment_store.read_fold(root, 0))
            np.testing.assert_array_equal(trained["oof_predictions"], np.arange(int(loaded["n"])))

    def test_a_consistent_report_passes_and_each_mismatch_fails(self) -> None:
        sequences = _consistent_sequences()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_consistent_run(root, sequences)
            self.assertTrue(experiment_store.report_ready(root, sequences))
            self.assertTrue(experiment_store.arms_ready(root))

        def reject(mutate) -> None:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_consistent_run(root, sequences)
                mutate(root)
                self.assertFalse(experiment_store.report_ready(root, sequences))

        def change_full_weights(root: Path) -> None:
            weights = experiment_store.weights_path(root, "full")
            weights.write_bytes(weights.read_bytes() + b"x")

        def change_oof_sequence_sha(root: Path) -> None:
            path = root / "oof.json"
            record = json.loads(path.read_text())
            record["sequence_sha256"] = "0" * 64
            path.write_text(json.dumps(record))

        def change_oof_e_s(root: Path) -> None:
            path = root / "oof.json"
            record = json.loads(path.read_text())
            record["e_s"] = int(record["e_s"]) + 1
            path.write_text(json.dumps(record))

        def drop_one_arm(root: Path) -> None:
            path = root / "arms.json"
            record = json.loads(path.read_text())
            record["arms"] = ["full"]
            path.write_text(json.dumps(record))

        def delete_one_arm_weight(root: Path) -> None:
            (root / "arms" / "sum.pt").unlink()

        def change_report_e_s(root: Path) -> None:
            path = root / "single_seed_report.json"
            record = json.loads(path.read_text())
            record["e_s"] = int(record["e_s"]) + 1
            path.write_text(json.dumps(record))

        def change_markdown_metric(root: Path) -> None:
            path = root / "single_seed_report.md"
            text = path.read_text()
            self.assertIn("1975576344", text)
            path.write_text(text.replace("0.1", "0.9", 1))
            self.assertIn("1975576344", path.read_text())

        def change_full_score(root: Path) -> None:
            path = root / "muformer" / "full.json"
            record = json.loads(path.read_text())
            record["score"][0] = float(record["score"][0]) + 1.0
            path.write_text(json.dumps(record))

        def change_fold_prediction(root: Path) -> None:
            path = root / "muformer" / "fold_0.json"
            record = json.loads(path.read_text())
            record["held_out_prediction"][0] = float(record["held_out_prediction"][0]) + 1.0
            path.write_text(json.dumps(record))

        def change_oof_prediction(root: Path) -> None:
            path = root / "oof.json"
            record = json.loads(path.read_text())
            record["predictions"][0] = float(record["predictions"][0]) + 1.0
            path.write_text(json.dumps(record))

        def change_arm_eta(root: Path) -> None:
            path = root / "arms.json"
            record = json.loads(path.read_text())
            record["by_arm"]["full"]["eta"] = 0.5
            path.write_text(json.dumps(record))

        def change_arm_metrics(root: Path) -> None:
            path = root / "arms.json"
            record = json.loads(path.read_text())
            record["by_arm"]["full"]["metrics"]["spearman"]["mean_delta"] = 0.5
            path.write_text(json.dumps(record))

        def change_report_metric_and_markdown(root: Path) -> None:
            path = root / "single_seed_report.json"
            record = json.loads(path.read_text())
            record["by_arm"]["full"]["metrics"]["spearman"]["mean_delta"] = 0.5
            path.write_text(json.dumps(record))
            (root / "single_seed_report.md").write_text(experiment_store._report_markdown(record))

        def change_report_oof_rows(root: Path) -> None:
            path = root / "single_seed_report.json"
            record = json.loads(path.read_text())
            record["oof_rows"] = 999
            path.write_text(json.dumps(record))
            (root / "single_seed_report.md").write_text(experiment_store._report_markdown(record))

        def change_fold_epochs(root: Path) -> None:
            path = root / "muformer" / "fold_0.json"
            record = json.loads(path.read_text())
            record["epochs_run"] = int(record["epochs_run"]) - 1
            path.write_text(json.dumps(record))

        def change_fold_restore_best(root: Path) -> None:
            path = root / "muformer" / "fold_0.json"
            record = json.loads(path.read_text())
            record["restore_best"] = True
            path.write_text(json.dumps(record))

        def change_fold_replaced_best(root: Path) -> None:
            path = root / "muformer" / "fold_1.json"
            record = json.loads(path.read_text())
            record["replaced_with_earlier_best"] = True
            path.write_text(json.dumps(record))

        def change_fold_id(root: Path) -> None:
            path = root / "muformer" / "fold_0.json"
            record = json.loads(path.read_text())
            record["fold"] = 1
            path.write_text(json.dumps(record))

        def shorten_fold_prediction(root: Path) -> None:
            path = root / "muformer" / "fold_0.json"
            record = json.loads(path.read_text())
            record["held_out_prediction"] = record["held_out_prediction"][:2]
            record["held_out_prediction_sha256"] = experiment_store.vector_sha256(record["held_out_prediction"])
            path.write_text(json.dumps(record))

        def change_fold_sequences(root: Path) -> None:
            path = root / "muformer" / "fold_0.json"
            record = json.loads(path.read_text())
            record["held_out_sequence"] = [f"not-training-{index}" for index in range(len(record["held_out_sequence"]))]
            record["sequence_sha256"] = experiment_store.sequence_sha256(record["held_out_sequence"])
            path.write_text(json.dumps(record))

        def overlap_fold_sequences(root: Path) -> None:
            first = json.loads((root / "muformer" / "fold_0.json").read_text())
            path = root / "muformer" / "fold_1.json"
            record = json.loads(path.read_text())
            record["held_out_sequence"] = list(first["held_out_sequence"])
            record["sequence_sha256"] = experiment_store.sequence_sha256(record["held_out_sequence"])
            path.write_text(json.dumps(record))

        def drop_one_fold_row(root: Path) -> None:
            path = root / "muformer" / "fold_2.json"
            record = json.loads(path.read_text())
            record["held_out_prediction"] = record["held_out_prediction"][:-1]
            record["held_out_sequence"] = record["held_out_sequence"][:-1]
            record["held_out_prediction_sha256"] = experiment_store.vector_sha256(record["held_out_prediction"])
            record["sequence_sha256"] = experiment_store.sequence_sha256(record["held_out_sequence"])
            path.write_text(json.dumps(record))

        def change_oof_but_keep_its_hash(root: Path) -> None:
            path = root / "oof.json"
            record = json.loads(path.read_text())
            record["predictions"][0] = float(record["predictions"][0]) + 1.0
            record["predictions_sha256"] = experiment_store.vector_sha256(record["predictions"])
            path.write_text(json.dumps(record))

        def delete_pmlm_identity(root: Path) -> None:
            (root / "pmlm_identity.json").unlink()

        def shorten_full_score(root: Path) -> None:
            path = root / "muformer" / "full.json"
            record = json.loads(path.read_text())
            record["score"] = record["score"][:-1]
            record["score_sha256"] = experiment_store.vector_sha256(record["score"])
            path.write_text(json.dumps(record))

        reject(change_full_weights)
        reject(change_oof_sequence_sha)
        reject(change_oof_e_s)
        reject(drop_one_arm)
        reject(delete_one_arm_weight)
        reject(change_report_e_s)
        reject(change_markdown_metric)
        reject(change_full_score)
        reject(change_fold_prediction)
        reject(change_oof_prediction)
        reject(change_arm_eta)
        reject(change_arm_metrics)
        reject(change_report_metric_and_markdown)
        reject(change_report_oof_rows)
        reject(change_fold_epochs)
        reject(change_fold_restore_best)
        reject(change_fold_replaced_best)
        reject(change_fold_id)
        reject(shorten_fold_prediction)
        reject(change_fold_sequences)
        reject(overlap_fold_sequences)
        reject(drop_one_fold_row)
        reject(change_oof_but_keep_its_hash)
        reject(delete_pmlm_identity)
        reject(shorten_full_score)

    def test_a_non_finite_prediction_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                experiment_store.write_oof(
                    root,
                    np.array([np.nan]),
                    ["seq"],
                    1,
                    schedule_only=False,
                    weights_saved=True,
                )

    def test_a_tampered_fold_prediction_is_not_reused(self) -> None:
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
            path = root / "muformer" / "fold_0.json"
            record = json.loads(path.read_text())
            record["held_out_prediction"][0] = float(record["held_out_prediction"][0]) + 1.0
            path.write_text(json.dumps(record))
            factory = _Factory()
            with self.assertRaises(RuntimeError):
                outer_train.train_one_outer_seed(
                    "n1000",
                    oof.OUTER_SEEDS[0],
                    loaded,
                    factory,
                    training_sequences=sequences,
                    artifact_dir=root,
                )
            self.assertEqual(factory.trains, 0)
