"""The five-fold entry records rows and epochs and refuses a formal ten-seed run."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import oof
import outer_train
import predict

SRC = Path(__file__).resolve().parents[3] / "Mu-Protein" / "mu-former" / "src"
sys.path.insert(0, str(SRC))
import data as mu_data
import trainer as mu_trainer


class _Factory:
    def __init__(self, e_s: int):
        self.e_s = int(e_s)
        self.requests = []
        self.trainers = []

    def __call__(self, request):
        trainer = _Trainer(request, self.e_s)
        self.requests.append(request)
        self.trainers.append(trainer)
        return trainer


class _Trainer:
    def __init__(self, request, e_s: int):
        self.request = request
        self.best_epochs = []
        self.e_s = int(e_s)
        self.scored = False
        self.valid_sequences = np.array([f"v{index}" for index in range(6)])
        self.valid_df = pd.DataFrame({"sequence": self.valid_sequences, "score": np.zeros(6)})
        self.valid_loader = ("validation", id(self))
        self.test_sequences = np.array([f"t{index}" for index in range(9)])
        self.test_df = pd.DataFrame({"sequence": self.test_sequences, "score": np.zeros(9)})
        self.test_loader = ("test", id(self))

    def train(self, epochs, eval_freq, patience, save_checkpoint, resume_path, restore_best=True,
            train_row_index=None, drop_last=True):
        self.epochs = int(epochs)
        self.restore_best = bool(restore_best)
        self.train_row_index = None if train_row_index is None else np.asarray(train_row_index, dtype=np.int64).copy()
        self.drop_last = bool(drop_last)
        if restore_best:
            self.best_epochs.append(self.e_s)

    def score_validation(self, valid_loader, valid_df, save_prediction=False, mode="ensemble"):
        if valid_loader is None or valid_df is None:
            raise AssertionError("validation set was not passed")
        if valid_loader is not self.valid_loader:
            raise AssertionError("validation scoring did not receive the validation loader")
        if not np.array_equal(np.asarray(valid_df["sequence"]), self.valid_sequences):
            raise AssertionError("validation scoring received a different split")
        self.scored = True
        self.scored_split = "validation"
        return {
            "predictions": np.linspace(0.0, 1.0, self.valid_sequences.shape[0]),
            "sequence": np.asarray(valid_df["sequence"]),
        }

    def score_test(self, test_loader, test_df, save_prediction=False, mode="ensemble"):
        raise AssertionError("test scoring ran while scheduling the outer model")


class _SplitTrainer:
    """Validation length 4 and test length 6, with different sequences."""

    def __init__(self):
        self.valid_sequences = np.array(["v0", "v1", "v2", "v3"])
        self.test_sequences = np.array(["t0", "t1", "t2", "t3", "t4", "t5"])
        self.valid_score = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        self.test_score = np.zeros(6, dtype=np.float64)
        self.valid_df = pd.DataFrame({"sequence": self.valid_sequences, "score": self.valid_score})
        self.test_df = pd.DataFrame({"sequence": self.test_sequences, "score": np.arange(6, dtype=np.float64)})
        self.valid_loader = ("validation",)
        self.test_loader = ("test",)
        self.best_epochs = []
        self.validation_calls = []
        self.test_calls = []

    def train(self, epochs, eval_freq, patience, save_checkpoint, resume_path, restore_best=True,
            train_row_index=None, drop_last=True):
        if restore_best:
            self.best_epochs.append(2)

    def score_validation(self, valid_loader, valid_df, save_prediction=False, mode="ensemble"):
        if valid_loader is not self.valid_loader:
            raise AssertionError("validation scoring did not receive the validation loader")
        if not np.array_equal(np.asarray(valid_df["sequence"]), self.valid_sequences):
            raise AssertionError("validation scoring did not receive the validation sequences")
        self.validation_calls.append(valid_loader)
        frame = valid_df.copy()
        frame["prediction"] = self.valid_score
        return {"df": frame}

    def score_test(self, test_loader, test_df, save_prediction=False, mode="ensemble"):
        if test_loader is not self.test_loader:
            raise AssertionError("test scoring did not receive the test loader")
        if not np.array_equal(np.asarray(test_df["sequence"]), self.test_sequences):
            raise AssertionError("test scoring did not receive the test sequences")
        self.test_calls.append(test_loader)
        frame = test_df.copy()
        frame["prediction"] = self.test_score
        return {"df": frame}


class OuterTrainTests(unittest.TestCase):
    def test_formal_ten_seed_entry_refuses_before_a_factory_runs(self) -> None:
        with self.assertRaises(RuntimeError):
            outer_train.run_ten_outer_seeds()
        loaded = oof.load_scale(Path(__file__).resolve().parent / "precheck", "n1000")
        factory = _Factory(5)
        with self.assertRaises(RuntimeError):
            outer_train.train_outer_models("n1000", oof.OUTER_SEEDS[0], loaded, factory, mode="formal")
        self.assertEqual(factory.requests, [])

    def test_folds_are_the_complement_and_share_one_e_s(self) -> None:
        precheck = Path(__file__).resolve().parent / "precheck"
        loaded = oof.load_scale(precheck, "n1000")
        full = oof.load_scale(precheck, "full")
        factory = _Factory(6)
        result = outer_train.train_outer_models("n1000", oof.OUTER_SEEDS[1], loaded, factory, mode="synthetic")
        self.assertEqual(result["e_s"], 6)
        self.assertFalse(result["formal_training_started"])
        self.assertEqual(len(result["folds"]), 5)
        self.assertTrue(factory.trainers[0].scored)
        self.assertTrue(factory.trainers[0].restore_best)
        self.assertIsNone(factory.trainers[0].train_row_index)
        self.assertTrue(factory.trainers[0].drop_last)
        self.assertEqual(factory.trainers[0].epochs, outer_train.BASELINE_EPOCHS)
        for trainer, fold in zip(factory.trainers[1:], loaded["folds"]):
            self.assertFalse(trainer.scored)
            self.assertFalse(trainer.restore_best)
            self.assertFalse(trainer.drop_last)
            self.assertEqual(trainer.epochs, 6)
            train_idx = trainer.train_row_index
            val_idx = trainer.request["val_idx"]
            self.assertEqual(set(train_idx.tolist()), set(range(1000)) - set(val_idx.tolist()))
            self.assertTrue(np.array_equal(train_idx, fold["train_idx"]))
            self.assertEqual(train_idx.shape, (800,))
        with self.assertRaises(ValueError):
            outer_train.train_outer_models("n1000", oof.OUTER_SEEDS[1], full, _Factory(6), mode="synthetic")
        self.assertEqual(factory.trainers[0].scored_split, "validation")
        for request in factory.requests:
            self.assertNotIn("batch_size", request)
            for key in ("encoder_lr", "decoder_lr", "weight_decay", "hidden", "dropout", "siamese", "warmup"):
                self.assertNotIn(key, request)
        self.assertNotEqual(result["e_s"], 229)
        self.assertEqual([item["epochs_run"] for item in result["folds"]], [6, 6, 6, 6, 6])

    def test_batch_size_64_and_seed_42_epoch_do_not_enter_muformer(self) -> None:
        trainer = _Trainer({"restore_best": True}, 6)
        with self.assertRaises(RuntimeError):
            outer_train.execute_muformer_request(
                trainer,
                {
                    "restore_best": True,
                    "epochs": 2,
                    "eval_freq": 1,
                    "patience": 1,
                    "batch_size": 64,
                    "select_on_held_out_fold": False,
                },
            )
        self.assertFalse(hasattr(trainer, "epochs"))
        fold_trainer = _Trainer({"restore_best": False}, 6)
        with self.assertRaises(RuntimeError):
            outer_train.execute_muformer_request(
                fold_trainer,
                {
                    "restore_best": False,
                    "epochs": 229,
                    "eval_freq": 1,
                    "patience": 300,
                    "train_idx": np.array([0, 1]),
                    "select_on_held_out_fold": False,
                    "baseline_best_epoch": 6,
                    "e_s_source": "this_outer_seed_baseline",
                },
            )
        self.assertFalse(hasattr(fold_trainer, "epochs"))
        held_out = _Trainer({"restore_best": False}, 6)
        with self.assertRaises(RuntimeError):
            outer_train.execute_muformer_request(
                held_out,
                {
                    "restore_best": False,
                    "epochs": 6,
                    "eval_freq": 1,
                    "patience": 300,
                    "train_idx": np.array([0, 1]),
                    "select_on_held_out_fold": True,
                    "baseline_best_epoch": 6,
                    "e_s_source": "this_outer_seed_baseline",
                },
            )
        self.assertFalse(hasattr(held_out, "epochs"))

    def test_validation_and_test_lengths_stay_separate_and_eta_uses_validation(self) -> None:
        trainer = _SplitTrainer()
        produced = outer_train.execute_muformer_request(
            trainer,
            {"restore_best": True, "epochs": 2, "eval_freq": 1, "patience": 1},
        )
        self.assertEqual(produced["split"], "validation")
        self.assertEqual(produced["score"].shape, (4,))
        self.assertTrue(np.array_equal(produced["sequences"], trainer.valid_sequences))
        self.assertEqual(trainer.test_calls, [])
        test_rows = outer_train.score_restored_test_split(trainer)
        self.assertEqual(test_rows["split"], "test")
        self.assertEqual(test_rows["score"].shape, (6,))
        self.assertTrue(np.array_equal(test_rows["sequences"], trainer.test_sequences))
        self.assertEqual(len(trainer.validation_calls), 1)
        self.assertEqual(len(trainer.test_calls), 1)
        y_valid = np.array([0.0, 1.0, 2.0, 3.0])
        residual_valid = np.zeros(4, dtype=np.float64)
        y_test = np.arange(6, dtype=np.float64)
        s_test = np.zeros(6, dtype=np.float64)
        residual_test = y_test.copy()

        class _Unread:
            def __array__(self, dtype=None):
                raise AssertionError("test labels were read before eta was locked")

            def __len__(self):
                raise AssertionError("test labels were read before eta was locked")

            def __getitem__(self, item):
                raise AssertionError("test labels were read before eta was locked")

            def __iter__(self):
                raise AssertionError("test labels were read before eta was locked")

        with self.assertRaises(RuntimeError):
            oof.fuse_test_after_eta(
                eta_locked=False,
                eta=None,
                y_test=_Unread(),
                s_test=_Unread(),
                residual_test=_Unread(),
                label_sequences=_Unread(),
                score_sequences=_Unread(),
                residual_sequences=_Unread(),
                mode="synthetic",
            )
        with self.assertRaises(RuntimeError):
            oof.fuse_test_after_eta(
                eta_locked=True,
                eta=0.0,
                y_test=_Unread(),
                s_test=_Unread(),
                residual_test=_Unread(),
                label_sequences=_Unread(),
                score_sequences=_Unread(),
                residual_sequences=_Unread(),
                mode="formal",
            )
        locked = oof.lock_eta_and_fuse_test(
            y_valid=y_valid,
            s_valid=produced["score"],
            residual_valid=residual_valid,
            valid_label_sequences=produced["sequences"],
            valid_score_sequences=produced["sequences"],
            valid_residual_sequences=produced["sequences"],
            y_test=y_test,
            s_test=test_rows["score"],
            residual_test=residual_test,
            test_label_sequences=test_rows["sequences"],
            test_score_sequences=test_rows["sequences"],
            test_residual_sequences=test_rows["sequences"],
            mode="synthetic",
        )
        self.assertEqual(locked["eta"], 0.0)
        self.assertEqual(
            locked["eta"],
            predict.choose_eta(y_valid, produced["score"], residual_valid),
        )
        self.assertNotEqual(locked["eta"], predict.choose_eta(y_test, s_test, residual_test))
        self.assertEqual(locked["prediction"].shape, (6,))
        self.assertTrue(np.array_equal(locked["prediction"], test_rows["score"]))
        self.assertFalse(locked["formal_metrics_computed"])
        self.assertFalse(locked["formal_experiment"])
        with self.assertRaises(ValueError):
            oof.select_eta(
                y_valid,
                test_rows["score"],
                residual_valid,
                produced["sequences"],
                test_rows["sequences"],
                produced["sequences"],
            )

    def test_fold_rows_enter_the_training_batches(self) -> None:
        rows = 10
        frame = pd.DataFrame({
            "sequence": [f"A{index}" for index in range(rows)],
            "score": np.arange(rows, dtype=np.float32),
            "row_id": np.arange(rows, dtype=np.int64),
        })
        train_idx = np.array([1, 3, 4, 7, 8], dtype=np.int64)
        dataset = mu_data.PairedDataset.__new__(mu_data.PairedDataset)
        dataset.train_df = frame
        dataset.valid_df = frame.iloc[:1].copy()
        dataset.can_resample = False
        dataset.distributed = False

        def build_data(mode, return_df=False):
            source = dataset.train_df if mode == "train" else dataset.valid_df
            df = source.copy()
            samples = []
            for position in range(len(df)):
                row_id = int(df["row_id"].iloc[position])
                samples.append({
                    "sequence": df["sequence"].iloc[position],
                    "label": torch.tensor(float(row_id)),
                    "x": torch.tensor([float(row_id)]),
                    "mask": torch.tensor([1.0]),
                })
            native = {
                "sequence": "A",
                "label": torch.tensor(0.0),
                "x": torch.tensor([0.0]),
                "mask": torch.tensor([1.0]),
            }
            built = mu_data.MetagenesisData(samples, mode, native, fasta=None, label_oper="minus", aug_ratio=-1)
            if return_df:
                return built, df
            return built

        dataset.build_data = build_data
        holder = mu_trainer.Trainer.__new__(mu_trainer.Trainer)
        holder.dataset = dataset
        holder.batch_size = 2
        holder.num_workers = 0
        holder.shuffle = False
        holder.distributed = False
        holder.device = torch.device("cpu")
        holder.clip = 1.0
        holder.criterion = nn.MSELoss()
        holder._train_row_index = train_idx
        holder._drop_last = False
        model = _RowModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = SimpleNamespace(step=lambda: None)
        for _epoch in range(2):
            model.seen = []
            holder._train_epoch(0, model, optimizer, scheduler, None, _epoch + 1)
            seen = [int(value) for value in torch.cat(model.seen).tolist()]
            self.assertEqual(sorted(seen), sorted(int(value) for value in train_idx.tolist()))
        self.assertEqual(len(dataset.train_df), rows)
        holder._drop_last = True
        model.seen = []
        holder._train_epoch(0, model, optimizer, scheduler, None, 3)
        dropped = [int(value) for value in torch.cat(model.seen).tolist()]
        self.assertEqual(len(dropped), len(train_idx) - 1)


class _RowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.seen = []

    def forward(self, x1, x2, x1_mask=None, x2_mask=None):
        self.seen.append(x1.detach().view(-1).cpu().clone())
        return self.bias.expand(x1.shape[0])


if __name__ == "__main__":
    unittest.main()
