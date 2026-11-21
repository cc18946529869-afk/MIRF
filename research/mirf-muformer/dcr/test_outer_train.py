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

    def train(self, epochs, eval_freq, patience, save_checkpoint, resume_path, restore_best=True,
            train_row_index=None, drop_last=True):
        self.epochs = int(epochs)
        self.restore_best = bool(restore_best)
        self.train_row_index = None if train_row_index is None else np.asarray(train_row_index, dtype=np.int64).copy()
        self.drop_last = bool(drop_last)
        if restore_best:
            self.best_epochs.append(self.e_s)

    def score_restored(self, **kwargs):
        if kwargs.get("checkpoint_dir") is not None:
            raise AssertionError("checkpoint_dir was passed")
        self.scored = True
        return {"predictions": np.linspace(0.0, 1.0, 6)}


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
