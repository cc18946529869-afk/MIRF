"""The best μFormer weights stay on the same model instance."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

SRC = Path(__file__).resolve().parents[3] / "Mu-Protein" / "mu-former" / "src"
sys.path.insert(0, str(SRC))
import trainer


class Score(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.weight * x


class BestWeightTests(unittest.TestCase):
    def test_non_best_evaluation_does_not_clear_the_epoch(self) -> None:
        record = trainer.BestEpochWeights()
        record.note(True, {"w": 1}, {"o": 1}, 3, {"loss": 1.0})
        record.note(False, None, None, 4, {"loss": 2.0})
        self.assertEqual(record.best_epoch, 3)
        self.assertEqual(record.best_model_state, {"w": 1})

    def test_train_restores_the_best_state_into_the_same_module(self) -> None:
        inner = Score()
        host = nn.Module()
        host.add_module("module", inner)
        holder = trainer.Trainer.__new__(trainer.Trainer)
        holder.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
        holder.distributed = False
        holder.local_rank = 0
        holder.metric = "corr"
        holder.batch_size = 2
        holder.args = SimpleNamespace(warmup_epochs=0)
        holder.models = [host]
        holder.optimizers = [torch.optim.Adam(inner.parameters(), lr=1e-4)]
        rows = [0]
        holder._prepare_dataloaders = lambda: (([0], rows), ([0], rows))
        holder._init_scheduler = lambda *args: SimpleNamespace(step=lambda: None)
        scores = [0.1, 0.05, 0.4, 0.2]
        weights = [1.0, 2.0, 3.0, 4.0]
        state = {"i": 0}

        def train_epoch(*args):
            model = args[1]
            with torch.no_grad():
                model.module.weight.fill_(weights[state["i"]])
            return 0.0

        def fake_test(**kwargs):
            score = scores[state["i"]]
            state["i"] += 1
            return {"loss": 0.5, "metric": {"corr": score}}

        holder._train_epoch = train_epoch
        holder.test = fake_test
        holder.train(epochs=4, eval_freq=1, patience=30, save_checkpoint=False, resume_path=None)
        self.assertEqual(holder.best_epochs, [3])
        self.assertTrue(torch.equal(inner.weight.detach().cpu(), torch.tensor([3.0])))
        self.assertTrue(host.module is inner)
        score = inner(torch.tensor([2.0]))
        self.assertTrue(torch.allclose(score, torch.tensor([6.0])))

    def test_torch_load_stays_commented_and_the_score_formula_is_unchanged(self) -> None:
        trainer_text = (SRC / "trainer.py").read_text()
        for line in trainer_text.splitlines():
            code = line.split("#", 1)[0]
            self.assertNotIn("torch.load(", code)
        self.assertIn("optim.Adam(", trainer_text)
        self.assertIn("InverseSqrtLRDecay", trainer_text)
        self.assertIn("self.criterion(output.view(-1), y_real)", trainer_text)
        model_text = (SRC / "model.py").read_text()
        self.assertIn("output = (s_sem + s_motif + s_prob) / 2", model_text)

    def test_fixed_epoch_training_does_not_reload_an_earlier_best_state(self) -> None:
        inner = Score()
        host = nn.Module()
        host.add_module("module", inner)
        holder = trainer.Trainer.__new__(trainer.Trainer)
        holder.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
        holder.distributed = False
        holder.local_rank = 0
        holder.metric = "corr"
        holder.batch_size = 2
        holder.args = SimpleNamespace(warmup_epochs=0)
        holder.models = [host]
        holder.optimizers = [torch.optim.Adam(inner.parameters(), lr=1e-4)]
        rows = [0]
        holder._prepare_dataloaders = lambda: (([0], rows), ([0], rows))
        holder._init_scheduler = lambda *args: SimpleNamespace(step=lambda: None)
        weights = [1.0, 2.0, 3.0, 4.0]
        state = {"i": 0}

        def train_epoch(*args):
            model = args[1]
            with torch.no_grad():
                model.module.weight.fill_(weights[state["i"]])
            state["i"] += 1
            return 0.0

        def fake_test(**kwargs):
            raise AssertionError("held-out rows were used to select a fold epoch")

        holder._train_epoch = train_epoch
        holder.test = fake_test
        holder.train(epochs=4, eval_freq=1, patience=30, save_checkpoint=False, resume_path=None, restore_best=False)
        self.assertEqual(holder.best_epochs, [])
        self.assertTrue(torch.equal(inner.weight.detach().cpu(), torch.tensor([4.0])))

    def test_score_restored_does_not_load_a_checkpoint(self) -> None:
        import inspect

        source = inspect.getsource(trainer.Trainer.score_restored)
        self.assertNotIn("load_pretrained_model", source)
        self.assertNotIn("checkpoint_dir", source)
        holder = trainer.Trainer.__new__(trainer.Trainer)

        def boom(*args, **kwargs):
            raise AssertionError("load_pretrained_model was called")

        seen = {}

        def fake_test(**kwargs):
            seen.update(kwargs)
            return {"loss": 0.0, "metric": {"corr": 0.0}}

        holder.load_pretrained_model = boom
        holder.test = fake_test
        holder.score_restored(mode="ensemble", save_prediction=False)
        self.assertNotIn("checkpoint_dir", seen)
        main_text = (SRC.parents[0] / "main.py").read_text()
        train_at = main_text.index("trainer.train(")
        after_train = main_text[train_at:]
        score_at = after_train.index("trainer.score_restored(")
        between = after_train[:score_at]
        self.assertNotIn("checkpoint_dir", between)
        self.assertNotIn("checkpoint_dir", after_train[score_at : score_at + 180])


if __name__ == "__main__":
    unittest.main()
