"""The single-seed path reads real rows and does not start training."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

import engineering_chain
import oof
import outer_train
import pmlm_coords
import single_seed
import test_outer_train


class SingleSeedTests(unittest.TestCase):
    def test_prepare_aligns_n1000_and_does_not_train(self) -> None:
        prepared = single_seed.prepare_single_outer_seed("n1000", oof.OUTER_SEEDS[0])
        full = single_seed.prepare_single_outer_seed("full", oof.OUTER_SEEDS[0])
        self.assertEqual(full["rows"], 3497)
        self.assertEqual(tuple(full["values"].shape), (3497, 14))
        self.assertFalse(full["training_started"])
        self.assertEqual(prepared["outer_seed"], 1975576344)
        self.assertFalse(prepared["training_started"])
        self.assertFalse(prepared["muformer_called"])
        self.assertFalse(prepared["oof_predictions_computed"])
        self.assertFalse(prepared["arms_trained"])
        self.assertEqual(prepared["arms"], ["full", "sum", "no_potts"])
        self.assertEqual(prepared["residual_config"]["batch_size"], 64)
        self.assertEqual(prepared["rows"], 1000)
        self.assertEqual(tuple(prepared["values"].shape), (1000, 14))
        self.assertEqual(tuple(prepared["mask"].shape), (1000, 14))
        self.assertEqual(prepared["sealed_batch_size"], 64)
        self.assertEqual(prepared["comparison_batch_size"], 16)
        self.assertEqual(len(prepared["folds"]), 5)
        self.assertEqual(prepared["full_request"]["epochs"], 300)
        self.assertTrue(prepared["full_request"]["restore_best"])
        self.assertEqual(len(prepared["fold_requests"]), 5)
        for request in prepared["fold_requests"]:
            self.assertIsNone(request["epochs"])
            self.assertFalse(request["restore_best"])
            self.assertEqual(len(np.intersect1d(request["train_idx"], request["val_idx"])), 0)
        with self.assertRaises(RuntimeError):
            single_seed.prepare_single_outer_seed("n1000", oof.OUTER_SEEDS[0], start_training=True)
        with self.assertRaises(RuntimeError):
            single_seed.run_single_outer_seed("n1000", oof.OUTER_SEEDS[0], start_training=True)
        with self.assertRaises(RuntimeError):
            single_seed.run_single_outer_seed(
                "full",
                oof.OUTER_SEEDS[0],
                start_training=True,
                trainer_factory=test_outer_train._Factory(4),
                loaded_scale=oof.load_scale(Path(__file__).resolve().parent / "precheck", "full"),
            )
        factory = test_outer_train._Factory(4)
        loaded = oof.load_scale(Path(__file__).resolve().parent / "precheck", "n1000")
        trained = outer_train.train_one_outer_seed(
            "n1000",
            oof.OUTER_SEEDS[0],
            loaded,
            factory,
        )
        self.assertEqual(trained["e_s"], 4)
        self.assertEqual(len(factory.requests), 6)
        self.assertEqual(factory.requests[0]["epochs"], 300)
        self.assertTrue(factory.requests[0]["restore_best"])
        for request in factory.requests[1:]:
            self.assertEqual(request["epochs"], 4)
            self.assertFalse(request["restore_best"])
        self.assertFalse(trained["formal_training_started"])
        self.assertTrue(trained["training_started"])
        self.assertTrue(trained["muformer_train_called"])
        self.assertFalse(trained["oof_predictions_computed"])

    def test_held_out_predictions_cover_every_training_row_once(self) -> None:
        loaded = oof.load_scale(Path(__file__).resolve().parent / "precheck", "n1000")

        class _PredictingFactory(test_outer_train._Factory):
            def __call__(self, request):
                trainer = super().__call__(request)
                def predict_held_out(val_idx):
                    return np.asarray(val_idx, dtype=np.float64)
                trainer.predict_held_out = predict_held_out
                return trainer

        trained = outer_train.train_one_outer_seed(
            "n1000",
            oof.OUTER_SEEDS[0],
            loaded,
            _PredictingFactory(4),
        )
        self.assertTrue(trained["oof_predictions_computed"])
        self.assertEqual(trained["oof_predictions"].shape, (1000,))
        self.assertTrue(np.array_equal(trained["oof_predictions"], np.arange(1000)))

    def test_oof_residual_requires_the_same_sequence_order(self) -> None:
        sequences = ["a", "b", "c"]
        residual = single_seed.oof_residual([1.0, 2.0, 4.0], [0.5, 0.5, 1.0], sequences, sequences)
        self.assertTrue(np.allclose(residual, [0.5, 1.5, 3.0]))
        with self.assertRaises(ValueError):
            single_seed.oof_residual([1.0, 2.0], [0.5, 0.5], sequences, ["b", "a", "c"])

    def test_n1000_stand_in_uses_real_features_and_keeps_test_labels_out(self) -> None:
        result = engineering_chain.run_n1000_stand_in()
        self.assertFalse(result["muformer_trained"])
        self.assertFalse(result["pmlm_weights_used"])
        self.assertEqual(result["rows"], 1000)
        self.assertEqual(result["oof_coverage"], 1000)
        self.assertEqual(result["feature_shape"], [1000, 14])
        self.assertTrue(np.allclose(result["values"], result["received_values"], rtol=0.0, atol=1e-6))
        self.assertEqual(result["valid_rows"], 500)
        self.assertEqual(result["test_rows"], 999)
        self.assertEqual(result["source_lengths"], [112, 151])
        self.assertTrue(result["shared_init"])
        self.assertTrue(result["shared_first_order"])
        self.assertEqual(set(result["arms"]), {"full", "sum", "no_potts"})
        self.assertIn(result["eta"], engineering_chain.predict.ETA_GRID)
        self.assertEqual(set(result["test_point"]), set(engineering_chain.metrics.METRIC_NAMES))
        with self.assertRaises(RuntimeError):
            engineering_chain.choose_eta_from_validation(
                y_valid=[0.0, 1.0],
                s_valid=[0.0, 1.0],
                residual_valid=[0.0, 0.0],
                sequences=["a", "b"],
                y_test=[0.0, 1.0],
            )

    def test_incomplete_arguments_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            single_seed.prepare_single_outer_seed("n300", oof.OUTER_SEEDS[0])
        with self.assertRaises(ValueError):
            single_seed.prepare_single_outer_seed("n1000", 1)

    def test_a_missing_oof_row_is_refused(self) -> None:
        folds = [
            {"train_idx": np.array([1, 2]), "val_idx": np.array([0]), "epochs": 1},
            {"train_idx": np.array([0, 2]), "val_idx": np.array([1]), "epochs": 1},
        ]
        with self.assertRaises(ValueError):
            oof.assemble_oof(3, folds, 1, lambda train_idx, val_idx, epochs: np.zeros(val_idx.shape))

    def test_an_unmappable_position_is_refused(self) -> None:
        dictionary = pmlm_coords.load_dictionary()
        with self.assertRaises(ValueError):
            pmlm_coords.token_index_for_reference(dictionary, "A" * 10, 24)

    def test_ten_seed_entry_and_formal_outer_training_still_refuse(self) -> None:
        with self.assertRaises(RuntimeError):
            outer_train.run_ten_outer_seeds()
        with self.assertRaises(RuntimeError):
            outer_train.train_outer_models("n1000", oof.OUTER_SEEDS[0], {"scale": "n1000"}, object(), mode="formal")
