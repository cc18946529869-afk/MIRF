"""Synthetic checks for the residual branch and the sealed precheck schedule."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, train_test_split

DCR = Path(__file__).resolve().parent
sys.path.insert(0, str(DCR))
import frozen_pmlm
import geometry
import identity_audit
import lobes
import oof
import predict
import residual

PREP = DCR.parent / "prep"
PRECHECK = DCR / "precheck"


class SplitTests(unittest.TestCase):
    def test_precheck_files_match_the_sealed_splits_and_are_not_rewritten(self) -> None:
        before = {path.name: path.stat().st_mtime_ns for path in PRECHECK.iterdir()}
        loaded = oof.load_precheck(PRECHECK)
        after = {path.name: path.stat().st_mtime_ns for path in PRECHECK.iterdir()}
        self.assertEqual(before, after)
        self.assertEqual(loaded["n1000"]["n"], 1000)
        self.assertEqual(loaded["full"]["n"], 3497)
        self.assertFalse(np.array_equal(loaded["n1000"]["fold_id"], loaded["full"]["fold_id"][:1000]))
        for scale, count in (("n1000", 1000), ("full", 3497)):
            expected = list(KFold(5, shuffle=True, random_state=20261121).split(np.arange(count)))
            for fold, (train_idx, val_idx) in enumerate(expected):
                self.assertTrue(np.array_equal(loaded[scale]["folds"][fold]["train_idx"], train_idx))
                self.assertTrue(np.array_equal(loaded[scale]["folds"][fold]["val_idx"], val_idx))
            train_idx, val_idx = train_test_split(
                np.arange(count), test_size=0.1, shuffle=True, random_state=20261122, stratify=None
            )
            self.assertTrue(np.array_equal(loaded[scale]["residual_train_idx"], train_idx))
            self.assertTrue(np.array_equal(loaded[scale]["residual_val_idx"], val_idx))
        scheduled = oof.folds_for_seed(loaded["n1000"], 7)
        self.assertEqual([item["epochs"] for item in scheduled], [7, 7, 7, 7, 7])
        full_scheduled = oof.folds_for_seed(loaded["full"], 7)
        self.assertEqual(full_scheduled[0]["val_idx"].shape, (700,))
        self.assertEqual(scheduled[0]["val_idx"].shape, (200,))


class GeometryTests(unittest.TestCase):
    def test_positions_82_and_182_stay_in_the_soft_weight_set(self) -> None:
        domain = PREP / "domain_map.tsv"
        pdb = PREP / "1BTL.pdb"
        masks = geometry.residual_geometry(domain, pdb)
        previous = identity_audit.geometry(domain, pdb)
        helical = lobes.token_indices_for_lobe(domain, lobes.HELICAL_LOBE)
        other = lobes.token_indices_for_lobe(domain, lobes.OTHER_LOBE)
        self.assertEqual(int(helical.shape[0]), 151)
        self.assertEqual(int(other.shape[0]), 112)
        self.assertNotIn(82, other.tolist())
        self.assertIn(82, helical.tolist())
        self.assertIn(182, helical.tolist())
        source, other_tokens = lobes.source_and_other_tokens(domain, 24)
        self.assertIn(24, source.tolist())
        self.assertNotIn(1, source.tolist())
        self.assertNotIn(82, source.tolist())
        self.assertTrue(bool(masks["coordinate"][58]) and bool(masks["coordinate"][158]))
        self.assertFalse(bool(previous["other_known"][0, 58]))
        self.assertTrue(bool(previous["other_unknown"][0, 58]))
        self.assertTrue(bool(masks["other_known"][0, 58]))
        self.assertFalse(bool(masks["other_unknown"][0, 58]))
        distance = masks["distance"][0, 58]
        expected = 1.0 / (1.0 + np.exp((distance - 8.0) / 1.5))
        self.assertAlmostEqual(float(masks["weight"][0, 58]), float(expected))
        self.assertGreater(float(masks["weight"][0, 58]), 0.0)
        recorded = masks["record"]["mismatches"]
        self.assertEqual([(item["ref_pos"], item["wt_aa"], item["pdb_aa"], item["atom"]) for item in recorded], [(82, "V", "I", "CB"), (182, "A", "V", "CB")])
        self.assertFalse(masks["record"]["used_in_coord_equals_not_mismatched"])
        code = [line.split("#", 1)[0].strip() for line in (DCR / "geometry.py").read_text().splitlines()]
        self.assertNotIn("in_coord = ~mismatched", code)
        audit_code = [line.split("#", 1)[0].strip() for line in (DCR / "identity_audit.py").read_text().splitlines()]
        self.assertIn("in_coord = ~mismatched", audit_code)
        other_pairs = masks["other"]
        self.assertTrue(np.array_equal(masks["other_known"] | masks["other_unknown"], other_pairs))
        self.assertFalse(np.any(masks["other_known"] & masks["other_unknown"]))


class FrozenTests(unittest.TestCase):
    def test_site_token_is_the_reference_position_and_the_head_is_not_used(self) -> None:
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.marker = nn.Parameter(torch.ones(1))

            def forward(self, tokens):
                batch, length = tokens.shape
                position = torch.arange(length, dtype=torch.float32).view(length, 1, 1)
                return position.expand(length, batch, 4).clone(), {}

            def prob(self, hidden):
                raise AssertionError("the residual branch called the language-model head")

        first = Encoder()
        second = Encoder()
        frozen = frozen_pmlm.FrozenPmlm(first)
        other = frozen_pmlm.FrozenPmlm(second)
        self.assertIsNot(frozen.encoder, other.encoder)
        self.assertFalse(frozen.encoder.marker.requires_grad)
        frozen.train(True)
        self.assertFalse(frozen.training)
        tokens = torch.zeros(2, 288, dtype=torch.long)
        site, source, other_lobe = frozen.representations(
            tokens,
            82,
            torch.tensor([60, 82, 210]),
            torch.tensor([24, 286]),
        )
        self.assertTrue(torch.equal(site, torch.full((2, 4), 82.0)))
        self.assertTrue(torch.allclose(source, torch.full((2, 4), (60 + 82 + 210) / 3)))
        self.assertTrue(torch.allclose(other_lobe, torch.full((2, 4), (24 + 286) / 2)))


class PredictorTests(unittest.TestCase):
    def test_widths_masks_and_fusion_stay_outside_the_score(self) -> None:
        self.assertEqual((residual.D_S, residual.D_L, residual.D_R, residual.CONCAT_WIDTH), (64, 32, 32, 160))
        model = residual.ResidualPredictor(4)
        self.assertIsInstance(model.activation, nn.GELU)
        self.assertEqual(model.dropout.p, 0.1)
        self.assertEqual(float(model.output.weight.abs().sum()), 0.0)
        self.assertEqual(float(model.output.bias.abs().sum()), 0.0)
        self.assertFalse(torch.equal(model.site.weight, torch.zeros_like(model.site.weight)))
        values = torch.tensor([[float(index) for index in range(14)]])
        observed = torch.ones(1, 14)
        observed[0, 0] = 0.0
        packed = residual.pack_observation(values, observed)
        self.assertEqual(tuple(packed.shape), (1, 28))
        self.assertEqual(float(packed[0, 0]), 0.0)
        self.assertEqual(float(packed[0, 14]), 0.0)
        self.assertEqual(float(packed[0, 1]), 1.0)
        zero_observed = residual.pack_observation(torch.zeros(1, 14), torch.ones(1, 14))
        masked = residual.pack_observation(torch.zeros(1, 14), torch.zeros(1, 14))
        self.assertEqual(float(zero_observed[0, 0]), 0.0)
        self.assertEqual(float(zero_observed[0, 14]), 1.0)
        self.assertEqual(float(masked[0, 14]), 0.0)
        full = residual.arm_mask(torch.ones(2, 14), residual.ARM_FULL)
        summed = residual.arm_mask(torch.ones(2, 14), residual.ARM_SUM)
        none = residual.arm_mask(torch.ones(2, 14), residual.ARM_NONE)
        self.assertEqual(int(full[0].sum()), 14)
        self.assertEqual(int(summed[0].sum()), 2)
        self.assertEqual(float(summed[0, residual.DELTA_E_INDEX]), 1.0)
        self.assertEqual(float(summed[0, residual.STD_DELTA_E_INDEX]), 1.0)
        self.assertEqual(int(none.sum()), 0)
        site = torch.zeros(1, 4)
        source = torch.zeros(1, 4)
        other = torch.zeros(1, 4)
        with torch.no_grad():
            model.site.weight.zero_()
            model.site.bias.zero_()
            model.site.bias[0] = 1.0
            model.lobe.weight.zero_()
            model.lobe.bias.zero_()
            model.dcr.weight.zero_()
            model.dcr.bias.zero_()
            encoded = model.project(site, source, other, values, observed)
        self.assertEqual(tuple(encoded.shape), (1, 160))
        self.assertEqual(float(encoded[0, 0]), 1.0)
        self.assertEqual(float(encoded[0, 64:].abs().sum()), 0.0)
        with torch.no_grad():
            model.site.bias.zero_()
            model.lobe.bias[0] = 1.0
            encoded = model.project(site, source, other, values, observed)
        self.assertEqual(float(encoded[0, 64]), 1.0)
        self.assertEqual(float(encoded[0, 96]), 1.0)
        self.assertEqual(float(encoded[0, 128]), 0.0)
        score = model(site, source, other, values, observed)
        self.assertEqual(tuple(score.shape), (1,))
        fused = predict.fuse(np.array([2.0, 4.0]), 0.0, np.array([9.0, 9.0]))
        self.assertTrue(np.array_equal(fused, np.array([2.0, 4.0])))
        self.assertEqual(predict.ETA_GRID[0], 0.0)
        self.assertEqual(predict.ETA_GRID[-1], 1.0)
        self.assertEqual(len(predict.ETA_GRID), 21)
        self.assertEqual(predict.choose_eta(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]), np.zeros(3)), 0.0)
        self.assertEqual(predict.choose_eta(np.array([0.0, 1.0, 2.0]), np.zeros(3), np.array([0.0, 1.0, 2.0])), 1.0)

    def test_mae_epoch_changes_only_on_a_new_low(self) -> None:
        best_epoch, best_mae, stall, stop = None, None, 0, False
        seen = []
        for epoch, mae in enumerate((0.5, 0.4, 0.45, 0.46), start=1):
            best_epoch, best_mae, stall, stop = residual.update_mae_epoch(best_epoch, best_mae, stall, epoch, mae, 2)
            seen.append(best_epoch)
            if stop:
                break
        self.assertEqual(seen, [1, 2, 2, 2])
        self.assertTrue(stop)
        self.assertEqual(best_epoch, 2)


class ProtocolTests(unittest.TestCase):
    def test_formal_mode_is_refused(self) -> None:
        with self.assertRaises(RuntimeError):
            oof.run_seed_protocol(
                scale="n1000",
                outer_seed=oof.OUTER_SEEDS[0],
                e_s=1,
                loaded_scale={"scale": "n1000", "n": 1, "folds": [], "residual_train_idx": [], "residual_val_idx": []},
                y_oof=np.zeros(1),
                train_batch={},
                y_valid=np.zeros(2),
                s_valid=np.zeros(2),
                valid_batch={},
                fit_fold=lambda *args: None,
                mode="formal",
                embed_dim=4,
            )

    def test_three_arms_share_one_score_and_one_epoch_on_synthetic_rows(self) -> None:
        loaded = oof.load_scale(PRECHECK, "n1000")
        generator = np.random.default_rng(0)
        y_oof = generator.normal(size=1000)
        features = generator.normal(size=(1000, 14)).astype(np.float32)
        features[:, residual.DELTA_E_INDEX] = y_oof.astype(np.float32)
        representation = generator.normal(size=(1000, 4)).astype(np.float32)
        calls = []

        def fit_fold(train_idx, val_idx, epochs):
            calls.append(int(epochs))
            return np.zeros(val_idx.shape[0], dtype=np.float64)

        train_batch = {
            "site": torch.tensor(representation),
            "source": torch.tensor(representation),
            "other": torch.tensor(representation),
            "values": torch.tensor(features),
            "mask": torch.ones(1000, 14),
        }
        y_valid = np.array([0.0, 1.0, 2.0, 3.0])
        s_valid = np.array([0.1, 0.2, 0.3, 0.4])
        sequences = np.array(["v0", "v1", "v2", "v3"])
        valid_features = np.zeros((4, 14), dtype=np.float32)
        valid_features[:, residual.DELTA_E_INDEX] = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        valid_rep = np.zeros((4, 4), dtype=np.float32)
        valid_batch = {
            "site": torch.tensor(valid_rep),
            "source": torch.tensor(valid_rep),
            "other": torch.tensor(valid_rep),
            "values": torch.tensor(valid_features),
            "mask": torch.ones(4, 14),
        }
        result = oof.run_seed_protocol(
            scale="n1000",
            outer_seed=oof.OUTER_SEEDS[0],
            e_s=5,
            loaded_scale=loaded,
            y_oof=y_oof,
            train_batch=train_batch,
            y_valid=y_valid,
            s_valid=s_valid,
            valid_batch=valid_batch,
            valid_sequences=sequences,
            score_sequences=sequences,
            batch_sequences=sequences,
            fit_fold=fit_fold,
            mode="synthetic",
            max_epochs=2,
            patience=1,
            batch_size=250,
            embed_dim=4,
        )
        self.assertEqual(calls, [5, 5, 5, 5, 5])
        self.assertEqual(result["e_s"], 5)
        self.assertEqual(result["fold_epochs"], (5, 5, 5, 5, 5))
        self.assertIs(result["baseline"], s_valid)
        self.assertIs(result["s_mu_f"], s_valid)
        self.assertFalse(result["formal_metrics_computed"])
        self.assertEqual(len(np.unique(result["oof_score"])), 1)
        self.assertEqual([result["arms"][arm]["observed_slots"] for arm in residual.ARMS], [14, 2, 0])
        models = [result["arms"][arm]["model"] for arm in residual.ARMS]
        self.assertEqual(len({id(model) for model in models}), 3)
        for arm in residual.ARMS:
            item = result["arms"][arm]
            self.assertEqual(item["retrain_rows"], 1000)
            self.assertEqual(item["retrain_epochs_run"], item["best_epoch"])
            self.assertGreaterEqual(item["selection_epochs_run"], 1)
            valid_mask = residual.arm_mask(valid_batch["mask"], arm)
            correction = predict.residual_vector(item["model"], {**valid_batch, "mask": valid_mask})
            self.assertTrue(np.allclose(item["prediction"], predict.fuse(s_valid, item["eta"], correction)))
            self.assertIn("oof_residual_mean", item)
            self.assertIn("retrained_residual_mean", item)
        reference = residual.shared_initial_state(4, oof.OUTER_SEEDS[0])
        other_seed = residual.shared_initial_state(4, oof.OUTER_SEEDS[1])
        self.assertFalse(torch.equal(reference["site.weight"], other_seed["site.weight"]))
        self.assertEqual(float(reference["output.weight"].abs().sum()), 0.0)
        first = result["arms"][residual.ARMS[0]]
        for arm in residual.ARMS:
            item = result["arms"][arm]
            for key in reference:
                self.assertTrue(torch.equal(item["initial_state"][key], reference[key]))
                self.assertTrue(torch.equal(item["selection_initial"][key], item["retrain_initial"][key]))
                self.assertTrue(torch.equal(item["selection_initial"][key], reference[key]))
            self.assertTrue(torch.equal(item["selection_orders"][0], first["selection_orders"][0]))
            self.assertTrue(torch.equal(item["retrain_orders"][0], first["retrain_orders"][0]))
            self.assertTrue(torch.equal(item["selection_first_dropout_mask"], first["selection_first_dropout_mask"]))
            self.assertTrue(torch.equal(item["retrain_first_dropout_mask"], first["retrain_first_dropout_mask"]))
        self.assertFalse(torch.equal(first["selection_first_dropout_mask"], first["retrain_first_dropout_mask"]))
        selection = torch.Generator()
        selection.manual_seed(residual.stream_seed(oof.OUTER_SEEDS[0], "selection"))
        for expected in first["selection_orders"]:
            replayed = residual.shuffled_row_order(first["selection_index_source"], selection)
            self.assertTrue(torch.equal(replayed, expected))
        continued = residual.shuffled_row_order(first["retrain_index_source"], selection)
        retrain = torch.Generator()
        retrain.manual_seed(residual.stream_seed(oof.OUTER_SEEDS[0], "retrain"))
        replayed_retrain = residual.shuffled_row_order(first["retrain_index_source"], retrain)
        self.assertTrue(torch.equal(replayed_retrain, first["retrain_orders"][0]))
        self.assertFalse(torch.equal(continued, first["retrain_orders"][0]))

    def test_same_length_different_sequences_are_rejected(self) -> None:
        loaded = oof.load_scale(PRECHECK, "n1000")
        y = np.array([0.0, 1.0, 2.0])
        score = np.array([0.2, 0.1, 0.0])
        correction = np.array([1.0, 0.0, 0.5])
        left = np.array(["a", "b", "c"])
        different = np.array(["a", "b", "d"])
        reordered = np.array(["a", "c", "b"])
        called = []

        def fit_fold(train_idx, val_idx, epochs):
            called.append(1)
            return np.zeros(len(val_idx), dtype=np.float64)

        for mismatched in (different, reordered):
            with self.assertRaisesRegex(ValueError, "sequence"):
                oof.run_seed_protocol(
                    scale="n1000",
                    outer_seed=oof.OUTER_SEEDS[0],
                    e_s=5,
                    loaded_scale=loaded,
                    y_oof=np.zeros(loaded["n"]),
                    train_batch={},
                    y_valid=y,
                    s_valid=score,
                    valid_batch={},
                    valid_sequences=left,
                    score_sequences=mismatched,
                    batch_sequences=left,
                    fit_fold=fit_fold,
                    mode="synthetic",
                    embed_dim=4,
                )
        self.assertEqual(called, [])
        with self.assertRaisesRegex(ValueError, "sequence"):
            oof.select_eta(y, score, correction, left, different, left)
        with self.assertRaisesRegex(ValueError, "sequence"):
            oof.select_eta(y, score, correction, left, reordered, left)

    def test_choose_eta_does_not_read_test_labels(self) -> None:
        y_valid = np.array([0.0, 1.0, 2.0, 3.0])
        s_valid = np.array([0.0, 0.2, 0.4, 0.6])
        residual_valid = np.array([0.0, 1.0, 2.0, 3.0])
        valid_sequences = np.array(["v0", "v1", "v2", "v3"])
        y_test_values = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
        s_test = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        residual_test = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        test_sequences = np.array(["t0", "t1", "t2", "t3", "t4"])
        state = {"locked": False}
        recorded = []

        class _TestLabels:
            def __array__(self, dtype=None):
                if not state["locked"]:
                    raise AssertionError("test labels were read")
                return np.array(y_test_values, dtype=dtype, copy=True)

            def __len__(self):
                if not state["locked"]:
                    raise AssertionError("test labels were read")
                return int(y_test_values.shape[0])

            def __getitem__(self, item):
                if not state["locked"]:
                    raise AssertionError("test labels were read")
                return y_test_values[item]

            def __iter__(self):
                if not state["locked"]:
                    raise AssertionError("test labels were read")
                return iter(y_test_values)

        original = predict.choose_eta

        def spy(y, score, residual):
            recorded.append(np.asarray(y, dtype=np.float64).copy())
            eta = original(y, score, residual)
            state["locked"] = True
            return eta

        predict.choose_eta = spy
        try:
            locked = oof.lock_eta_and_fuse_test(
                y_valid=y_valid,
                s_valid=s_valid,
                residual_valid=residual_valid,
                valid_label_sequences=valid_sequences,
                valid_score_sequences=valid_sequences,
                valid_residual_sequences=valid_sequences,
                y_test=_TestLabels(),
                s_test=s_test,
                residual_test=residual_test,
                test_label_sequences=test_sequences,
                test_score_sequences=test_sequences,
                test_residual_sequences=test_sequences,
                mode="synthetic",
            )
        finally:
            predict.choose_eta = original
        self.assertEqual(len(recorded), 1)
        self.assertTrue(np.array_equal(recorded[0], y_valid))
        self.assertEqual(recorded[0].shape, y_valid.shape)
        self.assertNotEqual(recorded[0].shape, y_test_values.shape)
        self.assertFalse(np.array_equal(recorded[0], y_test_values[: y_valid.shape[0]]))
        self.assertEqual(locked["prediction"].shape, y_test_values.shape)
        self.assertNotIn("y_test", oof.select_eta.__code__.co_varnames)
        protocol_source = oof.run_seed_protocol.__code__.co_varnames
        self.assertNotIn("y_test", protocol_source)


if __name__ == "__main__":
    unittest.main()
