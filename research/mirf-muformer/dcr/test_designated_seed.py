"""The designated n=1000 entry, real row prediction, and a small frozen PMLM batch."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

import designated_seed
import feature_align
import frozen_pmlm
import oof
import outer_train
import pmlm_coords
import sealed_config
import test_outer_train

CHECKPOINT = designated_seed.CHECKPOINT


class DesignatedEntryTests(unittest.TestCase):
    def test_only_the_designated_seed_can_enter(self) -> None:
        with self.assertRaises(RuntimeError):
            designated_seed.run_designated_single_seed(scale="full", outer_seed=designated_seed.DESIGNATED_SEED)
        with self.assertRaises(RuntimeError):
            designated_seed.run_designated_single_seed(scale="n1000", outer_seed=1)
        checked = designated_seed.run_designated_single_seed(execute=False)
        self.assertFalse(checked["training_started"])
        self.assertFalse(checked["executed"])
        self.assertTrue(checked["executable"])
        self.assertEqual(checked["outer_seed"], 1975576344)

    def test_execute_uses_predict_rows_and_cannot_start_twice(self) -> None:
        loaded = oof.load_scale(Path(__file__).resolve().parent / "precheck", "n1000")
        self.assertEqual(int(np.asarray(loaded["residual_train_idx"]).shape[0]), 900)
        self.assertEqual(int(np.asarray(loaded["residual_val_idx"]).shape[0]), 100)

        class _Factory(test_outer_train._Factory):
            def __call__(self, request):
                trainer = super().__call__(request)
                def predict_rows(sequences, row_index):
                    index = np.asarray(row_index, dtype=np.int64)
                    return {
                        "prediction": index.astype(np.float64),
                        "sequence": [str(item) for item in sequences],
                        "row_index": index,
                    }
                trainer.predict_rows = predict_rows
                return trainer

        with tempfile.TemporaryDirectory() as directory:
            first = designated_seed.run_designated_single_seed(
                execute=True,
                trainer_factory=_Factory(4),
                output_dir=Path(directory),
                compute_embeddings=False,
            )
            self.assertTrue(first["training_started"])
            self.assertTrue(first["muformer_train_called"])
            self.assertFalse(first["formal_training_started"])
            self.assertTrue(first["oof_predictions_computed"])
            self.assertEqual(first["oof_rows"], 1000)
            self.assertFalse(first["arms_trained"])
            with self.assertRaises(RuntimeError):
                designated_seed.run_designated_single_seed(
                    execute=True,
                    trainer_factory=_Factory(4),
                    output_dir=Path(directory),
                )
        with self.assertRaises(RuntimeError):
            outer_train.train_one_outer_seed(
                "n1000",
                oof.OUTER_SEEDS[0],
                loaded,
                test_outer_train._Factory(4),
                training_sequences=["seq"] * loaded["n"],
            )

    def test_production_factory_uses_the_real_trainer_arguments(self) -> None:
        request = {"muformer_config": sealed_config.muformer_config()}
        with tempfile.TemporaryDirectory() as directory:
            kwargs = designated_seed.production_trainer_kwargs(request, Path(directory))
        self.assertEqual(kwargs["batch_size"], 16)
        self.assertEqual(kwargs["hidden_size"], 256)
        self.assertEqual(kwargs["dropout"], 0.2)
        self.assertEqual(kwargs["decoder_name"], "siamese")
        self.assertEqual(kwargs["label_oper"], "minus")
        self.assertEqual(kwargs["aug_ratio"], -1)
        self.assertEqual(kwargs["loss"], "mae")
        self.assertIsNone(kwargs["lr"])
        self.assertFalse(kwargs["freeze_lm"])
        self.assertEqual(kwargs["random_seed"], 1975576344)
        self.assertEqual(kwargs["args"].encoder_lr, 1e-5)
        self.assertEqual(kwargs["args"].decoder_lr, 1e-4)
        self.assertEqual(kwargs["args"].weight_decay, 0.0)
        self.assertEqual(kwargs["args"].warmup_epochs, 10)
        self.assertTrue(Path(kwargs["train_tsv"]).is_file())
        self.assertTrue(Path(kwargs["valid_tsv"]).is_file())
        self.assertTrue(Path(kwargs["test_tsv"]).is_file())
        self.assertTrue(Path(kwargs["fasta"]).is_file())
        self.assertIs(designated_seed.load_trainer_class(), designated_seed.load_trainer_class())

    def test_production_factory_keeps_the_official_split_and_does_not_train(self) -> None:
        request = {"muformer_config": sealed_config.muformer_config()}
        with tempfile.TemporaryDirectory() as directory:
            trainer = designated_seed.production_trainer_factory(request, output_dir=Path(directory))
            dataset = trainer.dataset
            self.assertEqual(len(dataset.train_df), 1000)
            self.assertEqual(len(dataset.valid_df), 500)
            self.assertEqual(len(dataset.test_df), 999)
            self.assertFalse(dataset.can_resample)
            train_sequences, _scores = feature_align.read_scored_tsv(designated_seed.TRAIN_TSV)
            valid_sequences, _valid_scores = feature_align.read_scored_tsv(designated_seed.VALID_TSV)
            test_sequences, _test_scores = feature_align.read_scored_tsv(designated_seed.TEST_TSV)
            self.assertEqual(dataset.train_df["sequence"].tolist(), train_sequences)
            self.assertEqual(dataset.valid_df["sequence"].tolist(), valid_sequences)
            self.assertEqual(dataset.test_df["sequence"].tolist(), test_sequences)
            self.assertFalse(hasattr(trainer, "epochs_completed"))
            self.assertTrue(hasattr(trainer, "predict_rows"))
            outer_train._assert_trainer_carries_muformer_config(
                trainer,
                {"muformer_config": sealed_config.muformer_config()},
            )
            self.assertFalse(hasattr(trainer, "epochs_completed"))
            self.assertEqual(
                trainer.adam_groups,
                (
                    {"lr": 1e-5, "weight_decay": 0.0},
                    {"lr": 1e-4, "weight_decay": 0.0},
                ),
            )

    def test_small_real_pmlm_batch_enters_the_three_arms(self) -> None:
        self.assertTrue(CHECKPOINT.is_file())
        index = feature_align.FeatureIndex(feature_align.table_path())
        train_sequences, train_scores = feature_align.read_scored_tsv(feature_align.N1000_TRAIN)
        valid_sequences, valid_scores = feature_align.read_scored_tsv(designated_seed.VALID_TSV)
        test_sequences, test_scores = feature_align.read_scored_tsv(designated_seed.TEST_TSV)
        train_pick = self._two_lobe_rows(index, train_sequences, train_scores)
        valid_pick = self._first_rows(index, valid_sequences, valid_scores, "valid")
        test_pick = self._first_rows(index, test_sequences, test_scores, "test")
        model = frozen_pmlm.load_independent_pretrained(str(CHECKPOINT))
        self.assertFalse(model.training)
        self.assertFalse(any(bool(parameter.requires_grad) for parameter in model.parameters()))
        train_embedded = frozen_pmlm.embed_sequences(model, train_pick["sequence"], index, checkpoint=CHECKPOINT)
        valid_embedded = frozen_pmlm.embed_sequences(model, valid_pick["sequence"], index, checkpoint=CHECKPOINT)
        test_embedded = frozen_pmlm.embed_sequences(model, test_pick["sequence"], index, checkpoint=CHECKPOINT)
        self.assertEqual(tuple(train_embedded["site"].shape), (2, 1280))
        self.assertEqual(train_embedded["metadata"]["hidden"], 1280)
        self.assertEqual(len(train_embedded["metadata"]["checkpoint_sha256"]), 64)
        self.assertFalse(train_embedded["metadata"]["embeddings_stored"])
        self.assertFalse(train_embedded["metadata"]["lm_head_called"])
        calls = []

        def test_score_fn():
            calls.append("test")
            base = test_pick["values"][:, 1].astype(np.float64)
            return base, test_pick["sequence"]

        result = designated_seed.fit_three_arms(
            train_batch=designated_seed._batch_from_rows(train_pick["values"], train_pick["mask"], train_embedded),
            train_scores=train_pick["score"],
            train_sequences=train_pick["sequence"],
            oof_predictions=train_pick["values"][:, 1],
            oof_sequences=train_pick["sequence"],
            valid_batch=designated_seed._batch_from_rows(valid_pick["values"], valid_pick["mask"], valid_embedded),
            valid_scores=valid_pick["score"],
            valid_base=valid_pick["values"][:, 0],
            valid_sequences=valid_pick["sequence"],
            test_batch=designated_seed._batch_from_rows(test_pick["values"], test_pick["mask"], test_embedded),
            test_scores=test_pick["score"],
            test_sequences=test_pick["sequence"],
            test_score_fn=test_score_fn,
            train_idx=np.array([0]),
            val_idx=np.array([1]),
            outer_seed=designated_seed.DESIGNATED_SEED,
        )
        self.assertEqual(calls, ["test"])
        self.assertEqual(result["arms"], ["full", "sum", "no_potts"])
        self.assertTrue(result["shared_init"])
        self.assertTrue(result["shared_first_order"])
        self.assertEqual(set(result["n_bootstrap"].values()), {10000})
        self.assertTrue(all(value is False for value in result["spearman_passes"].values()))
        self.assertFalse(result["formal_experiment"])
        self.assertFalse(result["ten_seed_conclusion"])

    def test_real_trainer_predict_rows_does_not_train(self) -> None:
        import argparse
        import sys

        source = designated_seed.REPO_ROOT / "Mu-Protein" / "mu-former" / "src"
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        import protein  # noqa: F401
        from trainer import Trainer

        amino = "ACDEFGHIKLMNPQRSTVWY"
        rows = []
        for index in range(8):
            sequence = ["A"] * 30
            sequence[index] = amino[index]
            rows.append(("".join(sequence), float(index + 1)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / "rows.tsv"
            fasta = root / "wt.fasta"
            table.write_text("sequence\tscore\n" + "\n".join(f"{sequence}\t{score}" for sequence, score in rows) + "\n")
            fasta.write_text(">wt\n" + ("A" * 30) + "\n")
            trainer = Trainer(
                output_dir=str(root / "out"),
                train_tsv=str(table),
                valid_tsv=None,
                test_tsv=None,
                fasta=str(fasta),
                split_ratio=[0.5, 0.25, 0.25],
                batch_size=4,
                hidden_size=256,
                dropout=0.2,
                save_log=False,
                num_workers=0,
                encoder_name="pmlm",
                pretrained_model=str(CHECKPOINT),
                loss="mae",
                lr=None,
                label_oper="minus",
                decoder_name="siamese",
                freeze_lm=False,
                args=argparse.Namespace(
                    encoder_lr=1e-5,
                    decoder_lr=1e-4,
                    weight_decay=0.0,
                    warmup_epochs=10,
                    skip_bad_lines=False,
                ),
            )
            requested = [rows[0][0], rows[3][0]]
            predicted = trainer.predict_rows(requested, np.array([0, 3]))
        self.assertEqual(predicted["sequence"], requested)
        self.assertEqual(tuple(predicted["prediction"].shape), (2,))
        self.assertTrue(np.array_equal(predicted["row_index"], np.array([0, 3])))
        self.assertFalse(hasattr(trainer, "epochs_completed"))

    def _two_lobe_rows(self, index, sequences, scores):
        chosen = []
        seen = set()
        for sequence, score in zip(sequences, scores):
            row = index.row(sequence)
            if row["split"] != "train":
                continue
            source, _other = pmlm_coords.per_row_lobe_indices(pmlm_coords.DOMAIN_MAP, row["ref_pos"])
            length = int(source.shape[0])
            if length in seen:
                continue
            chosen.append((sequence, float(score)))
            seen.add(length)
            if len(chosen) == 2:
                break
        self.assertEqual(len(chosen), 2)
        return self._pack(index, chosen)

    def _first_rows(self, index, sequences, scores, split):
        chosen = []
        for sequence, score in zip(sequences, scores):
            if index.row(sequence)["split"] != split:
                continue
            chosen.append((sequence, float(score)))
            if len(chosen) == 2:
                break
        self.assertEqual(len(chosen), 2)
        return self._pack(index, chosen)

    def _pack(self, index, chosen):
        sequences = [item[0] for item in chosen]
        values, mask = index.lookup(sequences)
        return {
            "sequence": sequences,
            "score": np.array([item[1] for item in chosen], dtype=np.float64),
            "values": values,
            "mask": mask,
        }


class RefusalTests(unittest.TestCase):
    def test_synthetic_gates_and_ten_seed_entry_still_refuse(self) -> None:
        with self.assertRaises(RuntimeError):
            outer_train.run_ten_outer_seeds()
        with self.assertRaises(RuntimeError):
            oof.run_seed_protocol(
                scale="n1000",
                outer_seed=oof.OUTER_SEEDS[0],
                e_s=1,
                loaded_scale={"scale": "n1000"},
                y_oof=np.zeros(1),
                train_batch={},
                y_valid=np.zeros(1),
                s_valid=np.zeros(1),
                valid_batch={},
                fit_fold=lambda *args, **kwargs: None,
                embed_dim=8,
                mode="formal",
            )
        with self.assertRaises(RuntimeError):
            oof.fuse_test_after_eta(
                eta_locked=True,
                eta=0.0,
                y_test=[0.0],
                s_test=[0.0],
                residual_test=[0.0],
                label_sequences=["a"],
                score_sequences=["a"],
                residual_sequences=["a"],
                mode="formal",
            )
