"""Construct the real Trainer once and do not call train().

Run with CUDA_VISIBLE_DEVICES already empty, or let this file set it
before torch is imported. This script is not part of the short synthetic suite.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import traceback
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "Mu-Protein" / "mu-former" / "src"
sys.path.insert(0, str(SRC))

import protein  # registers fairseq task prot_pmlm before Trainer loads the checkpoint
from model import Muformer, SiameseMuformerDecoder
from trainer import Trainer

CHECKPOINT = REPO_ROOT / "Mu-Protein" / "pretrained" / "pmlm_650m.pt"
AMINO = "ACDEFGHIKLMNPQRSTVWY"


def write_rows(path: Path) -> None:
    rows = []
    for index in range(20):
        sequence = ["A"] * 40
        sequence[index] = AMINO[index % len(AMINO)]
        if sequence[index] == "A":
            sequence[index] = "C"
        rows.append({"sequence": "".join(sequence), "score": float(index)})
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def check(results, name, ok, detail):
    results.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}\t{name}\t{detail}")


def main() -> int:
    results = []
    print(f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}")
    print(f"cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
    print(f"checkpoint={CHECKPOINT} bytes={CHECKPOINT.stat().st_size}")
    trainer = None
    try:
        with tempfile.TemporaryDirectory(prefix="muformer-constructor-") as directory:
            root = Path(directory)
            table = root / "rows.tsv"
            output = root / "output"
            output.mkdir()
            write_rows(table)
            args = argparse.Namespace(
                encoder_lr=1e-5,
                decoder_lr=1e-4,
                weight_decay=0.0,
                warmup_epochs=10,
                skip_bad_lines=False,
            )
            trainer = Trainer(
                output_dir=str(output),
                train_tsv=str(table),
                valid_tsv=None,
                test_tsv=None,
                fasta=None,
                split_ratio=[0.7, 0.1, 0.2],
                batch_size=16,
                hidden_size=256,
                dropout=0.2,
                save_log=False,
                num_workers=0,
                encoder_name="pmlm",
                pretrained_model=str(CHECKPOINT),
                loss="mae",
                lr=None,
                decoder_name="siamese",
                freeze_lm=False,
                args=args,
            )
            dataset = trainer.dataset
            sequences = set(dataset.full_df["sequence"].tolist())
            parts = [
                set(dataset.train_df["sequence"].tolist()),
                set(dataset.valid_df["sequence"].tolist()),
                set(dataset.test_df["sequence"].tolist()),
            ]
            model = trainer.models[0].module
            optimizer = trainer.optimizers[0]
            encoder_ids = {id(item) for item in model.encoder.parameters()}
            decoder_ids = {id(item) for item in model.decoder.parameters()}
            groups = optimizer.param_groups
            group_ids = [{id(item) for item in group["params"]} for group in groups]
            written = [item.name for item in output.rglob("*") if item.is_file()]
            check(results, "device_count", torch.cuda.device_count() == 0, torch.cuda.device_count())
            check(results, "distributed", trainer.distributed is False, trainer.distributed)
            check(results, "pmlm_loaded", trainer.encoder is not None and sum(item.numel() for item in trainer.encoder.parameters()) > 0, sum(item.numel() for item in trainer.encoder.parameters()))
            check(results, "batch_size", trainer.batch_size == 16, trainer.batch_size)
            check(results, "encoder_lr", trainer.encoder_lr == 1e-5, trainer.encoder_lr)
            check(results, "decoder_lr", trainer.decoder_lr == 1e-4, trainer.decoder_lr)
            check(results, "weight_decay", trainer.weight_decay == 0, trainer.weight_decay)
            check(results, "hidden_size", trainer.hidden_size == 256 and model.hidden_size == 256, (trainer.hidden_size, model.hidden_size))
            check(results, "dropout", trainer.dropout == 0.2 and model.dropout == 0.2, (trainer.dropout, model.dropout))
            check(results, "decoder_name", trainer.decoder_name == "siamese" and isinstance(model, Muformer) and isinstance(model.decoder, SiameseMuformerDecoder), type(model.decoder).__name__)
            check(results, "adam_groups", len(trainer.optimizers) == 1 and len(groups) == 2, len(groups))
            check(results, "adam_encoder", group_ids[0] == encoder_ids and groups[0]["lr"] == 1e-5 and groups[0]["weight_decay"] == 0, (groups[0]["lr"], groups[0]["weight_decay"], len(group_ids[0])))
            check(results, "adam_decoder", group_ids[1] == decoder_ids and groups[1]["lr"] == 1e-4 and groups[1]["weight_decay"] == 0, (groups[1]["lr"], groups[1]["weight_decay"], len(group_ids[1])))
            check(results, "warmup_epochs", args.warmup_epochs == 10 and trainer.args.warmup_epochs == 10, trainer.args.warmup_epochs)
            check(results, "train_not_called", not hasattr(trainer, "epochs_completed"), hasattr(trainer, "epochs_completed"))
            check(results, "split_counts", (len(dataset.train_df), len(dataset.valid_df), len(dataset.test_df)) == (14, 2, 4), (len(dataset.train_df), len(dataset.valid_df), len(dataset.test_df)))
            check(results, "split_partition", len(sequences) == 20 and parts[0].isdisjoint(parts[1]) and parts[0].isdisjoint(parts[2]) and parts[1].isdisjoint(parts[2]) and set().union(*parts) == sequences, [len(item) for item in parts])
            check(results, "no_checkpoint", written == [], written)
    except Exception:
        traceback.print_exc()
        print("FAIL\tconstructor\traise")
        return 1
    failed = [name for name, ok, _detail in results if not ok]
    print(f"checks={len(results)} failed={len(failed)}")
    if failed:
        print("failed_names=" + ",".join(failed))
        return 1
    print("constructor_smoke=ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
