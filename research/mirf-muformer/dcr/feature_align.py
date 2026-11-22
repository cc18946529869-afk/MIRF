"""Read the saved soft-weight table by sequence identity.

This module does not recompute Potts features, does not rewrite the table,
and does not call the v0.1 writer.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import potts_features
import sealed_config

TABLE_NAME = potts_features.SOFT_WEIGHT_TABLE
FEATURE_NAMES = potts_features.FEATURE_NAMES
MASK_NAMES = tuple(f"mask_{index:02d}" for index in range(1, 15))
N1000_TRAIN = potts_features.N1000_TRAIN
FULL_TRAIN = potts_features.MAIN_SPLITS / "BLAT_ECOLX_Stiffler_2015_train.tsv"


def table_path(features: Path | None = None) -> Path:
    directory = sealed_config.features_directory() if features is None else Path(features)
    return directory / TABLE_NAME


class FeatureIndex:
    """One row per sequence. Lookup follows the caller's order."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.rows: dict[str, dict] = {}
        with self.path.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None:
                raise ValueError("the soft-weight table has no header")
            missing = [name for name in ("sequence", "split", "ref_pos", *FEATURE_NAMES, *MASK_NAMES) if name not in reader.fieldnames]
            if missing:
                raise ValueError(f"the soft-weight table is missing columns: {missing}")
            for line_number, row in enumerate(reader, start=2):
                sequence = row["sequence"]
                if sequence in self.rows:
                    raise ValueError(f"sequence at line {line_number} is duplicated in the soft-weight table")
                values = np.array([float(row[name]) for name in FEATURE_NAMES], dtype=np.float64)
                mask = np.array([float(row[name]) for name in MASK_NAMES], dtype=np.float64)
                if values.shape != (14,) or mask.shape != (14,) or not np.isin(mask, (0.0, 1.0)).all():
                    raise ValueError(f"line {line_number} does not have 14 features and a 0/1 mask")
                mature = row.get("mature_pos", "")
                self.rows[sequence] = {
                    "values": values,
                    "mask": mask,
                    "split": row["split"],
                    "ref_pos": int(row["ref_pos"]) if row["ref_pos"] != "" else None,
                    "mature_pos": int(mature) if mature != "" else None,
                    "mutation": row.get("mutation", ""),
                    "wt_aa": row.get("wt_aa", ""),
                    "mut_aa": row.get("mut_aa", ""),
                }
        if not self.rows:
            raise ValueError("the soft-weight table has no rows")

    def lookup(self, sequences) -> tuple[np.ndarray, np.ndarray]:
        """Return values and mask in the exact order of ``sequences``."""
        ordered = [str(sequence) for sequence in sequences]
        missing = [sequence for sequence in ordered if sequence not in self.rows]
        if missing:
            raise KeyError(f"sequence is not in the soft-weight table: {missing[0]}")
        values = np.stack([self.rows[sequence]["values"] for sequence in ordered])
        mask = np.stack([self.rows[sequence]["mask"] for sequence in ordered])
        return values, mask

    def row(self, sequence: str) -> dict:
        if sequence not in self.rows:
            raise KeyError(f"sequence is not in the soft-weight table: {sequence}")
        return self.rows[sequence]


def read_scored_tsv(path: Path) -> tuple[list[str], np.ndarray]:
    """Labels come from the training TSV ``score`` column, not from the feature table."""
    sequences = []
    scores = []
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or "sequence" not in reader.fieldnames or "score" not in reader.fieldnames:
            raise ValueError(f"{path} does not have sequence and score columns")
        for row in reader:
            sequences.append(row["sequence"])
            scores.append(float(row["score"]))
    if not sequences:
        raise ValueError(f"{path} has no rows")
    return sequences, np.asarray(scores, dtype=np.float64)


def align_scored_sequences(index: FeatureIndex, sequences, scores) -> dict:
    """Join score, values, and mask onto one sequence order."""
    ordered = [str(sequence) for sequence in sequences]
    labels = np.asarray(scores, dtype=np.float64)
    if labels.shape != (len(ordered),):
        raise ValueError("scores are not aligned to the requested sequences")
    values, mask = index.lookup(ordered)
    if values.shape != (len(ordered), 14) or mask.shape != values.shape:
        raise ValueError("aligned features do not have one 14-column row per sequence")
    return {
        "sequence": ordered,
        "score": labels,
        "values": values,
        "mask": mask,
    }


def alignment_report(index: FeatureIndex | None = None) -> dict:
    """Counts for the saved table. This does not build a training batch."""
    if index is None:
        index = FeatureIndex(table_path())
    valid = potts_features.MAIN_SPLITS / "BLAT_ECOLX_Stiffler_2015_valid.tsv"
    test = potts_features.MAIN_SPLITS / "BLAT_ECOLX_Stiffler_2015_test.tsv"
    by_split = {}
    for name in ("train", "valid", "test"):
        by_split[name] = sum(row["split"] == name for row in index.rows.values())
    return {
        "table": str(table_path()),
        "rows": len(index.rows),
        "unique_sequences": len(index.rows),
        "split_rows": by_split,
        "feature_names": list(FEATURE_NAMES),
        "mask_names": list(MASK_NAMES),
        "n1000_train": membership(index, N1000_TRAIN, "train"),
        "full_train": membership(index, FULL_TRAIN, "train"),
        "valid": membership(index, valid, "valid"),
        "test": membership(index, test, "test"),
        "labels_come_from": "training TSV score column",
        "lookup_key": "sequence",
    }


def membership(index: FeatureIndex, path: Path, split: str) -> dict:
    """Count how many training sequences exist in one table split. This is not row alignment."""
    sequences, _scores = read_scored_tsv(path)
    found = sum(sequence in index.rows and index.rows[sequence]["split"] == split for sequence in sequences)
    return {
        "path": str(path),
        "split": split,
        "rows": len(sequences),
        "unique": len(set(sequences)),
        "found_in_split": found,
    }


def write_alignment_report(destination: Path, features: Path | None = None) -> dict:
    """Write membership counts. The file does not contain feature values."""
    index = FeatureIndex(table_path(features))
    masked_rows = sum(not bool(np.all(row["mask"] == 1.0)) for row in index.rows.values())
    payload = alignment_report(index)
    payload["masked_rows"] = masked_rows
    payload["joined_into_a_training_batch_by_this_report"] = False
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    return payload
