"""Write the precheck OOF and internal-split indices. No model training."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.model_selection import KFold, train_test_split

OUT = Path(__file__).resolve().parent
MIRF = OUT.parents[3]
MUFORMER = Path("/data/cc233_data/work_dir/Muformer_DPFunc/new-Mu-Protein-main/data/dms")
OUTER_SEEDS = [
    1975576344,
    94706340,
    1220745215,
    508376101,
    1910795809,
    141521756,
    1119615152,
    929659347,
    120967996,
    1571307030,
]
KNOWN = {42, 20261121, 20261122, 20261123, 2847319051}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    lines = path.read_text().splitlines()
    header = lines[0].split("\t")
    rows = [dict(zip(header, line.split("\t"))) for line in lines[1:]]
    if any(len(line.split("\t")) != len(header) for line in lines[1:]):
        raise ValueError(f"{path} has a ragged row")
    return header, rows


def write_indices(path: Path, values: np.ndarray) -> str:
    path.write_text("".join(f"{int(value)}\n" for value in values))
    return sha256(path)


def sequences(rows: list[dict[str, str]]) -> list[str]:
    if any("sequence" not in row for row in rows):
        raise ValueError("a row has no sequence column")
    return [row["sequence"] for row in rows]


def main() -> None:
    scales = {
        "n1000": {
            "train": MUFORMER / "BLAT_ECOLX_Stiffler_2015_n1000" / "BLAT_ECOLX_Stiffler_2015_n1000_train.tsv",
            "valid": MUFORMER / "BLAT_ECOLX_Stiffler_2015_n1000" / "BLAT_ECOLX_Stiffler_2015_n1000_valid.tsv",
            "test": MUFORMER / "BLAT_ECOLX_Stiffler_2015_n1000" / "BLAT_ECOLX_Stiffler_2015_n1000_test.tsv",
        },
        "full": {
            "train": MUFORMER / "BLAT_ECOLX_Stiffler_2015" / "BLAT_ECOLX_Stiffler_2015_train.tsv",
            "valid": MUFORMER / "BLAT_ECOLX_Stiffler_2015" / "BLAT_ECOLX_Stiffler_2015_valid.tsv",
            "test": MUFORMER / "BLAT_ECOLX_Stiffler_2015" / "BLAT_ECOLX_Stiffler_2015_test.tsv",
        },
    }
    proteingym = {
        name: MIRF / "proteingym_v0.1_split" / "subs" / f"BLAT_ECOLX_Stiffler_2015_{name}.tsv"
        for name in ("train", "valid", "test")
    }
    report: dict = {
        "interpreter": sys.executable,
        "numpy_version": np.__version__,
        "numpy_file": np.__file__,
        "sklearn_version": sklearn.__version__,
        "sklearn_file": sklearn.__file__,
        "scales": {},
        "proteingym_sha256": {name: sha256(path) for name, path in proteingym.items()},
        "seeds": {
            "values": OUTER_SEEDS,
            "unique": len(set(OUTER_SEEDS)) == len(OUTER_SEEDS),
            "collision_with_known": sorted(set(OUTER_SEEDS) & KNOWN),
        },
    }
    stored = {}
    for scale, paths in scales.items():
        loaded = {name: load_table(path) for name, path in paths.items()}
        seq = {name: sequences(rows) for name, (_, rows) in loaded.items()}
        n = len(seq["train"])
        fold_id = np.full(n, -1, dtype=np.int64)
        fold_files = {}
        train_counts = []
        val_counts = []
        for fold, (train_idx, val_idx) in enumerate(KFold(5, shuffle=True, random_state=20261121).split(np.arange(n))):
            fold_id[val_idx] = fold
            train_counts.append(int(len(train_idx)))
            val_counts.append(int(len(val_idx)))
            fold_files[f"oof_fold{fold}_train.txt"] = write_indices(OUT / f"{scale}_oof_fold{fold}_train.txt", train_idx)
            fold_files[f"oof_fold{fold}_val.txt"] = write_indices(OUT / f"{scale}_oof_fold{fold}_val.txt", val_idx)
        residual_train, residual_val = train_test_split(
            np.arange(n), test_size=0.1, shuffle=True, random_state=20261122, stratify=None,
        )
        fold_files["oof_fold_id.txt"] = write_indices(OUT / f"{scale}_oof_fold_id.txt", fold_id)
        fold_files["residual_train.txt"] = write_indices(OUT / f"{scale}_residual_train.txt", residual_train)
        fold_files["residual_val.txt"] = write_indices(OUT / f"{scale}_residual_val.txt", residual_val)
        sets = {name: set(values) for name, values in seq.items()}
        stored[scale] = {"seq": seq, "fold_id": fold_id, "header": {name: header for name, (header, _) in loaded.items()}}
        report["scales"][scale] = {
            "paths": {name: str(path) for name, path in paths.items()},
            "sha256": {name: sha256(path) for name, path in paths.items()},
            "columns": {name: header for name, (header, _) in loaded.items()},
            "rows": {name: len(values) for name, values in seq.items()},
            "duplicate_sequences_within_file": {name: len(values) - len(set(values)) for name, values in seq.items()},
            "sequence_intersection_counts": {
                "train_valid": len(sets["train"] & sets["valid"]),
                "train_test": len(sets["train"] & sets["test"]),
                "valid_test": len(sets["valid"] & sets["test"]),
            },
            "oof_val_counts": val_counts,
            "oof_train_counts": train_counts,
            "each_row_one_oof_fold": False,
            "residual_train_count": int(len(residual_train)),
            "residual_val_count": int(len(residual_val)),
            "residual_partition": bool(len(set(residual_train.tolist()) & set(residual_val.tolist())) == 0 and set(residual_train.tolist()) | set(residual_val.tolist()) == set(range(n))),
            "index_sha256": fold_files,
        }
    full_seq = stored["full"]["seq"]["train"]
    n1000_seq = stored["n1000"]["seq"]["train"]
    full_pos = {sequence: index for index, sequence in enumerate(full_seq)}
    mapped = [full_pos[sequence] for sequence in n1000_seq]
    reused = stored["full"]["fold_id"][np.array(mapped, dtype=np.int64)]
    report["n1000_inside_full_train"] = len(set(n1000_seq) - set(full_seq)) == 0 and len(n1000_seq) == 1000
    report["n1000_rows_whose_fold_would_change_if_full_fold_ids_were_reused"] = int(np.sum(reused != stored["n1000"]["fold_id"]))
    report["n1000_valid_test_byte_equal_to_full"] = {
        "valid": report["scales"]["n1000"]["sha256"]["valid"] == report["scales"]["full"]["sha256"]["valid"],
        "test": report["scales"]["n1000"]["sha256"]["test"] == report["scales"]["full"]["sha256"]["test"],
    }
    report["muformer_full_byte_equal_to_proteingym"] = {
        name: report["scales"]["full"]["sha256"][name] == report["proteingym_sha256"][name]
        for name in ("train", "valid", "test")
    }
    for scale, item in report["scales"].items():
        n = item["rows"]["train"]
        fold_id = stored[scale]["fold_id"]
        counts = np.bincount(fold_id, minlength=5)
        complements = []
        for fold in range(5):
            val = set(np.flatnonzero(fold_id == fold).tolist())
            train = set(range(n)) - val
            saved_train = {int(line) for line in (OUT / f"{scale}_oof_fold{fold}_train.txt").read_text().splitlines()}
            saved_val = {int(line) for line in (OUT / f"{scale}_oof_fold{fold}_val.txt").read_text().splitlines()}
            complements.append(saved_train == train and saved_val == val and saved_train.isdisjoint(saved_val))
        item["each_row_one_oof_fold"] = bool(len(fold_id) == n and int((fold_id < 0).sum()) == 0 and set(fold_id.tolist()) == set(range(5)) and int(counts.sum()) == n)
        item["fold_train_is_exact_complement"] = all(complements)
        item["fold_train_indices_inside_outer_train"] = all(complements)
    manifest = OUT / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    sums = OUT / "SHA256SUMS"
    lines = []
    for path in sorted(OUT.glob("*")):
        if path.name == "SHA256SUMS" or path.suffix == ".py":
            continue
        lines.append(f"{sha256(path)}  {path.name}")
    sums.write_text("\n".join(lines) + "\n")
    checks = [
        report["seeds"]["unique"] and report["seeds"]["collision_with_known"] == [],
        report["n1000_inside_full_train"],
        report["n1000_rows_whose_fold_would_change_if_full_fold_ids_were_reused"] > 0,
        all(item["sequence_intersection_counts"][key] == 0 for item in report["scales"].values() for key in item["sequence_intersection_counts"]),
        all(value == 0 for item in report["scales"].values() for value in item["duplicate_sequences_within_file"].values()),
        all(item["each_row_one_oof_fold"] and item["fold_train_is_exact_complement"] and item["residual_partition"] for item in report["scales"].values()),
        report["scales"]["n1000"]["oof_train_counts"] == [800, 800, 800, 800, 800],
        report["scales"]["n1000"]["residual_val_count"] == 100,
        report["scales"]["full"]["residual_val_count"] == 350,
        set(report["scales"]["full"]["oof_train_counts"]) <= {2797, 2798},
        all(report["muformer_full_byte_equal_to_proteingym"].values()),
        all(report["n1000_valid_test_byte_equal_to_full"].values()),
    ]
    if not all(checks):
        raise SystemExit(f"precheck failed: {checks}")
    print(json.dumps({
        "sha256sums": sha256(sums),
        "n1000_fold_change_if_reused": report["n1000_rows_whose_fold_would_change_if_full_fold_ids_were_reused"],
        "full_oof_train_counts": report["scales"]["full"]["oof_train_counts"],
        "full_oof_val_counts": report["scales"]["full"]["oof_val_counts"],
        "intersections": {scale: item["sequence_intersection_counts"] for scale, item in report["scales"].items()},
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
