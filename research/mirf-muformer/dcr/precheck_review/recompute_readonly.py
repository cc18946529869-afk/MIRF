"""Read the saved precheck indices and recompute the checks. Do not write indices."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.model_selection import KFold, train_test_split

PRECHECK = Path("/home/cc233/data/cc233_data/work_dir/MIRF/research/mirf-muformer/dcr/precheck")
KNOWN_SEEDS = {42, 20261121, 20261122, 20261123, 2847319051}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_idx(name: str) -> list[int]:
    text = (PRECHECK / name).read_bytes()
    if not text.endswith(b"\n"):
        raise SystemExit(f"{name} has no trailing newline")
    lines = text.decode().splitlines()
    if any(line == "" for line in lines):
        raise SystemExit(f"{name} has an empty line")
    return [int(line) for line in lines]


def sequences(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    header = lines[0].split("\t")
    index = header.index("sequence")
    values = []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) != len(header):
            raise SystemExit(f"{path} has a ragged row")
        values.append(parts[index])
    return values


def main() -> None:
    print("executable", sys.executable)
    print("numpy", np.__version__)
    print("numpy_file", np.__file__)
    print("sklearn", sklearn.__version__)
    print("sklearn_file", sklearn.__file__)
    mismatches = []
    for line in (PRECHECK / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        got = sha256(PRECHECK / name)
        print("sha256", name, got, "match", got == digest)
        if got != digest:
            mismatches.append(name)
    print("sha256_mismatch_count", len(mismatches))
    manifest = json.loads((PRECHECK / "manifest.json").read_text())
    for scale, n in (("n1000", 1000), ("full", 3497)):
        fold_id = load_idx(f"{scale}_oof_fold_id.txt")
        val_sets = []
        for fold in range(5):
            val = load_idx(f"{scale}_oof_fold{fold}_val.txt")
            train = load_idx(f"{scale}_oof_fold{fold}_train.txt")
            val_set, train_set = set(val), set(train)
            val_sets.append(val_set)
            from_id = {i for i, value in enumerate(fold_id) if value == fold}
            print(
                scale, "fold", fold,
                "val", len(val_set), "train", len(train_set),
                "disjoint", val_set.isdisjoint(train_set),
                "union_is_0_n", val_set | train_set == set(range(n)),
                "val_matches_fold_id", val_set == from_id,
                "indices_inside", max(val_set | train_set) < n and min(val_set | train_set) >= 0,
            )
        pairwise = all(val_sets[i].isdisjoint(val_sets[j]) for i in range(5) for j in range(i + 1, 5))
        print(scale, "val_pairwise_disjoint", pairwise, "val_cover", set().union(*val_sets) == set(range(n)))
        for fold, (train_idx, val_idx) in enumerate(KFold(5, shuffle=True, random_state=20261121).split(np.arange(n))):
            saved_train = set(load_idx(f"{scale}_oof_fold{fold}_train.txt"))
            saved_val = set(load_idx(f"{scale}_oof_fold{fold}_val.txt"))
            print(scale, "kfold_set_match", fold, saved_train == set(train_idx.tolist()), saved_val == set(val_idx.tolist()))
        fresh_train, fresh_val = train_test_split(
            np.arange(n), test_size=0.1, shuffle=True, random_state=20261122, stratify=None,
        )
        saved_train = load_idx(f"{scale}_residual_train.txt")
        saved_val = load_idx(f"{scale}_residual_val.txt")
        print(
            scale, "split_set_match",
            set(saved_train) == set(fresh_train.tolist()),
            set(saved_val) == set(fresh_val.tolist()),
            "order_match", saved_train == fresh_train.tolist(), saved_val == fresh_val.tolist(),
            "disjoint", set(saved_train).isdisjoint(saved_val),
            "cover", set(saved_train) | set(saved_val) == set(range(n)),
            "counts", len(saved_train), len(saved_val),
        )
    loaded = {}
    for scale, item in manifest["scales"].items():
        loaded[scale] = {name: sequences(Path(path)) for name, path in item["paths"].items()}
        sets = {name: set(values) for name, values in loaded[scale].items()}
        print(scale, "rows", {name: len(values) for name, values in loaded[scale].items()})
        print(scale, "duplicate_sequences", {name: len(values) - len(sets[name]) for name, values in loaded[scale].items()})
        print(
            scale, "intersections",
            len(sets["train"] & sets["valid"]),
            len(sets["train"] & sets["test"]),
            len(sets["valid"] & sets["test"]),
        )
        for name, path in item["paths"].items():
            print(scale, name, "file_sha256", sha256(Path(path)))
    missing = set(loaded["n1000"]["train"]) - set(loaded["full"]["train"])
    print("n1000_sequences_absent_from_full_train", len(missing))
    print("n1000_valid_equal_full_valid", loaded["n1000"]["valid"] == loaded["full"]["valid"])
    print("n1000_test_equal_full_test", loaded["n1000"]["test"] == loaded["full"]["test"])
    position = {sequence: index for index, sequence in enumerate(loaded["full"]["train"])}
    full_id = load_idx("full_oof_fold_id.txt")
    small_id = load_idx("n1000_oof_fold_id.txt")
    changed = sum(full_id[position[sequence]] != small_id[index] for index, sequence in enumerate(loaded["n1000"]["train"]))
    print("fold_ids_changed_if_full_ids_reused", changed)
    seeds = [int(value) for value in manifest["seeds"]["values"]]
    print("seeds_unique", len(seeds) == len(set(seeds)) == 10)
    print("seed_collision", sorted(set(seeds) & KNOWN_SEEDS))


if __name__ == "__main__":
    main()
