"""Independently recompute the two reported heaviest-cluster chain diagnostics."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("assignments_dir", type=Path)
    parser.add_argument("prep", type=Path)
    args = parser.parse_args()
    data = json.loads(args.ledger.read_text())
    msa = np.load(args.prep / "mature_codes.npy", mmap_mode="r", allow_pickle=False)
    neighbors = np.loadtxt(args.prep / "neighbor_counts.tsv", usecols=1,
                           skiprows=1, dtype=np.int64)
    weights = 1.0 / neighbors
    output = {}
    for key, report in data["granularities"].items():
        labels = np.load(args.assignments_dir / f"clusters_{key}.npy", allow_pickle=False)
        groups, size = np.unique(labels, return_counts=True)
        assert len(groups) == report["components"] and size.max() == report["maximum_rows"]
        order = np.argsort(labels, kind="stable")
        starts = np.r_[0, np.cumsum(size)[:-1]]
        mass = np.add.reduceat(weights[order], starts)
        heavy = int(np.argmax(mass))
        pick = order[starts[heavy]:starts[heavy] + size[heavy]]
        selected_weights = weights[pick]
        assert len(pick) == report["heaviest_component_rows"] and len(pick) > 1
        subset = msa[pick]
        neighbors = np.zeros((len(pick), len(pick)), dtype=bool)
        for start in range(0, len(pick), 32):
            ii = slice(start, start + 32)
            matches = (subset[ii, None, :] == subset[None, :, :]).sum(axis=2)
            neighbors[ii] = matches >= int(key)
        reached = np.zeros(len(pick), dtype=bool)
        reached[0] = True
        while True:
            extended = reached | neighbors[reached].any(axis=0)
            if np.array_equal(reached, extended):
                break
            reached = extended
        assert reached.all()
        left, right = np.triu_indices(len(pick), k=1)
        pair_weights = selected_weights[left] * selected_weights[right]
        fraction = float(pair_weights[~neighbors[left, right]].sum() / pair_weights.sum())
        assert abs(fraction - report["heaviest_component_chain_fraction"]) < 1e-12
        assert (fraction > 0.50) and not report["eligible"]
        output[key] = {
            "components_from_archived_labels": len(groups),
            "heaviest_component_rows": len(pick),
            "heaviest_component_weight_share": float(mass[heavy] / weights.sum()),
            "heaviest_component_pair_count": int(len(left)),
            "heaviest_component_chain_fraction_brute_force": fraction,
            "chain_fraction_difference": abs(fraction - report["heaviest_component_chain_fraction"]),
            "connectivity_rechecked": True,
        }
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
