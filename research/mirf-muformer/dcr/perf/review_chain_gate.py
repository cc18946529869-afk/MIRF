"""Evaluate proposed descriptive cluster gates against archived aggregate evidence."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_dir", type=Path)
    parser.add_argument("original_gate", type=Path)
    args = parser.parse_args()
    archive = load(args.audit_dir / "global-influence.json")
    original = load(args.original_gate)
    expected = archive["granularities"]
    with (args.audit_dir / "all-cluster-influence.csv.gz").open("rb") as blob:
        sha = hashlib.file_digest(blob, "sha256").hexdigest()
    if sha != archive["csv_sha256"]:
        raise ValueError("Archived aggregate table SHA-256 mismatch")
    cutoffs = (0.40, 0.50, 0.60)
    limits = {"chained_weight_share": .05, "chained_concentration_share": .10, "within_pair_low_share": .15}
    data = {p: [] for p in expected}
    with gzip.open(args.audit_dir / "all-cluster-influence.csv.gz", "rt", encoding="utf-8") as table:
        for x in csv.DictReader(table):
            data[x["threshold"]].append((float(x["weight_share"]), float(x["concentration_contribution"]),
                                         float(x["low_similarity_pair_mass"]), float(x["within_cluster_pair_mass"]),
                                         float(x["chain_fraction"]) if x["chain_fraction"] else 0.))
    review = {}
    for name, rows in data.items():
        if len(rows) != expected[name]["clusters"]:
            raise ValueError("Inconsistent number of cluster rows")
        _, _, low, total, _ = zip(*rows)
        low_mass_share = math.fsum(low) / math.fsum(total)
        p = original["granularities"][name]
        structural_pass = (p["maximum_row_share"] < .5 and p["largest_weight_share"] < .5
                           and p["effective_weight_components"] >= 4)
        snapshots = []
        for cutoff in cutoffs:
            chosen = [item for item in rows if item[-1] > cutoff]
            diagnostics = {"chained_weight_share": math.fsum(x[0] for x in chosen),
                           "chained_concentration_share": math.fsum(x[1] for x in chosen),
                           "within_pair_low_share": low_mass_share}
            snapshots.append({"chain_cutoff": cutoff, "chained_components": len(chosen), **diagnostics,
                              "criteria_above_candidate_limit": [key for key, val in diagnostics.items() if val > limits[key]]})
        primary = snapshots[1]
        for key, old_key in (("chained_weight_share", "chained_cluster_weight_share"),
                             ("chained_concentration_share", "chained_clusters_concentration_share"),
                             ("within_pair_low_share", "low_similarity_mass_share_of_within_cluster_pairs")):
            assert abs(primary[key] - expected[name][old_key]) < 1e-12
        review[name] = {"clusters": len(rows), "legacy_global_structure_checks_pass": structural_pass,
                        "matches_candidate_global_rule": structural_pass and not primary["criteria_above_candidate_limit"],
                        "cutoff_sensitivity": snapshots}
    report = {"scope": "retrospective candidate thresholds for independent design review, no permission to run",
              "candidate_limits": limits, "archive_sha256": sha, "granularities": review}
    print(json.dumps(report, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
