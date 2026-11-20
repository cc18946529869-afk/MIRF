"""Independent denominator and threshold recomputation from aggregate cluster records."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cluster_table", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    candidate = json.loads(args.candidate.read_text())
    with args.cluster_table.open("rb") as input_file:
        assert hashlib.file_digest(input_file, "sha256").hexdigest() == candidate["archive_sha256"]
    partitions = defaultdict(list)
    with gzip.open(args.cluster_table, "rt", newline="", encoding="utf-8") as desc:
        for record in csv.DictReader(desc):
            rows = int(record["rows"])
            w, low, total = (float(record[key]) for key in
                             ("weight_share", "low_similarity_pair_mass", "within_cluster_pair_mass"))
            if rows == 1:
                assert low == 0 and abs(total) < 1e-14
                total = 0.
            else:
                assert total > 0 and 0 <= low <= total
            label = record["threshold"]
            partitions[label].append((w, low, total))
    assert set(partitions) == {"211", "237"}
    summary = {}
    for name, items in partitions.items():
        number = len(items)
        original = candidate["granularities"][name]
        assert number == original["clusters"]
        total_weight = math.fsum(w for w,_,_ in items)
        total_pair = math.fsum(pair for _,_,pair in items)
        weight_sq = math.fsum(w*w for w,_,_ in items)
        assert abs(total_weight-1) < 1e-12
        points = []
        for test in original["cutoff_sensitivity"]:
            cutoff = test["chain_cutoff"]
            flagged = [w for w,low,pair in items if pair and low > cutoff*pair]
            current = {
                "chained_weight_share": math.fsum(flagged) / total_weight,
                "chained_concentration_share": math.fsum(w*w for w in flagged) / weight_sq,
                "within_pair_low_share": math.fsum(low for _,low,_ in items) / total_pair,
            }
            assert len(flagged) == test["chained_components"]
            assert max(abs(current[k]-test[k]) for k in current) < 1e-12
            points.append({"cutoff": cutoff, "flagged": len(flagged), **current})
        summary[name] = {"components": number,
                         "relative_total_mass_variance_under_uniform_component_resampling":
                             weight_sq / (total_weight**2) - 1/number,
                         "pair_mass_denominator": total_pair,
                         "concentration_denominator": weight_sq,
                         "checks": points}
    report = {"scope": "recomputed from pair masses and cluster weights; no independent approval or resampling",
              "threshold_provenance": "retrospective design limits for descriptive screening only",
              "granularities": summary}
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
