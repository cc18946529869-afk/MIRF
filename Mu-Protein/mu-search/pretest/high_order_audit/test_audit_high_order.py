"""Synthetic checks for Hamming, deduplication, bins, and mutation lists."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_high_order as audit


def test_hamming_and_mutations():
    wild_type = "AAAA"
    sequence = "ABCA"
    assert audit.hamming(sequence, wild_type) == 2
    assert audit.mutation_list(sequence, wild_type) == ["A2B", "A3C"]
    assert audit.order_label(10) == "10"
    assert audit.order_label(11) == "10+"
    try:
        audit.hamming("AAA", wild_type)
    except ValueError:
        pass
    else:
        raise AssertionError("different lengths were accepted")


def test_first_score_is_not_a_new_mutant():
    rows = [("AAA", 1.0), ("BBB", 2.0), ("AAA", 9.0)]
    assert audit.first_scores(rows) == [("AAA", 1.0), ("BBB", 2.0)]


def test_bins_and_top10_rule():
    repeated = {
        "sequence": "BAAA",
        "score": 1.0,
        "distance": 1,
        "order": "1",
        "candidate_index": 1,
    }
    records = [dict(repeated, candidate_index=index, score=float(index)) for index in range(1, 12)]
    records.append(dict(repeated, candidate_index=12, score=100.0))
    records.append(
        {
            "sequence": "CCCC",
            "score": 0.0,
            "distance": 4,
            "order": "4",
            "candidate_index": 13,
        }
    )
    query = {row["order"]: row for row in audit.grouped_stats(records, "query")}
    unique = {row["order"]: row for row in audit.grouped_stats(records, "unique")}
    assert query["1"]["calls_or_unique"] == 12
    assert query["1"]["unique_sequences"] == 1
    assert abs(query["1"]["duplicate_fraction"] - 11 / 12) < 1e-12
    assert unique["1"]["calls_or_unique"] == 1
    assert unique["1"]["maximum"] == 1.0
    assert query["1"]["top10_mean"] is not None
    assert unique["4"]["top10_mean"] is None
    assert unique["4"]["calls_or_unique"] == 1


def test_percentile_endpoints():
    assert audit.percentile([1.0], 0.99) == 1.0
    assert audit.percentile([1.0, 3.0], 0.5) == 2.0


def test_recurrent_counts_each_run_once():
    rows = [
        {"method": "m", "seed": 0, "set": "top10", "rank": 1, "mutations": "A1C;A2D"},
        {"method": "m", "seed": 0, "set": "top10", "rank": 2, "mutations": "A1C;A2D"},
        {"method": "m", "seed": 1, "set": "top10", "rank": 1, "mutations": "A1C"},
    ]
    found = [
        row for row in audit.recurrent_rows(rows)
        if row["kind"] == "substitution" and row["item"] == "A1C"
    ]
    assert found[0]["n_sequences"] == 3
    assert found[0]["n_runs"] == 2
    pairs = [row for row in audit.recurrent_rows(rows) if row["kind"] == "pair"]
    assert pairs[0]["n_runs"] == 1


def main():
    test_hamming_and_mutations()
    test_first_score_is_not_a_new_mutant()
    test_bins_and_top10_rule()
    test_percentile_endpoints()
    test_recurrent_counts_each_run_once()
    print("audit_unit_ok")


if __name__ == "__main__":
    main()
