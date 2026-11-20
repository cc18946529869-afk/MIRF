"""Audit boundary-study logs and compare with the immutable six-case baseline."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import statistics
import tarfile


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stripped(rows):
    return [{k:v for k,v in row.items() if k != "seconds"} for row in rows]


def trace(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def six_case(run, old):
    output = {}
    for n, controls in old["trials"].items():
        trials = []
        for baseline in controls:
            start = baseline["start"]
            case_name = f"case-{n}-{start}"
            path = run / f"{case_name}.json"
            result = load(path)
            assert result["sample_size"] == int(n) and result["start"] == start
            for key in ("sites", "edges", "indices_sha256", "topology_sha256",
                        "alignment_weights_sha256", "reference_profile_sha256", "initial_range",
                        "library_sha256", "threads"):
                assert result[key] == baseline[key], key
            native = result["optimizer"]
            previous = baseline["optimizer"]
            for key in ("restricted_curvature_to_free_coordinates", "maxiter", "maxeval", "gtol_per_site",
                        "initial_log_likelihood"):
                assert native[key] == previous[key], key
            checked = trace(run/f"{case_name}.jsonl")
            bounds = [x["initial_step"] for x in checked if x["event"] == "zero_boundary_cap"]
            guard = native["zero_boundary"]
            rejected = guard["precheck_rejections"]
            assert rejected == guard["infeasible_likelihood_calls_avoided"]
            assert len(bounds) == guard["capped_searches"]
            accepted = [x for x in checked if x["event"] == "accepted"]
            evaluations = [x for x in checked if x["event"] == "evaluation"]
            assert len(accepted) == native["iterations"] <= 150
            assert len(evaluations) == native["solver_likelihood_gradient_evaluations"] <= 260
            assert sum(not x["finite"] for x in evaluations) == native["infeasible_trials"]
            cross = result["independent_terminal"]
            assert cross["terminal_log_domain_sites"] == result["sites"]
            gain = native["terminal_log_likelihood"] - previous["terminal_log_likelihood"]
            agreement = cross["likelihood_score_agreement"]
            res = cross["projected_gradient_per_site"]
            trials.append({
                "case":case_name,
                "old":previous,
                "new":result,
                "terminal_likelihood_gain_over_archived_baseline":gain,
                "terminal_agreement_required_and_met":agreement,
                "guard_direct_cost_seconds":guard["precheck_seconds"] + guard["ray_limit_seconds"],
                "first_cap_step_min": min(bounds, default=None),
                "last_ten_capped_search_start_steps_median": statistics.median(bounds[-10:]) if bounds else None,
                "capped_searches_below_one_millionth":sum(v < 1e-6 for v in bounds),
                "score_difference_divided_by_max_abs_projected_gradient":(
                    cross["terminal_max_per_site_score_difference"] / max(1, res)),
            })
        assert len(trials) == 3
        output[n] = trials
    return output


def compare_threads(run):
    result = {}
    for n in (1000, 10000):
        primary = f"case-{n}-short"
        alternate = f"threadcheck-{n}-short"
        main, other = (load(run/f"{name}.json") for name in (primary,alternate))
        a, b = (stripped(trace(run/f"{name}.jsonl")) for name in (primary,alternate))
        first_different = next((i+1 for i,(x,y) in enumerate(zip(a,b)) if x != y),None)
        if first_different is None and len(a) != len(b):
            first_different = min(len(a),len(b))+1
        result[str(n)] = {
            "initial_log_likelihood_difference":abs(main["optimizer"]["initial_log_likelihood"]
                                                    - other["optimizer"]["initial_log_likelihood"]),
            "trace_bitwise_values_match_excluding_times":a==b,
            "first_different_event_1based":first_different,
            "termination": [main["optimizer"]["status"],other["optimizer"]["status"]],
            "search_evaluations": [main["optimizer"]["solver_likelihood_gradient_evaluations"],
                                   other["optimizer"]["solver_likelihood_gradient_evaluations"]],
            "terminal_independent_gradient": [main["independent_terminal"]["projected_gradient_per_site"],
                                               other["independent_terminal"]["projected_gradient_per_site"]],
            "terminal_zero_edges": [main["optimizer"]["terminal_zero_edges"],other["optimizer"]["terminal_zero_edges"]],
            "terminal_log_likelihood_difference":abs(main["independent_terminal"]["terminal_log_likelihood"]
                                                    - other["independent_terminal"]["terminal_log_likelihood"]),
            "passes": [main["passed_bridge_gate"],other["passed_bridge_gate"]],
        }
    return result


def log_archive(run, destination):
    with tarfile.open(destination, "w:gz") as out:
        for path in sorted(run.rglob("*")):
            if path.is_file() and path.suffix in (".json", ".jsonl", ".stderr", ".stdout", ".time", ".txt"):
                content = path.read_bytes()
                entry = tarfile.TarInfo(str(path.relative_to(run)))
                entry.mtime = 0
                entry.size = len(content)
                out.addfile(entry, io.BytesIO(content))
    with destination.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("old_audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert (args.run/"finished.txt").is_file()
    expected = load(args.old_audit/"bridge-results.json")
    results = six_case(args.run, expected)
    with tarfile.open(args.old_audit/"run-logs.tar.gz") as raw:
        orig = [json.loads(line) for line in raw.extractfile("scale-1000-short.jsonl")]
    replay = trace(args.run/"legacy-1000-short.jsonl")
    assert stripped(orig) == stripped(replay)
    assert "zero_boundary" not in load(args.run/"legacy-1000-short.json")["optimizer"]
    tests = [load(args.run/f"{name}.json") for name in ("correctness","pressure")]
    assert all(x["passed"] for x in tests)
    threadcheck = compare_threads(args.run)
    passed = all(x["new"]["passed_bridge_gate"] for records in results.values() for x in records)
    args.output.mkdir(parents=True, exist_ok=True)
    artifact = args.output/"raw-logs.tar.gz"
    raw_hash = log_archive(args.run, artifact)
    evidence = {
        "bridge_gate_passed":passed,
        "same_case_identity_and_optimizer_constants_verified":True,
        "old_1000_short_trace_reproduced_excluding_timing": True,
        "raw_archive_sha256":raw_hash,
        "tests":tests,
        "runs":results,
        "threads_4_vs_1":threadcheck,
    }
    (args.output/"results.json").write_text(json.dumps(evidence,indent=2,allow_nan=False)+"\n")
    for n, records in results.items():
        for row in records:
            new = row["new"]
            o = new["optimizer"]
            audit = new["independent_terminal"]
            s = o["zero_boundary"]
            print(n, new["start"], new["passed_bridge_gate"], o["status"], s["line_searches"],
                  s["prechecks"], s["precheck_rejections"], o["iterations"],
                  o["solver_likelihood_gradient_evaluations"], o["terminal_zero_edges"],
                  o["history_resets"], s["infeasible_likelihood_calls_avoided"], o["infeasible_trials"],
                  audit["projected_gradient_per_site"], audit["likelihood_score_agreement"],
                  row["terminal_likelihood_gain_over_archived_baseline"],
                  row["last_ten_capped_search_start_steps_median"],
                  round(o["search_seconds"],3), round(row["guard_direct_cost_seconds"],6),
                  round(new["peak_rss_kib"]/1024,2))
    print(json.dumps(threadcheck, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
