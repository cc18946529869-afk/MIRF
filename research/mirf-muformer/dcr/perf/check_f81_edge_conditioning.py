"""Stage 2.5 fixed-checkpoint precision and constrained local conditioning."""
from __future__ import annotations

import argparse
import ctypes
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import struct
from time import perf_counter

import numpy as np

from .check_f81_gradient_attribution import recover_warm_terminal
from .check_f81_observed_bridge import load_case
from .check_f81_split_gradient import Reference
from .f81_edge_conditioning import C1, EdgeObjective, physical, solve_edge
from .f81_optimize import projected_norm
from .f81_zero_boundary import ZeroBoundary


ABS = Decimal("1e-8")
METHODS = ("direct_diag", "hybrid_log", "hybrid_log_diag")


class TangentReference:
    """Forward-mode derivative on a separate internal rooting."""
    def __init__(self, auditor, library):
        self.pruning = auditor.pruning
        self.order = auditor.topology.child_to_edge
        self.edge_index = np.argsort(self.order)
        native = ctypes.CDLL(str(library.resolve()))
        self.run = native.f81_dual_gradient
        ptr, integer = ctypes.c_void_p, ctypes.c_int
        self.run.argtypes = [ptr, ptr, integer, integer, integer, ptr, ptr, integer, integer, ctypes.c_char_p]
        self.run.restype = integer

    def score(self, point, edge, bits):
        pruning = self.pruning
        tree = pruning.topology
        ordered = np.ascontiguousarray(point[self.order])
        sites, leaves = pruning.states.shape
        output = ctypes.create_string_buffer(128)
        status = self.run(pruning.states.ctypes.data, pruning.profiles.ctypes.data,
                          sites, leaves, len(ordered)+1, tree.parent.ctypes.data, ordered.ctypes.data,
                          int(self.edge_index[edge]), bits, output)
        if status != 0:
            raise ArithmeticError(f"Tangent derivative failed: {status}")
        return Decimal(output.value.decode("ascii"))


def ordered_binary64(value):
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    sign_bit = 1 << 63
    return sign_bit - (bits & (sign_bit - 1)) if bits & sign_bit else sign_bit + bits


def relative_error(x, y):
    denominator = max(abs(x), abs(y))
    return abs(x-y)/denominator if denominator else Decimal(0)


def precision_audit(point, edge, split, tangent, scan):
    reference = split[256]
    nearest = float(reference)
    representation = abs(Decimal.from_float(nearest) - reference)
    ordinary, logged = (float(s[edge]) for s in scan["vectors"])
    high = {bits: tangent.score(point, edge, bits) for bits in (192, 256)}
    cross_path = {}
    for bits in (192, 256):
        delta = abs(high[bits]-split[bits])
        cross_path[str(bits)] = {"split": str(split[bits]), "tangent": str(high[bits]),
                                "absolute_difference": str(delta),
                                "relative_difference": str(relative_error(high[bits], split[bits])),
                                "passes_absolute_gate": delta <= ABS}
    return {
        "binary64": {
            "ordinary": ordinary, "rerooted_log": logged, "nearest_to_reference": nearest,
            "representation_error_abs": str(representation),
            "representation_meets_absolute_gate": representation <= ABS,
            "cross_path_abs": abs(ordinary-logged),
            "cross_path_relative": str(relative_error(physical(ordinary), physical(logged))),
            "cross_path_ulp_distance": abs(ordered_binary64(ordinary)-ordered_binary64(logged)),
            "ordinary_ulp_distance_to_nearest": abs(ordered_binary64(ordinary)-ordered_binary64(nearest)),
            "rerooted_ulp_distance_to_nearest": abs(ordered_binary64(logged)-ordered_binary64(nearest)),
            "ordinary_abs_error_to_reference": str(abs(physical(ordinary)-reference)),
            "rerooted_abs_error_to_reference": str(abs(physical(logged)-reference)),
            "cross_path_meets_absolute_gate": abs(physical(ordinary)-physical(logged)) <= ABS,
        },
        "higher_precision_paths": cross_path,
        "tangent_calls": 2,
    }


def scan_vectors(engine, auditor, point):
    normal = engine.evaluate(point, threads=4)
    logged = auditor.evaluate(point, threads=4, log_domain=True)
    n = len(normal.site_log_likelihood)
    first = -normal.length_gradient / n
    second = -logged.length_gradient / n
    return {
        "vectors": (first, second),
        "total_log_likelihood": normal.log_likelihood,
        "rerooted_log_likelihood": logged.log_likelihood,
        "projected_gradient_inf": projected_norm(point, first),
        "rerooted_projected_gradient_inf": projected_norm(point, second),
        "cross_path_gradient_abs": float(np.abs(first-second).max()),
        "cross_path_gradient_passed": bool(np.abs(first-second).max() <= 1e-8),
    }


def public(scan):
    return {k: v for k, v in scan.items() if k != "vectors"}


def ref_terms(engine, point, library, edge):
    ref = Reference(engine, point, library)
    return {bits: ref.site_terms(edge, bits) for bits in (192, 256)}


def local_attempt(engine, auditor, guard, point, scan, edge, reference_library, dual, *, compare):
    split = ref_terms(engine, point, reference_library, edge)
    initial = float(point[edge])
    precision = precision_audit(point, edge, {b: split[b][0] for b in split}, dual, scan)
    alternatives = [solve_edge(split[256][1], initial, method)
                    for method in (METHODS if compare else ("hybrid_log_diag",))]
    proposed = alternatives[-1]
    trial = point.copy()
    trial[edge] = proposed["end_length"]
    # Split odds depend on the held edges only. Evaluate the same candidate
    # in the original topology and in the independent tangent path.
    same_point = trial[edge] == point[edge]
    if not guard.feasible(trial):
        raise RuntimeError("Local result fails fixed-topology zero-edge support")
    after = scan if same_point else scan_vectors(engine, auditor, trial)
    post_split = {bits: EdgeObjective(split[bits][1], physical(initial)).evaluate(physical(trial[edge]))[1]
                  for bits in split}
    post_precision = (precision if same_point
                      else precision_audit(trial, edge, post_split, dual, after))
    independent_delta = -(after["rerooted_log_likelihood"] - scan["rerooted_log_likelihood"])/len(split[256][1])
    relative = Decimal(proposed["relative_objective"])
    local_agrees = all(x["passes_absolute_gate"] for x in post_precision["higher_precision_paths"].values())
    # Holding other edge coordinates fixed can move the maximum residual to
    # another edge. Commit only when both full binary64 checks improve.
    accept = (not same_point and relative < 0 and independent_delta < 0 and local_agrees
              and after["projected_gradient_inf"] < scan["projected_gradient_inf"]
              and after["rerooted_projected_gradient_inf"] < scan["rerooted_projected_gradient_inf"])
    return {"edge": edge, "before": public(scan), "after_proposal": public(after),
            "precision_before": precision, "precision_after": post_precision,
            "local_strategies": alternatives,
            "independent_mean_objective_change": independent_delta,
            "abs_objective_change_difference": str(abs(physical(independent_delta)-relative)),
            "whole_tree_update_retained": bool(accept),
            "split_calls": 2, "tangent_calls": 2 if same_point else 4,
            "candidate_full_gradient_calls": 0 if same_point else 2,
            "candidate_high_precision_response_calls": 2}, trial, after


def run(args):
    engine, auditor, facts = load_case(args.prep, args.gate, 10000, args.library)
    point = np.load(args.checkpoint, allow_pickle=False)
    manifest = json.loads(args.manifest.read_text())
    for key in ("alignment_weights_sha256", "indices_sha256", "reference_profile_sha256",
                "topology_sha256", "library_sha256"):
        assert facts[key] == manifest[key]
    replay = 0
    if args.warm_trace:
        assert hashlib.sha256(point.tobytes()).hexdigest() == manifest["initial_sha256"]
        point, replay = recover_warm_terminal(engine, point, args.warm_trace, manifest)
    digest = hashlib.sha256(point.tobytes()).hexdigest()
    assert digest == manifest["terminal_sha256" if replay else "checkpoint_sha256"]
    attribution = json.loads(args.attribution.read_text())[args.case]
    assert digest == attribution["checkpoint_sha256"]
    guard = ZeroBoundary(engine, args.boundary)
    dual = TangentReference(auditor, args.dual)
    before = scan_vectors(engine, auditor, point)
    targeted = [entry["edge_index"] for entry in attribution["selected_edge_site_attributions"]
                if "max_projected_residual" in entry["selection"]
                or args.case == "previous-10000" and "shortest_positive" in entry["selection"]]
    assert targeted
    isolated = [local_attempt(engine, auditor, guard, point, before, e, args.reference, dual, compare=True)
                for e in targeted]
    chain = []
    current = before
    for iteration in range(3):
        residual = point - np.maximum(0, point-current["vectors"][1])
        edge = int(np.abs(residual).argmax())
        first = next((entry for entry in isolated if entry[0]["edge"] == edge), None) if not iteration else None
        proposal, trial, output = (first if first is not None else
                                   local_attempt(engine, auditor, guard, point, current, edge,
                                                 args.reference, dual, compare=False))
        chain.append({k: v for k, v in proposal.items() if k != "local_strategies"} | {
            "local_trials": proposal["local_strategies"][-1],
            "reused_isolated_result": first is not None,
        })
        if not proposal["whole_tree_update_retained"]:
            break
        point, current = trial, output
    counted = [x[0] for x in isolated]+[x for x in chain if not x["reused_isolated_result"]]
    trial_calls = sum(sum(v["evaluations"] for v in x["local_strategies"])
                      if "local_strategies" in x else x["local_trials"]["evaluations"] for x in counted)
    return {**facts, "archived_case": args.case, "initial_sha256": digest, "replay_evaluations": replay,
            "isolated_branches": [x[0] for x in isolated], "retained_chain": chain,
            "before": public(before), "after": public(current),
            "local_eval_budget_each_run": 96, "greedy_updates_budget": 3,
            "scalar_conditioned_objective_evaluations": trial_calls,
            "full_gradient_evaluations": 2+sum(x["candidate_full_gradient_calls"] for x in counted),
            "high_precision_split_evaluations": sum(x["split_calls"] for x in counted),
            "high_precision_tangent_evaluations": sum(x["tangent_calls"] for x in counted),
            "high_precision_scalar_terminal_checks": sum(x["candidate_high_precision_response_calls"] for x in counted),
            "declared_terminal_full_projection_gate": 1e-5,
            "declared_absolute_agreement_gate": 1e-8}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("short-10000", "warm-10000", "previous-10000"), required=True)
    for key in ("prep", "gate", "library", "boundary", "reference", "dual", "checkpoint", "manifest",
                "attribution", "warm-trace"):
        parser.add_argument("--" + key, type=Path, required=key != "warm-trace")
    args = parser.parse_args()
    started = perf_counter()
    with localcontext() as ctx:
        ctx.prec = 115
        output = run(args)
    output["elapsed_seconds"] = perf_counter()-started
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
