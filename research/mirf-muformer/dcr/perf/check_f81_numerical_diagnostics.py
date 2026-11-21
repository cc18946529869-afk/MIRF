"""Audit thread scheduling, Armijo trials and independent gradient errors."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from decimal import Decimal, localcontext
from pathlib import Path
import resource
from time import perf_counter

import numpy as np

from .check_f81_observed_bridge import load_case, start_lengths, audit_trace
from .f81_optimize import optimize_projected, projected_norm
from .f81_pruning import build
from .f81_unrooted import UnrootedPruning, UnrootedTopology
from .f81_zero_boundary import ZeroBoundary, build_boundary


ABS = 1e-8


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def exact_star(library):
    """Closed-form Decimal likelihood for one edge of an artificial star."""
    topology = UnrootedTopology.from_edges(4, np.column_stack([np.arange(4), np.full(4,4)]))
    frequencies = np.zeros((1, 21))
    frequencies[0,:4] = .25
    results = []
    with localcontext() as ctx:
        ctx.prec = 115
        for compatible, vals in ((True,(0.,1e-16)),(False,(1e-6,1e-8,1e-12,1e-16,1e-24))):
            rows = np.array([[1 if compatible else 0], [1], [1], [1]],dtype=np.uint8)
            evaluator = UnrootedPruning(topology, rows, frequencies, library)
            for s in vals:
                x = Decimal.from_float(s)
                quarter = Decimal(1)/4
                weight = Decimal(3)/4 if compatible else -quarter

                def evidence(t):
                    return quarter*(quarter + weight*(-t).exp())

                exact = -weight*(-x).exp() / (quarter + weight*(-x).exp())
                h = max(x, Decimal("1e-28")) * Decimal("1e-28")
                finite_difference = (4*(evidence(x+h).ln()-evidence(x).ln())
                                    -(evidence(x+2*h).ln()-evidence(x).ln()))/(2*h)
                lengths = np.array([s,0.,0.,0.])
                normal = evaluator.evaluate(lengths, threads=1)
                logged = evaluator.evaluate(lengths, threads=1, log_domain=True)
                exp_err = [abs(Decimal.from_float(float(item.length_gradient[0]))-exact)
                           for item in (normal, logged)]
                representability = abs(Decimal.from_float(float(exact))-exact)
                results.append({"compatible":compatible,"branch_length":s,
                                "oracle_log_likelihood":str(evidence(x).ln()),
                                "oracle_gradient":str(exact),
                                "decimal_one_sided_abs_error":float(abs(finite_difference-exact)),
                                "nearest_binary64_abs_error": float(representability),
                                "normal_abs_gradient_error":float(exp_err[0]),
                                "log_abs_gradient_error":float(exp_err[1]),
                                "paths_abs_difference":abs(float(normal.length_gradient[0]-logged.length_gradient[0])),
                                "paths_pass_frozen_absolute_test": bool(abs(normal.length_gradient[0]-logged.length_gradient[0])<=ABS),
                                "both_pass_oracle_absolute_test":max(exp_err)<=Decimal("1e-8")})
                assert abs(finite_difference-exact) < Decimal("1e-8")
        assert all(x["both_pass_oracle_absolute_test"] for x in results if x["compatible"])
        assert any(x["nearest_binary64_abs_error"]>ABS for x in results)
    return results


def snapshots(engine, fixed):
    comparisons = []
    for label,x in fixed:
        for domain in (False,True):
            runs = [engine.evaluate(x,threads=t,log_domain=domain) for t in (1,2,4)]
            gradients = [digest(z.length_gradient) for z in runs]
            values = [digest(z.site_log_likelihood) for z in runs]
            comparisons.append({"point":label,"forced_log":domain,"threads":[1,2,4],
                                "block_size":engine.pruning.gradient_block_size,
                                "gradient_sha256":gradients,"site_likelihood_sha256":values,
                                "bitwise_equal":len(set(gradients))==len(set(values))==1})
    assert all(x["bitwise_equal"] for x in comparisons)
    return comparisons


def diagnose_terminal(point, derivative, value, last, status, auditor, guard, scaled_probe):
    exact = auditor.evaluate(point, threads=4, log_domain=True)
    n = len(exact.site_log_likelihood)
    ref_grad = -exact.length_gradient/n
    fit_grad = -value.length_gradient/n
    residual = point - np.clip(point-ref_grad,0,None)
    assert float(np.abs(residual).max()) == projected_norm(point,ref_grad)
    current = -value.log_likelihood/n

    def change(candidate):
        evaluated = auditor.evaluate(candidate, gradient=False, threads=4, log_domain=True)
        if not math.isfinite(evaluated.log_likelihood):
            raise ValueError("Feasibility and independent likelihood disagree")
        return -math.fsum(evaluated.site_log_likelihood-exact.site_log_likelihood)/n

    moves = []
    for label, initial_step, direction in last:
        probes = []
        halvings = (0, 8, 16, 22, 23, 24, 26, 28, 32, 38, 46, 54, 62)
        if scaled_probe:
            # A raw linear prediction is used only to locate diagnostic
            # probes. The projected displacement is recomputed for each one.
            prediction = initial_step * abs(float(derivative @ direction))
            if prediction > 0 and math.isfinite(prediction):
                center = math.floor(math.log2(prediction))
                halvings = sorted(set(halvings).union(center + k for k in (-8, 0, 8, 16)
                                                     if center + k > 62))
        for k in halvings:
            step = math.ldexp(initial_step,-k)
            trial = np.maximum(0,point+step*direction)
            delta = trial-point
            slope = float(fit_grad@delta)
            entry = {"halvings":k,"step":step,"dot_with_gradient":slope,
                     "max_displacement":float(np.abs(delta).max()),"moved":bool(np.any(delta))}
            if slope < 0:
                if not guard.feasible(trial):
                    entry["outcome"] = "boundary_conflict"
                else:
                    difference = change(trial)
                    entry.update(independent_objective_change=difference,
                                 independent_armijo_gap=difference-1e-4*slope,
                                 requested_decrease=1e-4*slope)
            probes.append(entry)
        moves.append({"direction":label,"probes_following_same_ray":probes})
    coord = int(np.abs(residual).argmax())
    one_sided = []
    coordinate_scale = point[coord] if point[coord] > 0 else 1e-3
    increments = [ratio * coordinate_scale for ratio in (1e-2, 1e-3, 1e-4)]
    if scaled_probe:
        local_scale = 1 / max(1, abs(ref_grad[coord]))
        increments += [ratio * local_scale for ratio in (1e-1, 1e-3, 1e-5, 1e-7)]
    for step in increments:
        trial = point.copy()
        trial[coord] += step
        actual = float(trial[coord]-point[coord])
        if not actual:
            continue
        fd = change(trial) / actual
        one_sided.append({"positive_step":actual,"independent_difference_quotient":fd,
                          "absolute_disagreement":abs(fd-ref_grad[coord]),
                          "relative_disagreement":abs(fd-ref_grad[coord])/max(1,abs(ref_grad[coord]))})
    return {"termination":status,"trace_probes_outside_solver_budget":True,
            "objective":current,"normal_projected_gradient":projected_norm(point,fit_grad),
            "rerooted_log_projected_gradient":float(np.abs(residual).max()),
            "normal_vs_rerooted_log_gradient_max_abs":float(np.abs(fit_grad-ref_grad).max()),
            "passes_gradient_absolute_gate":bool(np.abs(fit_grad-ref_grad).max()<=ABS),
            "max_site_likelihood_path_difference":float(np.abs(value.site_log_likelihood-exact.site_log_likelihood).max()),
            "dominant_projected_coordinate_length":float(point[coord]),
            "dominant_coordinate_exact_log_path_gradient":float(ref_grad[coord]),
            "one_sided_dominant_coordinate_probes":one_sided,
            "rays":moves}


def observed(args):
    engine,auditor,facts=load_case(args.prep,args.gate,args.rows,args.library)
    guard=ZeroBoundary(engine,args.boundary)
    start=start_lengths("short",facts["edges"])
    checks=snapshots(engine,[("short",start),("heterogeneous",start_lengths("heterogeneous",facts["edges"]))]) if args.snapshot else []
    events=[]
    terminal=[]
    with args.trace.open("x",encoding="utf-8") as stream:
        def emit(event):
            events.append(event)
            stream.write(json.dumps(event,allow_nan=False)+"\n")
            stream.flush()

        def finished(*data):
            if args.probe:
                terminal.append(diagnose_terminal(*data, auditor, guard, args.scaled_probe))

        lengths, fit=optimize_projected(engine,start,emit=emit,threads=args.threads,
                                       maxeval=260,maxiter=150,gtol=1e-5,
                                       restrict_history=True,zero_boundary=guard,
                                       search_diagnostics=True,
                                       terminal_diagnostic=finished if args.probe else None)
    final_check = snapshots(engine,[("terminal",lengths)]) if args.snapshot else []
    return {**facts,"threads":args.threads,"search":fit,"trace_audit":audit_trace(events,fit),
            "start_thread_checks":checks,"terminal_thread_checks":final_check,
            "terminal_diagnostics":terminal,
            "peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library",type=Path,required=True)
    parser.add_argument("--boundary",type=Path,required=True)
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--prep",type=Path)
    parser.add_argument("--gate",type=Path)
    parser.add_argument("--rows",type=int,choices=(1000,10000))
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--trace",type=Path)
    parser.add_argument("--snapshot",action="store_true")
    parser.add_argument("--probe",action="store_true")
    parser.add_argument("--scaled-probe", action="store_true", help="Use representable scale-aware terminal probes")
    args=parser.parse_args()
    if args.check:
        out={"build_commands":[build(args.library),build_boundary(args.boundary)],
             "decimal_gradient_oracle":exact_star(args.library)}
    elif args.prep and args.gate and args.rows and args.trace:
        before=perf_counter()
        out=observed(args)
        out["whole_case_wall_seconds"]=perf_counter()-before
    else:
        parser.error("Choose --check or provide an observed fixture and trace path")
    print(json.dumps(out,allow_nan=False,indent=2))


if __name__=="__main__":
    main()
