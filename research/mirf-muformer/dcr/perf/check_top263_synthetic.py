"""Bounded Top-263 experiment on fixed synthetic fixtures; never reads research data."""

import argparse
from decimal import Context, Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN
import json
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np

import bootstrap_gauge
import contact_validation
import potts_fit
from perf.fit_kernel_b import BoundBKernel, build

PRECISION = 42
TOP = 263
ZERO = Decimal(0)


def point(value):
    if isinstance(value, Decimal):
        return value, value
    exact = Decimal(value) if isinstance(value, int) else Decimal.from_float(float(value))
    return exact, exact


class Intervals:
    def __init__(self):
        self.lo = Context(prec=PRECISION, rounding=ROUND_FLOOR)
        self.hi = Context(prec=PRECISION, rounding=ROUND_CEILING)
        self.nearest = Context(prec=PRECISION, rounding=ROUND_HALF_EVEN)

    def add(self, x, y):
        return self.lo.add(x[0], y[0]), self.hi.add(x[1], y[1])

    def sub(self, x, y):
        return self.lo.subtract(x[0], y[1]), self.hi.subtract(x[1], y[0])

    def mean(self, values, count):
        lower, upper = self.sum(values)
        divisor = Decimal(count)
        return self.lo.divide(lower, divisor), self.hi.divide(upper, divisor)

    def sum(self, values):
        acc = point(0)
        for item in values:
            acc = self.add(acc, item)
        return acc

    def scale_positive(self, interval, scalar):
        if scalar < 0:
            raise ValueError("expected a nonnegative scalar")
        return self.lo.multiply(interval[0], scalar), self.hi.multiply(interval[1], scalar)

    def mul(self, x, y):
        return (
            min(self.lo.multiply(a, b) for a in x for b in y),
            max(self.hi.multiply(a, b) for a in x for b in y),
        )

    def positive_ratio(self, x, y):
        if x[0] < 0 or y[0] <= 0:
            raise ValueError("expected a nonnegative numerator and positive denominator")
        return self.lo.divide(x[0], y[1]), self.hi.divide(x[1], y[0])

    def exp(self, x):
        # Decimal exp is correctly rounded to nearest. Neighboring numbers enclose it.
        return max(ZERO, self.lo.next_minus(self.nearest.exp(x[0]))), self.hi.next_plus(self.nearest.exp(x[1]))

    def sqrt(self, x):
        if x[0] < 0:
            raise ValueError("negative square root interval")
        lo = self.lo.next_minus(self.nearest.sqrt(x[0])) if x[0] else ZERO
        hi = self.hi.next_plus(self.nearest.sqrt(x[1])) if x[1] else ZERO
        # Check the brackets with directed multiplication, independently of sqrt rounding.
        if self.hi.multiply(lo, lo) > x[0] or self.lo.multiply(hi, hi) < x[1]:
            raise ArithmeticError("square root enclosure failed")
        return lo, hi

    def square(self, x):
        magnitudes = [part.copy_abs() for part in x]
        min_abs = ZERO if x[0] <= 0 <= x[1] else min(magnitudes)
        max_abs = max(magnitudes)
        return self.lo.multiply(min_abs, min_abs), self.hi.multiply(max_abs, max_abs)

    def discrepancy(self, value, interval):
        exact = point(value)[0]
        return max(ZERO, self.hi.subtract(exact, interval[0]), self.hi.subtract(interval[1], exact))


def gradient_enclosure(iv, matrix, site, weights, total, theta, q):
    """Enclose the real gradient at the saved binary64 inputs."""
    parameters = [point(x) for x in theta]
    _, length = matrix.shape
    partners = [j for j in range(length) if j != site]
    grad = [point(0)] * len(theta)
    for row, weight in zip(matrix.tolist(), weights):
        offsets = [q + slot * q + row[j] for slot, j in enumerate(partners)]
        logits = [
            iv.sum([parameters[a]] + [parameters[base + a * (length - 1) * q] for base in offsets])
            for a in range(q)
        ]
        shift = point(max(x[1] for x in logits))
        exps = [iv.exp(iv.sub(x, shift)) for x in logits]
        den = iv.sum(exps)
        weight_exact = point(weight)[0]
        for a, exp in enumerate(exps):
            prob = iv.positive_ratio(exp, den)
            contribution = iv.scale_positive(iv.sub(prob, point(int(a == row[site]))), weight_exact)
            grad[a] = iv.add(grad[a], contribution)
            for base in offsets:
                idx = base + a * (length - 1) * q
                grad[idx] = iv.add(grad[idx], contribution)

    w2 = iv.scale_positive(point(total), Decimal(2))
    field = iv.mul(w2, point(potts_fit.LAMBDA_H))
    coupling = iv.mul(w2, point(potts_fit.LAMBDA_J_PRIME))
    for k, value in enumerate(parameters):
        grad[k] = iv.add(grad[k], iv.mul(value, field if k < q else coupling))
    return grad, min(field[0], coupling[0])


def zero_sum(iv, values):
    size = len(values)
    rows = [iv.mean(row, size) for row in values]
    cols = [iv.mean(column, size) for column in zip(*values)]
    mean = iv.mean(rows, size)
    return [
        [iv.add(iv.sub(iv.sub(x, rows[i]), cols[j]), mean) for j, x in enumerate(row)]
        for i, row in enumerate(values)
    ]


def raw_enclosure(iv, first, second):
    u = zero_sum(iv, [[point(x) for x in row] for row in first])
    v = zero_sum(iv, [[point(x) for x in row] for row in second])
    return iv.sqrt(iv.sum(
        iv.square(iv.scale_positive(iv.add(u[i][j], v[j][i]), Decimal("0.5")))
        for i in range(1, len(u)) for j in range(1, len(u))
    ))


def apc_enclosure(iv, raw):
    count = len(raw)
    rows = [iv.mean(line, count - 1) for line in raw]
    center = iv.mean(rows, count)
    if center[1] == 0:
        return [[point(0)] * count for _ in raw], "exact-zero"
    if center[0] <= 0:
        return None, "center-not-separated"
    return [
        [point(0) if i == j else iv.sub(value, iv.positive_ratio(iv.mul(rows[i], rows[j]), center))
         for j, value in enumerate(line)]
        for i, line in enumerate(raw)
    ], "positive"


def check_bounds():
    iv = Intervals()
    assert iv.exp(point(0))[0] <= 1 <= iv.exp(point(0))[1]
    assert iv.sqrt(point(4))[0] <= 2 <= iv.sqrt(point(4))[1]
    zero = [point(0)] * 3
    assert apc_enclosure(iv, [zero] * 3)[1] == "exact-zero"
    uncertain = (ZERO, Decimal(1))
    assert apc_enclosure(iv, [zero, [point(0), point(0), uncertain], [point(0), uncertain, point(0)]])[0] is None


def run_fixture(kernel, *, group_fixture, weight_scale=1):
    # Fixed artificial numeric arrays. No paths or datasets are accepted.
    rng = np.random.default_rng(511 if group_fixture else 733)
    length, q, rows = (45, 3, 160) if group_fixture else (27, 21, 64)
    matrix = rng.integers(0, q, size=(rows, length), dtype=np.int8)
    if group_fixture:
        matrix[:, :23] = rng.integers(0, q, size=(rows, 1), dtype=np.int8)
        matrix[:, 23:28] = rng.integers(0, q, size=(rows, 1), dtype=np.int8)
    weights = np.full(rows, weight_scale, dtype=np.float64)
    total = float(weights.sum())
    start = perf_counter()
    fits = [potts_fit.fit_site(matrix, weights, i, total, q=q, b_kernel=kernel) for i in range(length)]
    fit_seconds = perf_counter() - start
    iv = Intervals()
    eta, xi, e = ZERO, ZERO, []
    start = perf_counter()
    lib = kernel.library_for_worker()
    for site, fit in enumerate(fits):
        _, _, gradient = potts_fit.site_terms_c(fit.theta, matrix, weights, site, total, q=q, library=lib)
        enclosure, kappa = gradient_enclosure(iv, matrix, site, weights, total, fit.theta, q)
        eta = max(eta, max(iv.discrepancy(g, bound) for g, bound in zip(gradient, enclosure)))
        norm = iv.sqrt(iv.sum(iv.square(x) for x in enclosure))[1]
        e.append(iv.hi.divide(norm, kappa))
    gradient_seconds = perf_counter() - start

    start = perf_counter()
    fields, directional = bootstrap_gauge.assemble_directional([f.h for f in fits], [f.couplings for f in fits])
    _, gauge = bootstrap_gauge.gauge_parameters(fields, directional)
    numeric_raw = np.sqrt(np.sum(gauge[:, :, 1:, 1:] ** 2, axis=(2, 3)))
    raw = [[point(0)] * length for _ in range(length)]
    for i in range(length):
        for j in range(i + 1, length):
            nominal = raw_enclosure(iv, directional[i,j], directional[j,i])
            xi = max(xi, iv.discrepancy(numeric_raw[i,j], nominal))
            pad = iv.hi.divide(iv.hi.add(e[i], e[j]), Decimal(2))
            raw[i][j] = raw[j][i] = (
                max(ZERO, iv.lo.subtract(nominal[0], pad)), iv.hi.add(nominal[1], pad)
            )
    corrected, center_status = apc_enclosure(iv, raw)
    if center_status == "exact-zero":
        # The exact-zero extension is tested here only on the toy scoring path.
        numeric_apc = np.zeros_like(numeric_raw)
    else:
        numeric_apc = contact_validation.average_product_correction(numeric_raw)
    left, right = np.triu_indices(length, k=1)
    order = contact_validation.ranking_order(left, right, numeric_apc[left, right])
    inside = list(zip(left[order[:TOP]], right[order[:TOP]]))
    outside = list(zip(left[order[TOP:]], right[order[TOP:]]))
    slack = None
    if corrected is not None and center_status == "positive":
        slack = iv.lo.subtract(
            min(corrected[i][j][0] for i,j in inside),
            max(corrected[i][j][1] for i,j in outside),
        )
    certifies = bool(center_status == "exact-zero" or (slack is not None and slack > 0))
    return {
        "fixture": "synthetic_numeric_groups" if group_fixture else "synthetic_numeric_diffuse",
        "rows": rows, "columns": length, "states": q, "candidate_pairs": len(order), "top_k": TOP,
        "uniform_weight_scale": weight_scale,
        "stored_gradient_inf_max": max(f.grad_inf for f in fits),
        "stored_fit_status_converged": sum(f.status == "converged" for f in fits),
        "eta_upper_max": str(eta),
        "xi_raw_upper_max": str(xi),
        "parameter_error_upper_max": str(max(e)),
        "apc_center_status": center_status,
        "top_boundary_interval_slack_lower": None if slack is None else str(slack),
        "numeric_boundary_gap": float(numeric_apc[inside[-1]] - numeric_apc[outside[0]]),
        "top_set_certified": certifies,
        "seconds": {
            "synthetic_fits": fit_seconds,
            "gradient_reference": gradient_seconds,
            "raw_apc_and_top_set": perf_counter() - start,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New JSON file for the synthetic fixtures")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    check_bounds()
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp) / "b"
        manifest = build(directory)
        artifact = manifest["artifact_sha256"]
        kernel = BoundBKernel(directory, expected_sha256=artifact)
        results = [
            run_fixture(kernel, group_fixture=True),
            run_fixture(kernel, group_fixture=False),
            # Positive-path control: same exact minimizers, scaled unnormalized objective.
            run_fixture(kernel, group_fixture=False, weight_scale=32),
        ]
    report = {
        "scope": "Synthetic numeric arrays only; one toy group (all pairs), no production contact labels.",
        "decimal_precision": PRECISION,
        "artifact_sha256": artifact,
        "compiler": manifest["compiler_version"],
        "fixture_sets_certified": sum(r["top_set_certified"] for r in results),
        "fixture_sets_total": len(results),
        "fixtures": results,
    }
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
