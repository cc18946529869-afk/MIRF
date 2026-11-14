"""Relative Frobenius deviation and the fixed bootstrap decision table.

These functions do not fit parameters. A numerator and its denominator use the
same array layout. ``J*`` therefore keeps both triangles together.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

PASS_LIMIT = 0.10
STAGE_MINIMUM = {100: 90, 200: 180}


def sample_deviation(samples: np.ndarray) -> np.ndarray:
    """Return the per-component sample standard deviation.

    ``samples`` has the replicate axis first. The deviation is centered at the
    sample mean and divides by ``B_s - 1``.
    """
    values = np.asarray(samples, dtype=np.float64)
    count = int(values.shape[0])
    if values.ndim < 1 or count < 2:
        raise ValueError("a sample deviation needs at least two replicates")
    if not np.all(np.isfinite(values)):
        raise ValueError("a sample deviation needs finite parameter values")
    centered = values - values.mean(axis=0)
    variance = np.sum(centered * centered, axis=0) / float(count - 1)
    return np.sqrt(variance)


def relative_frobenius(samples: np.ndarray, reference: np.ndarray) -> float | None:
    """Return ``||s||_F / ||reference||_F``, or ``None`` when the reference vanishes."""
    values = np.asarray(samples, dtype=np.float64)
    scale_array = np.asarray(reference, dtype=np.float64)
    if values.ndim < 1 or values.shape[1:] != scale_array.shape:
        raise ValueError("bootstrap samples and the reference must share a parameter shape")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(scale_array)):
        raise ValueError("bootstrap samples and the reference must be finite")
    scale = float(np.linalg.norm(scale_array))
    if scale == 0.0:
        return None
    deviation = sample_deviation(values)
    return float(np.linalg.norm(deviation) / scale)


def relative_change(full: float | None, half: float | None) -> float | None:
    """Return the relative change of one statistic, with both zeros defined as zero."""
    if full is None or half is None:
        return None
    if full == 0.0 and half == 0.0:
        return 0.0
    if full == 0.0:
        return None
    return abs(full - half) / full


def _stack(success: list[bool], values: Mapping[int, np.ndarray], start: int, stop: int) -> np.ndarray:
    chosen = []
    shape = None
    for identifier in range(start, stop + 1):
        if not success[identifier - 1]:
            continue
        if identifier not in values:
            raise ValueError(f"replicate {identifier} is marked successful but has no parameters")
        array = np.asarray(values[identifier], dtype=np.float64)
        if shape is None:
            shape = array.shape
        elif array.shape != shape:
            raise ValueError("replicates in one statistic do not share a parameter shape")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"replicate {identifier} contains a nonfinite parameter")
        chosen.append(array)
    if len(chosen) < 2:
        raise ValueError(f"replicates {start}-{stop} do not contain two successes")
    return np.stack(chosen, axis=0)


def evaluate_stage(
    success: list[bool],
    fields: Mapping[int, np.ndarray],
    couplings: Mapping[int, np.ndarray],
    field_reference: np.ndarray,
    coupling_reference: np.ndarray,
    stage: int,
) -> dict:
    """Apply the 100-replicate decision, or the 200-replicate decision.

    ``success[0]`` belongs to replicate 1. The half sample is the successful
    replicates inside the preassigned lower IDs, not a resampled half of the
    successes. ``R`` used for acceptance is the statistic on the whole stage.
    """
    if stage not in STAGE_MINIMUM:
        raise ValueError("the stage must be 100 or 200")
    if len(success) != stage:
        raise ValueError(f"the stage needs {stage} replicate flags")
    count = int(sum(bool(item) for item in success))
    minimum = STAGE_MINIMUM[stage]
    outcome = {
        "stage": stage,
        "success_count": count,
        "minimum": minimum,
        "action": "extend" if stage == 100 else "no_conclusion",
        "passed": False,
        "parameter_stable": None,
        "r_h": None,
        "r_j": None,
        "d_h": None,
        "d_j": None,
    }
    if count < minimum:
        return outcome
    half = stage // 2
    field_all = relative_frobenius(_stack(success, fields, 1, stage), field_reference)
    coupling_all = relative_frobenius(_stack(success, couplings, 1, stage), coupling_reference)
    field_half = relative_frobenius(_stack(success, fields, 1, half), field_reference)
    coupling_half = relative_frobenius(_stack(success, couplings, 1, half), coupling_reference)
    outcome["r_h"] = field_all
    outcome["r_j"] = coupling_all
    if None in (field_all, coupling_all, field_half, coupling_half):
        outcome["action"] = "undefined"
        return outcome
    field_change = relative_change(field_all, field_half)
    coupling_change = relative_change(coupling_all, coupling_half)
    outcome["d_h"] = field_change
    outcome["d_j"] = coupling_change
    if field_change is None or coupling_change is None:
        outcome["action"] = "undefined"
        return outcome
    if max(field_change, coupling_change) > PASS_LIMIT:
        return outcome
    stable = field_all <= PASS_LIMIT and coupling_all <= PASS_LIMIT
    outcome["parameter_stable"] = stable
    outcome["passed"] = stable
    outcome["action"] = "pass" if stable else "parameter_unstable"
    return outcome
