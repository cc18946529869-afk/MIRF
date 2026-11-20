"""Fixed-topology pruning for a 21-category profile model.

Each tree is a postorder parent vector. Leaves are nodes 0..N-1, one
per row, and the edge length/gradient slot is the child node id.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
import math
from pathlib import Path
import subprocess

import numpy as np


@dataclass(frozen=True)
class Topology:
    leaves: int
    parent: np.ndarray
    offsets: np.ndarray
    children: np.ndarray
    max_degree: int

    @classmethod
    def from_parents(cls, leaves, parent):
        raw = np.asarray(parent)
        nodes = raw.size + 1
        if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("Parent indices must be a vector of integers")
        if not 1 <= leaves <= nodes < (1 << 31):
            raise ValueError("Invalid leaf or node count")
        if np.any(raw >= nodes) or np.any(raw <= np.arange(nodes - 1)):
            raise ValueError("Every parent must appear after its child")
        parents = np.ascontiguousarray(raw, dtype=np.int32)
        order = np.ascontiguousarray(np.argsort(parents, kind="stable"), dtype=np.int32)
        degree = np.bincount(parents, minlength=nodes)
        if np.any(degree[:leaves]) or np.any(degree[leaves:] == 0):
            raise ValueError("Exactly the first N nodes must be leaves")
        offsets = np.empty(nodes + 1, dtype=np.int32)
        offsets[0] = 0
        np.cumsum(degree, dtype=np.int32, out=offsets[1:])
        return cls(leaves, parents, offsets, order, max(1, int(degree.max())))


@dataclass(frozen=True)
class Evaluation:
    log_likelihood: float
    site_log_likelihood: np.ndarray
    length_gradient: np.ndarray | None
    log_domain_sites: int


def build(output: Path):
    source = Path(__file__).with_suffix(".c")
    flags = ["gcc", "-O3", "-std=c11", "-Wall", "-Wextra", "-Werror", "-fPIC", "-shared",
             "-fopenmp", "-fno-fast-math", "-ffp-contract=off"]
    command = [*flags, str(source), "-o", str(output.resolve()), "-lm"]
    subprocess.run(command, check=True)
    return command


class Pruning:
    def __init__(self, topology: Topology, observations, profiles, library: Path):
        states = np.asarray(observations)
        if states.ndim != 2 or states.shape[0] != topology.leaves or states.shape[1] == 0:
            raise ValueError("Expected a nonempty sites matrix in observation-row order")
        if not np.issubdtype(states.dtype, np.integer) or (states.min() < 0 or states.max() >= 21):
            raise ValueError("Observations must contain only integer codes 0..20")
        law = np.asarray(profiles, dtype=np.float64)
        if (law.shape != (states.shape[1], 21) or not np.isfinite(law).all()
            or np.any(law < 0) or np.any(law > 1)
            or not np.allclose(law.sum(axis=1), 1.0, rtol=0, atol=1e-12)):
            raise ValueError("Invalid site profiles")
        self.topology = topology
        self.states = np.ascontiguousarray(states.T, dtype=np.uint8)
        self.profiles = np.ascontiguousarray(law)
        self._native = ctypes.CDLL(str(library.resolve())).f81_sites
        self._native.argtypes = ([ctypes.c_void_p] * 2 + [ctypes.c_int32] * 3 + [ctypes.c_void_p] * 6
                                + [ctypes.c_int32] * 4 + [ctypes.c_void_p] * 4)
        self._native.restype = None

    def evaluate(self, lengths, *, gradient=True, threads=1, log_domain=False):
        lengths = np.ascontiguousarray(lengths, dtype=np.float64)
        tree = self.topology
        if lengths.shape != tree.parent.shape or not np.isfinite(lengths).all() or np.any(lengths < 0):
            raise ValueError("One finite nonnegative length is required for each edge")
        if not 1 <= threads <= 40:
            raise ValueError("threads must be between 1 and 40")
        stay = np.exp(-lengths)
        redraw = -np.expm1(-lengths)
        with np.errstate(divide="ignore"):
            log_redraw = np.log(redraw)
        use_log = log_domain or np.any(stay < np.finfo(float).tiny)
        nsites = len(self.states)
        by_thread = np.zeros((threads, len(lengths))) if gradient else None
        status = np.zeros(nsites, dtype=np.int32)
        loglik = np.full(nsites, -np.inf, dtype=np.float64)
        log_used = np.zeros(nsites, dtype=np.uint8)
        self._native(self.states.ctypes.data, self.profiles.ctypes.data, nsites, len(lengths) + 1,
                     tree.leaves, tree.offsets.ctypes.data, tree.children.ctypes.data,
                     stay.ctypes.data, redraw.ctypes.data, lengths.ctypes.data, log_redraw.ctypes.data,
                     int(use_log), tree.max_degree, threads, int(gradient),
                     None if by_thread is None else by_thread.ctypes.data,
                     loglik.ctypes.data, status.ctypes.data, log_used.ctypes.data)
        if np.any(status == 3):
            raise MemoryError("Native partial likelihood workspace allocation failed")
        if np.any(status == 2):
            raise FloatingPointError(f"At least one site exceeds the available likelihood or gradient range"
                                     f" ({int(np.sum(status == 2))}/{nsites})")
        if np.any(status == 1):
            loglik[status == 1] = -np.inf
            return Evaluation(-math.inf, loglik, None, int(log_used.sum()))
        result = by_thread.sum(axis=0) if gradient else None
        if result is not None and not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite total gradient")
        return Evaluation(math.fsum(loglik), loglik, result, int(log_used.sum()))
