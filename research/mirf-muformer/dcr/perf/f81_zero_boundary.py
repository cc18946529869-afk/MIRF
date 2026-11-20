"""Exact observed-leaf equality and feasible ray steps for F81 pruning."""
from __future__ import annotations

import ctypes
import math
from pathlib import Path
import subprocess

import numpy as np


# Leave a proportional interior margin only on rays with an incompatible
# boundary. Feasible zero edges still have their original lower bound of zero.
FRACTION_TO_CONFLICT = 0.95


def build_boundary(output: Path):
    command = ["gcc", "-O3", "-std=c11", "-Wall", "-Wextra", "-Werror", "-fPIC", "-shared",
               "-fno-fast-math", "-ffp-contract=off", str(Path(__file__).with_suffix(".c")),
               "-o", str(output.resolve()), "-lm"]
    subprocess.run(command, check=True)
    return command


class ZeroBoundary:
    def __init__(self, engine, library: Path):
        self.topology = engine.topology
        self.order = self.topology.child_to_edge
        self.parent = self.topology.layout.parent
        # Axis-wise unique sorts and compares the actual observation bytes.
        # Labels encode equality only and are never used as a hash surrogate.
        _, labels = np.unique(engine.pruning.states.T, axis=0, return_inverse=True)
        self.labels = np.ascontiguousarray(labels.reshape(-1), dtype=np.int32)
        native = ctypes.CDLL(str(library.resolve()))
        signatures = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32]
        self._feasible = native.f81_zero_feasible
        self._feasible.argtypes = [*signatures, ctypes.c_void_p]
        self._feasible.restype = ctypes.c_int
        self._conflict = native.f81_first_conflict
        self._conflict.argtypes = [*signatures, ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.POINTER(ctypes.c_double)]
        self._conflict.restype = ctypes.c_int

    def _in_child_order(self, vector):
        vector = np.asarray(vector, dtype=np.float64)
        if vector.shape != self.order.shape or not np.isfinite(vector).all():
            raise ValueError("Expected one finite value per unrooted edge")
        return vector[self.order]

    def feasible(self, lengths):
        child = self._in_child_order(lengths)
        if np.any(child < 0):
            raise ValueError("Lengths cannot be negative")
        result = self._feasible(self.parent.ctypes.data, self.labels.ctypes.data,
                                len(self.labels), len(child), child.ctypes.data)
        if result < 0:
            raise MemoryError("Zero-edge component allocation failed")
        return bool(result)

    def first_conflict(self, point, direction):
        point = self._in_child_order(point)
        direction = self._in_child_order(direction)
        if np.any(point < 0):
            raise ValueError("Lengths cannot be negative")
        limit = ctypes.c_double()
        result = self._conflict(self.parent.ctypes.data, self.labels.ctypes.data, len(self.labels),
                                len(point), point.ctypes.data, direction.ctypes.data,
                                ctypes.byref(limit))
        if result:
            raise MemoryError("Zero-edge ray workspace allocation failed")
        return limit.value

    @staticmethod
    def bounded_step(conflict):
        if not math.isfinite(conflict) or conflict > 1 / FRACTION_TO_CONFLICT:
            return 1.0
        return min(FRACTION_TO_CONFLICT * conflict, math.nextafter(conflict, 0.0))
