"""Independently audit selected derivatives using Edge-split Boost binary floats."""
from __future__ import annotations

import argparse
import ctypes
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import resource
from time import perf_counter

import numpy as np

from .check_f81_observed_bridge import load_case
from .f81_unrooted import UnrootedPruning, UnrootedTopology


def adjacency(topology):
    layout = topology.layout
    pairs = np.empty((len(layout.parent), 2), dtype=np.int32)
    pairs[topology.child_to_edge] = np.column_stack((np.arange(len(layout.parent)), layout.parent))
    neighbors = [[] for _ in range(len(pairs) + 1)]
    for edge, (left, right) in enumerate(pairs.tolist()):
        neighbors[left].append((right, edge))
        neighbors[right].append((left, edge))
    return pairs, neighbors


class Reference:
    def __init__(self, engine, lengths, library):
        self.pruning = engine.pruning
        self.lengths = lengths
        self.pairs, self.neighbors = adjacency(engine.topology)
        native = ctypes.CDLL(str(library.resolve()))
        self.native = native.f81_split_gradient
        pointer, integer = ctypes.c_void_p, ctypes.c_int
        self.native.argtypes = [pointer, pointer, integer, integer, integer,
                                pointer, pointer, pointer, integer, integer,
                                ctypes.c_double, integer, ctypes.c_char_p]
        self.native.restype = integer

        self.native_site_terms = getattr(native, "f81_split_site_terms", None)
        if self.native_site_terms is not None:
            self.native_site_terms.argtypes = [*self.native.argtypes, pointer]
            self.native_site_terms.restype = integer

    def split(self, edge):
        left, right = self.pairs[edge]
        nodes = len(self.neighbors)
        parents = np.full(nodes, -1, dtype=np.int32)
        lengths = np.zeros(nodes)
        traversal = [int(left), int(right)]
        for node in traversal:
            for other, index in self.neighbors[node]:
                if index != edge and other != parents[node]:
                    parents[other] = node
                    lengths[other] = self.lengths[index]
                    traversal.append(other)
        if len(traversal) != nodes:
            raise ValueError("Edge split must span the two complete components")
        order = np.array(traversal[::-1], dtype=np.int32)
        return left, right, order, parents, lengths

    def score(self, edge, bits):
        average, _ = self.evaluate(edge, bits, include_sites=False)
        return average

    def site_terms(self, edge, bits):
        """Return A, B, mass, negative-log score and diagonal by site."""
        return self.evaluate(edge, bits, include_sites=True)

    def evaluate(self, edge, bits, *, include_sites):
        left, right, order, parents, lengths = self.split(edge)
        states, profiles = self.pruning.states, self.pruning.profiles
        sites, leaves = states.shape
        width = 128
        encoded = ctypes.create_string_buffer(width)
        args = (states.ctypes.data, profiles.ctypes.data, sites, leaves, len(order),
                order.ctypes.data, parents.ctypes.data, lengths.ctypes.data,
                int(left), int(right), float(self.lengths[edge]), bits, encoded)
        terms = None
        if include_sites:
            if self.native_site_terms is None:
                raise ValueError("Build the reference library with the site-term entry point")
            buffer = ctypes.create_string_buffer(width * sites * 5)
            status = self.native_site_terms(*args, ctypes.addressof(buffer))
        else:
            status = self.native(*args)
        if status != 0:
            raise ArithmeticError(f"Independent {bits}-bit split computation failed: {status}")
        if include_sites:
            slots = np.frombuffer(buffer, dtype=f"S{width}").reshape(sites, 5)
            terms = [tuple(Decimal(x.decode("ascii")) for x in row) for row in slots]
        return Decimal(encoded.value.decode("ascii")), terms



def check(library, native_reference):
    # All three zero-length leaves force state 1 at the internal node.
    pairs = np.column_stack((np.arange(4), np.repeat(4, 4)))
    tree = UnrootedTopology.from_edges(4, pairs)
    pi = np.zeros((1, 21))
    pi[0, :4] = 0.25
    findings = []
    for same, length in ((True, 0.), (False, 1e-12), (False, 1e-24)):
        states = np.array([[int(same)], [1], [1], [1]], dtype=np.uint8)
        engine = UnrootedPruning(tree, states, pi, library)
        point = np.array([length, 0., 0., 0.])
        ref = Reference(engine, point, native_reference)
        with localcontext() as ctx:
            ctx.prec = 110
            decay = (-Decimal.from_float(length)).exp()
            oracle = 3 * decay / (1 + 3 * decay) if same else -decay / (1 - decay)
            delta = abs(ref.score(0, 256) - oracle)
            assert delta <= Decimal("1e-45")
            findings.append({"compatible": same, "length": length, "absolute_error": str(delta)})
    return {"tests": findings, "passed": True}


def compare(args):
    engine, auditor, facts = load_case(args.prep, args.gate, 10000, args.library)
    point = np.load(args.checkpoint, allow_pickle=False)
    normal = -engine.evaluate(point, threads=4).length_gradient / facts["sites"]
    logged = -auditor.evaluate(point, threads=4, log_domain=True).length_gradient / facts["sites"]
    mismatch = np.abs(normal - logged)
    positive = np.flatnonzero(point > 0)
    zero = np.flatnonzero(point == 0)
    selections = [("largest_cross_path_error", int(np.argmax(mismatch)))]
    if len(positive):
        selections.append(("smallest_positive_edge", int(positive[np.argmin(point[positive])])))
    if len(zero):
        selections.append(("zero_edge_largest_cross_path_error", int(zero[np.argmax(mismatch[zero])])))

    oracle = Reference(engine, point, args.reference)
    records = {}
    for label, edge in selections:
        if edge in records:
            records[edge]["selection"].append(label)
            continue
        started = perf_counter()
        lower, higher = (oracle.score(edge, bits) for bits in (192, 256))
        with localcontext() as ctx:
            ctx.prec = 115
            gate = Decimal("1e-8")
            ref_gap = abs(lower - higher)
            nearest = float(higher)
            nearest_exact = Decimal.from_float(nearest)
            representation = nearest_exact - higher
            record = {"edge_index": edge, "selection": [label], "length": float(point[edge]),
                      "reference_192bit": str(lower), "reference_256bit": str(higher),
                      "reference_pair_abs_difference": str(ref_gap),
                      "nearest_binary64": nearest, "nearest_binary64_hex": nearest.hex(),
                      "signed_representation_error": str(representation),
                      "absolute_representation_error": str(abs(representation)),
                      "representable_at_frozen_absolute_gate": abs(representation) <= gate,
                      "reference_pair_passes_absolute_gate": ref_gap <= gate,
                      "cross_binary64_absolute_difference": float(mismatch[edge]),
                      "paths": {}}
            for key, binary in (("ordinary", normal[edge]), ("rerooted_log", logged[edge])):
                error = Decimal.from_float(float(binary)) - nearest_exact
                total = error + representation
                record["paths"][key] = {"binary64": float(binary), "binary64_hex": float(binary).hex(),
                                        "signed_algorithm_error_to_nearest": str(error),
                                        "absolute_algorithm_error_to_nearest": str(abs(error)),
                                        "signed_total_error_to_reference": str(total),
                                        "absolute_total_error_to_reference": str(abs(total)),
                                        "passes_absolute_gate_vs_reference": abs(total) <= gate}
            record["elapsed_seconds"] = perf_counter() - started
            records[edge] = record
    return {**facts, "checkpoint_sha256": hashlib.sha256(point.tobytes()).hexdigest(),
            "high_precision_library_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
            "absolute_gate": "1e-8", "coordinates": list(records.values()),
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for opt in ("prep", "gate", "checkpoint", "library", "reference"):
        parser.add_argument("--" + opt, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        result = check(args.library, args.reference)
    else:
        result = compare(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
