"""Identifiable edge coordinates for fixed-topology reversible pruning."""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

from .f81_pruning import Pruning, Topology


@dataclass(frozen=True)
class UnrootedTopology:
    """Input edge order is the parameter order, independent of the pruning root."""

    layout: Topology
    child_to_edge: np.ndarray

    @classmethod
    def from_edges(cls, leaves, edges, root=None):
        pairs = np.asarray(edges)
        if (pairs.ndim != 2 or pairs.shape[1] != 2
                or not np.issubdtype(pairs.dtype, np.integer)):
            raise ValueError("Expected an integer edge-by-two array")
        nodes = len(pairs) + 1
        if not 3 <= leaves < nodes < (1 << 31):
            raise ValueError("An unrooted optimization tree needs at least three leaves")
        if np.any(pairs < 0) or np.any(pairs >= nodes) or np.any(pairs[:, 0] == pairs[:, 1]):
            raise ValueError("Invalid edge endpoints")
        adjacent = [[] for _ in range(nodes)]
        for edge, (left, right) in enumerate(pairs.tolist()):
            adjacent[left].append((right, edge))
            adjacent[right].append((left, edge))
        if (any(len(x) != 1 for x in adjacent[:leaves])
                or any(len(x) < 3 for x in adjacent[leaves:])):
            raise ValueError("Leaves need degree one and all internal nodes degree at least three")

        root = leaves if root is None else root
        if not isinstance(root, (int, np.integer)) or not leaves <= root < nodes:
            raise ValueError("Choose an internal node as the computation root")
        old_parent = np.full(nodes, -1, dtype=np.int32)
        old_parent[root] = root
        old_edge = np.empty(nodes, dtype=np.int32)
        traversal = [root]
        for node in traversal:
            for neighbor, edge in adjacent[node]:
                if neighbor == old_parent[node]:
                    continue
                if old_parent[neighbor] != -1:
                    raise ValueError("Unrooted input contains a cycle or duplicate edge")
                old_parent[neighbor] = node
                old_edge[neighbor] = edge
                traversal.append(neighbor)
        if len(traversal) != nodes:
            raise ValueError("Unrooted input is disconnected")
        postorder = np.r_[np.arange(leaves), [node for node in reversed(traversal) if node >= leaves]]
        new_id = np.empty(nodes, dtype=np.int32)
        new_id[postorder] = np.arange(nodes)
        layout = Topology.from_parents(leaves, new_id[old_parent[postorder[:-1]]])
        return cls(layout, old_edge[postorder[:-1]])

    @classmethod
    def from_rooted(cls, topology, lengths):
        """Suppress a degree-two computation root and map its legacy edge slots."""
        parent = topology.parent
        values = np.asarray(lengths, dtype=float)
        if values.shape != parent.shape or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Invalid original edge lengths")
        root = len(parent)
        pairs = np.column_stack((np.arange(root), parent))
        children = np.flatnonzero(parent == root)
        old_to_new = np.arange(root)
        if len(children) == 2:
            keep = parent != root
            pairs = np.vstack((pairs[keep], children))
            old_to_new[keep] = np.arange(int(keep.sum()))
            old_to_new[children] = len(pairs) - 1
            with np.errstate(over="ignore"):
                merged = values[children].sum()
            if not np.isfinite(merged):
                raise ValueError("Sum of the split-root lengths is outside binary64")
            values = np.r_[values[keep], merged]
        return cls.from_edges(topology.leaves, pairs), values, old_to_new


class UnrootedPruning:
    def __init__(self, topology, observations, profiles, library):
        self.topology = topology
        self.pruning = Pruning(topology.layout, observations, profiles, library)

    def evaluate(self, lengths, **kwargs):
        values = np.asarray(lengths, dtype=float)
        order = self.topology.child_to_edge
        if values.shape != order.shape:
            raise ValueError("Expected one length per unrooted edge")
        evaluated = self.pruning.evaluate(values[order], **kwargs)
        if evaluated.length_gradient is None:
            return evaluated
        gradient = np.empty_like(evaluated.length_gradient)
        gradient[order] = evaluated.length_gradient
        return replace(evaluated, length_gradient=gradient)
