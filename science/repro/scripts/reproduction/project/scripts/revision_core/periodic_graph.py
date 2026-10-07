"""Periodic quotient-graph transport analysis with explicit lattice image shifts.

The module never infers transport from a unit-cell component fraction or from
fractional-coordinate extrema.  A direction percolates only when a node in the
central cell reaches its own translated image in a bounded periodic supercell.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


Vector3 = Tuple[int, int, int]
CellState = Tuple[str, Vector3]
AXES: Dict[str, Vector3] = {"a": (1, 0, 0), "b": (0, 1, 0), "c": (0, 0, 1)}


@dataclass(frozen=True)
class PeriodicEdge:
    """Undirected quotient-graph edge from node_i to an image of node_j."""

    edge_uid: str
    node_i: str
    node_j: str
    image_shift: Vector3
    barrier_eV: Optional[float]

    def __post_init__(self) -> None:
        if len(self.image_shift) != 3 or any(not isinstance(value, int) for value in self.image_shift):
            raise ValueError("image_shift must contain three integers")
        if self.node_i == self.node_j and self.image_shift == (0, 0, 0):
            raise ValueError("zero-shift self edges are not valid migration edges")
        if self.barrier_eV is not None and (not math.isfinite(self.barrier_eV) or self.barrier_eV < 0):
            raise ValueError("barrier_eV must be finite, nonnegative, or None")


def add_vector(left: Vector3, right: Vector3) -> Vector3:
    return tuple(left[index] + right[index] for index in range(3))  # type: ignore[return-value]


def subtract_vector(left: Vector3, right: Vector3) -> Vector3:
    return tuple(left[index] - right[index] for index in range(3))  # type: ignore[return-value]


def negate_vector(vector: Vector3) -> Vector3:
    return (-vector[0], -vector[1], -vector[2])


def in_supercell(cell: Vector3, radius: int) -> bool:
    return all(-radius <= value <= radius for value in cell)


def active_edges(
    edges: Sequence[PeriodicEdge],
    threshold_eV: float,
    excluded_edge_uids: Optional[Set[str]] = None,
) -> List[PeriodicEdge]:
    excluded = excluded_edge_uids or set()
    return [
        edge
        for edge in edges
        if edge.edge_uid not in excluded
        and edge.barrier_eV is not None
        and edge.barrier_eV <= threshold_eV + 1e-12
    ]


def quotient_adjacency(edges: Iterable[PeriodicEdge]) -> Dict[str, List[Tuple[str, Vector3, str]]]:
    adjacency: Dict[str, List[Tuple[str, Vector3, str]]] = defaultdict(list)
    for edge in edges:
        adjacency[edge.node_i].append((edge.node_j, edge.image_shift, edge.edge_uid))
        adjacency[edge.node_j].append((edge.node_i, negate_vector(edge.image_shift), edge.edge_uid))
    return adjacency


def reaches_adjacent_image(
    nodes: Sequence[str],
    edges: Sequence[PeriodicEdge],
    threshold_eV: float,
    axis: str,
    supercell_radius: int,
    excluded_edge_uids: Optional[Set[str]] = None,
) -> bool:
    """Return whether any central node reaches its own adjacent periodic image."""
    if axis not in AXES:
        raise ValueError(f"Unknown axis: {axis}")
    if supercell_radius < 1:
        raise ValueError("supercell_radius must be at least one")
    adjacency = quotient_adjacency(active_edges(edges, threshold_eV, excluded_edge_uids))
    target_shifts = {AXES[axis], negate_vector(AXES[axis])}
    for start_node in nodes:
        start: CellState = (start_node, (0, 0, 0))
        queue = deque([start])
        visited: Set[CellState] = {start}
        while queue:
            node, cell = queue.popleft()
            if node == start_node and cell in target_shifts:
                return True
            for neighbour, shift, _edge_uid in adjacency.get(node, []):
                next_cell = add_vector(cell, shift)
                state = (neighbour, next_cell)
                if in_supercell(next_cell, supercell_radius) and state not in visited:
                    visited.add(state)
                    queue.append(state)
    return False


def unique_thresholds(edges: Sequence[PeriodicEdge]) -> List[float]:
    return sorted({float(edge.barrier_eV) for edge in edges if edge.barrier_eV is not None})


def directional_percolation_threshold(
    nodes: Sequence[str],
    edges: Sequence[PeriodicEdge],
    axis: str,
    supercell_radius: int,
) -> Optional[float]:
    thresholds = unique_thresholds(edges)
    lower = 0
    upper = len(thresholds)
    while lower < upper:
        middle = (lower + upper) // 2
        if reaches_adjacent_image(nodes, edges, thresholds[middle], axis, supercell_radius):
            upper = middle
        else:
            lower = middle + 1
    return thresholds[lower] if lower < len(thresholds) else None


def rational_rank(vectors: Iterable[Vector3]) -> int:
    matrix = [[Fraction(value) for value in vector] for vector in vectors if vector != (0, 0, 0)]
    if not matrix:
        return 0
    row = 0
    for column in range(3):
        pivot = next((candidate for candidate in range(row, len(matrix)) if matrix[candidate][column]), None)
        if pivot is None:
            continue
        matrix[row], matrix[pivot] = matrix[pivot], matrix[row]
        pivot_value = matrix[row][column]
        matrix[row] = [value / pivot_value for value in matrix[row]]
        for other in range(len(matrix)):
            if other == row or not matrix[other][column]:
                continue
            factor = matrix[other][column]
            matrix[other] = [matrix[other][index] - factor * matrix[row][index] for index in range(3)]
        row += 1
        if row == len(matrix):
            break
    return row


def maximal_minor_gcd(vectors: Sequence[Vector3], rank: int) -> int:
    """Greatest common divisor of all nonzero maximal minors in Z^3."""
    if rank == 0:
        return 0
    divisor = 0
    if rank == 1:
        for vector in vectors:
            for value in vector:
                if value:
                    divisor = math.gcd(divisor, abs(value))
                    if divisor == 1:
                        return 1
        return divisor
    elif rank == 2:
        for left, right in combinations(vectors, 2):
            cross = (
                left[1] * right[2] - left[2] * right[1],
                left[2] * right[0] - left[0] * right[2],
                left[0] * right[1] - left[1] * right[0],
            )
            for value in cross:
                if value:
                    divisor = math.gcd(divisor, abs(value))
                    if divisor == 1:
                        return 1
        return divisor
    elif rank == 3:
        for first, second, third in combinations(vectors, 3):
            determinant = (
                first[0] * (second[1] * third[2] - second[2] * third[1])
                - first[1] * (second[0] * third[2] - second[2] * third[0])
                + first[2] * (second[0] * third[1] - second[1] * third[0])
            )
            if determinant:
                divisor = math.gcd(divisor, abs(determinant))
                if divisor == 1:
                    return 1
        return divisor
    else:
        raise ValueError("Integer winding rank cannot exceed three")


def integer_winding_lattice_contains(generators: Sequence[Vector3], target: Vector3) -> bool:
    """Test exact membership in the integer row lattice generated by windings."""
    nonzero = [vector for vector in generators if vector != (0, 0, 0)]
    rank = rational_rank(nonzero)
    if rank == 0 or rational_rank(nonzero + [target]) != rank:
        return False
    return maximal_minor_gcd(nonzero, rank) == maximal_minor_gcd(nonzero + [target], rank)


def winding_vectors(nodes: Sequence[str], edges: Sequence[PeriodicEdge]) -> List[Vector3]:
    """Return nonzero integer residuals from a translated spanning forest."""
    adjacency = quotient_adjacency(edges)
    potentials: Dict[str, Vector3] = {}
    residuals: Set[Vector3] = set()
    for root in nodes:
        if root in potentials:
            continue
        potentials[root] = (0, 0, 0)
        queue = deque([root])
        while queue:
            node = queue.popleft()
            for neighbour, shift, _edge_uid in adjacency.get(node, []):
                expected = add_vector(potentials[node], shift)
                if neighbour not in potentials:
                    potentials[neighbour] = expected
                    queue.append(neighbour)
                    continue
                residual = subtract_vector(expected, potentials[neighbour])
                if residual != (0, 0, 0):
                    first_nonzero = next(value for value in residual if value)
                    canonical = residual if first_nonzero > 0 else negate_vector(residual)
                    residuals.add(canonical)
    return sorted(residuals)


def winding_vector_rank(nodes: Sequence[str], edges: Sequence[PeriodicEdge]) -> int:
    ranks = []
    for component in quotient_components(nodes, edges):
        component_edges = [edge for edge in edges if edge.node_i in component and edge.node_j in component]
        ranks.append(rational_rank(winding_vectors(sorted(component), component_edges)))
    return max(ranks, default=0)


def exact_axis_percolates(nodes: Sequence[str], edges: Sequence[PeriodicEdge], axis: str) -> bool:
    if axis not in AXES:
        raise ValueError(f"Unknown axis: {axis}")
    for component in quotient_components(nodes, edges):
        component_edges = [edge for edge in edges if edge.node_i in component and edge.node_j in component]
        if integer_winding_lattice_contains(winding_vectors(sorted(component), component_edges), AXES[axis]):
            return True
    return False


def exact_directional_percolation_threshold(
    nodes: Sequence[str],
    edges: Sequence[PeriodicEdge],
    axis: str,
) -> Optional[float]:
    thresholds = unique_thresholds(edges)
    lower = 0
    upper = len(thresholds)
    while lower < upper:
        middle = (lower + upper) // 2
        if exact_axis_percolates(nodes, active_edges(edges, thresholds[middle]), axis):
            upper = middle
        else:
            lower = middle + 1
    return thresholds[lower] if lower < len(thresholds) else None


def quotient_components(nodes: Sequence[str], edges: Sequence[PeriodicEdge]) -> List[Set[str]]:
    adjacency = quotient_adjacency(edges)
    remaining = set(nodes)
    components: List[Set[str]] = []
    while remaining:
        root = min(remaining)
        component = {root}
        queue = deque([root])
        remaining.remove(root)
        while queue:
            node = queue.popleft()
            for neighbour, _shift, _edge_uid in adjacency.get(node, []):
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    component.add(neighbour)
                    queue.append(neighbour)
        components.append(component)
    return components


def edge_is_quotient_bridge(edge: PeriodicEdge, edges: Sequence[PeriodicEdge]) -> bool:
    if edge.node_i == edge.node_j:
        return False
    remaining = [candidate for candidate in edges if candidate.edge_uid != edge.edge_uid]
    adjacency = quotient_adjacency(remaining)
    queue = deque([edge.node_i])
    visited = {edge.node_i}
    while queue:
        node = queue.popleft()
        if node == edge.node_j:
            return False
        for neighbour, _shift, _edge_uid in adjacency.get(node, []):
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
    return True


def critical_periodic_edges(
    nodes: Sequence[str],
    edges: Sequence[PeriodicEdge],
    thresholds_by_axis: Dict[str, Optional[float]],
    supercell_radius: int,
) -> List[Dict[str, Any]]:
    critical_axes: Dict[str, List[str]] = defaultdict(list)
    for axis, threshold in thresholds_by_axis.items():
        if threshold is None:
            continue
        at_threshold = active_edges(edges, threshold)
        for edge in at_threshold:
            remaining = [candidate for candidate in at_threshold if candidate.edge_uid != edge.edge_uid]
            if not exact_axis_percolates(nodes, remaining, axis):
                critical_axes[edge.edge_uid].append(axis)
    maximum_threshold = max(unique_thresholds(edges), default=0.0)
    fully_active = active_edges(edges, maximum_threshold)
    rank_critical: Set[str] = set()
    baseline_rank = winding_vector_rank(nodes, fully_active)
    if baseline_rank:
        for edge in fully_active:
            remaining = [candidate for candidate in fully_active if candidate.edge_uid != edge.edge_uid]
            if winding_vector_rank(nodes, remaining) < baseline_rank:
                rank_critical.add(edge.edge_uid)
    all_critical = sorted(set(critical_axes) | rank_critical)
    return [
        {
            "edge_uid": edge_uid,
            "directions": sorted(critical_axes.get(edge_uid, [])),
            "winding_rank_critical": edge_uid in rank_critical,
        }
        for edge_uid in all_critical
    ]


def periodic_redundancy_count(
    nodes: Sequence[str],
    edges: Sequence[PeriodicEdge],
) -> int:
    """Return sum(beta_1 - winding rank) over periodic quotient components."""
    redundant = 0
    for component in quotient_components(nodes, edges):
        component_edges = [edge for edge in edges if edge.node_i in component and edge.node_j in component]
        rank = rational_rank(winding_vectors(sorted(component), component_edges))
        if rank == 0:
            continue
        cycle_rank = len(component_edges) - len(component) + 1
        redundant += max(0, cycle_rank - rank)
    return redundant


def validate_periodic_graph(nodes: Sequence[str], edges: Sequence[PeriodicEdge]) -> None:
    if len(nodes) != len(set(nodes)):
        raise ValueError("Periodic graph node IDs must be unique")
    edge_uids = [edge.edge_uid for edge in edges]
    if len(edge_uids) != len(set(edge_uids)):
        raise ValueError("Periodic graph edge_uids must be unique")
    node_set = set(nodes)
    missing_endpoints = sorted(
        {endpoint for edge in edges for endpoint in (edge.node_i, edge.node_j) if endpoint not in node_set}
    )
    if missing_endpoints:
        raise ValueError(f"Periodic edge endpoints are absent from node table: {missing_endpoints}")


def threshold_values_equal(left: Optional[float], right: Optional[float], tolerance: float = 1e-12) -> bool:
    if left is None or right is None:
        return left is right
    return abs(left - right) <= tolerance


def supercell_graph_size(nodes: Sequence[str], edges: Sequence[PeriodicEdge], radius: int) -> Dict[str, int]:
    cells = [
        (a, b, c)
        for a in range(-radius, radius + 1)
        for b in range(-radius, radius + 1)
        for c in range(-radius, radius + 1)
    ]
    edge_instances = 0
    for cell in cells:
        for edge in edges:
            if in_supercell(add_vector(cell, edge.image_shift), radius):
                edge_instances += 1
    return {"cell_count": len(cells), "node_image_count": len(cells) * len(nodes), "edge_image_count": edge_instances}


def periodic_readout(
    nodes: Sequence[str],
    edges: Sequence[PeriodicEdge],
    validation_radii: Tuple[int, int] = (1, 2),
) -> Dict[str, Any]:
    """Compute periodic thresholds and topology, gated by barrier completeness."""
    if len(validation_radii) != 2 or validation_radii[0] != 1 or validation_radii[1] != 2:
        raise ValueError("validation_radii must be (1, 2), representing 3x3x3 and 5x5x5")
    validate_periodic_graph(nodes, edges)
    missing = [edge.edge_uid for edge in edges if edge.barrier_eV is None]
    coverage = 0.0 if not edges else (len(edges) - len(missing)) / float(len(edges))
    base: Dict[str, Any] = {
        "barrier_coverage": coverage,
        "missing_barrier_edge_uids": missing,
        "supercell_3x3x3": supercell_graph_size(nodes, edges, 1),
        "supercell_5x5x5": supercell_graph_size(nodes, edges, 2),
    }
    if missing:
        base.update(
            {
                "status": "PREDICTED_BARRIER_COVERAGE_INCOMPLETE",
                "Eperc_a": None,
                "Eperc_b": None,
                "Eperc_c": None,
                "Eperc_a_3x3x3": None,
                "Eperc_b_3x3x3": None,
                "Eperc_c_3x3x3": None,
                "Eperc_a_5x5x5": None,
                "Eperc_b_5x5x5": None,
                "Eperc_c_5x5x5": None,
                "Eperc_a_exact": None,
                "Eperc_b_exact": None,
                "Eperc_c_exact": None,
                "periodic_dimensionality": None,
                "winding_vector_rank": None,
                "global_winding_union_rank": None,
                "winding_vectors": [],
                "critical_periodic_edges": [],
                "redundancy_count": None,
                "supercell_converged": False,
            }
        )
        return base
    by_radius: Dict[int, Dict[str, Optional[float]]] = {
        radius: {
            axis: directional_percolation_threshold(nodes, edges, axis, radius)
            for axis in AXES
        }
        for radius in validation_radii
    }
    exact = {
        axis: exact_directional_percolation_threshold(nodes, edges, axis)
        for axis in AXES
    }
    convergence = {
        axis: (
            threshold_values_equal(by_radius[1][axis], by_radius[2][axis])
            and threshold_values_equal(by_radius[2][axis], exact[axis])
        )
        for axis in AXES
    }
    for axis in AXES:
        smaller = by_radius[1][axis]
        larger = by_radius[2][axis]
        if smaller is not None and larger is not None and larger > smaller + 1e-12:
            raise RuntimeError(f"5x5x5 threshold exceeds 3x3x3 threshold along {axis}; expansion invariant failed")
    maximum_threshold = max(unique_thresholds(edges), default=0.0)
    fully_active = active_edges(edges, maximum_threshold)
    vectors = winding_vectors(nodes, fully_active)
    rank = winding_vector_rank(nodes, fully_active)
    global_union_rank = rational_rank(vectors)
    critical = critical_periodic_edges(nodes, edges, exact, 2)
    redundancy = periodic_redundancy_count(nodes, fully_active)
    base.update(
        {
            "status": "PASSED" if all(convergence.values()) else "SUPERCELL_NOT_CONVERGED",
            "Eperc_a": exact["a"] if convergence["a"] else None,
            "Eperc_b": exact["b"] if convergence["b"] else None,
            "Eperc_c": exact["c"] if convergence["c"] else None,
            "Eperc_a_3x3x3": by_radius[1]["a"],
            "Eperc_b_3x3x3": by_radius[1]["b"],
            "Eperc_c_3x3x3": by_radius[1]["c"],
            "Eperc_a_5x5x5": by_radius[2]["a"],
            "Eperc_b_5x5x5": by_radius[2]["b"],
            "Eperc_c_5x5x5": by_radius[2]["c"],
            "Eperc_a_exact": exact["a"],
            "Eperc_b_exact": exact["b"],
            "Eperc_c_exact": exact["c"],
            "periodic_dimensionality": f"{rank}D",
            "winding_vector_rank": rank,
            "global_winding_union_rank": global_union_rank,
            "winding_vectors": vectors,
            "critical_periodic_edges": critical,
            "redundancy_count": redundancy,
            "supercell_converged": all(convergence.values()),
            "direction_convergence": convergence,
        }
    )
    return base


def run_toy_validation() -> Dict[str, Any]:
    """Exercise 0D, 1D, 2D, and 3D quotient graphs in both supercell sizes."""
    cases = {
        "0D": {
            "nodes": ["n0", "n1"],
            "edges": [PeriodicEdge("e0", "n0", "n1", (0, 0, 0), 0.1)],
            "expected_rank": 0,
            "expected_thresholds": {"a": None, "b": None, "c": None},
        },
        "1D": {
            "nodes": ["n0"],
            "edges": [PeriodicEdge("ea", "n0", "n0", (1, 0, 0), 0.2)],
            "expected_rank": 1,
            "expected_thresholds": {"a": 0.2, "b": None, "c": None},
        },
        "2D": {
            "nodes": ["n0"],
            "edges": [
                PeriodicEdge("ea", "n0", "n0", (1, 0, 0), 0.2),
                PeriodicEdge("eb", "n0", "n0", (0, 1, 0), 0.3),
            ],
            "expected_rank": 2,
            "expected_thresholds": {"a": 0.2, "b": 0.3, "c": None},
        },
        "3D": {
            "nodes": ["n0"],
            "edges": [
                PeriodicEdge("ea", "n0", "n0", (1, 0, 0), 0.2),
                PeriodicEdge("eb", "n0", "n0", (0, 1, 0), 0.3),
                PeriodicEdge("ec", "n0", "n0", (0, 0, 1), 0.4),
            ],
            "expected_rank": 3,
            "expected_thresholds": {"a": 0.2, "b": 0.3, "c": 0.4},
        },
    }
    results: Dict[str, Any] = {}
    for name, case in cases.items():
        result = periodic_readout(case["nodes"], case["edges"])
        expected = case["expected_thresholds"]
        passed = (
            result["status"] == "PASSED"
            and result["winding_vector_rank"] == case["expected_rank"]
            and result["Eperc_a"] == expected["a"]
            and result["Eperc_b"] == expected["b"]
            and result["Eperc_c"] == expected["c"]
            and result["supercell_converged"] is True
        )
        results[name] = {"passed": passed, "readout": result}
    return {"passed": all(item["passed"] for item in results.values()), "cases": results}
