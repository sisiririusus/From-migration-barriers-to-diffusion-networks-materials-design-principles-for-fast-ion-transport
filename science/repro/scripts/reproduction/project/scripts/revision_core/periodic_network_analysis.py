"""Reusable post-prediction helpers for periodic-network candidate analysis.

The functions in this module operate only on explicit periodic image shifts
and traceable barrier columns.  They do not infer source/sink nodes, clip
prediction intervals, or combine acquisition objectives into a weighted
score.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from revision_core.periodic_graph import (
    AXES,
    PeriodicEdge,
    periodic_readout,
    threshold_values_equal,
)


REQUIRED_EDGE_COLUMNS: Tuple[str, ...] = (
    "edge_uid",
    "node_i",
    "node_j",
    "image_shift_a",
    "image_shift_b",
    "image_shift_c",
)

REQUIRED_INTERVAL_SCENARIOS: Tuple[str, ...] = (
    "point",
    "lower80",
    "upper80",
    "lower90",
    "upper90",
)


class PeriodicNetworkAnalysisError(ValueError):
    """Raised when a network-analysis input fails a scientific guard."""


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _required_text(value: Any, field: str, row_label: Any) -> str:
    if _is_missing(value):
        raise PeriodicNetworkAnalysisError(
            "Missing %s at edge-frame row %s" % (field, row_label)
        )
    text = str(value).strip()
    if not text:
        raise PeriodicNetworkAnalysisError(
            "Blank %s at edge-frame row %s" % (field, row_label)
        )
    return text


def _integer_shift(value: Any, field: str, edge_uid: str) -> int:
    if _is_missing(value) or isinstance(value, (bool, np.bool_)):
        raise PeriodicNetworkAnalysisError(
            "Invalid %s for edge %s: %r" % (field, edge_uid, value)
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise PeriodicNetworkAnalysisError(
            "Invalid %s for edge %s: %r" % (field, edge_uid, value)
        )
    if not math.isfinite(numeric) or numeric != math.floor(numeric):
        raise PeriodicNetworkAnalysisError(
            "%s must be an exact integer for edge %s: %r" % (field, edge_uid, value)
        )
    return int(numeric)


def _barrier_value(value: Any, barrier_column: str, edge_uid: str, allow_missing: bool) -> Optional[float]:
    if _is_missing(value):
        if allow_missing:
            return None
        raise PeriodicNetworkAnalysisError(
            "Missing barrier column %s for edge %s" % (barrier_column, edge_uid)
        )
    try:
        barrier = float(value)
    except (TypeError, ValueError):
        raise PeriodicNetworkAnalysisError(
            "Non-numeric barrier column %s for edge %s: %r"
            % (barrier_column, edge_uid, value)
        )
    if not math.isfinite(barrier) or barrier < 0:
        raise PeriodicNetworkAnalysisError(
            "Barrier column %s must be finite and nonnegative for edge %s; "
            "interval values are never clipped: %r"
            % (barrier_column, edge_uid, value)
        )
    return barrier


def build_periodic_edges(
    edge_frame: pd.DataFrame,
    barrier_column: str,
    allow_missing: bool = False,
) -> List[PeriodicEdge]:
    """Build :class:`PeriodicEdge` objects without changing barrier values.

    Row order is preserved.  Missing barriers are rejected by default; callers
    that only need to audit incomplete coverage may opt into ``None`` values.
    Negative interval bounds fail rather than being clipped to zero.
    """
    missing_columns = [
        column for column in REQUIRED_EDGE_COLUMNS + (barrier_column,)
        if column not in edge_frame.columns
    ]
    if missing_columns:
        raise PeriodicNetworkAnalysisError(
            "Edge frame is missing required columns: %s" % sorted(set(missing_columns))
        )

    edges: List[PeriodicEdge] = []
    seen_uids = set()
    for row_label, row in edge_frame.iterrows():
        edge_uid = _required_text(row["edge_uid"], "edge_uid", row_label)
        if edge_uid in seen_uids:
            raise PeriodicNetworkAnalysisError("Duplicate edge_uid: %s" % edge_uid)
        seen_uids.add(edge_uid)
        node_i = _required_text(row["node_i"], "node_i", row_label)
        node_j = _required_text(row["node_j"], "node_j", row_label)
        shift = (
            _integer_shift(row["image_shift_a"], "image_shift_a", edge_uid),
            _integer_shift(row["image_shift_b"], "image_shift_b", edge_uid),
            _integer_shift(row["image_shift_c"], "image_shift_c", edge_uid),
        )
        barrier = _barrier_value(row[barrier_column], barrier_column, edge_uid, allow_missing)
        try:
            edges.append(PeriodicEdge(edge_uid, node_i, node_j, shift, barrier))
        except ValueError as error:
            raise PeriodicNetworkAnalysisError(
                "Invalid periodic edge %s: %s" % (edge_uid, error)
            )
    return edges


def _validate_single_graph(edge_frame: pd.DataFrame) -> None:
    if "graph_id" not in edge_frame.columns:
        return
    graph_ids = {
        _required_text(value, "graph_id", index)
        for index, value in edge_frame["graph_id"].items()
    }
    if len(graph_ids) != 1:
        raise PeriodicNetworkAnalysisError(
            "Periodic readout requires exactly one graph_id, found %s" % sorted(graph_ids)
        )


def require_converged_periodic_readout(readout: Mapping[str, Any], context: str) -> None:
    """Fail unless 3x3x3, 5x5x5, and exact thresholds agree on every axis."""
    convergence = readout.get("direction_convergence")
    converged = (
        readout.get("status") == "PASSED"
        and readout.get("supercell_converged") is True
        and isinstance(convergence, Mapping)
        and all(convergence.get(axis) is True for axis in AXES)
    )
    if not converged:
        raise PeriodicNetworkAnalysisError(
            "%s did not pass mandatory 3x3x3/5x5x5/exact convergence: status=%r"
            % (context, readout.get("status"))
        )
    for axis in AXES:
        released = readout.get("Eperc_%s" % axis)
        exact = readout.get("Eperc_%s_exact" % axis)
        three = readout.get("Eperc_%s_3x3x3" % axis)
        five = readout.get("Eperc_%s_5x5x5" % axis)
        if not (
            threshold_values_equal(released, exact)
            and threshold_values_equal(three, exact)
            and threshold_values_equal(five, exact)
        ):
            raise PeriodicNetworkAnalysisError(
                "%s has inconsistent released thresholds along %s" % (context, axis)
            )


def _validate_interval_order(
    edge_frame: pd.DataFrame,
    barrier_columns: Mapping[str, str],
) -> None:
    ordered_scenarios = ("lower90", "lower80", "point", "upper80", "upper90")
    missing_columns = [
        barrier_columns[scenario]
        for scenario in ordered_scenarios
        if barrier_columns[scenario] not in edge_frame.columns
    ]
    if missing_columns:
        raise PeriodicNetworkAnalysisError(
            "Edge frame is missing interval columns: %s" % sorted(set(missing_columns))
        )
    for row_label, row in edge_frame.iterrows():
        edge_uid = _required_text(row["edge_uid"], "edge_uid", row_label)
        values = [
            _barrier_value(row[barrier_columns[scenario]], barrier_columns[scenario], edge_uid, False)
            for scenario in ordered_scenarios
        ]
        numeric = [float(value) for value in values if value is not None]
        if len(numeric) != len(ordered_scenarios) or any(
            numeric[index] > numeric[index + 1]
            for index in range(len(numeric) - 1)
        ):
            raise PeriodicNetworkAnalysisError(
                "Prediction intervals are not nested for edge %s: %s"
                % (edge_uid, dict(zip(ordered_scenarios, numeric)))
            )


def run_periodic_interval_readouts(
    nodes: Sequence[str],
    edge_frame: pd.DataFrame,
    barrier_columns: Mapping[str, str],
) -> Dict[str, Dict[str, Any]]:
    """Run converged point and 80/90 percent interval network scenarios.

    ``barrier_columns`` must map exactly ``point``, ``lower80``, ``upper80``,
    ``lower90``, and ``upper90`` to columns in ``edge_frame``.  Values are
    passed through unchanged; negative lower bounds fail closed.
    """
    keys = set(barrier_columns)
    required = set(REQUIRED_INTERVAL_SCENARIOS)
    if keys != required:
        raise PeriodicNetworkAnalysisError(
            "barrier_columns must contain exactly %s; found %s"
            % (sorted(required), sorted(keys))
        )
    if len(set(barrier_columns.values())) != len(barrier_columns):
        raise PeriodicNetworkAnalysisError("Each interval scenario must use a distinct barrier column")
    _validate_single_graph(edge_frame)
    _validate_interval_order(edge_frame, barrier_columns)

    results: Dict[str, Dict[str, Any]] = {}
    for scenario in REQUIRED_INTERVAL_SCENARIOS:
        column = barrier_columns[scenario]
        edges = build_periodic_edges(edge_frame, column, allow_missing=False)
        readout = periodic_readout(nodes, edges, validation_radii=(1, 2))
        require_converged_periodic_readout(readout, "scenario %s (%s)" % (scenario, column))
        result = dict(readout)
        result["scenario_id"] = scenario
        result["barrier_column"] = column
        results[scenario] = result
    return results


def _threshold_removal_fields(
    baseline: Optional[float],
    removed: Optional[float],
    axis: str,
) -> Dict[str, Any]:
    if baseline is None:
        if removed is not None:
            raise PeriodicNetworkAnalysisError(
                "Removing an edge cannot create %s-direction percolation" % axis
            )
        delta: Optional[float] = None
        lost = False
    elif removed is None:
        delta = None
        lost = True
    else:
        if removed < baseline - 1e-12:
            raise PeriodicNetworkAnalysisError(
                "Removing an edge lowered Eperc_%s from %.12g to %.12g"
                % (axis, baseline, removed)
            )
        delta = max(0.0, float(removed - baseline))
        lost = False
    return {
        "baseline_Eperc_%s" % axis: baseline,
        "removed_Eperc_%s" % axis: removed,
        "delta_Eperc_%s" % axis: delta,
        "direction_lost_%s" % axis: lost,
    }


def compute_point_edge_removal_sensitivity(
    nodes: Sequence[str],
    point_edges: Sequence[PeriodicEdge],
) -> List[Dict[str, Any]]:
    """Compute exact point-estimate removal effects for all critical edges.

    ``periodic_graph.critical_periodic_edges`` proves that an edge absent from
    its result cannot raise a directional point threshold and cannot lower the
    fully-active winding rank.  Such edges are recorded as
    ``NONCRITICAL_ZERO_SCREENED`` without an unnecessary readout.  Every
    direction/rank-critical edge is removed and recomputed with the full
    3x3x3, 5x5x5, and exact convergence gate.
    """
    edges = list(point_edges)
    baseline = periodic_readout(nodes, edges, validation_radii=(1, 2))
    require_converged_periodic_readout(baseline, "point baseline")
    critical_by_uid = {
        str(item["edge_uid"]): item
        for item in baseline.get("critical_periodic_edges", [])
    }
    rows: List[Dict[str, Any]] = []
    for edge in sorted(edges, key=lambda item: item.edge_uid):
        critical = critical_by_uid.get(edge.edge_uid)
        recomputed = critical is not None
        if recomputed:
            remaining = [candidate for candidate in edges if candidate.edge_uid != edge.edge_uid]
            removed_readout = periodic_readout(nodes, remaining, validation_radii=(1, 2))
            require_converged_periodic_readout(
                removed_readout, "point removal edge %s" % edge.edge_uid
            )
            status = "CRITICAL_EXACTLY_RECOMPUTED"
        else:
            removed_readout = baseline
            status = "NONCRITICAL_ZERO_SCREENED"

        row: Dict[str, Any] = {
            "edge_uid": edge.edge_uid,
            "screen_status": status,
            "exactly_recomputed": recomputed,
            "point_critical_directions": sorted(critical.get("directions", [])) if critical else [],
            "point_winding_rank_critical": bool(critical.get("winding_rank_critical")) if critical else False,
            "baseline_winding_vector_rank": baseline["winding_vector_rank"],
            "removed_winding_vector_rank": removed_readout["winding_vector_rank"],
            "winding_rank_loss": int(
                baseline["winding_vector_rank"] - removed_readout["winding_vector_rank"]
            ),
            "baseline_periodic_dimensionality": baseline["periodic_dimensionality"],
            "removed_periodic_dimensionality": removed_readout["periodic_dimensionality"],
            "removed_readout_status": removed_readout["status"],
            "removed_supercell_converged": removed_readout["supercell_converged"],
        }
        if row["winding_rank_loss"] < 0:
            raise PeriodicNetworkAnalysisError(
                "Removing edge %s increased winding rank" % edge.edge_uid
            )
        for axis in AXES:
            row.update(
                _threshold_removal_fields(
                    baseline.get("Eperc_%s" % axis),
                    removed_readout.get("Eperc_%s" % axis),
                    axis,
                )
            )
        rows.append(row)
    return rows


def _numeric_objectives(
    values: Sequence[Sequence[float]],
    directions: Sequence[str],
    item_ids: Optional[Sequence[Any]],
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    if not directions:
        raise PeriodicNetworkAnalysisError("Pareto objectives must not be empty")
    if len(values) == 0:
        matrix = np.empty((0, len(directions)), dtype=float)
    else:
        matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise PeriodicNetworkAnalysisError("Pareto values must be a two-dimensional matrix")
    if matrix.shape[1] != len(directions) or matrix.shape[1] == 0:
        raise PeriodicNetworkAnalysisError(
            "Pareto direction count must equal the nonzero objective count"
        )
    normalized_directions = tuple(str(direction).lower() for direction in directions)
    if any(direction not in ("min", "max") for direction in normalized_directions):
        raise PeriodicNetworkAnalysisError("Pareto directions must be 'min' or 'max'")
    if not np.isfinite(matrix).all():
        raise PeriodicNetworkAnalysisError("Pareto objectives contain NaN or infinity")
    if item_ids is None:
        ids = tuple("%012d" % index for index in range(matrix.shape[0]))
    else:
        if len(item_ids) != matrix.shape[0]:
            raise PeriodicNetworkAnalysisError("item_ids length does not match Pareto rows")
        if any(_is_missing(item_id) for item_id in item_ids):
            raise PeriodicNetworkAnalysisError("Pareto item_ids must be nonmissing")
        ids = tuple(str(item_id) for item_id in item_ids)
        if any(not item_id for item_id in ids) or len(ids) != len(set(ids)):
            raise PeriodicNetworkAnalysisError("Pareto item_ids must be nonblank and unique")
    signs = np.asarray([1.0 if direction == "min" else -1.0 for direction in normalized_directions])
    return matrix * signs, ids


def _dominates(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.all(left <= right) and np.any(left < right))


def pareto_nondominated_indices(
    values: Sequence[Sequence[float]],
    directions: Sequence[str],
    item_ids: Optional[Sequence[Any]] = None,
) -> List[int]:
    """Return deterministic lexical-order indices of the nondominated set."""
    matrix, ids = _numeric_objectives(values, directions, item_ids)
    nondominated = []
    for candidate in range(matrix.shape[0]):
        if not any(
            other != candidate and _dominates(matrix[other], matrix[candidate])
            for other in range(matrix.shape[0])
        ):
            nondominated.append(candidate)
    return sorted(nondominated, key=lambda index: ids[index])


def pareto_fronts(
    values: Sequence[Sequence[float]],
    directions: Sequence[str],
    item_ids: Optional[Sequence[Any]] = None,
) -> List[List[int]]:
    """Return successive nondominated fronts with lexical order within fronts."""
    matrix, ids = _numeric_objectives(values, directions, item_ids)
    remaining = set(range(matrix.shape[0]))
    fronts: List[List[int]] = []
    while remaining:
        front = [
            candidate
            for candidate in remaining
            if not any(
                other != candidate and _dominates(matrix[other], matrix[candidate])
                for other in remaining
            )
        ]
        if not front:
            raise RuntimeError("Pareto sorting failed to identify a front")
        ordered = sorted(front, key=lambda index: ids[index])
        fronts.append(ordered)
        remaining.difference_update(front)
    return fronts


def _validate_diversity_inputs(
    features: Sequence[Sequence[float]],
    item_ids: Sequence[Any],
    n_select: Optional[int],
    strata: Optional[Sequence[Any]],
) -> Tuple[np.ndarray, Tuple[str, ...], int, Optional[Tuple[str, ...]]]:
    if len(features) == 0:
        matrix = np.empty((0, 0), dtype=float)
    else:
        matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise PeriodicNetworkAnalysisError("Diversity features must be a two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise PeriodicNetworkAnalysisError("Diversity features contain NaN or infinity")
    if len(item_ids) != matrix.shape[0]:
        raise PeriodicNetworkAnalysisError("item_ids length does not match diversity rows")
    if any(_is_missing(item_id) for item_id in item_ids):
        raise PeriodicNetworkAnalysisError("Diversity item_ids must be nonmissing")
    ids = tuple(str(item_id) for item_id in item_ids)
    if any(not item_id for item_id in ids) or len(ids) != len(set(ids)):
        raise PeriodicNetworkAnalysisError("Diversity item_ids must be nonblank and unique")
    if n_select is None:
        count = matrix.shape[0]
    else:
        if isinstance(n_select, (bool, np.bool_)):
            raise PeriodicNetworkAnalysisError("n_select must be an integer")
        try:
            numeric_count = float(n_select)
        except (TypeError, ValueError):
            raise PeriodicNetworkAnalysisError("n_select must be an integer")
        if not math.isfinite(numeric_count) or numeric_count != math.floor(numeric_count):
            raise PeriodicNetworkAnalysisError("n_select must be an integer")
        count = int(numeric_count)
    if count < 0 or count > matrix.shape[0]:
        raise PeriodicNetworkAnalysisError("n_select must be between zero and the row count")
    resolved_strata: Optional[Tuple[str, ...]] = None
    if strata is not None:
        if len(strata) != matrix.shape[0]:
            raise PeriodicNetworkAnalysisError("strata length does not match diversity rows")
        converted = []
        for value in strata:
            if _is_missing(value) or not str(value).strip():
                raise PeriodicNetworkAnalysisError("strata values must be nonmissing and nonblank")
            converted.append(str(value))
        resolved_strata = tuple(converted)
    return matrix, ids, count, resolved_strata


def farthest_first_indices(
    features: Sequence[Sequence[float]],
    item_ids: Sequence[Any],
    n_select: Optional[int] = None,
    strata: Optional[Sequence[Any]] = None,
) -> List[int]:
    """Return deterministic farthest-first indices in caller-scaled space.

    Without strata, the lexically smallest ID seeds the sequence.  With
    strata, strata are visited in lexical round-robin order; each visit picks
    the member farthest from the global selected set, with lexical ID as the
    exact-distance tie breaker.  The caller remains responsible for scaling
    heterogeneous feature dimensions before invoking this function.
    """
    matrix, ids, count, resolved_strata = _validate_diversity_inputs(
        features, item_ids, n_select, strata
    )
    if count == 0:
        return []

    selected: List[int] = []
    remaining = set(range(matrix.shape[0]))

    def choose(candidates: Sequence[int]) -> int:
        if not selected:
            return min(candidates, key=lambda index: ids[index])
        selected_matrix = matrix[np.asarray(selected, dtype=int)]
        ranked = []
        for index in candidates:
            distances = np.sum((selected_matrix - matrix[index]) ** 2, axis=1)
            minimum_distance = float(np.min(distances))
            ranked.append((-minimum_distance, ids[index], index))
        return min(ranked)[2]

    if resolved_strata is None:
        while remaining and len(selected) < count:
            index = choose(sorted(remaining, key=lambda item: ids[item]))
            selected.append(index)
            remaining.remove(index)
        return selected

    stratum_order = sorted(set(resolved_strata))
    while remaining and len(selected) < count:
        progress = False
        for stratum in stratum_order:
            candidates = [
                index for index in remaining if resolved_strata[index] == stratum
            ]
            if not candidates:
                continue
            index = choose(candidates)
            selected.append(index)
            remaining.remove(index)
            progress = True
            if len(selected) == count:
                break
        if not progress:
            raise RuntimeError("Stratified farthest-first selection made no progress")
    return selected
