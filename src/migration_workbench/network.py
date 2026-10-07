"""Run the original exact periodic readout on one graph at a time."""

from __future__ import annotations

import math

from .paths import ensure_science_path


def _number(value):
    if value is None or value == "":
        return None
    return float(value)


def periodic_analysis(nodes, edge_rows, with_removal=False):
    ensure_science_path()
    from revision_core.periodic_graph import PeriodicEdge, periodic_readout
    from revision_core.periodic_network_analysis import (
        compute_point_edge_removal_sensitivity,
        require_converged_periodic_readout,
    )

    node_ids = [str(node["id"]) for node in nodes]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("周期图节点 id 重复")
    known = set(node_ids)
    edges = []
    for row in edge_rows:
        if row["node_i"] not in known or row["node_j"] not in known:
            raise ValueError("边 %s 的节点未列在 nodes 中" % row["edge_uid"])
        shift = tuple(int(item) for item in row["image_shift"])
        barrier = _number(row.get("predicted_barrier_eV"))
        if barrier is None or not math.isfinite(barrier) or barrier < 0:
            raise ValueError("边 %s 缺少有效非负势垒，不能发布周期阈值" % row["edge_uid"])
        edges.append(PeriodicEdge(str(row["edge_uid"]), str(row["node_i"]), str(row["node_j"]), shift, barrier))
    result = periodic_readout(node_ids, edges, validation_radii=(1, 2))
    require_converged_periodic_readout(result, "图 %s" % (edge_rows[0].get("graph_id", "user") if edge_rows else "empty"))
    if with_removal:
        result["edge_removal"] = compute_point_edge_removal_sensitivity(node_ids, edges)
    result["interpretation"] = "模型或用户给定势垒上的静态周期可达性；并非实验电导率"
    return result


def rank_candidates(edges, readout, measured_excluded=True):
    """Expose transparent lexicographic channels without fitted scalar weights."""
    ordered = sorted(float(row["predicted_barrier_eV"]) for row in edges)
    if not ordered:
        return []
    critical = {str(row["edge_uid"]): row for row in readout.get("critical_periodic_edges", [])}
    removal = {str(row["edge_uid"]): row for row in readout.get("edge_removal", [])}
    spreads = sorted(float(row.get("ensemble_disagreement_eV") or 0.0) for row in edges)
    ranked = []
    for edge in edges:
        if measured_excluded and edge.get("is_labeled"):
            continue
        uid = str(edge["edge_uid"])
        barrier = float(edge["predicted_barrier_eV"])
        low = 1.0 - sum(value <= barrier + 1e-12 for value in ordered) / float(len(ordered))
        width = float(edge.get("ensemble_disagreement_eV") or 0.0)
        uncertainty_rank = sum(value <= width + 1e-12 for value in spreads) / float(len(spreads))
        impact = removal.get(uid, {})
        deltas = [float(impact.get("delta_Eperc_%s" % axis) or 0.0) for axis in "abc"]
        delta = max(deltas) if deltas else 0.0
        if any(impact.get("direction_lost_%s" % axis) for axis in "abc"):
            delta = max(delta, max(ordered) + 1.0)
        is_critical = 1 if uid in critical else 0
        ood = 1 if edge.get("ood_warning_95") else 0
        # The paper's channels prioritize criticality and applicability. D is
        # omitted here until a validated path-family novelty input is present.
        exploit = (is_critical, 1 - ood, low, delta, uncertainty_rank, uid)
        explore = (is_critical, ood, delta, uncertainty_rank, low, uid)
        ranked.append({
            "edge_uid": uid,
            "predicted_barrier_eV": barrier,
            "applicability_score": edge.get("applicability_score"),
            "ood_warning_95": ood,
            "critical": bool(is_critical),
            "critical_directions": critical.get(uid, {}).get("directions", []),
            "max_threshold_delta_eV": delta if uid in removal else None,
            "ensemble_disagreement_eV": width,
            "start_frac_coords": edge.get("start_frac_coords"),
            "end_frac_coords": edge.get("end_frac_coords"),
            "image_shift": edge["image_shift"],
            "exploit_order": exploit,
            "explore_order": explore,
            "ranking_note": "字典序预筛；未计算论文的路径家族多样性分量，不能与终版候选表等同",
        })
    for channel in ("exploit", "explore"):
        sorted_rows = sorted(ranked, key=lambda row: row[channel + "_order"], reverse=True)
        for rank, row in enumerate(sorted_rows, 1):
            row[channel + "_rank"] = rank
    for row in ranked:
        del row["exploit_order"]
        del row["explore_order"]
    return sorted(ranked, key=lambda row: row["exploit_rank"])
