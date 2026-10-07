"""Paper examples from frozen source tables, with optional exact recomputation."""

from __future__ import annotations

import csv
import json
import math

from .paths import TABLES, EXTENDED
from .network import periodic_analysis


BASE_METRICS = TABLES / "revision_periodic_network_metrics.csv"
BASE_NODES = TABLES / "revision_source_data" / "periodic_trial_nodes.csv"
BASE_EDGES = TABLES / "revision_source_data" / "revision_periodic_oof_edge_predictions.csv"
COMPARISON_METRICS = EXTENDED / "tables" / "tableS7_cross_composition_periodic_readouts.csv"
COMPARISON_SOURCE = EXTENDED / "source_snapshot" / "current_tables" / "revision_source_data"
COMPARISON_NODES = COMPARISON_SOURCE / "figure67_nasicon_nodes.csv"
COMPARISON_EDGES = COMPARISON_SOURCE / "figure67_nasicon_oof_candidate_predictions.csv"


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _float_or_none(value):
    if value is None or value == "":
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def catalog():
    cases = []
    for family, path in (("base_periodic", BASE_METRICS), ("composition_comparison", COMPARISON_METRICS)):
        for row in rows(path):
            group = row["formula_species_group"]
            cases.append({
                "id": group.replace("|", "_"),
                "formula_species_group": group,
                "graph_id": row["graph_id"],
                "family": family,
                "n_nodes": int(row["n_nodes"]),
                "n_edges": int(row["n_edges"]),
                "winding_rank": int(row["winding_vector_rank"]),
                "Eperc_a": _float_or_none(row.get("Eperc_a")),
                "Eperc_b": _float_or_none(row.get("Eperc_b")),
                "Eperc_c": _float_or_none(row.get("Eperc_c")),
            })
    return cases


def load_case(case_id, recompute=False):
    matches = [case for case in catalog() if case["id"] == case_id]
    if len(matches) != 1:
        raise ValueError("未知论文示例：%s" % case_id)
    case = matches[0]
    family = case["family"]
    metrics = rows(BASE_METRICS if family == "base_periodic" else COMPARISON_METRICS)
    metrics = next(row for row in metrics if row["graph_id"] == case["graph_id"])
    nodes_source = BASE_NODES if family == "base_periodic" else COMPARISON_NODES
    edges_source = BASE_EDGES if family == "base_periodic" else COMPARISON_EDGES
    nodes = []
    for row in rows(nodes_source):
        if row["graph_id"] == case["graph_id"]:
            nodes.append({"id": row["node_id"], "frac_coords": json.loads(row["frac_coords"])})
    edges = []
    for row in rows(edges_source):
        if row["graph_id"] != case["graph_id"]:
            continue
        edges.append({
            "graph_id": row["graph_id"],
            "edge_uid": row["edge_uid"],
            "node_i": row["node_i"],
            "node_j": row["node_j"],
            "image_shift": [int(row["image_shift_%s" % axis]) for axis in "abc"],
            "start_frac_coords": json.loads(row["start_frac_coords"]),
            "end_frac_coords": json.loads(row["end_frac_coords"]),
            "predicted_barrier_eV": float(row["oof_predicted_barrier_eV"]),
            "ensemble_disagreement_eV": _float_or_none(row.get("ensemble_disagreement_eV")),
            "applicability_score": _float_or_none(row.get("applicability_score")),
            "ood_warning_95": int(row.get("ood_warning_95") or 0),
            "is_labeled": int(row.get("is_labeled") or 0) == 1,
            "prediction_role": "group_purged_OOF",
        })
    if len(nodes) != case["n_nodes"] or len(edges) != case["n_edges"]:
        raise RuntimeError("论文示例节点或边数与来源指标表不符")
    frozen = {
        "Eperc_a": case["Eperc_a"], "Eperc_b": case["Eperc_b"], "Eperc_c": case["Eperc_c"],
        "winding_vector_rank": case["winding_rank"],
        "critical_periodic_edge_count": int(metrics.get("critical_periodic_edge_count") or 0),
        "graph_applicability_score": _float_or_none(metrics.get("graph_applicability_score")),
        "status": "frozen_paper_source",
    }
    current = None
    if recompute:
        current = periodic_analysis(nodes, edges)
        for axis in "abc":
            observed = current["Eperc_%s" % axis]
            expected = frozen["Eperc_%s" % axis]
            if (observed is None) != (expected is None) or (
                observed is not None and abs(observed - expected) > 1e-8
            ):
                raise RuntimeError("重算 %s 阈值与论文表不符" % axis)
        if current["winding_vector_rank"] != frozen["winding_vector_rank"]:
            raise RuntimeError("重算周期维数与论文表不符")

    candidates = []
    if family == "base_periodic":
        source = TABLES / "revision_active_neb_candidates.csv"
        candidates = [row for row in rows(source) if row["graph_id"] == case["graph_id"]]
    elif case["formula_species_group"] == "Na10Mn2Al2P9O24|Na":
        source = COMPARISON_SOURCE / "figure7_prospective_case_source.csv"
        candidates = [
            row for row in rows(source)
            if row["graph_id"] == case["graph_id"] and row["prospective_candidate"] == "1"
        ]

    return {
        "kind": "paper_example",
        "case": case,
        "name": case["formula_species_group"],
        "graph_id": case["graph_id"],
        "formula": case["formula_species_group"].split("|")[0],
        "migrating_species": case["formula_species_group"].split("|")[1],
        "nodes": nodes,
        "edges": edges,
        "readout": current if current is not None else frozen,
        "frozen_readout": frozen,
        "computed_now": bool(recompute),
        "recommendations": candidates,
        "source_files": [str(nodes_source), str(edges_source), str(BASE_METRICS if family == "base_periodic" else COMPARISON_METRICS)],
        "interpretation": "每个结构/离子独立成图；OOF 预测上的静态周期筛选，并非电导率",
    }
