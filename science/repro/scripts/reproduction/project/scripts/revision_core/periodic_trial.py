"""Build and audit a small real-structure periodic migration-graph trial.

Candidate nodes and edges are generated from raw structures. Reported NEB paths
are retained automatically, including paths beyond the empirical candidate
distance cutoff. No historical graph, hand-authored path, or paper figure is
read by this module.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from revision_core.data_governance import (
    ROOT,
    canonical_json,
    identify_migrating_species,
    periodic_cartesian_displacements,
    project_path,
    read_config,
    reduced_formula,
    require_authorised_raw_dataset,
    sha256_file,
    sha256_json,
    source_group_from_bibtex,
    valid_structure,
    write_csv_artifacts,
    write_json,
)
from revision_core.periodic_graph import PeriodicEdge, periodic_readout, run_toy_validation


Frac3 = Tuple[float, float, float]
Int3 = Tuple[int, int, int]
Matrix3 = List[List[float]]

EDGE_REQUIRED_COLUMNS = [
    "edge_uid",
    "graph_id",
    "node_i",
    "node_j",
    "image_shift_a",
    "image_shift_b",
    "image_shift_c",
    "start_frac_coords",
    "end_frac_coords",
    "displacement_frac",
    "displacement_cart",
    "lattice_matrix",
    "hop_distance_A",
    "edge_source",
    "is_labeled",
    "source_record_id",
]
EDGE_EXTRA_COLUMNS = [
    "forced_reported_edge",
    "reported_barrier_eV",
    "predicted_barrier_eV",
    "candidate_distance_threshold_A",
    "candidate_threshold_source",
]


def normalize_frac(values: Sequence[float]) -> Frac3:
    return tuple(float(value) % 1.0 for value in values)  # type: ignore[return-value]


def frac_to_cart(vector: Sequence[float], lattice: Sequence[Sequence[float]]) -> Frac3:
    return tuple(
        sum(float(vector[row]) * float(lattice[row][column]) for row in range(3))
        for column in range(3)
    )  # type: ignore[return-value]


def vector_norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in vector))


@lru_cache(maxsize=2048)
def _cached_minimum_singular_value(flat_lattice: Tuple[float, ...]) -> float:
    import numpy as np

    value = float(np.linalg.svd(np.asarray(flat_lattice, dtype=float).reshape(3, 3), compute_uv=False).min())
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Lattice matrix is singular or invalid")
    return value


def lattice_minimum_singular_value(lattice: Matrix3) -> float:
    return _cached_minimum_singular_value(tuple(float(value) for row in lattice for value in row))


def raw_cell_offset(values: Sequence[float]) -> Int3:
    return tuple(int(math.floor(float(value))) for value in values)  # type: ignore[return-value]


def minimum_image_solution(
    start_raw: Sequence[float],
    end_raw: Sequence[float],
    lattice: Matrix3,
    tie_tolerance_A: float,
) -> Dict[str, Any]:
    """Find a metric minimum image, preserving the raw image hint when degenerate."""
    start = normalize_frac(start_raw)
    end = normalize_frac(end_raw)
    start_offset = raw_cell_offset(start_raw)
    end_offset = raw_cell_offset(end_raw)
    raw_hint: Int3 = tuple(end_offset[axis] - start_offset[axis] for axis in range(3))  # type: ignore[assignment]
    base_delta = tuple(end[axis] - start[axis] for axis in range(3))
    hinted_delta = tuple(base_delta[axis] + raw_hint[axis] for axis in range(3))
    best_known = vector_norm(frac_to_cart(hinted_delta, lattice))
    sigma_min = lattice_minimum_singular_value(lattice)
    radius = max(1, int(math.ceil(best_known / sigma_min + vector_norm(base_delta))) + 1)
    candidates = []
    best_distance = math.inf
    for shift_values in product(range(-radius, radius + 1), repeat=3):
        shift: Int3 = (int(shift_values[0]), int(shift_values[1]), int(shift_values[2]))
        displacement = tuple(base_delta[axis] + shift[axis] for axis in range(3))
        cartesian = frac_to_cart(displacement, lattice)
        distance = vector_norm(cartesian)
        if distance < best_distance - tie_tolerance_A:
            best_distance = distance
            candidates = [(shift, displacement, cartesian)]
        elif abs(distance - best_distance) <= tie_tolerance_A:
            candidates.append((shift, displacement, cartesian))
    candidates.sort(key=lambda item: item[0])
    chosen = next((item for item in candidates if item[0] == raw_hint), candidates[0])
    return {
        "start_frac": start,
        "end_frac": end,
        "raw_image_hint": raw_hint,
        "image_shift": chosen[0],
        "displacement_frac": chosen[1],
        "displacement_cart": chosen[2],
        "hop_distance_A": best_distance,
        "image_shift_degenerate": len(candidates) > 1,
        "minimum_image_count": len(candidates),
        "minimum_image_search_radius": radius,
    }


def periodic_distance(
    left: Sequence[float],
    right: Sequence[float],
    lattice: Matrix3,
    tie_tolerance_A: float = 1e-8,
) -> float:
    return float(minimum_image_solution(left, right, lattice, tie_tolerance_A)["hop_distance_A"])


def derived_candidate_shift_radius(lattice: Matrix3, cutoff_A: float) -> int:
    sigma_min = lattice_minimum_singular_value(lattice)
    return max(1, int(math.ceil(cutoff_A / sigma_min + math.sqrt(3.0))))


def lattice_relative_change(initial: Sequence[Sequence[float]], final: Sequence[Sequence[float]]) -> float:
    numerator = math.sqrt(
        sum((float(initial[row][column]) - float(final[row][column])) ** 2 for row in range(3) for column in range(3))
    )
    denominator = math.sqrt(sum(float(initial[row][column]) ** 2 for row in range(3) for column in range(3)))
    return numerator / denominator if denominator else math.inf


def interpolated_quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot calculate a candidate cutoff from an empty hop-distance set")
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def record_id(raw_index: int, jid: Any) -> str:
    return f"em_complete_{raw_index:04d}_jid_{jid}"


def reported_transition(
    raw_index: int,
    record: Dict[str, Any],
    minimum_displacement: float,
    tie_tolerance_A: float,
    fixed_cell_tolerance: float,
    source_fallback: str,
) -> Dict[str, Any]:
    initial = record.get("structure_ini")
    final = record.get("structure_fin")
    if not valid_structure(initial) or not valid_structure(final):
        raise ValueError(f"Invalid structure in raw record {raw_index}")
    lattice = [[float(value) for value in row] for row in initial["lattice_mat"]]
    solutions = [
        minimum_image_solution(start, end, lattice, tie_tolerance_A)
        for start, end in zip(initial["coords"], final["coords"])
    ]
    moved_indices = [
        index for index, solution in enumerate(solutions)
        if solution["hop_distance_A"] > minimum_displacement
    ]
    if not moved_indices:
        raise ValueError(f"No moved atom in raw record {raw_index}")
    atom_index = max(moved_indices, key=lambda index: solutions[index]["hop_distance_A"])
    solution = solutions[atom_index]
    species = str(initial["elements"][atom_index])
    relative_lattice_change = lattice_relative_change(initial["lattice_mat"], final["lattice_mat"])
    source_group, source_resolved = source_group_from_bibtex(str(record.get("bibtex", "")), source_fallback)
    return {
        "raw_index": raw_index,
        "record_id": record_id(raw_index, record.get("jid")),
        "jid": str(record.get("jid")),
        "formula": str(record.get("formula")),
        "sys_name": str(record.get("sys_name")),
        "migrating_species": species,
        "atom_index": atom_index,
        "moved_atom_count": len(moved_indices),
        "start_frac": solution["start_frac"],
        "end_frac": solution["end_frac"],
        "raw_image_hint": solution["raw_image_hint"],
        "image_shift": solution["image_shift"],
        "displacement_frac": solution["displacement_frac"],
        "displacement_cart": solution["displacement_cart"],
        "hop_distance_A": solution["hop_distance_A"],
        "image_shift_degenerate": solution["image_shift_degenerate"],
        "minimum_image_count": solution["minimum_image_count"],
        "minimum_image_search_radius": solution["minimum_image_search_radius"],
        "reported_barrier_eV": float(record["target"]),
        "structure_ini_hash": sha256_json(initial),
        "lattice": lattice,
        "lattice_relative_change": relative_lattice_change,
        "fixed_cell": relative_lattice_change <= fixed_cell_tolerance,
        "graph_safe": relative_lattice_change <= fixed_cell_tolerance and len(moved_indices) == 1,
        "source_group": source_group,
        "source_group_resolved": source_resolved,
        "XC": str(record.get("XC", "")),
        "space_group": str(record.get("space_group", "")).strip(),
    }


def empirical_cutoffs(
    transitions: Sequence[Dict[str, Any]],
    probability: float,
    minimum_species_records: int,
) -> Tuple[Dict[str, float], Dict[str, str], float]:
    all_distances = [transition["hop_distance_A"] for transition in transitions]
    global_cutoff = interpolated_quantile(all_distances, probability)
    by_species: Dict[str, List[float]] = defaultdict(list)
    for transition in transitions:
        by_species[transition["migrating_species"]].append(transition["hop_distance_A"])
    cutoffs: Dict[str, float] = {}
    sources: Dict[str, str] = {}
    for species, values in sorted(by_species.items()):
        if len(values) >= minimum_species_records:
            cutoffs[species] = interpolated_quantile(values, probability)
            sources[species] = f"species_reported_hop_q{probability:.3f}_n{len(values)}"
        else:
            cutoffs[species] = global_cutoff
            sources[species] = f"global_reported_hop_q{probability:.3f}_species_n{len(values)}"
    return cutoffs, sources, global_cutoff


def graph_id_for(host_fingerprint: str) -> str:
    digest = hashlib.sha256(host_fingerprint.encode("utf-8")).hexdigest()[:18]
    return f"pg_{digest}"


def quantized_periodic_coordinate(value: float, tolerance: float) -> int:
    period_bins = max(1, int(round(1.0 / tolerance)))
    return int(round((float(value) % 1.0) / tolerance)) % period_bins


def host_fingerprint(
    transition: Dict[str, Any],
    raw_record: Dict[str, Any],
    coordinate_tolerance: float,
    lattice_tolerance: float,
) -> str:
    structure = raw_record["structure_ini"]
    species = transition["migrating_species"]
    host_sites = sorted(
        (
            element,
            tuple(quantized_periodic_coordinate(value, coordinate_tolerance) for value in coords),
        )
        for element, coords in zip(structure["elements"], structure["coords"])
        if element != species
    )
    lattice_bins = [
        [int(round(float(value) / lattice_tolerance)) for value in row]
        for row in structure["lattice_mat"]
    ]
    return canonical_json(
        {
            "reduced_formula": reduced_formula(transition["formula"]),
            "migrating_species": species,
            "lattice_bins": lattice_bins,
            "host_sites": host_sites,
        }
    )


def group_transitions(
    transitions: Sequence[Dict[str, Any]],
    raw_records: Sequence[Dict[str, Any]],
    coordinate_tolerance: float,
    lattice_tolerance: float,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for transition in transitions:
        fingerprint = host_fingerprint(
            transition,
            raw_records[transition["raw_index"]],
            coordinate_tolerance,
            lattice_tolerance,
        )
        grouped[fingerprint].append(transition)
    groups = []
    for fingerprint, records in grouped.items():
        records = sorted(records, key=lambda item: item["raw_index"])
        first = records[0]
        structure_hashes = sorted({record["structure_ini_hash"] for record in records})
        groups.append(
            {
                "graph_id": graph_id_for(fingerprint),
                "host_fingerprint_sha256": hashlib.sha256(fingerprint.encode("utf-8")).hexdigest(),
                "structure_ini_hash": first["structure_ini_hash"],
                "source_structure_ini_hashes": structure_hashes,
                "formula": first["formula"],
                "reduced_formula": reduced_formula(first["formula"]),
                "migrating_species": first["migrating_species"],
                "transitions": records,
            }
        )
    return groups


def select_trial_groups(
    groups: Sequence[Dict[str, Any]],
    raw_records: Sequence[Dict[str, Any]],
    trial_count: int,
    maximum_lattice_change: float,
    cutoffs: Dict[str, float],
    minimum_mobile_sites: int,
    node_tolerance_A: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    eligible = []
    rejected_lattice = 0
    rejected_mobile_site_count = 0
    rejected_no_periodic_candidate = 0
    for group in groups:
        records = group["transitions"]
        max_change = max(record["lattice_relative_change"] for record in records)
        if max_change > maximum_lattice_change:
            rejected_lattice += 1
            continue
        first_raw = raw_records[records[0]["raw_index"]]
        species = group["migrating_species"]
        mobile_coords = [
            normalize_frac(coords)
            for element, coords in zip(first_raw["structure_ini"]["elements"], first_raw["structure_ini"]["coords"])
            if element == species
        ]
        if len(mobile_coords) < minimum_mobile_sites:
            rejected_mobile_site_count += 1
            continue
        lattice = [[float(value) for value in row] for row in first_raw["structure_ini"]["lattice_mat"]]
        radius = derived_candidate_shift_radius(lattice, cutoffs[species])
        has_periodic_edge = False
        for left in mobile_coords:
            for right in mobile_coords:
                for shift in product(range(-radius, radius + 1), repeat=3):
                    if shift == (0, 0, 0):
                        continue
                    displacement = tuple(right[axis] + shift[axis] - left[axis] for axis in range(3))
                    distance = vector_norm(frac_to_cart(displacement, lattice))
                    if node_tolerance_A < distance <= cutoffs[species] + 1e-12:
                        has_periodic_edge = True
                        break
                if has_periodic_edge:
                    break
            if has_periodic_edge:
                break
        if not has_periodic_edge:
            rejected_no_periodic_candidate += 1
            continue
        candidate = dict(group)
        candidate["n_atoms"] = len(first_raw["structure_ini"]["elements"])
        candidate["initial_mobile_site_count"] = len(mobile_coords)
        candidate["max_lattice_relative_change"] = max_change
        candidate["derived_candidate_shift_radius"] = radius
        eligible.append(candidate)
    eligible.sort(key=lambda item: (item["n_atoms"], min(record["raw_index"] for record in item["transitions"]), item["graph_id"]))
    species_frequency = Counter(
        transition["migrating_species"]
        for group in groups
        for transition in group["transitions"]
    )
    species_order = [species for species, _count in sorted(species_frequency.items(), key=lambda item: (-item[1], item[0]))]
    selected: List[Dict[str, Any]] = []
    for species in species_order:
        candidates = [group for group in eligible if group["migrating_species"] == species]
        if not candidates:
            continue
        selected.append(candidates[0])
        if len(selected) == trial_count:
            break
    return selected, {
        "total_graph_groups": len(groups),
        "eligible_graph_groups": len(eligible),
        "rejected_for_lattice_change": rejected_lattice,
        "rejected_for_mobile_site_count": rejected_mobile_site_count,
        "rejected_without_periodic_candidate": rejected_no_periodic_candidate,
        "species_frequency": dict(sorted(species_frequency.items())),
    }


def node_match(nodes: Sequence[Dict[str, Any]], coords: Frac3, lattice: Matrix3, tolerance_A: float) -> Optional[int]:
    for index, node in enumerate(nodes):
        if periodic_distance(node["frac_coords_raw"], coords, lattice) <= tolerance_A:
            return index
    return None


def canonical_edge_key(node_i: str, node_j: str, shift: Int3) -> Tuple[str, str, Int3]:
    forward = (node_i, node_j, shift)
    reverse = (node_j, node_i, (-shift[0], -shift[1], -shift[2]))
    return min(forward, reverse)


def make_edge_row(
    graph_id: str,
    nodes_by_id: Dict[str, Dict[str, Any]],
    key: Tuple[str, str, Int3],
    lattice: Matrix3,
    source_record_ids: Sequence[str],
    cutoff: float,
    cutoff_source: str,
) -> Dict[str, str]:
    node_i, node_j, shift = key
    start = nodes_by_id[node_i]["frac_coords_raw"]
    end = nodes_by_id[node_j]["frac_coords_raw"]
    displacement_frac = tuple(end[axis] + shift[axis] - start[axis] for axis in range(3))
    displacement_cart = frac_to_cart(displacement_frac, lattice)
    identity = canonical_json({"graph_id": graph_id, "node_i": node_i, "node_j": node_j, "shift": shift})
    return {
        "edge_uid": "edge_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:22],
        "graph_id": graph_id,
        "node_i": node_i,
        "node_j": node_j,
        "image_shift_a": str(shift[0]),
        "image_shift_b": str(shift[1]),
        "image_shift_c": str(shift[2]),
        "start_frac_coords": canonical_json(start),
        "end_frac_coords": canonical_json(end),
        "displacement_frac": canonical_json(displacement_frac),
        "displacement_cart": canonical_json(displacement_cart),
        "lattice_matrix": canonical_json(lattice),
        "hop_distance_A": repr(vector_norm(displacement_cart)),
        "edge_source": "distance_enumeration_from_raw_structure",
        "is_labeled": "0",
        "source_record_id": ";".join(sorted(source_record_ids)),
        "forced_reported_edge": "0",
        "reported_barrier_eV": "{}",
        "predicted_barrier_eV": "",
        "candidate_distance_threshold_A": repr(cutoff),
        "candidate_threshold_source": cutoff_source,
    }


def build_graph(
    group: Dict[str, Any],
    raw_records: Sequence[Dict[str, Any]],
    cutoff: float,
    cutoff_source: str,
    node_tolerance_A: float,
    validation_radii: Tuple[int, int],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]], Dict[str, Any]]:
    transitions = group["transitions"]
    first_raw = raw_records[transitions[0]["raw_index"]]
    structure = first_raw["structure_ini"]
    lattice: Matrix3 = [[float(value) for value in row] for row in structure["lattice_mat"]]
    species = group["migrating_species"]
    graph_id = group["graph_id"]
    group_record_ids = [transition["record_id"] for transition in transitions]
    nodes: List[Dict[str, Any]] = []
    for transition in transitions:
        source_raw = raw_records[transition["raw_index"]]
        for atom_index, (element, coords) in enumerate(zip(source_raw["structure_ini"]["elements"], source_raw["structure_ini"]["coords"])):
            if element != species:
                continue
            normalized = normalize_frac(coords)
            index = node_match(nodes, normalized, lattice, node_tolerance_A)
            if index is None:
                nodes.append(
                    {
                        "frac_coords_raw": normalized,
                        "node_source": "raw_initial_structure_mobile_site",
                        "source_atom_indices": [atom_index],
                        "source_record_ids": {transition["record_id"]},
                    }
                )
            else:
                nodes[index]["source_record_ids"].add(transition["record_id"])
                if atom_index not in nodes[index]["source_atom_indices"]:
                    nodes[index]["source_atom_indices"].append(atom_index)
    for transition in transitions:
        index = node_match(nodes, transition["end_frac"], lattice, node_tolerance_A)
        if index is None:
            nodes.append(
                {
                    "frac_coords_raw": transition["end_frac"],
                    "node_source": "reported_neb_final_endpoint",
                    "source_atom_indices": [transition["atom_index"]],
                    "source_record_ids": {transition["record_id"]},
                }
            )
        else:
            nodes[index]["source_record_ids"].add(transition["record_id"])
    nodes.sort(key=lambda item: tuple(round(value, 12) for value in item["frac_coords_raw"]))
    node_rows: List[Dict[str, str]] = []
    nodes_by_id: Dict[str, Dict[str, Any]] = {}
    for index, node in enumerate(nodes):
        node_id = f"{graph_id}_n{index:04d}"
        node["node_id"] = node_id
        nodes_by_id[node_id] = node
        frac = node["frac_coords_raw"]
        node_rows.append(
            {
                "graph_id": graph_id,
                "node_id": node_id,
                "formula": group["formula"],
                "reduced_formula": reduced_formula(group["formula"]),
                "migrating_species": species,
                "frac_coords": canonical_json(frac),
                "cart_coords": canonical_json(frac_to_cart(frac, lattice)),
                "lattice_matrix": canonical_json(lattice),
                "node_source": node["node_source"],
                "source_atom_indices": canonical_json(sorted(node["source_atom_indices"])),
                "source_record_id": ";".join(sorted(node["source_record_ids"])),
                "source_structure_ini_hashes": canonical_json(group["source_structure_ini_hashes"]),
            }
        )
    edge_map: Dict[Tuple[str, str, Int3], Dict[str, str]] = {}
    shift_radius = derived_candidate_shift_radius(lattice, cutoff)
    shifts = list(product(range(-shift_radius, shift_radius + 1), repeat=3))
    node_ids = sorted(nodes_by_id)
    for node_i in node_ids:
        for node_j in node_ids:
            for shift_tuple in shifts:
                shift: Int3 = (int(shift_tuple[0]), int(shift_tuple[1]), int(shift_tuple[2]))
                if node_i == node_j and shift == (0, 0, 0):
                    continue
                key = canonical_edge_key(node_i, node_j, shift)
                if key in edge_map:
                    continue
                row = make_edge_row(graph_id, nodes_by_id, key, lattice, group_record_ids, cutoff, cutoff_source)
                distance = float(row["hop_distance_A"])
                if distance > node_tolerance_A and distance <= cutoff + 1e-12:
                    edge_map[key] = row
    expanded_only = []
    for node_i in node_ids:
        for node_j in node_ids:
            for shift_tuple in product(range(-(shift_radius + 1), shift_radius + 2), repeat=3):
                if max(abs(int(value)) for value in shift_tuple) <= shift_radius:
                    continue
                shift = (int(shift_tuple[0]), int(shift_tuple[1]), int(shift_tuple[2]))
                if node_i == node_j and shift == (0, 0, 0):
                    continue
                key = canonical_edge_key(node_i, node_j, shift)
                if key in edge_map:
                    continue
                row = make_edge_row(graph_id, nodes_by_id, key, lattice, group_record_ids, cutoff, cutoff_source)
                distance = float(row["hop_distance_A"])
                if distance > node_tolerance_A and distance <= cutoff + 1e-12:
                    expanded_only.append(row["edge_uid"])
    if expanded_only:
        raise RuntimeError(
            f"Derived image-shift radius {shift_radius} missed {len(expanded_only)} cutoff-qualified edges in {graph_id}"
        )
    reported_path_rows: List[Dict[str, str]] = []
    reported_values_by_key: Dict[Tuple[str, str, Int3], Dict[str, float]] = defaultdict(dict)
    for transition in transitions:
        start_index = node_match(nodes, transition["start_frac"], lattice, node_tolerance_A)
        end_index = node_match(nodes, transition["end_frac"], lattice, node_tolerance_A)
        if start_index is None or end_index is None:
            raise RuntimeError(f"Reported path endpoints did not map to graph nodes: {transition['record_id']}")
        start_id = nodes[start_index]["node_id"]
        end_id = nodes[end_index]["node_id"]
        key = canonical_edge_key(start_id, end_id, transition["image_shift"])
        if key not in edge_map:
            edge_map[key] = make_edge_row(graph_id, nodes_by_id, key, lattice, group_record_ids, cutoff, cutoff_source)
        edge_row = edge_map[key]
        forced = float(edge_row["hop_distance_A"]) > cutoff + 1e-12
        edge_row["is_labeled"] = "1"
        edge_row["forced_reported_edge"] = "1" if forced else "0"
        edge_row["edge_source"] = (
            "reported_neb_forced_beyond_empirical_cutoff"
            if forced
            else "distance_enumeration_plus_reported_neb"
        )
        current_sources = set(filter(None, edge_row["source_record_id"].split(";")))
        current_sources.add(transition["record_id"])
        edge_row["source_record_id"] = ";".join(sorted(current_sources))
        reported_values_by_key[key][transition["record_id"]] = transition["reported_barrier_eV"]
        reported_path_rows.append(
            {
                "record_id": transition["record_id"],
                "jid": transition["jid"],
                "graph_id": graph_id,
                "edge_uid": edge_row["edge_uid"],
                "formula": transition["formula"],
                "sys_name": transition["sys_name"],
                "migrating_species": species,
                "reported_start_frac_coords": canonical_json(transition["start_frac"]),
                "reported_end_frac_coords": canonical_json(transition["end_frac"]),
                "reported_image_shift": canonical_json(transition["image_shift"]),
                "raw_image_hint": canonical_json(transition["raw_image_hint"]),
                "image_shift_degenerate": "1" if transition["image_shift_degenerate"] else "0",
                "minimum_image_count": str(transition["minimum_image_count"]),
                "reported_hop_distance_A": repr(transition["hop_distance_A"]),
                "reported_barrier_eV": repr(transition["reported_barrier_eV"]),
                "candidate_distance_threshold_A": repr(cutoff),
                "forced_reported_edge": "1" if forced else "0",
                "lattice_relative_change": repr(transition["lattice_relative_change"]),
            }
        )
    for key, values in reported_values_by_key.items():
        edge_map[key]["reported_barrier_eV"] = canonical_json(values)
    edge_rows = sorted(edge_map.values(), key=lambda row: row["edge_uid"])
    periodic_edges = [
        PeriodicEdge(
            edge_uid=row["edge_uid"],
            node_i=row["node_i"],
            node_j=row["node_j"],
            image_shift=(int(row["image_shift_a"]), int(row["image_shift_b"]), int(row["image_shift_c"])),
            barrier_eV=float(row["predicted_barrier_eV"]) if row["predicted_barrier_eV"] else None,
        )
        for row in edge_rows
    ]
    readout = periodic_readout([row["node_id"] for row in node_rows], periodic_edges, validation_radii)
    readout["candidate_image_shift_radius"] = shift_radius
    readout["candidate_image_shift_radius_plus_one_verified"] = True
    return node_rows, edge_rows, reported_path_rows, readout


def diagnostic_plot(
    graph_id: str,
    formula: str,
    species: str,
    node_rows: Sequence[Dict[str, str]],
    edge_rows: Sequence[Dict[str, str]],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    node_cart = {row["node_id"]: np.asarray(json.loads(row["cart_coords"]), dtype=float) for row in node_rows}
    lattice = np.asarray(json.loads(node_rows[0]["lattice_matrix"]), dtype=float)
    point_cloud = list(node_cart.values())
    edge_segments = []
    for row in edge_rows:
        start = node_cart[row["node_i"]]
        shift = np.asarray([int(row["image_shift_a"]), int(row["image_shift_b"]), int(row["image_shift_c"])], dtype=float)
        end = node_cart[row["node_j"]] + shift @ lattice
        edge_segments.append((row, start, end))
        point_cloud.extend([start, end])
    points = np.vstack(point_cloud)
    centered = points - points.mean(axis=0)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:2].T if vh.shape[0] >= 2 else np.eye(3)[:, :2]
    origin = points.mean(axis=0)

    def project(vector: Any) -> Any:
        return (np.asarray(vector, dtype=float) - origin) @ basis

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for row, start, end in edge_segments:
        projected = np.vstack([project(start), project(end)])
        if row["forced_reported_edge"] == "1":
            color, width, alpha = "#c0392b", 2.0, 0.9
        elif row["is_labeled"] == "1":
            color, width, alpha = "#2878b5", 1.7, 0.85
        else:
            color, width, alpha = "#9aa0a6", 0.6, 0.22
        ax.plot(projected[:, 0], projected[:, 1], color=color, lw=width, alpha=alpha, zorder=1)
    central = np.vstack([project(value) for value in node_cart.values()])
    ax.scatter(central[:, 0], central[:, 1], s=30, color="#202124", edgecolor="white", linewidth=0.5, zorder=3)
    if len(node_rows) <= 20:
        for row, point in zip(node_rows, central):
            ax.text(point[0], point[1], row["node_id"].rsplit("_", 1)[-1], fontsize=6, ha="left", va="bottom")
    ax.set_title(f"{formula} / {species}: real periodic-graph geometry diagnostic\n{graph_id}; no predicted-barrier Eperc reported")
    ax.set_xlabel("Cartesian PCA coordinate 1 (Å)")
    ax.set_ylabel("Cartesian PCA coordinate 2 (Å)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(color="#eceff1", linewidth=0.5)
    ax.legend(
        handles=[
            Line2D([0], [0], color="#9aa0a6", lw=1.2, label="distance-enumerated candidate"),
            Line2D([0], [0], color="#2878b5", lw=2.0, label="reported NEB edge"),
            Line2D([0], [0], color="#c0392b", lw=2.0, label="forced reported edge"),
        ],
        loc="best",
        fontsize=7,
    )
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_graph_json(
    path: Path,
    graph_metadata: Dict[str, Any],
    nodes: Sequence[Dict[str, str]],
    edges: Sequence[Dict[str, str]],
    readout: Dict[str, Any],
) -> None:
    write_json(
        path,
        {
            "schema_version": "1.0",
            "graph_metadata": graph_metadata,
            "nodes": list(nodes),
            "edges": list(edges),
            "periodic_readout": readout,
        },
    )


def summary_row(
    group: Dict[str, Any],
    node_rows: Sequence[Dict[str, str]],
    edge_rows: Sequence[Dict[str, str]],
    readout: Dict[str, Any],
    cutoff: float,
    cutoff_source: str,
    selection_rank: int,
) -> Dict[str, str]:
    return {
        "graph_id": group["graph_id"],
        "formula": group["formula"],
        "reduced_formula": reduced_formula(group["formula"]),
        "migrating_species": group["migrating_species"],
        "source_record_id": ";".join(record["record_id"] for record in group["transitions"]),
        "selection_rank": str(selection_rank),
        "selection_rule": "top_frequency_graph_safe_species_then_smallest_host_equivalent_graph",
        "n_nodes": str(len(node_rows)),
        "n_edges": str(len(edge_rows)),
        "n_labeled_edges": str(sum(row["is_labeled"] == "1" for row in edge_rows)),
        "n_forced_reported_edges": str(sum(row["forced_reported_edge"] == "1" for row in edge_rows)),
        "candidate_distance_threshold_A": repr(cutoff),
        "candidate_threshold_source": cutoff_source,
        "candidate_image_shift_radius": str(readout["candidate_image_shift_radius"]),
        "candidate_image_shift_radius_plus_one_verified": "1" if readout["candidate_image_shift_radius_plus_one_verified"] else "0",
        "predicted_barrier_coverage": repr(readout["barrier_coverage"]),
        "Eperc_a": "" if readout["Eperc_a"] is None else repr(readout["Eperc_a"]),
        "Eperc_b": "" if readout["Eperc_b"] is None else repr(readout["Eperc_b"]),
        "Eperc_c": "" if readout["Eperc_c"] is None else repr(readout["Eperc_c"]),
        "Eperc_a_3x3x3": "" if readout["Eperc_a_3x3x3"] is None else repr(readout["Eperc_a_3x3x3"]),
        "Eperc_b_3x3x3": "" if readout["Eperc_b_3x3x3"] is None else repr(readout["Eperc_b_3x3x3"]),
        "Eperc_c_3x3x3": "" if readout["Eperc_c_3x3x3"] is None else repr(readout["Eperc_c_3x3x3"]),
        "Eperc_a_5x5x5": "" if readout["Eperc_a_5x5x5"] is None else repr(readout["Eperc_a_5x5x5"]),
        "Eperc_b_5x5x5": "" if readout["Eperc_b_5x5x5"] is None else repr(readout["Eperc_b_5x5x5"]),
        "Eperc_c_5x5x5": "" if readout["Eperc_c_5x5x5"] is None else repr(readout["Eperc_c_5x5x5"]),
        "Eperc_a_exact": "" if readout["Eperc_a_exact"] is None else repr(readout["Eperc_a_exact"]),
        "Eperc_b_exact": "" if readout["Eperc_b_exact"] is None else repr(readout["Eperc_b_exact"]),
        "Eperc_c_exact": "" if readout["Eperc_c_exact"] is None else repr(readout["Eperc_c_exact"]),
        "periodic_dimensionality": readout["periodic_dimensionality"] or "",
        "winding_vector_rank": "" if readout["winding_vector_rank"] is None else str(readout["winding_vector_rank"]),
        "global_winding_union_rank": "" if readout["global_winding_union_rank"] is None else str(readout["global_winding_union_rank"]),
        "critical_periodic_edges": canonical_json(readout["critical_periodic_edges"]),
        "redundancy_count": "" if readout["redundancy_count"] is None else str(readout["redundancy_count"]),
        "supercell_converged": "1" if readout["supercell_converged"] else "0",
        "readout_status": readout["status"],
        "lattice_relative_change_max": repr(group["max_lattice_relative_change"]),
    }


def run_periodic_trial(config_path: Optional[Path] = None) -> Dict[str, Any]:
    config_path = (config_path or ROOT / "config" / "revision_config.yaml").resolve()
    config = read_config(config_path)
    periodic_config = config["periodic_graph"]
    validation_radii = tuple(int(value) for value in periodic_config["supercell_validation_radii"])
    if validation_radii != (1, 2):
        raise ValueError("periodic_graph.supercell_validation_radii must be [1, 2] for 3x3x3 and 5x5x5")
    if periodic_config["barrier_column_for_periodic_readout"] != "predicted_barrier_eV":
        raise ValueError("Periodic readout must use predicted_barrier_eV, never reported or hand-entered barriers")
    raw_path, raw_hash = require_authorised_raw_dataset(config)
    records = json.loads(raw_path.read_text(encoding="utf-8"))
    minimum_displacement = float(config["quality_rules"]["minimum_migration_displacement_A"])
    transitions = [
        reported_transition(
            index,
            record,
            minimum_displacement,
            float(periodic_config["minimum_image_tie_tolerance_A"]),
            float(periodic_config["fixed_cell_lattice_relative_tolerance"]),
            config["source_group"]["fallback"],
        )
        for index, record in enumerate(records)
    ]
    graph_safe_transitions = [transition for transition in transitions if transition["graph_safe"]]
    if not graph_safe_transitions:
        raise RuntimeError("No fixed-cell single-mover transitions are available for the real-structure trial")
    cutoffs, cutoff_sources, global_cutoff = empirical_cutoffs(
        graph_safe_transitions,
        float(periodic_config["reported_hop_distance_quantile_for_candidate_cutoff"]),
        int(periodic_config["minimum_species_records_for_species_cutoff"]),
    )
    groups = group_transitions(
        graph_safe_transitions,
        records,
        float(periodic_config["host_fingerprint_coordinate_tolerance"]),
        float(periodic_config["host_fingerprint_lattice_tolerance_A"]),
    )
    selected, selection_audit = select_trial_groups(
        groups,
        records,
        int(periodic_config["trial_graph_count"]),
        float(periodic_config["fixed_cell_lattice_relative_tolerance"]),
        cutoffs,
        int(periodic_config["minimum_initial_mobile_sites"]),
        float(periodic_config["node_merge_tolerance_A"]),
    )
    if len(selected) != int(periodic_config["trial_graph_count"]):
        raise RuntimeError("Not enough eligible real structures for the configured periodic-graph trial")
    toy_validation = run_toy_validation()
    if not toy_validation["passed"]:
        raise RuntimeError("Periodic 0D-3D toy validation failed; real graph generation is blocked")

    table_dir = project_path(config["outputs"]["source_data_directory"])
    derived_dir = project_path(periodic_config["derived_data_directory"])
    diagnostics_dir = derived_dir / "diagnostics"
    graph_dir = derived_dir / "graphs"
    table_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)
    all_nodes: List[Dict[str, str]] = []
    all_edges: List[Dict[str, str]] = []
    all_reported: List[Dict[str, str]] = []
    summaries: List[Dict[str, str]] = []
    graph_files = []
    diagnostic_files = []
    for selection_rank, group in enumerate(selected, start=1):
        species = group["migrating_species"]
        node_rows, edge_rows, reported_rows, readout = build_graph(
            group,
            records,
            cutoffs[species],
            cutoff_sources[species],
            float(periodic_config["node_merge_tolerance_A"]),
            validation_radii,
        )
        all_nodes.extend(node_rows)
        all_edges.extend(edge_rows)
        all_reported.extend(reported_rows)
        summaries.append(
            summary_row(
                group,
                node_rows,
                edge_rows,
                readout,
                cutoffs[species],
                cutoff_sources[species],
                selection_rank,
            )
        )
        graph_path = graph_dir / f"{group['graph_id']}.json"
        diagnostic_path = diagnostics_dir / f"{group['graph_id']}_geometry_diagnostic.png"
        metadata = {
            "graph_id": group["graph_id"],
            "formula": group["formula"],
            "migrating_species": species,
            "structure_ini_hash": group["structure_ini_hash"],
            "source_structure_ini_hashes": group["source_structure_ini_hashes"],
            "source_record_ids": [record["record_id"] for record in group["transitions"]],
            "candidate_distance_threshold_A": cutoffs[species],
            "candidate_threshold_source": cutoff_sources[species],
            "selection_rank": selection_rank,
            "candidate_image_shift_radius": readout["candidate_image_shift_radius"],
            "candidate_image_shift_radius_plus_one_verified": readout["candidate_image_shift_radius_plus_one_verified"],
        }
        write_graph_json(graph_path, metadata, node_rows, edge_rows, readout)
        diagnostic_plot(group["graph_id"], group["formula"], species, node_rows, edge_rows, diagnostic_path)
        graph_files.append(graph_path)
        diagnostic_files.append(diagnostic_path)

    node_columns = list(all_nodes[0])
    edge_columns = EDGE_REQUIRED_COLUMNS + EDGE_EXTRA_COLUMNS
    reported_columns = list(all_reported[0])
    summary_columns = list(summaries[0])
    source_outputs = periodic_config["source_data_outputs"]
    csv_artifacts = {
        source_outputs["nodes"]: write_csv_artifacts(
            table_dir,
            source_outputs["nodes"],
            all_nodes,
            node_columns,
            raw_hash,
            {"scope": "real periodic-graph trial nodes", "row_count": len(all_nodes)},
        ),
        source_outputs["edges"]: write_csv_artifacts(
            table_dir,
            source_outputs["edges"],
            all_edges,
            edge_columns,
            raw_hash,
            {
                "scope": "real periodic-graph trial candidate and reported edges",
                "row_count": len(all_edges),
                "required_columns": EDGE_REQUIRED_COLUMNS,
                "forced_reported_edge_count": sum(row["forced_reported_edge"] == "1" for row in all_edges),
            },
        ),
        source_outputs["graphs"]: write_csv_artifacts(
            table_dir,
            source_outputs["graphs"],
            summaries,
            summary_columns,
            raw_hash,
            {"scope": "real periodic-graph trial summaries", "row_count": len(summaries)},
        ),
        source_outputs["reported_paths"]: write_csv_artifacts(
            table_dir,
            source_outputs["reported_paths"],
            all_reported,
            reported_columns,
            raw_hash,
            {"scope": "reported NEB path to periodic-edge mapping", "row_count": len(all_reported)},
        ),
    }
    complete_prediction_graphs = sum(float(row["predicted_barrier_coverage"]) == 1.0 for row in summaries)
    converged_graphs = sum(row["supercell_converged"] == "1" for row in summaries)
    gate = {
        "toy_tests_passed": toy_validation["passed"],
        "required_threshold_agreement": ["3x3x3", "5x5x5", "exact_integer_winding_lattice"],
        "real_graph_count": len(summaries),
        "real_graphs_with_complete_predicted_barriers": complete_prediction_graphs,
        "real_graphs_with_3x3x3_5x5x5_convergence": converged_graphs,
        "real_graphs_with_3x3x3_5x5x5_and_exact_convergence": converged_graphs,
        "allow_replace_paper_source_data": bool(
            toy_validation["passed"]
            and complete_prediction_graphs == len(summaries)
            and converged_graphs == len(summaries)
        ),
    }
    blockers = []
    if complete_prediction_graphs != len(summaries):
        blockers.append("Candidate edges do not yet have complete traceable predicted barriers; real Eperc is withheld.")
    if converged_graphs != len(summaries):
        blockers.append("Real 3x3x3/5x5x5/exact-winding Eperc agreement cannot pass until complete predictions exist.")
    audit = {
        "audit_schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "small real-structure periodic migration-graph trial; diagnostics only",
        "input": {"path": str(raw_path), "sha256": raw_hash, "record_count": len(records)},
        "forbidden_legacy_inputs_used": [],
        "candidate_cutoff": {
            "method": "empirical quantile of real reported periodic hop distances by migrating species",
            "quantile": periodic_config["reported_hop_distance_quantile_for_candidate_cutoff"],
            "global_fallback_A": global_cutoff,
            "by_species_A": dict(sorted(cutoffs.items())),
            "source_by_species": dict(sorted(cutoff_sources.items())),
        },
        "selection": {
            "rule": periodic_config["trial_selection"],
            **selection_audit,
            "selected_graphs": [
                {
                    "graph_id": group["graph_id"],
                    "formula": group["formula"],
                    "migrating_species": group["migrating_species"],
                    "n_atoms": group["n_atoms"],
                    "source_record_ids": [item["record_id"] for item in group["transitions"]],
                }
                for group in selected
            ],
        },
        "graph_safety": {
            "all_raw_records": len(transitions),
            "fixed_cell_single_mover_records": len(graph_safe_transitions),
            "excluded_lattice_change_or_multi_mover_records": len(transitions) - len(graph_safe_transitions),
            "image_shift_degenerate_records_all": sum(transition["image_shift_degenerate"] for transition in transitions),
            "image_shift_degenerate_records_graph_safe": sum(transition["image_shift_degenerate"] for transition in graph_safe_transitions),
            "records_with_out_of_cell_mover_endpoint": sum(
                any(value < 0 or value >= 1 for value in records[transition["raw_index"]]["structure_fin"]["coords"][transition["atom_index"]])
                for transition in transitions
            ),
        },
        "counts": {
            "trial_graphs": len(summaries),
            "nodes": len(all_nodes),
            "edges": len(all_edges),
            "labeled_edges": sum(row["is_labeled"] == "1" for row in all_edges),
            "forced_reported_edges": sum(row["forced_reported_edge"] == "1" for row in all_edges),
            "reported_paths": len(all_reported),
            "edge_source_counts": dict(sorted(Counter(row["edge_source"] for row in all_edges).items())),
        },
        "toy_validation": toy_validation,
        "publication_source_gate": gate,
        "blockers": blockers,
        "graph_summaries": summaries,
    }
    audit_path = table_dir / "periodic_trial_audit.json"
    audit_markdown_path = table_dir / "periodic_trial_audit.md"
    gate_path = table_dir / "periodic_graph_release_gate.json"
    write_json(audit_path, audit)
    write_json(derived_dir / "periodic_trial_audit.json", audit)
    write_json(gate_path, gate)
    lines = [
        "# Periodic migration-graph trial audit",
        "",
        "This is a geometry/provenance diagnostic. It is not a paper figure source and reports no real Eperc while candidate-edge predictions are incomplete.",
        "",
        f"- Trial graphs: {len(summaries)}",
        f"- Nodes: {len(all_nodes)}",
        f"- Candidate/reported edges: {len(all_edges)}",
        f"- Forced reported edges: {audit['counts']['forced_reported_edges']}",
        f"- 0D–3D toy validation: {'passed' if toy_validation['passed'] else 'failed'}",
        f"- Paper source replacement allowed: {gate['allow_replace_paper_source_data']}",
        "",
        "## Blockers",
        "",
    ]
    lines.extend(f"- {blocker}" for blocker in blockers)
    audit_markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log_dir = project_path(config["outputs"]["log_directory"])
    write_json(log_dir / "periodic_graph_toy_validation.json", toy_validation)
    write_json(log_dir / "revision_periodic_trial_run.json", {"status": "passed_with_gate_blocked", "audit": str(audit_path.relative_to(ROOT)), "gate": gate})
    (log_dir / "revision_periodic_trial_run.log").write_text(
        f"status=passed_with_gate_blocked\ntrial_graphs={len(summaries)}\ntoy_tests_passed={toy_validation['passed']}\nallow_replace_paper_source_data={gate['allow_replace_paper_source_data']}\n",
        encoding="utf-8",
    )
    manifest = {
        "manifest_schema_version": "1.0",
        "raw_input_sha256": raw_hash,
        "config_sha256": sha256_file(config_path),
        "runner_sha256": sha256_file(ROOT / "scripts" / "run_periodic_graph_trial.py"),
        "generator_sha256": sha256_file(Path(__file__)),
        "periodic_engine_sha256": sha256_file(Path(__file__).with_name("periodic_graph.py")),
        "data_governance_writer_sha256": sha256_file(Path(__file__).with_name("data_governance.py")),
        "test_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in sorted((Path(__file__).parent / "tests").glob("test_periodic_*.py"))
        },
        "csv_artifacts": {
            name: {
                "csv": str(artifact["csv"].relative_to(ROOT)),
                "sha256": artifact["sha256"],
                "schema": str(artifact["schema"].relative_to(ROOT)),
                "schema_sha256": sha256_file(artifact["schema"]),
                "checksum": str(artifact["checksum"].relative_to(ROOT)),
                "checksum_sha256": sha256_file(artifact["checksum"]),
                "audit": str(artifact["audit"].relative_to(ROOT)),
                "audit_sha256": sha256_file(artifact["audit"]),
            }
            for name, artifact in csv_artifacts.items()
        },
        "graph_json": {str(path.relative_to(ROOT)): sha256_file(path) for path in graph_files},
        "diagnostic_png": {str(path.relative_to(ROOT)): sha256_file(path) for path in diagnostic_files},
        "run_logs": {
            "logs/periodic_graph_toy_validation.json": sha256_file(log_dir / "periodic_graph_toy_validation.json"),
            "logs/revision_periodic_trial_run.json": sha256_file(log_dir / "revision_periodic_trial_run.json"),
            "logs/revision_periodic_trial_run.log": sha256_file(log_dir / "revision_periodic_trial_run.log"),
        },
        "audit_sha256": sha256_file(audit_path),
        "release_gate_sha256": sha256_file(gate_path),
        "publication_source_gate": gate,
    }
    write_json(table_dir / "periodic_trial_manifest.json", manifest)
    write_json(derived_dir / "periodic_trial_manifest.json", manifest)
    return audit
