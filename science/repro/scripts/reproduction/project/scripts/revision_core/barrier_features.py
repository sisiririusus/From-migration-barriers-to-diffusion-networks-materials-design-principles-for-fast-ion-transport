"""Traceable four-block features for the revision migration-barrier model.

Only the authorised raw JSON and revision provenance/candidate tables are read.
No historical predictions, barrier-weighted topology, or paper metrics enter the
feature builder.  The raw structures contain endpoints but no NEB images, so all
path constriction quantities are explicitly named straight-path proxies.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from revision_core.data_governance import ROOT, canonical_json, reduced_formula, sha256_file, sha256_json
from revision_core.element_properties import (
    NEUTRAL_PROPERTY_COLUMNS,
    load_neutral_element_properties,
    weighted_property_stats,
)
from revision_core.periodic_trial import frac_to_cart, normalize_frac, reported_transition, vector_norm


METADATA_COLUMNS = [
    "record_id",
    "raw_index",
    "jid",
    "formula",
    "reduced_formula",
    "migrating_species",
    "barrier_eV",
    "source_group",
    "XC",
    "crystal_class",
    "structure_family_normalized",
    "space_group",
    "formula_species_group",
    "is_nasicon",
    "structure_ini_hash",
    "structure_fin_hash",
    "raw_record_sha256",
    "cohort_id",
]

BLOCK_A = "A_composition"
BLOCK_B = "B_local_geometry"
BLOCK_C = "C_local_chemistry"
BLOCK_D = "D_occupancy_dopant_configuration"


class LeakageFeatureError(ValueError):
    """Raised when a proposed input feature violates the fail-closed policy."""


def model_config_path() -> Path:
    return ROOT / "config" / "revision_model_config.yaml"


def load_model_config(path: Optional[Path] = None) -> Dict[str, Any]:
    config_path = (path or model_config_path()).resolve()
    return json.loads(config_path.read_text(encoding="utf-8"))


def resolve_project_input(relative_path: str) -> Path:
    return (ROOT / relative_path).resolve()


def normalize_structure_family(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value).strip().upper())
    synonyms = {"SPINELS": "SPINEL"}
    return synonyms.get(text, text or "UNSPECIFIED")


def assert_no_leakage_features(
    feature_columns: Sequence[str],
    config: Dict[str, Any],
    allowed_feature_columns: Optional[Sequence[str]] = None,
) -> None:
    blocker = config["leakage_blocker"]
    normalize = lambda value: re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")
    forbidden_patterns = [normalize(value) for value in blocker["forbidden_name_patterns"]]
    forbidden_exact = {normalize(value) for value in blocker["forbidden_metadata_features"]}
    violations = []
    for column in feature_columns:
        normalized = normalize(column)
        padded = "_%s_" % normalized
        metadata_match = any(("_%s_" % forbidden) in padded for forbidden in forbidden_exact)
        if metadata_match or any(pattern in normalized for pattern in forbidden_patterns):
            violations.append(column)
    if blocker.get("allowlist_only"):
        allowed = (
            list(allowed_feature_columns)
            if allowed_feature_columns is not None
            else blocker.get("resolved_model_input_allowlist")
        )
        if not isinstance(allowed, list) or not allowed:
            raise LeakageFeatureError("Resolved four-block model-input allowlist is missing")
        unknown = sorted(set(str(column) for column in feature_columns) - set(str(column) for column in allowed))
        violations.extend(unknown)
    if violations:
        raise LeakageFeatureError("Forbidden model-input fields: %s" % sorted(set(violations)))


def _stable_id(prefix: str, payload: Dict[str, Any], length: int = 20) -> str:
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:length]
    return "%s_%s" % (prefix, digest)


def _lattice_volume(lattice: Sequence[Sequence[float]]) -> float:
    return float(abs(np.linalg.det(np.asarray(lattice, dtype=float))))


def _minimum_image_distances(
    center_frac: Sequence[float],
    atom_frac: Sequence[Sequence[float]],
    lattice: Sequence[Sequence[float]],
) -> np.ndarray:
    """Full-metric periodic distances, vectorized and bounded without component wrap."""
    coords = np.mod(np.asarray(atom_frac, dtype=float), 1.0)
    center = np.mod(np.asarray(center_frac, dtype=float), 1.0)
    if coords.size == 0:
        return np.asarray([], dtype=float)
    base = coords - center[None, :]
    lattice_array = np.asarray(lattice, dtype=float)
    rounded_shift = -np.rint(base)
    initial_cart = np.dot(base + rounded_shift, lattice_array)
    initial_best = np.sqrt(np.sum(initial_cart ** 2, axis=1))
    sigma_min = float(np.linalg.svd(lattice_array, compute_uv=False).min())
    if not math.isfinite(sigma_min) or sigma_min <= 0:
        raise ValueError("Singular lattice in local environment")
    radius = max(1, int(math.ceil(float(np.max(initial_best)) / sigma_min + 1.0)) + 1)
    shifts = np.asarray(list(product(range(-radius, radius + 1), repeat=3)), dtype=float)
    minimum_squared = np.full(len(coords), np.inf, dtype=float)
    for begin in range(0, len(shifts), 512):
        chunk = shifts[begin : begin + 512]
        displacement = base[:, None, :] + chunk[None, :, :]
        cartesian = np.einsum("nsk,kl->nsl", displacement, lattice_array)
        squared = np.einsum("nsl,nsl->ns", cartesian, cartesian)
        minimum_squared = np.minimum(minimum_squared, np.min(squared, axis=1))
    return np.sqrt(np.maximum(minimum_squared, 0.0))


def _effective_coordination(distances: np.ndarray, count: int) -> float:
    valid = np.sort(distances[np.isfinite(distances) & (distances > 1e-10)])[:count]
    if len(valid) == 0:
        return math.nan
    nearest = max(float(valid[0]), 1e-12)
    exponent = np.clip(1.0 - (valid / nearest) ** 6, -700.0, 1.0)
    return float(np.sum(np.exp(exponent)))


def _nearest_summary(prefix: str, distances: np.ndarray, neighbor_count: int) -> Dict[str, float]:
    ordered = np.sort(distances[np.isfinite(distances) & (distances > 1e-10)])
    nearest = ordered[:neighbor_count]
    output = {
        prefix + "nearest_1_A": float(nearest[0]) if len(nearest) >= 1 else math.nan,
        prefix + "nearest_2_A": float(nearest[1]) if len(nearest) >= 2 else math.nan,
        prefix + "nearest_4_mean_A": float(np.mean(nearest[:4])) if len(nearest) >= 1 else math.nan,
        prefix + "nearest_8_mean_A": float(np.mean(nearest[:8])) if len(nearest) >= 1 else math.nan,
        prefix + "effective_coordination": _effective_coordination(distances, neighbor_count),
    }
    return output


def _composition_block(
    elements: Sequence[str],
    migrating_species: str,
    all_symbols: Sequence[str],
    property_lookup: Dict[str, Dict[str, float]],
    lattice: Sequence[Sequence[float]],
) -> Dict[str, float]:
    counts = Counter(str(symbol) for symbol in elements)
    atom_count = float(sum(counts.values()))
    fractions = {symbol: counts.get(symbol, 0) / atom_count for symbol in all_symbols}
    nonzero = np.asarray([value for value in fractions.values() if value > 0], dtype=float)
    output: Dict[str, float] = {
        "A_atom_count": atom_count,
        "A_element_count": float(len(counts)),
        "A_composition_entropy": float(-np.sum(nonzero * np.log(nonzero))) if len(nonzero) else math.nan,
        "A_max_element_fraction": float(np.max(nonzero)) if len(nonzero) else math.nan,
        "A_min_nonzero_element_fraction": float(np.min(nonzero)) if len(nonzero) else math.nan,
        "A_migrating_species_fraction": counts.get(migrating_species, 0) / atom_count,
        "A_cell_volume_A3": _lattice_volume(lattice),
        "A_volume_per_atom_A3": _lattice_volume(lattice) / atom_count if atom_count else math.nan,
    }
    for symbol in all_symbols:
        output["A_element_fraction_%s" % symbol] = float(fractions[symbol])

    symbols = sorted(counts)
    weights = [float(counts[symbol]) for symbol in symbols]
    host_symbols = [symbol for symbol in symbols if symbol != migrating_species]
    host_weights = [float(counts[symbol]) for symbol in host_symbols]
    for property_name in NEUTRAL_PROPERTY_COLUMNS:
        for scope, scope_symbols, scope_weights in (
            ("all", symbols, weights),
            ("host", host_symbols, host_weights),
        ):
            stats = weighted_property_stats(scope_symbols, scope_weights, property_lookup, property_name)
            for statistic, value in stats.items():
                output["A_%s_%s_%s" % (scope, property_name, statistic)] = value
        output["A_migrant_%s" % property_name] = property_lookup[migrating_species][property_name]
    return output


def _local_environment_chemistry(
    label: str,
    distances: np.ndarray,
    environment_elements: Sequence[str],
    all_symbols: Sequence[str],
    property_lookup: Dict[str, Dict[str, float]],
    neighbor_count: int,
) -> Dict[str, float]:
    valid_indices = np.where(np.isfinite(distances) & (distances > 1e-10))[0]
    ordered_indices = valid_indices[np.argsort(distances[valid_indices])][:neighbor_count]
    output: Dict[str, float] = {}
    if len(ordered_indices) == 0:
        for symbol in all_symbols:
            output["C_%s_element_fraction_%s" % (label, symbol)] = math.nan
        for property_name in NEUTRAL_PROPERTY_COLUMNS:
            for statistic in ("mean", "std", "min", "max", "coverage"):
                output["C_%s_%s_%s" % (label, property_name, statistic)] = math.nan
        return output

    selected_distances = distances[ordered_indices]
    nearest = max(float(np.min(selected_distances)), 1e-12)
    weights = np.exp(np.clip(1.0 - (selected_distances / nearest) ** 6, -700.0, 1.0))
    selected_elements = [str(environment_elements[index]) for index in ordered_indices]
    total_weight = float(np.sum(weights))
    for symbol in all_symbols:
        symbol_weight = float(np.sum([weight for element, weight in zip(selected_elements, weights) if element == symbol]))
        output["C_%s_element_fraction_%s" % (label, symbol)] = symbol_weight / total_weight if total_weight else math.nan
    for property_name in NEUTRAL_PROPERTY_COLUMNS:
        stats = weighted_property_stats(selected_elements, weights, property_lookup, property_name)
        for statistic, value in stats.items():
            output["C_%s_%s_%s" % (label, property_name, statistic)] = value

    # These remain structurally unavailable.  The missing indicators, not a
    # guessed common oxidation state or ion radius, are model inputs.
    output["C_%s_local_oxidation_state_mean" % label] = math.nan
    output["C_%s_local_oxidation_state_missing_indicator" % label] = 1.0
    output["C_%s_coordination_conditioned_ionic_radius_A_mean" % label] = math.nan
    output["C_%s_coordination_conditioned_ionic_radius_missing_indicator" % label] = 1.0
    return output


def edge_feature_blocks(
    structure: Dict[str, Any],
    migrating_species: str,
    start_frac: Sequence[float],
    end_frac: Sequence[float],
    image_shift: Sequence[int],
    moving_atom_index: Optional[int],
    all_symbols: Sequence[str],
    property_lookup: Dict[str, Dict[str, float]],
    config: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    lattice = np.asarray(structure["lattice_mat"], dtype=float)
    elements = [str(value) for value in structure["elements"]]
    coords = np.asarray(structure["coords"], dtype=float)
    keep = np.ones(len(elements), dtype=bool)
    if moving_atom_index is not None:
        if moving_atom_index < 0 or moving_atom_index >= len(elements):
            raise IndexError("moving_atom_index outside structure")
        keep[moving_atom_index] = False
    environment_coords = coords[keep]
    environment_elements = [element for index, element in enumerate(elements) if keep[index]]

    start = np.asarray(normalize_frac(start_frac), dtype=float)
    end = np.asarray(normalize_frac(end_frac), dtype=float)
    shift = np.asarray(image_shift, dtype=float)
    displacement_frac = end + shift - start
    displacement_cart = np.dot(displacement_frac, lattice)
    hop_distance = float(np.linalg.norm(displacement_cart))
    n_samples = int(config["featurizer"]["path_sample_count"])
    neighbor_count = int(config["featurizer"]["local_neighbor_count"])
    path_t = np.linspace(0.0, 1.0, n_samples)
    path_points = np.asarray([start + value * displacement_frac for value in path_t])
    distance_profiles = np.asarray(
        [_minimum_image_distances(point, environment_coords, lattice) for point in path_points],
        dtype=float,
    )

    center_minimum = np.asarray(
        [float(np.min(row[row > 1e-10])) if np.any(row > 1e-10) else math.nan for row in distance_profiles]
    )
    neutral_radii = np.asarray(
        [property_lookup[element]["atomic_radius_A"] for element in environment_elements], dtype=float
    )
    migrant_radius = property_lookup[migrating_species]["atomic_radius_A"]
    clearance_available = bool(math.isfinite(migrant_radius) and np.all(np.isfinite(neutral_radii)))
    if clearance_available:
        gap_profiles = distance_profiles - (migrant_radius + neutral_radii)[None, :]
        minimum_gap_profile = np.min(gap_profiles, axis=1)
        bottleneck_index = int(np.argmin(minimum_gap_profile))
        signed_clearance = float(np.min(minimum_gap_profile))
        scale = float(config["featurizer"]["crowding_log_scale_A"])
        log_crowding_profile = np.asarray(
            [float(np.logaddexp(0.0, logsumexp(-row / scale))) for row in gap_profiles]
        )
        robust_log_crowding = float(np.max(log_crowding_profile))
        overlap_flag = 1.0 if signed_clearance < 0.0 else 0.0
    else:
        minimum_gap_profile = np.full(n_samples, np.nan)
        bottleneck_index = int(np.nanargmin(center_minimum))
        signed_clearance = math.nan
        robust_log_crowding = math.nan
        overlap_flag = math.nan

    singular = np.linalg.svd(lattice, compute_uv=False)
    block_b: Dict[str, float] = {
        "B_hop_distance_A": hop_distance,
        "B_lattice_volume_A3": _lattice_volume(lattice),
        "B_lattice_singular_min_A": float(np.min(singular)),
        "B_lattice_singular_max_A": float(np.max(singular)),
        "B_lattice_condition_number": float(np.max(singular) / np.min(singular)),
        "B_straight_path_center_distance_min_A": float(np.nanmin(center_minimum)),
        "B_straight_path_center_distance_mean_A": float(np.nanmean(center_minimum)),
        "B_straight_path_center_distance_std_A": float(np.nanstd(center_minimum)),
        "B_straight_path_bottleneck_proxy_t": float(path_t[bottleneck_index]),
        "signed_clearance_A": signed_clearance,
        "overlap_flag": overlap_flag,
        "robust_log_crowding": robust_log_crowding,
        "B_straight_path_clearance_proxy_available_indicator": 1.0 if clearance_available else 0.0,
        "B_straight_path_clearance_proxy_missing_indicator": 0.0 if clearance_available else 1.0,
    }
    location_indices = {
        "start": 0,
        "midpoint": int(np.argmin(np.abs(path_t - 0.5))),
        "end": n_samples - 1,
        "bottleneck_proxy": bottleneck_index,
    }
    for label, index in location_indices.items():
        block_b.update(_nearest_summary("B_%s_" % label, distance_profiles[index], neighbor_count))

    block_c: Dict[str, float] = {}
    for label, index in location_indices.items():
        block_c.update(
            _local_environment_chemistry(
                label,
                distance_profiles[index],
                environment_elements,
                all_symbols,
                property_lookup,
                neighbor_count,
            )
        )
    closest_index = int(np.nanargmin(distance_profiles[bottleneck_index]))
    closest_element = environment_elements[closest_index]
    for property_name in NEUTRAL_PROPERTY_COLUMNS:
        block_c["C_bottleneck_proxy_nearest_element_%s" % property_name] = property_lookup[closest_element][property_name]

    mobile_indices = [index for index, element in enumerate(elements) if element == migrating_species and index != moving_atom_index]
    mobile_coords = coords[mobile_indices]
    mobile_distances_by_location = {
        label: _minimum_image_distances(path_points[index], mobile_coords, lattice)
        for label, index in location_indices.items()
    }
    cell_volume = _lattice_volume(lattice)
    block_d: Dict[str, float] = {
        "D_initial_mobile_site_count": float(sum(element == migrating_species for element in elements)),
        "D_initial_mobile_site_fraction": float(sum(element == migrating_species for element in elements)) / len(elements),
        "D_initial_mobile_number_density_A3": float(sum(element == migrating_species for element in elements)) / cell_volume,
        "D_endpoint_to_nearest_other_mobile_A": (
            float(np.min(mobile_distances_by_location["end"]))
            if len(mobile_distances_by_location["end"])
            else math.nan
        ),
        "D_explicit_start_site_occupancy": math.nan,
        "D_explicit_end_site_occupancy": math.nan,
        "D_explicit_vacancy_fraction": math.nan,
        "D_explicit_dopant_fraction": math.nan,
        "D_explicit_site_occupancy_missing_indicator": 1.0,
        "D_explicit_vacancy_annotation_missing_indicator": 1.0,
        "D_explicit_dopant_reference_missing_indicator": 1.0,
        "D_ordered_coordinate_configuration_available_indicator": 1.0,
    }
    for label, distances in mobile_distances_by_location.items():
        valid = distances[np.isfinite(distances) & (distances > 1e-10)]
        block_d["D_%s_nearest_other_mobile_A" % label] = float(np.min(valid)) if len(valid) else math.nan
        block_d["D_%s_other_mobile_effective_coordination" % label] = _effective_coordination(distances, neighbor_count)
    return {BLOCK_B: block_b, BLOCK_C: block_c, BLOCK_D: block_d}


def _metadata_row(provenance: pd.Series, cohort_id: str) -> Dict[str, Any]:
    family = normalize_structure_family(provenance["crystal_class"])
    formula_species = "%s|%s" % (provenance["reduced_formula"], provenance["migrating_species"])
    return {
        "record_id": str(provenance["record_id"]),
        "raw_index": int(provenance["raw_index"]),
        "jid": str(provenance["jid"]),
        "formula": str(provenance["formula"]),
        "reduced_formula": str(provenance["reduced_formula"]),
        "migrating_species": str(provenance["migrating_species"]),
        "barrier_eV": float(provenance["barrier_eV"]),
        "source_group": str(provenance["source_group"]),
        "XC": str(provenance["XC"]),
        "crystal_class": str(provenance["crystal_class"]),
        "structure_family_normalized": family,
        "space_group": str(provenance["space_group"]),
        "formula_species_group": formula_species,
        "is_nasicon": 1 if family == "NASICON" else 0,
        "structure_ini_hash": str(provenance["structure_ini_hash"]),
        "structure_fin_hash": str(provenance["structure_fin_hash"]),
        "raw_record_sha256": str(provenance["raw_record_sha256"]),
        "cohort_id": cohort_id,
    }


def validate_authorised_inputs(config: Dict[str, Any]) -> Tuple[Path, Path, Dict[str, str]]:
    inputs = config["inputs"]
    provenance_path = resolve_project_input(inputs["provenance_csv"])
    raw_path = resolve_project_input(inputs["raw_json"])
    authorised_raw = (ROOT.parent / "00_raw" / "scidata2025" / "EM-COMPLETE-DATASET.json").resolve()
    if raw_path != authorised_raw:
        raise PermissionError("Model pipeline raw input is not the authorised EM dataset")
    hashes = {
        "raw_json": sha256_file(raw_path),
        "provenance_csv": sha256_file(provenance_path),
    }
    if hashes["raw_json"] != inputs["expected_raw_sha256"]:
        raise RuntimeError("Raw JSON SHA256 mismatch")
    if hashes["provenance_csv"] != inputs["expected_provenance_sha256"]:
        raise RuntimeError("Provenance CSV SHA256 mismatch")
    return raw_path, provenance_path, hashes


def _validate_provenance_record(index: int, raw: Dict[str, Any], row: pd.Series) -> None:
    expected_record_id = "em_complete_%04d_jid_%s" % (index, raw.get("jid"))
    checks = {
        "raw_index": int(row["raw_index"]) == index,
        "record_id": str(row["record_id"]) == expected_record_id,
        "jid": str(row["jid"]) == str(raw.get("jid")),
        "formula": str(row["formula"]) == str(raw.get("formula")),
        "barrier_eV": abs(float(row["barrier_eV"]) - float(raw.get("target"))) <= 1e-12,
        "structure_ini_hash": str(row["structure_ini_hash"]) == sha256_json(raw["structure_ini"]),
        "structure_fin_hash": str(row["structure_fin_hash"]) == sha256_json(raw["structure_fin"]),
        "raw_record_sha256": str(row["raw_record_sha256"]) == sha256_json(raw),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError("Provenance mismatch at raw index %d: %s" % (index, failed))


def build_feature_tables(config: Dict[str, Any]) -> Dict[str, Any]:
    raw_path, provenance_path, input_hashes = validate_authorised_inputs(config)
    raw_records = json.loads(raw_path.read_text(encoding="utf-8"))
    provenance = pd.read_csv(provenance_path).sort_values("raw_index").reset_index(drop=True)
    expected_count = int(config["inputs"]["expected_record_count"])
    if len(raw_records) != expected_count or len(provenance) != expected_count:
        raise RuntimeError("Expected %d raw/provenance records" % expected_count)
    used_elements = sorted({str(element) for record in raw_records for element in record["structure_ini"]["elements"]})
    element_frame, property_lookup, element_manifest = load_neutral_element_properties(used_elements)

    strict_rows: List[Dict[str, Any]] = []
    composition_rows: List[Dict[str, Any]] = []
    coverage_rows: List[Dict[str, Any]] = []
    transitions: Dict[str, Dict[str, Any]] = {}
    block_columns: Dict[str, set] = {BLOCK_A: set(), BLOCK_B: set(), BLOCK_C: set(), BLOCK_D: set()}
    strict_config = config["cohorts"]["strict_local"]
    for index, raw in enumerate(raw_records):
        provenance_row = provenance.iloc[index]
        _validate_provenance_record(index, raw, provenance_row)
        transition = reported_transition(
            index,
            raw,
            float(strict_config["minimum_mover_displacement_A"]),
            float(config["featurizer"]["minimum_image_tie_tolerance_A"]),
            float(strict_config["lattice_relative_tolerance"]),
            "UNRESOLVED_BIBTEX_SOURCE",
        )
        transitions[str(provenance_row["record_id"])] = transition
        species_match = transition["migrating_species"] == str(provenance_row["migrating_species"])
        strict_eligible = bool(transition["graph_safe"] and species_match)
        exclusion_reason = "INCLUDED_NOT_APPLICABLE"
        if not transition["fixed_cell"]:
            exclusion_reason = "FRAME_REGISTRATION_REQUIRED_LATTICE_CHANGE"
        elif transition["moved_atom_count"] != 1:
            exclusion_reason = "FRAME_REGISTRATION_REQUIRED_MULTIPLE_MOVERS"
        elif not species_match:
            exclusion_reason = "MIGRATING_SPECIES_MISMATCH"

        metadata_all = _metadata_row(provenance_row, "composition_all619")
        composition_block = _composition_block(
            raw["structure_ini"]["elements"],
            str(provenance_row["migrating_species"]),
            used_elements,
            property_lookup,
            raw["structure_ini"]["lattice_mat"],
        )
        block_columns[BLOCK_A].update(composition_block)
        composition_rows.append(dict(metadata_all, **composition_block))

        coverage_rows.append(
            {
                "record_id": str(provenance_row["record_id"]),
                "raw_index": index,
                "jid": str(provenance_row["jid"]),
                "strict_local_eligible": 1 if strict_eligible else 0,
                "composition_all619_eligible": 1,
                "strict_local_exclusion_reason": exclusion_reason,
                "fixed_cell": 1 if transition["fixed_cell"] else 0,
                "moved_atom_count": int(transition["moved_atom_count"]),
                "derived_migrating_species": transition["migrating_species"],
                "provenance_migrating_species": str(provenance_row["migrating_species"]),
                "local_path_proxy_status": "AVAILABLE_ENDPOINT_STRAIGHT_PATH_PROXY" if strict_eligible else "UNAVAILABLE",
                "true_NEB_saddle_geometry_available": 0,
                "oxidation_state_available": 0,
                "coordination_conditioned_ionic_radius_available": 0,
                "explicit_occupancy_available": 0,
                "explicit_dopant_reference_available": 0,
                "explicit_vacancy_fraction_available": 0,
            }
        )
        if not strict_eligible:
            continue

        metadata_strict = _metadata_row(provenance_row, "strict_local")
        local_blocks = edge_feature_blocks(
            raw["structure_ini"],
            transition["migrating_species"],
            transition["start_frac"],
            transition["end_frac"],
            transition["image_shift"],
            transition["atom_index"],
            used_elements,
            property_lookup,
            config,
        )
        row = dict(metadata_strict, **composition_block)
        for block_name, values in local_blocks.items():
            block_columns[block_name].update(values)
            row.update(values)
        strict_rows.append(row)

    strict_frame = pd.DataFrame(strict_rows).sort_values("record_id").reset_index(drop=True)
    composition_frame = pd.DataFrame(composition_rows).sort_values("record_id").reset_index(drop=True)
    coverage_frame = pd.DataFrame(coverage_rows).sort_values("record_id").reset_index(drop=True)
    expected_strict = int(strict_config["expected_record_count"])
    if len(strict_frame) != expected_strict:
        raise RuntimeError("Strict-local cohort has %d records; expected %d" % (len(strict_frame), expected_strict))
    if len(composition_frame) != expected_count:
        raise RuntimeError("Composition cohort does not retain all raw records")
    feature_blocks = {name: sorted(columns) for name, columns in block_columns.items()}
    resolved_allowlist = sorted({column for columns in feature_blocks.values() for column in columns})
    for name, columns in feature_blocks.items():
        assert_no_leakage_features(columns, config, allowed_feature_columns=resolved_allowlist)
        if name != BLOCK_A and not set(columns).issubset(strict_frame.columns):
            raise RuntimeError("Feature block %s missing from strict table" % name)
    return {
        "strict": strict_frame,
        "composition": composition_frame,
        "coverage": coverage_frame,
        "feature_blocks": feature_blocks,
        "element_frame": element_frame,
        "element_manifest": element_manifest,
        "property_lookup": property_lookup,
        "used_elements": used_elements,
        "raw_records": raw_records,
        "transitions": transitions,
        "input_hashes": input_hashes,
    }


def feature_schema(
    strict_frame: pd.DataFrame,
    composition_frame: pd.DataFrame,
    feature_blocks: Dict[str, Sequence[str]],
) -> Dict[str, Any]:
    block_by_column = {
        column: block
        for block, columns in feature_blocks.items()
        for column in columns
    }
    fields = []
    all_columns = list(dict.fromkeys(list(composition_frame.columns) + list(strict_frame.columns)))
    for column in all_columns:
        if column == "barrier_eV":
            role = "target_only"
            block = "target"
        elif column in block_by_column:
            role = "model_input_allowlist"
            block = block_by_column[column]
        else:
            role = "identifier_grouping_or_audit_only"
            block = "metadata"
        nullable = bool(
            (column in strict_frame and strict_frame[column].isna().any())
            or (column in composition_frame and composition_frame[column].isna().any())
        )
        source = "authorised raw structure"
        if column.startswith("C_") and ("oxidation_state" in column or "ionic_radius" in column):
            source = "unavailable in raw data; null plus missing indicator"
        elif column.startswith("A_") or column.startswith("C_"):
            source = "authorised raw structure plus versioned neutral element-property snapshot"
        elif column.startswith("B_") or column in {"signed_clearance_A", "overlap_flag", "robust_log_crowding"}:
            source = "authorised initial structure and reported endpoints; straight-path proxy only"
        elif column.startswith("D_"):
            source = "authorised ordered initial configuration; explicit defect annotations absent"
        fields.append(
            {
                "name": column,
                "dtype": "number" if column in block_by_column or column == "barrier_eV" else "string_or_integer",
                "unit": "eV" if column == "barrier_eV" else ("angstrom" if column.endswith("_A") else "dimensionless_or_declared_in_name"),
                "nullable": nullable,
                "block": block,
                "role": role,
                "source": source,
                "availability_rule": "strict_local only" if block in {BLOCK_B, BLOCK_C, BLOCK_D} else "all records",
                "missing_code": "empty CSV cell / JSON null; never semantic zero" if nullable else "not nullable",
            }
        )
    return {
        "schema_version": "1.0",
        "fields": fields,
        "feature_blocks": {name: list(columns) for name, columns in feature_blocks.items()},
        "target": "barrier_eV",
        "path_geometry_warning": "No NEB images or saddle structures are present; bottleneck quantities are straight-line proxies.",
        "topology_policy": "Excluded from the main feature allowlist.",
    }


def match_candidate_start_atom(
    structure: Dict[str, Any],
    species: str,
    start_frac: Sequence[float],
    tolerance_A: float,
) -> Tuple[Optional[int], float]:
    indices = [index for index, element in enumerate(structure["elements"]) if str(element) == species]
    distances = _minimum_image_distances(
        start_frac,
        [structure["coords"][index] for index in indices],
        structure["lattice_mat"],
    )
    if not len(distances):
        return None, math.nan
    local_index = int(np.argmin(distances))
    distance = float(distances[local_index])
    return (indices[local_index] if distance <= tolerance_A else None), distance


def build_periodic_candidate_features(
    config: Dict[str, Any],
    feature_result: Dict[str, Any],
) -> pd.DataFrame:
    edge_path = resolve_project_input(config["inputs"]["periodic_candidate_edges_csv"])
    edges = pd.read_csv(edge_path, keep_default_na=False)
    provenance = pd.read_csv(resolve_project_input(config["inputs"]["provenance_csv"]))
    provenance_by_id = {str(row["record_id"]): row for _, row in provenance.iterrows()}
    raw_records = feature_result["raw_records"]
    lookup = feature_result["property_lookup"]
    all_symbols = feature_result["used_elements"]
    rows: List[Dict[str, Any]] = []
    tolerance = float(config["featurizer"]["candidate_start_match_tolerance_A"])
    for _, edge in edges.iterrows():
        source_ids = [value for value in str(edge["source_record_id"]).split(";") if value]
        if not source_ids:
            raise RuntimeError("Candidate edge lacks source_record_id")
        source_id = sorted(source_ids)[0]
        provenance_row = provenance_by_id[source_id]
        raw = raw_records[int(provenance_row["raw_index"])]
        species = str(provenance_row["migrating_species"])
        start = json.loads(edge["start_frac_coords"])
        end = json.loads(edge["end_frac_coords"])
        shift = [int(edge["image_shift_a"]), int(edge["image_shift_b"]), int(edge["image_shift_c"])]
        mover_index, match_distance = match_candidate_start_atom(raw["structure_ini"], species, start, tolerance)
        feature_orientation = "stored_edge_orientation"
        if mover_index is None:
            reverse_mover_index, reverse_match_distance = match_candidate_start_atom(
                raw["structure_ini"], species, end, tolerance
            )
            if reverse_mover_index is None:
                raise RuntimeError(
                    "Neither endpoint of candidate edge %s matches an occupied %s site within %.6g A"
                    % (edge["edge_uid"], species, tolerance)
                )
            start, end = end, start
            shift = [-value for value in shift]
            mover_index = reverse_mover_index
            match_distance = reverse_match_distance
            feature_orientation = "reversed_to_occupied_start"
        composition = _composition_block(
            raw["structure_ini"]["elements"], species, all_symbols, lookup, raw["structure_ini"]["lattice_mat"]
        )
        local = edge_feature_blocks(
            raw["structure_ini"], species, start, end, shift, mover_index, all_symbols, lookup, config
        )
        row: Dict[str, Any] = {
            "edge_uid": str(edge["edge_uid"]),
            "graph_id": str(edge["graph_id"]),
            "source_record_id": str(edge["source_record_id"]),
            "representative_structure_record_id": source_id,
            "migrating_species": species,
            "start_atom_match_distance_A": match_distance,
            "start_atom_removed_from_environment": 1,
            "feature_orientation": feature_orientation,
        }
        row.update(composition)
        for values in local.values():
            row.update(values)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("edge_uid").reset_index(drop=True)
