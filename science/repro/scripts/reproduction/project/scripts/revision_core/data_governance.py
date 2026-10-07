"""Build a fully traceable governance table from the authorised raw NEB dataset.

Only the configured JSON file may be read as scientific input.  Every raw record
is emitted once; quality findings are flags and never exclusion criteria.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from functools import reduce
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
SOURCE_FIELDS = (
    "jid",
    "formula",
    "target",
    "XC",
    "bibtex",
    "crystal_class",
    "sys_name",
    "space_group",
    "structure_ini",
    "structure_fin",
)
FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d+(?:\.\d+)?)?")
BIBTEX_KEY = re.compile(r"@\w+\s*\{\s*([^,\s]+)", flags=re.IGNORECASE)


def canonical_json(value: Any) -> str:
    """Return a stable serialization suitable for structural provenance hashes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def read_config(config_path: Path) -> dict[str, Any]:
    """Read JSON syntax stored in a .yaml file (valid YAML 1.2 and dependency-free)."""
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{config_path} must use JSON syntax (which is valid YAML 1.2) so the pipeline has no YAML-parser dependency."
        ) from exc


def project_path(relative_path: str) -> Path:
    path = (ROOT / relative_path).resolve()
    try:
        path.relative_to(ROOT.parent)
    except ValueError:
        raise ValueError(f"Path resolves outside permitted project parent: {relative_path}")
    return path


def require_authorised_raw_dataset(config: dict[str, Any]) -> tuple[Path, str]:
    raw_path = project_path(config["input"]["allowed_raw_dataset"])
    expected = (ROOT.parent / "00_raw" / "scidata2025" / "EM-COMPLETE-DATASET.json").resolve()
    if raw_path != expected:
        raise ValueError(f"Raw input is not the sole authorised dataset: {raw_path}")
    if not raw_path.is_file():
        raise FileNotFoundError(f"Authorised raw dataset does not exist: {raw_path}")
    actual_hash = sha256_file(raw_path)
    if actual_hash.lower() != config["input"]["expected_sha256"].lower():
        raise ValueError("Authorised raw dataset SHA256 does not match revision_config.yaml.")
    return raw_path, actual_hash


def value_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (dict, list, tuple)):
        return bool(value)
    return True


def numeric_matrix(value: Any, rows: int, columns: int) -> bool:
    if not isinstance(value, list) or len(value) != rows:
        return False
    try:
        return all(isinstance(row, list) and len(row) == columns and all(math.isfinite(float(x)) for x in row) for row in value)
    except (TypeError, ValueError):
        return False


def valid_structure(structure: Any) -> bool:
    if not isinstance(structure, dict):
        return False
    elements = structure.get("elements")
    coords = structure.get("coords")
    if not isinstance(elements, list) or not elements or not isinstance(coords, list) or len(coords) != len(elements):
        return False
    if not all(isinstance(element, str) and element.strip() for element in elements):
        return False
    return numeric_matrix(coords, len(elements), 3) and numeric_matrix(structure.get("lattice_mat"), 3, 3)


def reduced_formula(formula: str) -> str:
    compact = formula.replace(" ", "")
    tokens = list(FORMULA_TOKEN.finditer(compact))
    if not compact or not tokens or "".join(token.group(0) for token in tokens) != compact:
        raise ValueError(f"Unsupported formula syntax: {formula!r}")
    elements: list[str] = []
    amounts: list[Fraction] = []
    for token in tokens:
        element, count = token.groups()
        amount = Fraction(count) if count else Fraction(1)
        elements.append(element)
        amounts.append(amount)
    denominator_lcm = 1
    for amount in amounts:
        denominator_lcm = abs(denominator_lcm * amount.denominator) // math.gcd(
            denominator_lcm, amount.denominator
        )
    integer_amounts = [int(amount * denominator_lcm) for amount in amounts]
    common_divisor = reduce(math.gcd, integer_amounts)
    integer_amounts = [amount // common_divisor for amount in integer_amounts]
    return "".join(element + ("" if amount == 1 else str(amount)) for element, amount in zip(elements, integer_amounts))


def source_group_from_bibtex(bibtex: str, fallback: str) -> tuple[str, bool]:
    normalized = (bibtex or "").strip()
    if normalized.upper() == "SELF":
        return "SELF", True
    match = BIBTEX_KEY.search(normalized)
    return (match.group(1), True) if match else (fallback, False)


def periodic_cartesian_displacements(structure_ini: dict[str, Any], structure_fin: dict[str, Any]) -> list[float]:
    lattice = structure_ini["lattice_mat"]
    distances = []
    for initial, final in zip(structure_ini["coords"], structure_fin["coords"]):
        fractional = [float(final[axis]) - float(initial[axis]) for axis in range(3)]
        wrapped = [value - math.floor(value + 0.5) for value in fractional]
        cartesian = [sum(wrapped[row] * float(lattice[row][column]) for row in range(3)) for column in range(3)]
        distances.append(math.sqrt(sum(component * component for component in cartesian)))
    return distances


def identify_migrating_species(
    structure_ini: Any,
    structure_fin: Any,
    minimum_displacement: float,
) -> tuple[str, float | None, list[str]]:
    """Infer the species at the largest periodic initial-to-final displacement."""
    if not valid_structure(structure_ini) or not valid_structure(structure_fin):
        return "UNRESOLVED", None, ["MIGRATING_SPECIES_UNRESOLVED"]
    if structure_ini["elements"] != structure_fin["elements"] or len(structure_ini["coords"]) != len(structure_fin["coords"]):
        return "UNRESOLVED", None, ["STRUCTURE_ELEMENT_ORDER_MISMATCH", "MIGRATING_SPECIES_UNRESOLVED"]
    distances = periodic_cartesian_displacements(structure_ini, structure_fin)
    maximum = max(distances, default=0.0)
    if not math.isfinite(maximum) or maximum <= minimum_displacement:
        return "UNRESOLVED", maximum, ["MIGRATING_SPECIES_UNRESOLVED"]
    tolerance = max(1e-12, maximum * 1e-9)
    maximum_indices = [index for index, value in enumerate(distances) if abs(value - maximum) <= tolerance]
    species = sorted({structure_ini["elements"][index] for index in maximum_indices})
    if len(species) != 1:
        return "UNRESOLVED", maximum, ["MIGRATING_SPECIES_AMBIGUOUS"]
    return species[0], maximum, []


def interpolated_quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def barrier_fences(records: list[dict[str, Any]]) -> tuple[float, float, dict[int, float | None], dict[int, str | None]]:
    parsed: dict[int, float | None] = {}
    errors: dict[int, str | None] = {}
    values = []
    for index, record in enumerate(records):
        try:
            barrier = float(record.get("target"))
        except (TypeError, ValueError):
            parsed[index] = None
            errors[index] = "BARRIER_NOT_NUMERIC"
            continue
        if not math.isfinite(barrier):
            parsed[index] = barrier
            errors[index] = "BARRIER_NOT_FINITE"
        elif barrier < 0:
            parsed[index] = barrier
            errors[index] = "BARRIER_NEGATIVE"
        else:
            parsed[index] = barrier
            errors[index] = None
            values.append(barrier)
    if not values:
        raise ValueError("No finite nonnegative barrier values are available for the audit.")
    q1 = interpolated_quantile(values, 0.25)
    q3 = interpolated_quantile(values, 0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr, parsed, errors


def append_flag(flags: list[str], flag: str) -> None:
    if flag not in flags:
        flags.append(flag)


def compact_float(value: float | None) -> str:
    return "" if value is None else repr(value)


def make_rows(records: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    lower_fence, upper_fence, parsed_barriers, barrier_errors = barrier_fences(records)
    minimum_displacement = float(config["quality_rules"]["minimum_migration_displacement_A"])
    rows: list[dict[str, str]] = []
    raw_by_index: dict[int, dict[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Raw record at index {index} is not a JSON object.")
        raw_by_index[index] = record
        flags: list[str] = []
        missing = [field for field in SOURCE_FIELDS if not value_present(record.get(field))]
        if missing:
            append_flag(flags, "MISSING_METADATA_" + "_".join(missing))
        initial = record.get("structure_ini")
        final = record.get("structure_fin")
        if not value_present(initial):
            append_flag(flags, "MISSING_STRUCTURE_INI")
        elif not valid_structure(initial):
            append_flag(flags, "INVALID_STRUCTURE_INI")
        if not value_present(final):
            append_flag(flags, "MISSING_STRUCTURE_FIN")
        elif not valid_structure(final):
            append_flag(flags, "INVALID_STRUCTURE_FIN")
        barrier = parsed_barriers[index]
        if barrier_errors[index]:
            append_flag(flags, str(barrier_errors[index]))
        elif barrier is not None and (barrier < lower_fence or barrier > upper_fence):
            append_flag(flags, "BARRIER_IQR_OUTLIER")
        try:
            canonical_formula = reduced_formula(str(record.get("formula", "")))
        except ValueError:
            canonical_formula = "UNRESOLVED"
            append_flag(flags, "FORMULA_REDUCTION_FAILED")
        migrating_species, displacement, migration_flags = identify_migrating_species(initial, final, minimum_displacement)
        for flag in migration_flags:
            append_flag(flags, flag)
        source_group, source_resolved = source_group_from_bibtex(
            str(record.get("bibtex", "")), config["source_group"]["fallback"]
        )
        if not source_resolved:
            append_flag(flags, "MISSING_BIBTEX_CITATION_KEY")
        present_count = sum(value_present(record.get(field)) for field in SOURCE_FIELDS)
        jid = record.get("jid", "UNRESOLVED")
        rows.append(
            {
                "record_id": f"em_complete_{index:04d}_jid_{jid}",
                "raw_index": str(index),
                "jid": str(jid),
                "formula": str(record.get("formula", "")),
                "reduced_formula": canonical_formula,
                "migrating_species": migrating_species,
                "barrier_eV": compact_float(barrier),
                "XC": str(record.get("XC", "")),
                "bibtex": str(record.get("bibtex", "")),
                "source_group": source_group,
                "crystal_class": str(record.get("crystal_class", "")),
                "sys_name": str(record.get("sys_name", "")),
                "space_group": str(record.get("space_group", "")),
                "structure_ini_hash": sha256_json(initial) if valid_structure(initial) else "",
                "structure_fin_hash": sha256_json(final) if valid_structure(final) else "",
                "metadata_completeness": f"{present_count / len(SOURCE_FIELDS):.3f}",
                "duplicate_group_id": "",
                "quality_flag": ";".join(flags) if flags else "OK",
                "migration_displacement_A": compact_float(displacement),
                "raw_record_sha256": sha256_json(record),
            }
        )

    def add_groups(key_name: str, group_prefix: str, flag: str) -> None:
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            if row[key_name]:
                groups[row[key_name]].append(row)
        for key, grouped_rows in groups.items():
            if len(grouped_rows) <= 1:
                continue
            group_id = f"{group_prefix}_{key[:16]}"
            for row in grouped_rows:
                row["duplicate_group_id"] = ";".join(filter(None, [row["duplicate_group_id"], group_id]))
                current_flags = [] if row["quality_flag"] == "OK" else row["quality_flag"].split(";")
                append_flag(current_flags, flag)
                row["quality_flag"] = ";".join(current_flags)

    add_groups("structure_ini_hash", "initial_structure", "REPEATED_INITIAL_STRUCTURE")
    add_groups("structure_fin_hash", "final_structure", "REPEATED_FINAL_STRUCTURE")
    transition_groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        transition_groups[(row["structure_ini_hash"], row["structure_fin_hash"], row["barrier_eV"])].append(row)
    for key, grouped_rows in transition_groups.items():
        if key[0] and len(grouped_rows) > 1:
            group_id = "transition_" + sha256_json(key)[:16]
            for row in grouped_rows:
                row["duplicate_group_id"] = ";".join(filter(None, [row["duplicate_group_id"], group_id]))
                current_flags = [] if row["quality_flag"] == "OK" else row["quality_flag"].split(";")
                append_flag(current_flags, "DUPLICATE_TRANSITION_RECORD")
                row["quality_flag"] = ";".join(current_flags)
    jid_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        jid_groups[row["jid"]].append(row)
    for jid, grouped_rows in jid_groups.items():
        if len(grouped_rows) > 1:
            for row in grouped_rows:
                current_flags = [] if row["quality_flag"] == "OK" else row["quality_flag"].split(";")
                append_flag(current_flags, "DUPLICATE_JID")
                row["quality_flag"] = ";".join(current_flags)

    audit_context = {
        "barrier_iqr_lower_fence_eV": lower_fence,
        "barrier_iqr_upper_fence_eV": upper_fence,
        "raw_by_index": raw_by_index,
    }
    return rows, audit_context


def field_schema(name: str, artifact: str = "") -> dict[str, Any]:
    descriptions = {
        "record_id": "Stable identifier generated from raw array index and raw jid.",
        "raw_index": "Zero-based index in the authorised raw JSON list.",
        "jid": "Unmodified raw record jid.",
        "formula": "Unmodified raw formula.",
        "reduced_formula": "Formula reduced by greatest common divisor without changing element order.",
        "migrating_species": "Species inferred from largest periodic initial-to-final displacement.",
        "barrier_eV": "Unmodified raw target interpreted as migration barrier in eV.",
        "XC": "Unmodified raw exchange-correlation label; governance/sensitivity use only.",
        "bibtex": "Unmodified raw BibTeX provenance text.",
        "source_group": "BibTeX citation key derived from bibtex; governance/sensitivity use only.",
        "crystal_class": "Unmodified raw crystal class.",
        "sys_name": "Unmodified raw system name.",
        "space_group": "Unmodified raw space group.",
        "structure_ini_hash": "SHA256 of canonical initial-structure JSON.",
        "structure_fin_hash": "SHA256 of canonical final-structure JSON.",
        "metadata_completeness": "Fraction of ten required raw fields that are present.",
        "duplicate_group_id": "One or more IDs for reused initial/final structures or exact duplicate transitions.",
        "quality_flag": "Semicolon-delimited review flags. OK means no governance finding.",
        "migration_displacement_A": "Largest periodic initial-to-final site displacement used only for species identification.",
        "raw_record_sha256": "SHA256 of the complete canonical raw record JSON.",
        "review_reason": "Quality flag that requires a manual review.",
        "review_guidance": "Non-exclusion guidance for manual review.",
        "record_count": "Number of raw records in the stated audit category.",
        "edge_uid": "Stable SHA256-derived identifier for one canonical undirected periodic edge.",
        "graph_id": "Stable SHA256-derived identifier for one material/composition-specific host graph.",
        "node_id": "Graph-local mobile-site identifier.",
        "node_i": "Canonical start-node identifier for the stored edge orientation.",
        "node_j": "Canonical end-node identifier for the stored edge orientation.",
        "image_shift_a": "Integer a-lattice image shift applied to node_j relative to node_i.",
        "image_shift_b": "Integer b-lattice image shift applied to node_j relative to node_i.",
        "image_shift_c": "Integer c-lattice image shift applied to node_j relative to node_i.",
        "start_frac_coords": "Fractional coordinates of node_i, encoded as a JSON length-3 array.",
        "end_frac_coords": "Fractional coordinates of node_j before image translation, encoded as a JSON length-3 array.",
        "frac_coords": "Wrapped fractional coordinates of a mobile-site node, encoded as a JSON length-3 array.",
        "cart_coords": "Cartesian coordinates in angstrom of a mobile-site node, encoded as a JSON length-3 array.",
        "displacement_frac": "Periodic edge displacement end + image_shift - start, encoded as a JSON length-3 array.",
        "displacement_cart": "Cartesian edge displacement in angstrom, encoded as a JSON length-3 array.",
        "lattice_matrix": "Three row lattice vectors in angstrom, encoded as a JSON 3-by-3 array.",
        "hop_distance_A": "Euclidean norm of displacement_cart in angstrom.",
        "edge_source": "Algorithmic provenance of the candidate edge and any reported-path association.",
        "node_source": "Algorithmic provenance of the mobile-site node.",
        "source_record_id": "Semicolon-delimited authorised raw record IDs associated with this node or edge.",
        "is_labeled": "Whether at least one authorised raw NEB record maps to this edge (1 or 0).",
        "forced_reported_edge": "Whether a reported NEB path was retained despite exceeding the empirical candidate cutoff (1 or 0).",
        "reported_barrier_eV": "Reported raw NEB barrier provenance in eV; representation is artifact-specific.",
        "predicted_barrier_eV": "Traceable model-predicted barrier in eV; blank until a separately validated prediction pipeline supplies it.",
        "candidate_distance_threshold_A": "Empirical candidate-hop distance cutoff in angstrom.",
        "candidate_threshold_source": "Rule and sample count used to derive the candidate distance cutoff.",
        "source_atom_indices": "Sorted raw structure atom indices contributing to this node, encoded as a JSON array.",
        "source_structure_ini_hashes": "Sorted initial-structure SHA256 values contributing to this graph node, encoded as a JSON array.",
        "selection_rank": "One-based deterministic real-graph pilot selection rank.",
        "selection_rule": "Deterministic data-driven rule used to select the pilot graph.",
        "n_nodes": "Number of quotient-graph mobile-site nodes.",
        "n_edges": "Number of canonical undirected periodic candidate edges.",
        "n_labeled_edges": "Number of candidate edges associated with at least one reported raw NEB path.",
        "n_forced_reported_edges": "Number of reported edges retained beyond the empirical distance cutoff.",
        "candidate_image_shift_radius": "Automatically derived integer image-search radius used for candidate enumeration.",
        "candidate_image_shift_radius_plus_one_verified": "Whether radius+1 produced no additional cutoff-qualified canonical edges (1 or 0).",
        "predicted_barrier_coverage": "Fraction of graph edges with a traceable predicted barrier.",
        "Eperc_a": "Released periodic a-direction predicted-barrier threshold in eV; blank unless 3x3x3, 5x5x5, and exact winding tests agree.",
        "Eperc_b": "Released periodic b-direction predicted-barrier threshold in eV; blank unless 3x3x3, 5x5x5, and exact winding tests agree.",
        "Eperc_c": "Released periodic c-direction predicted-barrier threshold in eV; blank unless 3x3x3, 5x5x5, and exact winding tests agree.",
        "Eperc_a_3x3x3": "a-direction threshold from the 3x3x3 validation supercell in eV.",
        "Eperc_b_3x3x3": "b-direction threshold from the 3x3x3 validation supercell in eV.",
        "Eperc_c_3x3x3": "c-direction threshold from the 3x3x3 validation supercell in eV.",
        "Eperc_a_5x5x5": "a-direction threshold from the 5x5x5 validation supercell in eV.",
        "Eperc_b_5x5x5": "b-direction threshold from the 5x5x5 validation supercell in eV.",
        "Eperc_c_5x5x5": "c-direction threshold from the 5x5x5 validation supercell in eV.",
        "Eperc_a_exact": "a-direction threshold from exact integer winding-lattice membership in eV.",
        "Eperc_b_exact": "b-direction threshold from exact integer winding-lattice membership in eV.",
        "Eperc_c_exact": "c-direction threshold from exact integer winding-lattice membership in eV.",
        "periodic_dimensionality": "0D, 1D, 2D, or 3D from the maximum component winding-vector rank.",
        "winding_vector_rank": "Maximum integer winding-vector rank among connected quotient-graph components.",
        "global_winding_union_rank": "Audit-only rank after pooling windings from disconnected components; not used as dimensionality.",
        "critical_periodic_edges": "JSON array of direction-critical and/or winding-rank-critical edge descriptors.",
        "redundancy_count": "Cycle-space redundancy after subtracting independent periodic winding rank component-wise.",
        "supercell_converged": "Whether 3x3x3, 5x5x5, and exact winding thresholds agree in all directions (1 or 0).",
        "readout_status": "Periodic readout status, including incomplete-prediction and nonconvergence blockers.",
        "lattice_relative_change_max": "Maximum initial-to-final lattice relative change among records assigned to a graph.",
        "reported_start_frac_coords": "Raw reported-path start fractional coordinates, encoded as a JSON length-3 array.",
        "reported_end_frac_coords": "Raw reported-path end fractional coordinates, encoded as a JSON length-3 array.",
        "reported_image_shift": "Metric-minimizing periodic image shift, encoded as a JSON integer length-3 array.",
        "raw_image_hint": "Image shift implied only by raw out-of-cell coordinate offsets, encoded as a JSON integer length-3 array.",
        "image_shift_degenerate": "Whether more than one lattice shift has the same minimum hop distance within tolerance (1 or 0).",
        "minimum_image_count": "Number of tied metric-minimum lattice images within tolerance.",
        "reported_hop_distance_A": "Metric-minimum reported start-to-end hop distance in angstrom.",
        "lattice_relative_change": "Initial-to-final lattice relative change for the raw reported path.",
    }
    integers = {
        "raw_index", "jid", "image_shift_a", "image_shift_b", "image_shift_c",
        "selection_rank", "n_nodes", "n_edges", "n_labeled_edges", "n_forced_reported_edges",
        "candidate_image_shift_radius", "winding_vector_rank", "global_winding_union_rank",
        "redundancy_count", "minimum_image_count",
    }
    numbers = {
        "barrier_eV", "metadata_completeness", "migration_displacement_A", "hop_distance_A",
        "predicted_barrier_eV", "candidate_distance_threshold_A", "predicted_barrier_coverage",
        "Eperc_a", "Eperc_b", "Eperc_c", "Eperc_a_3x3x3", "Eperc_b_3x3x3", "Eperc_c_3x3x3",
        "Eperc_a_5x5x5", "Eperc_b_5x5x5", "Eperc_c_5x5x5", "Eperc_a_exact", "Eperc_b_exact",
        "Eperc_c_exact", "lattice_relative_change_max", "reported_hop_distance_A",
        "lattice_relative_change",
    }
    booleans = {
        "is_labeled", "forced_reported_edge", "candidate_image_shift_radius_plus_one_verified",
        "supercell_converged", "image_shift_degenerate",
    }
    arrays = {
        "start_frac_coords", "end_frac_coords", "frac_coords", "cart_coords", "displacement_frac",
        "displacement_cart", "lattice_matrix", "source_atom_indices",
        "source_structure_ini_hashes", "critical_periodic_edges", "reported_start_frac_coords",
        "reported_end_frac_coords", "reported_image_shift", "raw_image_hint",
    }
    json_objects = set()
    if name == "reported_barrier_eV" and artifact == "periodic_trial_edges.csv":
        json_objects.add(name)
        descriptions[name] = "JSON object mapping each authorised raw record_id to its reported NEB barrier in eV; empty object for unlabeled candidates."
    elif name == "reported_barrier_eV":
        numbers.add(name)
        descriptions[name] = "Raw NEB barrier for this reported path in eV."
    logical_type = "string"
    if name in integers:
        logical_type = "integer"
    elif name in numbers:
        logical_type = "number"
    elif name in booleans:
        logical_type = "boolean"
    elif name in arrays:
        logical_type = "array"
    elif name in json_objects:
        logical_type = "object"
    schema: dict[str, Any] = {
        "name": name,
        "type": logical_type,
        "description": descriptions.get(name, ""),
    }
    if name in arrays or name in json_objects:
        schema["csv_encoding"] = "JSON text"
    if name.startswith("Eperc_") or name in {
        "predicted_barrier_eV", "periodic_dimensionality", "winding_vector_rank",
        "global_winding_union_rank", "redundancy_count",
    }:
        schema["nullable"] = True
    return schema


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv_artifacts(
    output_dir: Path,
    filename: str,
    rows: list[dict[str, str]],
    columns: list[str],
    source_hash: str,
    audit_payload: dict[str, Any],
) -> dict[str, Any]:
    csv_path = output_dir / filename
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    csv_hash = sha256_file(csv_path)
    schema_path = output_dir / f"{csv_path.stem}.schema.json"
    checksum_path = output_dir / f"{csv_path.stem}.sha256.json"
    audit_path = output_dir / f"{csv_path.stem}.audit.json"
    write_json(
        schema_path,
        {
            "schema_version": "1.0",
            "artifact": csv_path.name,
            "row_count": len(rows),
            "columns": [field_schema(column, filename) for column in columns],
            "source_raw_dataset_sha256": source_hash,
        },
    )
    write_json(
        checksum_path,
        {
            "algorithm": "SHA256",
            "artifact": csv_path.name,
            "sha256": csv_hash,
            "bytes": csv_path.stat().st_size,
            "row_count": len(rows),
            "source_raw_dataset_sha256": source_hash,
        },
    )
    write_json(audit_path, audit_payload)
    return {
        "csv": csv_path,
        "schema": schema_path,
        "checksum": checksum_path,
        "audit": audit_path,
        "sha256": csv_hash,
        "row_count": len(rows),
    }


def review_guidance(flag: str) -> str:
    if flag == "BARRIER_IQR_OUTLIER":
        return "Verify the reported NEB barrier and its units against the cited source; this is a statistical review flag, not an exclusion."
    if flag.startswith("REPEATED_"):
        return "Compare the structures, route definition, and NEB endpoint context; reused structures can represent legitimate distinct transitions."
    if flag.startswith("DUPLICATE_"):
        return "Confirm whether records represent an intended replicate or an accidental duplicate."
    if "STRUCTURE" in flag or "MIGRATING" in flag:
        return "Inspect the raw initial/final structure payload and migration assignment before any downstream use."
    return "Inspect the original raw record and preserve the finding in the provenance table."


def quality_counts(rows: Iterable[dict[str, str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        if row["quality_flag"] == "OK":
            continue
        counts.update(row["quality_flag"].split(";"))
    return counts


def count_rows(counter: Counter[str], category_column: str) -> list[dict[str, str]]:
    return [
        {category_column: category, "record_count": str(count)}
        for category, count in sorted(counter.items())
    ]


def build_audit(
    rows: list[dict[str, str]],
    source_path: Path,
    source_hash: str,
    audit_context: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    flags = quality_counts(rows)
    review_rows = []
    for row in rows:
        if row["quality_flag"] == "OK":
            continue
        for flag in row["quality_flag"].split(";"):
            review_rows.append(
                {
                    "record_id": row["record_id"],
                    "raw_index": row["raw_index"],
                    "jid": row["jid"],
                    "formula": row["formula"],
                    "migrating_species": row["migrating_species"],
                    "barrier_eV": row["barrier_eV"],
                    "duplicate_group_id": row["duplicate_group_id"],
                    "review_reason": flag,
                    "review_guidance": review_guidance(flag),
                }
            )
    records_requiring_review = sorted({row["record_id"] for row in review_rows})
    audit = {
        "audit_schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "data governance only; no model training, screening, or figure generation",
        "input": {
            "path": str(source_path),
            "sha256": source_hash,
            "record_count": len(rows),
            "sole_allowed_scientific_input": config["input"]["allowed_raw_dataset"],
        },
        "quality_rule_parameters": {
            "barrier_outlier_method": config["quality_rules"]["barrier_outlier_method"],
            "barrier_iqr_lower_fence_eV": audit_context["barrier_iqr_lower_fence_eV"],
            "barrier_iqr_upper_fence_eV": audit_context["barrier_iqr_upper_fence_eV"],
            "minimum_migration_displacement_A": config["quality_rules"]["minimum_migration_displacement_A"],
        },
        "counts": {
            "by_XC": dict(sorted(Counter(row["XC"] for row in rows).items())),
            "by_source_group": dict(sorted(Counter(row["source_group"] for row in rows).items())),
            "by_migrating_species": dict(sorted(Counter(row["migrating_species"] for row in rows).items())),
            "quality_flags": dict(sorted(flags.items())),
            "metadata_incomplete_records": sum(float(row["metadata_completeness"]) < 1.0 for row in rows),
            "missing_or_invalid_structure_records": sum(
                any(token in row["quality_flag"] for token in ("MISSING_STRUCTURE", "INVALID_STRUCTURE", "STRUCTURE_ELEMENT_ORDER_MISMATCH"))
                for row in rows
            ),
            "migrating_species_unresolved_records": flags["MIGRATING_SPECIES_UNRESOLVED"] + flags["MIGRATING_SPECIES_AMBIGUOUS"],
            "formula_reduction_failure_records": flags["FORMULA_REDUCTION_FAILED"],
            "invalid_barrier_records": sum(
                any(token in row["quality_flag"] for token in ("BARRIER_NOT_NUMERIC", "BARRIER_NOT_FINITE", "BARRIER_NEGATIVE"))
                for row in rows
            ),
            "barrier_iqr_outlier_records": flags["BARRIER_IQR_OUTLIER"],
            "exact_duplicate_transition_records": flags["DUPLICATE_TRANSITION_RECORD"],
            "repeated_initial_structure_records": flags["REPEATED_INITIAL_STRUCTURE"],
            "repeated_final_structure_records": flags["REPEATED_FINAL_STRUCTURE"],
            "records_requiring_manual_confirmation": len(records_requiring_review),
            "manual_review_rows": len(review_rows),
        },
        "feature_governance": config["feature_governance"],
        "manual_review_record_ids": records_requiring_review,
        "invariants": {
            "all_raw_records_retained": len(rows) == config["input"]["expected_record_count"],
            "jid_is_unique_in_output": len({row["jid"] for row in rows}) == len(rows),
            "quality_flags_do_not_drop_records": True,
        },
    }
    return audit, review_rows


def write_markdown_audit(path: Path, audit: dict[str, Any], artifacts: dict[str, dict[str, Any]]) -> None:
    counts = audit["counts"]
    lines = [
        "# Revision data-governance audit",
        "",
        "This report is a data-governance artifact only. It does not train a model or generate a figure.",
        "",
        f"- Raw records retained: {audit['input']['record_count']}",
        f"- Raw SHA256: `{audit['input']['sha256']}`",
        f"- Incomplete metadata: {counts['metadata_incomplete_records']}",
        f"- Missing or invalid structures: {counts['missing_or_invalid_structure_records']}",
        f"- Unresolved or ambiguous migrating species: {counts['migrating_species_unresolved_records']}",
        f"- Invalid barriers: {counts['invalid_barrier_records']}",
        f"- IQR review barriers: {counts['barrier_iqr_outlier_records']}",
        f"- Exact duplicate transitions: {counts['exact_duplicate_transition_records']}",
        f"- Records requiring manual confirmation: {counts['records_requiring_manual_confirmation']}",
        "",
        "## Output checksums",
        "",
    ]
    for artifact_name, artifact in artifacts.items():
        lines.append(f"- `{artifact_name}`: `{artifact['sha256']}` ({artifact['row_count']} rows)")
    lines.extend(
        [
            "",
            "## Feature governance",
            "",
            "XC, source group, metadata completeness, and BibTeX provenance are retained for quality/sensitivity analysis. They are forbidden as primary new-material screening inputs.",
            "",
            "The JSON audit file contains complete XC, source-group, migrating-species, flag, and manual-review lists.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(config_path: Path | None = None) -> dict[str, Any]:
    config_path = (config_path or ROOT / "config" / "revision_config.yaml").resolve()
    config = read_config(config_path)
    raw_path, raw_hash = require_authorised_raw_dataset(config)
    records = json.loads(raw_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Authorised raw dataset must be a JSON list.")
    if len(records) != int(config["input"]["expected_record_count"]):
        raise ValueError(f"Expected {config['input']['expected_record_count']} raw records, found {len(records)}.")
    rows, audit_context = make_rows(records, config)
    columns = list(config["required_columns"])
    if any(set(row) != set(columns) for row in rows):
        raise ValueError("Generated row columns do not match the configured provenance schema.")
    output_dir = project_path(config["outputs"]["source_data_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    audit, review_rows = build_audit(rows, raw_path, raw_hash, audit_context, config)
    labeled_audit = {
        "artifact_scope": "all raw records, one output row per raw record",
        "row_count": len(rows),
        "input_raw_sha256": raw_hash,
        "quality_flag_counts": audit["counts"]["quality_flags"],
    }
    review_columns = [
        "record_id",
        "raw_index",
        "jid",
        "formula",
        "migrating_species",
        "barrier_eV",
        "duplicate_group_id",
        "review_reason",
        "review_guidance",
    ]
    artifacts = {
        config["outputs"]["labeled_edges_filename"]: write_csv_artifacts(
            output_dir,
            config["outputs"]["labeled_edges_filename"],
            rows,
            columns,
            raw_hash,
            labeled_audit,
        ),
        config["outputs"]["manual_review_filename"]: write_csv_artifacts(
            output_dir,
            config["outputs"]["manual_review_filename"],
            review_rows,
            review_columns,
            raw_hash,
            {
                "artifact_scope": "one row per quality finding requiring manual confirmation",
                "row_count": len(review_rows),
                "unique_record_count": audit["counts"]["records_requiring_manual_confirmation"],
                "input_raw_sha256": raw_hash,
            },
        ),
    }
    count_artifacts = (
        ("xc_counts_filename", Counter(row["XC"] for row in rows), "XC"),
        ("source_group_counts_filename", Counter(row["source_group"] for row in rows), "source_group"),
        ("migrating_species_counts_filename", Counter(row["migrating_species"] for row in rows), "migrating_species"),
    )
    for filename_key, counter, category_column in count_artifacts:
        filename = config["outputs"][filename_key]
        artifacts[filename] = write_csv_artifacts(
            output_dir,
            filename,
            count_rows(counter, category_column),
            [category_column, "record_count"],
            raw_hash,
            {
                "artifact_scope": f"record counts grouped by {category_column}",
                "row_count": len(counter),
                "summed_record_count": sum(counter.values()),
                "input_raw_sha256": raw_hash,
            },
        )
    audit["artifacts"] = {
        name: {
            "csv": str(value["csv"].relative_to(ROOT)),
            "schema": str(value["schema"].relative_to(ROOT)),
            "checksum": str(value["checksum"].relative_to(ROOT)),
            "audit": str(value["audit"].relative_to(ROOT)),
            "sha256": value["sha256"],
            "row_count": value["row_count"],
        }
        for name, value in artifacts.items()
    }
    source_data_manifest_path = output_dir / "revision_source_data_manifest.json"
    write_json(
        source_data_manifest_path,
        {
            "manifest_schema_version": "1.0",
            "purpose": "SHA256 manifest for revision data-governance source-data CSV artifacts",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "authorised_raw_dataset": {
                "path": str(raw_path),
                "sha256": raw_hash,
                "record_count": len(rows),
            },
            "generator": {
                "config": str(config_path.relative_to(ROOT)),
                "config_sha256": sha256_file(config_path),
                "script": str(Path(__file__).relative_to(ROOT)),
                "script_sha256": sha256_file(Path(__file__)),
            },
            "csv_artifacts": audit["artifacts"],
        },
    )
    audit["source_data_manifest"] = str(source_data_manifest_path.relative_to(ROOT))
    audit_json_path = output_dir / "data_governance_audit.json"
    audit_markdown_path = output_dir / "data_governance_audit.md"
    write_json(audit_json_path, audit)
    write_markdown_audit(audit_markdown_path, audit, artifacts)
    log_dir = project_path(config["outputs"]["log_directory"])
    log_dir.mkdir(parents=True, exist_ok=True)
    run_summary = {
        "pipeline": config["pipeline_name"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "input_sha256": raw_hash,
        "record_count": len(rows),
        "audit_path": str(audit_json_path.relative_to(ROOT)),
        "artifacts": audit["artifacts"],
    }
    write_json(log_dir / "revision_data_governance_run.json", run_summary)
    (log_dir / "revision_data_governance_run.log").write_text(
        f"status=passed\nrecord_count={len(rows)}\ninput_sha256={raw_hash}\naudit={audit_json_path.relative_to(ROOT)}\n",
        encoding="utf-8",
    )
    return audit
