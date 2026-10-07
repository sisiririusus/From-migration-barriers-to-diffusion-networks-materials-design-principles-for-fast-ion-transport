"""Version-locked neutral elemental properties for revision model features.

The raw migration-barrier data do not contain oxidation states or coordination
numbers.  This module therefore exposes neutral elemental properties only.  It
deliberately does not convert Bokeh's context-free ``ion radius`` strings into
Shannon-like radii or treat their parenthetical charge as a structure valence.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from revision_core.data_governance import sha256_file


NEUTRAL_PROPERTY_COLUMNS = [
    "atomic_number",
    "atomic_mass_amu",
    "electronegativity_pauling",
    "atomic_radius_A",
    "first_ionization_kJmol",
    "electron_affinity_kJmol",
    "period",
    "group",
    "is_metal",
]


def _numeric(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        numeric = float(value)
        return numeric if math.isfinite(numeric) else math.nan
    except (TypeError, ValueError):
        match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
        return float(match.group(0)) if match else math.nan


def _metal_indicator(category: Any) -> float:
    text = str(category).strip().lower()
    metallic = {
        "alkali metal",
        "alkaline earth metal",
        "metal",
        "transition metal",
        "lanthanoid",
        "actinoid",
    }
    return 1.0 if text in metallic else 0.0


def load_neutral_element_properties(
    used_elements: Iterable[str],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], Dict[str, Any]]:
    """Load and normalize the installed, versioned Bokeh periodic table."""
    import bokeh
    from bokeh.sampledata import periodic_table
    from bokeh.sampledata.periodic_table import elements

    requested = sorted(set(str(symbol) for symbol in used_elements))
    available = set(elements["symbol"].astype(str))
    missing_symbols = sorted(set(requested) - available)
    if missing_symbols:
        raise ValueError("Element-property table lacks symbols: %s" % missing_symbols)

    rows: List[Dict[str, Any]] = []
    lookup: Dict[str, Dict[str, float]] = {}
    for symbol in requested:
        raw = elements[elements["symbol"].astype(str).eq(symbol)].iloc[0]
        row = {
            "symbol": symbol,
            "atomic_number": _numeric(raw["atomic number"]),
            "atomic_mass_amu": _numeric(raw["atomic mass"]),
            "electronegativity_pauling": _numeric(raw["electronegativity"]),
            "atomic_radius_A": _numeric(raw["atomic radius"]) / 100.0,
            "first_ionization_kJmol": _numeric(raw["IE-1"]),
            "electron_affinity_kJmol": _numeric(raw["EA"]),
            "period": _numeric(raw["period"]),
            "group": _numeric(raw["group"]),
            "is_metal": _metal_indicator(raw["metal"]),
            "source_category": str(raw["metal"]),
            "raw_ion_radius_not_used": "" if pd.isna(raw["ion radius"]) else str(raw["ion radius"]),
            "oxidation_state_available": 0,
            "coordination_conditioned_ionic_radius_available": 0,
        }
        rows.append(row)
        lookup[symbol] = {name: float(row[name]) for name in NEUTRAL_PROPERTY_COLUMNS}

    frame = pd.DataFrame(rows).sort_values("atomic_number").reset_index(drop=True)
    source_csv = Path(periodic_table.__file__).resolve().parent / "_data" / "elements.csv"
    source_module = Path(periodic_table.__file__).resolve()
    manifest = {
        "provider": "bokeh.sampledata.periodic_table.elements",
        "provider_version": str(bokeh.__version__),
        "source_module": str(source_module),
        "source_module_sha256": sha256_file(source_module),
        "source_csv": str(source_csv),
        "source_csv_sha256": sha256_file(source_csv),
        "used_element_count": len(requested),
        "used_elements": requested,
        "neutral_properties": list(NEUTRAL_PROPERTY_COLUMNS),
        "explicitly_excluded": [
            "raw ion radius because oxidation state and coordination are absent",
            "structure valence because formal charges are absent",
        ],
    }
    return frame, lookup, manifest


def weighted_property_stats(
    symbols: Iterable[str],
    weights: Iterable[float],
    lookup: Dict[str, Dict[str, float]],
    property_name: str,
) -> Dict[str, float]:
    symbol_list = list(symbols)
    weight_array = np.asarray(list(weights), dtype=float)
    if len(symbol_list) != len(weight_array):
        raise ValueError("symbols and weights must have equal length")
    total_weight = float(np.sum(weight_array))
    values = np.asarray([lookup[symbol][property_name] for symbol in symbol_list], dtype=float)
    valid = np.isfinite(values) & np.isfinite(weight_array) & (weight_array > 0)
    known_weight = float(np.sum(weight_array[valid]))
    coverage = known_weight / total_weight if total_weight > 0 else math.nan
    if known_weight <= 0:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan, "coverage": coverage}
    normalized = weight_array[valid] / known_weight
    selected = values[valid]
    mean = float(np.sum(normalized * selected))
    variance = float(np.sum(normalized * (selected - mean) ** 2))
    return {
        "mean": mean,
        "std": math.sqrt(max(variance, 0.0)),
        "min": float(np.min(selected)),
        "max": float(np.max(selected)),
        "coverage": coverage,
    }
