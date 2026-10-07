"""Locate the copied scientific sources and writeable application directories."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def workspace_root():
    configured = os.environ.get("MIGRATION_WORKBENCH_HOME")
    if configured:
        return Path(configured).resolve()
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


ROOT = workspace_root()
REPRO_ROOT = ROOT / "science" / "repro" / "scripts" / "reproduction"
PROJECT = REPRO_ROOT / "project"
LEGACY_SCRIPTS = PROJECT / "scripts"
MODEL_DIR = PROJECT / "new_data" / "revision_models"
TABLES = PROJECT / "tables"
EXTENDED = ROOT / "science" / "extended_analysis"
def run_root():
    configured = os.environ.get("MIGRATION_WORKBENCH_RUNS")
    if configured:
        return Path(configured).resolve()
    if getattr(sys, "frozen", False):
        if sys.platform == "win32":
            local = os.environ.get("LOCALAPPDATA")
            base = Path(local) if local else Path.home() / "AppData" / "Local"
        else:
            xdg = os.environ.get("XDG_DATA_HOME")
            base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
        return base / "MigrationNetworkWorkbench" / "runs"
    return ROOT / "runs"


RUNS = run_root()


def ensure_science_path():
    path = str(LEGACY_SCRIPTS)
    if path not in sys.path:
        sys.path.insert(0, path)


def require_inputs():
    required = [
        MODEL_DIR / "selected_barrier_model_pipeline.joblib",
        MODEL_DIR / "revision_uncertainty_network" / "full_strict_X_only_diversity_embedding.joblib",
    ]
    required.extend([
        MODEL_DIR / "element_properties_snapshot.csv",
        MODEL_DIR / "element_properties_manifest.json",
        PROJECT / "config" / "revision_model_config.yaml",
        TABLES / "revision_source_data" / "revision_periodic_oof_edge_predictions.csv",
    ])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少科研资源：" + "; ".join(missing))
    return required
