"""Build one portable desktop folder on the current operating system.

Run this from a clean virtual environment with the platform requirements installed.
Existing output is left intact; choose a fresh build directory for another build.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "packaging"
MANIFEST = PACKAGE / "resource_manifest.json"
PROJECT_SCRIPTS = ROOT / "science" / "repro" / "scripts" / "reproduction" / "project" / "scripts"
PLATFORM = "Windows" if sys.platform == "win32" else "Linux" if sys.platform.startswith("linux") else None
DIST = ROOT / "build" / PLATFORM if PLATFORM else None
APP = DIST / "MigrationWorkbench" if DIST else None


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_resources():
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
    for entry in entries:
        # The source manifest was originally authored on Windows.
        relative = Path(entry["path"].replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe resource path: %s" % relative)
        source = ROOT / relative
        if not source.is_file() or sha256(source) != entry["sha256"]:
            raise ValueError("Resource missing or changed: %s" % source)
        target = APP / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copy2(MANIFEST, APP / "resource_manifest.json")
    shutil.copy2(ROOT / "README.md", APP / "README.md")


def finish_bundle():
    """Finish a built folder, including a PyInstaller run interrupted after EXE creation."""
    import ase
    import PyQt5

    executable = APP / ("MigrationWorkbench.exe" if PLATFORM == "Windows" else "MigrationWorkbench")
    if not executable.is_file():
        raise FileNotFoundError(executable)
    copy_resources()
    spacegroup = Path(ase.__file__).resolve().parent / "spacegroup" / "spacegroup.dat"
    target = APP / "ase" / "spacegroup" / "spacegroup.dat"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(spacegroup, target)
    if PLATFORM == "Windows":
        qt_bin = Path(PyQt5.__file__).resolve().parent / "Qt5" / "bin"
        for name in ("Qt5Core.dll", "Qt5Gui.dll", "Qt5Widgets.dll"):
            shutil.copy2(qt_bin / name, APP / name)
    print("PACKAGE_READY=" + str(APP))


def main():
    if PLATFORM is None:
        raise RuntimeError("Only Windows and Linux are supported")
    if sys.argv[1:] == ["--finish-only"]:
        finish_bundle()
        return
    if APP.exists():
        raise FileExistsError("Existing build is preserved: %s" % APP)
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)
    from PyInstaller.__main__ import run
    import PyQt5

    qt_dir = Path(PyQt5.__file__).resolve().parent
    qt_bin = qt_dir / "Qt5" / "bin"
    if PLATFORM == "Windows":
        os.environ["PATH"] = str(qt_bin) + os.pathsep + os.environ.get("PATH", "")
        for name in ("Qt5Core.dll", "Qt5Gui.dll", "Qt5Widgets.dll"):
            ctypes.WinDLL(str(qt_bin / name))

    work = ROOT / "build" / "pyinstaller-work" / PLATFORM
    hidden = [
        "joblib", "pandas", "numpy", "scipy", "scipy.special", "scipy.stats",
        "sklearn", "sklearn.ensemble", "sklearn.tree", "sklearn.preprocessing",
        "sklearn.decomposition", "sklearn.model_selection", "sklearn.metrics",
        "ase.io.cif", "ase.io.vasp", "ase.spacegroup",
        "revision_core.data_governance", "revision_core.barrier_features",
        "revision_core.barrier_modeling", "revision_core.element_properties",
        "revision_core.periodic_trial", "revision_core.periodic_graph",
        "revision_core.periodic_network_analysis", "revision_core.uncertainty_calibration",
    ]
    arguments = [
        str(ROOT / "start.py"), "--name=MigrationWorkbench", "--onedir",
        "--distpath=" + str(DIST), "--workpath=" + str(work),
        "--specpath=" + str(work), "--paths=" + str(ROOT / "src"),
        "--paths=" + str(PROJECT_SCRIPTS), "--collect-submodules=ase.io",
        "--exclude-module=xgboost", "--exclude-module=lightgbm",
        "--exclude-module=catboost", "--exclude-module=bokeh",
        "--exclude-module=matplotlib", "--log-level=WARN",
    ]
    if PLATFORM == "Windows":
        arguments.append("--windowed")
    arguments.extend("--hidden-import=" + module for module in hidden)
    run(arguments)
    finish_bundle()


if __name__ == "__main__":
    main()
