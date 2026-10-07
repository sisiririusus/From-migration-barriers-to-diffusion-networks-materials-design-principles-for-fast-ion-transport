"""Exercise the frozen program, its model, import paths, and English report."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import uuid


ROOT = Path(__file__).resolve().parents[1]
PLATFORM = "Windows" if sys.platform == "win32" else "Linux"
APP = ROOT / "build" / PLATFORM / "MigrationWorkbench"
EXE = APP / ("MigrationWorkbench.exe" if PLATFORM == "Windows" else "MigrationWorkbench")
RUNS = ROOT / "build" / "qa-runs" / (PLATFORM + "_" + uuid.uuid4().hex[:8])


def check(label, arguments):
    status_file = RUNS / (label + "_status.json")
    environment = os.environ.copy()
    environment["MIGRATION_WORKBENCH_RUNS"] = str(RUNS)
    process = subprocess.run(
        [str(EXE), "--cli", "--status-file", str(status_file)] + arguments,
        cwd=str(APP), env=environment, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, universal_newlines=True, timeout=180,
    )
    if process.returncode or not status_file.is_file():
        raise AssertionError("%s failed (%s): %s" % (label, process.returncode, process.stdout[-2000:]))
    status = json.loads(status_file.read_text(encoding="utf-8"))
    if status.get("state") != "success":
        raise AssertionError("%s: %s" % (label, status))
    folder = Path(status["result_dir"]) if status.get("result_dir") else None
    return json.loads((folder / "result.json").read_text(encoding="utf-8")) if folder else None, folder


def main():
    if not EXE.is_file():
        raise FileNotFoundError(EXE)
    RUNS.mkdir(parents=True, exist_ok=True)
    check("validate", ["validate"])
    paper, _ = check("paper", ["paper", "Na23Se8Cl8_Na", "--recompute"])
    assert len(paper["nodes"]) == 24 and paper["readout"]["winding_vector_rank"] == 3
    assert abs(paper["readout"]["Eperc_a"] - 1.1829268333333336) < 1e-9
    project, folder = check("json_en", ["--language", "en", "analyze", str(APP / "examples" / "user_path_project.json")])
    assert abs(project["edges"][0]["predicted_barrier_eV"] - 0.52805) < 1e-5
    assert 'html lang="en"' in (folder / "report.html").read_text(encoding="utf-8")
    for suffix, name in (("cif", "user_path_structure.cif"), ("vasp", "user_path_structure.vasp")):
        result, _ = check(suffix, ["structure", str(APP / "examples" / name), "--species", "Na", "--moving-atom-index", "0", "--end-frac", "0,0.5,0.5"])
        assert len(result["edges"]) == 1 and result["edges"][0]["predicted_barrier_eV"] > 0
    graph, _ = check("graph", ["graph-csv", "--nodes", str(APP / "examples" / "user_network_nodes.csv"), "--edges", str(APP / "examples" / "user_network_edges.csv")])
    assert graph["readout"]["winding_vector_rank"] == 2
    print("PORTABLE_QA_PASS=" + str(EXE))
    print("QA_RUNS=" + str(RUNS))


if __name__ == "__main__":
    main()
