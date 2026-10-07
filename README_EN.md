# Migration Network Workbench

Migration Network Workbench is a local desktop application accompanying this repository's migration-barrier and diffusion-network research. It loads eight saved paper examples, predicts migration barriers for explicitly specified pre-NEB paths, analyzes exact periodic graphs, and exports traceable results. The Windows and Linux packages use the same scientific core.

This repository also contains the original manuscript scripts in [`scripts/`](scripts/). The desktop application lives in [`src/migration_workbench/`](src/migration_workbench/), with the audited model and required tables in [`science/`](science/). The full LaTeX manuscript is maintained separately; the app does not copy it into analysis output.

## Download and run

Download the ZIP for your system from [Releases](https://github.com/sisiririusus/From-migration-barriers-to-diffusion-networks-materials-design-principles-for-fast-ion-transport/releases). Extract it, then:

| Platform | Launch |
| --- | --- |
| Windows x86-64 | `MigrationWorkbench/MigrationWorkbench.exe` |
| Linux x86-64 | `chmod +x MigrationWorkbench/MigrationWorkbench && ./MigrationWorkbench/MigrationWorkbench` |

The Linux binary was built on glibc 2.28 and requires a graphical desktop with a working Qt X11/Wayland platform plugin. Older glibc versions are outside the tested binary target.

Use the **中文 / English** selector in the upper-right corner to switch the interface without changing the selected example or calculated values. Newly generated HTML reports follow the selected language; JSON and CSV field names remain stable.

## What the application does

- Opens eight paper cases: four base periodic networks, three composition comparisons, and the Na10 prospective-candidate case. Opening a case reads saved group-held-out OOF data; **Recompute** runs the exact periodic-network algorithm.
- Imports a project JSON file, a CIF/POSCAR structure with an explicit migrating ion, starting atom and target site, or node/edge CSV tables for a user-supplied graph.
- Preserves each edge's `edge_uid`, endpoints and integer periodic `image_shift` rather than merging distinct periodic images.
- Predicts barriers on specified paths with the saved ExtraTrees model, reports applicability/OOD information, computes winding rank, directional `Eperc` thresholds and critical edge orbits, and provides a traceable NEB prescreen.
- Writes `result.json`, `edges.csv`, `recommendations.csv`, `report.html` and `manifest.json` for each calculated run.

The output folder defaults to `%LOCALAPPDATA%\MigrationNetworkWorkbench\runs` on Windows and `${XDG_DATA_HOME:-~/.local/share}/MigrationNetworkWorkbench/runs` on Linux. Set `MIGRATION_WORKBENCH_RUNS` to choose another folder.

## Scientific interpretation

The paper example graphs use their saved OOF edge predictions, while new user paths use the frozen full-data model. These are different prediction roles. `Eperc` is a static-network percolation threshold under supplied or predicted barriers; it is **not** experimental ionic conductivity. An unreachable direction is not `0 eV`. Tree-to-tree disagreement is not a calibrated prediction interval. Recommendations for a new graph are a transparent prescreen and should not be read as the paper's final ranked candidate list. The application does not automatically identify vacancies or migration mechanisms in a new crystal structure.

## Build from source

Use an isolated environment and the pinned dependencies. The saved model was validated with scikit-learn 1.0.2. Use Python 3.7 on Windows or Python 3.9 on Linux x86-64.

```bash
# Linux
python3.9 -m venv .venv
.venv/bin/python -m pip install -r packaging/requirements-linux.txt
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python packaging/build_portable.py
.venv/bin/python packaging/verify_portable.py
.venv/bin/python packaging/make_portable_zip.py
```

```powershell
# Windows PowerShell
py -3.7 -m venv .venv
.venv\Scripts\python.exe -m pip install -r packaging\requirements-windows.txt
$env:PYTHONPATH='src'
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe packaging\build_portable.py
.venv\Scripts\python.exe packaging\verify_portable.py
.venv\Scripts\python.exe packaging\make_portable_zip.py
```

For development, start the UI with `python start.py`, or use the CLI, for example `python start.py --cli --language en validate` and `python start.py --cli --language en analyze examples/user_path_project.json`. Python builds go to `build/`; ZIP files go to `release-assets/`. Neither directory belongs in Git history. The packaged binaries are distributed as Release assets.

## Validation

The six source tests passed on both Windows and Linux. They include numerical parity against all 713 saved candidate predictions (maximum difference below `1e-8 eV`), paper network readouts, custom input paths and preservation of values across language switching. Both frozen applications passed model/resource validation, Na23 recomputation, JSON/CIF/POSCAR/CSV runs, and English report generation. Each GUI passed an offscreen launch test. The [GitHub Actions build](https://github.com/sisiririusus/From-migration-barriers-to-diffusion-networks-materials-design-principles-for-fast-ion-transport/actions/runs/37577442388) passed on Windows and Linux on 7 October 2026. Details are in [`docs/VALIDATION.md`](docs/VALIDATION.md). A visual review across multiple Linux desktop distributions has not yet been completed.

No software license has been selected in this repository; redistribution terms should be set by the repository owner.
