# Migration Network Workbench / 迁移网络工作台

[Full English README / 英文完整说明](README_EN.md)

基于研究项目的本地桌面软件：对指定的预 NEB 路径预测迁移势垒，计算精确周期网络读数，并查看论文的八个案例。Windows 和 Linux 发行包使用同一份科学核心。界面右上角可在 **中文 / English** 之间即时切换；实际计算导出的 HTML 报告跟随所选语言，数值 JSON/CSV 保持原字段名。

A local desktop application for predicting barriers on specified pre-NEB paths, computing exact periodic-network readouts, and inspecting eight paper examples. Windows and Linux releases use the same scientific core. Switch **中文 / English** in the top-right corner; newly generated HTML reports follow the selected language, while numerical JSON/CSV field names stay stable.

## 下载与启动 / Download and launch

从 GitHub Releases 下载对应系统的 `MigrationWorkbench_0.2_...zip`，解压后：

- Windows：运行 `MigrationWorkbench/MigrationWorkbench.exe`。
- Linux x86-64：运行 `chmod +x MigrationWorkbench/MigrationWorkbench && ./MigrationWorkbench/MigrationWorkbench`。需要图形桌面和可用的 X11/Wayland Qt 平台插件。

Download the ZIP for your platform from GitHub Releases. Extract it, then launch the executable above. Linux requires a graphical desktop. The bundled Linux build was made on glibc 2.28 x86-64; older glibc systems are not supported by this binary.

实际计算输出默认位于 Windows `%LOCALAPPDATA%\MigrationNetworkWorkbench\runs` 或 Linux `${XDG_DATA_HOME:-~/.local/share}/MigrationNetworkWorkbench/runs`。可用 `MIGRATION_WORKBENCH_RUNS` 自定义目录。论文示例的“打开示例”读取保存的 OOF 表；“实际重算”才运行周期算法。用户结构需要明确指定迁移离子、起点原子和目标位点；软件不会自动发现真实空位。

Calculated runs are written to the platform's user-data directory; set `MIGRATION_WORKBENCH_RUNS` to override it. “Open example” reads saved OOF tables, while “Recompute” runs the periodic algorithm. Imported structures require an explicit migrating ion, starting atom, and target site; the app does not discover vacancies automatically.

## 功能与科学范围 / Scientific scope

- 导入项目 JSON、CIF/POSCAR，或节点/边 CSV；逐边保留 `edge_uid` 和整数 `image_shift`。
- 预测路径势垒与适用域，计算 winding rank、各方向 `Eperc` 和关键边轨道，导出 `result.json`、`edges.csv`、`recommendations.csv`、`report.html` 和 `manifest.json`。
- 八个论文图案例包括四个基础周期网络、四个扩展比较/候选案例；Na10 候选与已标注边明确分开。
- 冻结 ExtraTrees 模型沿用 scikit-learn 1.0.2 的序列化环境。论文 OOF 图与用户新路径的全数据模型预测具有不同的数据角色，界面分别标示。

`Eperc` 是当前静态网络在给定/预测势垒下的周期贯通阈值，不是实验电导率；不可达方向不是 `0 eV`。树间分歧不是已校准预测区间。用户新图的 NEB 列表是透明预筛，并不等同于论文最终候选排名。

The app imports project JSON, CIF/POSCAR, or node/edge CSV; preserves edge IDs and integer image shifts; predicts path barriers and applicability; computes winding rank, directional `Eperc`, and critical edge orbits; and exports traceable results. `Eperc` is a static-network threshold, not conductivity. Unreachable directions are not `0 eV`; tree disagreement is not a calibrated interval. New-graph NEB recommendations are a prescreen, distinct from the paper's final candidate table.

## 源码构建 / Build from source

Use an isolated Python environment. The saved model was verified with scikit-learn 1.0.2. On Linux x86-64, use Python 3.9; on Windows, use Python 3.7.

```bash
# Linux
python3.9 -m venv .venv
.venv/bin/python -m pip install -r packaging/requirements-linux.txt
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python packaging/build_portable.py
.venv/bin/python packaging/make_portable_zip.py
```

```powershell
# Windows PowerShell (Python 3.7)
py -3.7 -m venv .venv
.venv\Scripts\python.exe -m pip install -r packaging\requirements-windows.txt
$env:PYTHONPATH='src'
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe packaging\build_portable.py
.venv\Scripts\python.exe packaging\make_portable_zip.py
```

To run the source UI, use `python start.py`; to use the CLI, run `python start.py --cli --language en validate` or `python start.py --cli --language en analyze examples/user_path_project.json`. `--language zh` is the default. Builds write to `build/`, and final archives to `release-assets/`; both are excluded from Git.

## GitHub 发布 / GitHub publishing

The desktop application's source is in this repository alongside the manuscript scripts in `scripts/`. Download the binaries from GitHub Releases; keep them out of Git history. Each ZIP contains a per-file `release_manifest.json`; the Release also includes `SHA256SUMS.txt`. The paper's complete LaTeX source is maintained separately; `science/` contains only the audited code, model, and tables needed by the application and validation.

软件源码与原有论文脚本并存；Windows/Linux 压缩包见本仓库的 Releases。公开许可证和正式论文引用信息由作者决定。
