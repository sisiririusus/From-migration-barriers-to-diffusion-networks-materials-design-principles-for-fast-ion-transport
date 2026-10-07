"""Command line interface for actual prediction and exact periodic analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import uuid

from .examples import catalog, load_case
from .inputs import load_graph_csv, project_from_structure
from .runner import analyze_file, analyze_project, save_run


def _parser():
    parser = argparse.ArgumentParser(prog="MigrationWorkbench", description=__doc__)
    parser.add_argument("--status-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--language", choices=("zh", "en"), default="zh", help="界面/报告语言 (zh/en) / UI and report language")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("examples", help="列出 8 个论文周期图示例")
    paper = sub.add_parser("paper", help="查看或重算一个论文示例")
    paper.add_argument("case_id")
    paper.add_argument("--recompute", action="store_true", help="实际重算精确周期阈值")
    paper.add_argument("--save", action="store_true", help="保存结果到 runs")
    analyze = sub.add_parser("analyze", help="分析带结构/路径或用户给定周期图的 JSON")
    analyze.add_argument("path", type=Path)
    analyze.add_argument("--skip-removal", action="store_true", help="只算周期读数，省略逐边移除")
    structure = sub.add_parser("structure", help="从 CIF/POSCAR 和指定端点建立并运行路径项目")
    structure.add_argument("path", type=Path)
    structure.add_argument("--species", required=True)
    structure.add_argument("--moving-atom-index", type=int, required=True)
    structure.add_argument("--end-frac", required=True, help="目标分数坐标，逗号分隔")
    structure.add_argument("--shift", default="0,0,0", help="终点周期像整数位移")
    graph = sub.add_parser("graph-csv", help="从节点/边 CSV 计算周期网络")
    graph.add_argument("--nodes", required=True, type=Path)
    graph.add_argument("--edges", required=True, type=Path)
    graph.add_argument("--barrier-column", default="barrier_eV")
    sub.add_parser("validate", help="检查模型、示例和原始周期图算法")
    return parser


def _triple(text, integer=False):
    bits = text.split(",")
    if len(bits) != 3:
        raise ValueError("坐标或周期位移需要三个逗号分隔的值")
    return [int(item) if integer else float(item) for item in bits]


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    def status(state, folder=None, error=None):
        if args.status_file:
            args.status_file.parent.mkdir(parents=True, exist_ok=True)
            args.status_file.write_text(json.dumps({
                "state": state, "result_dir": str(folder) if folder else None,
                "error": error,
            }, ensure_ascii=False), encoding="utf-8")
    try:
        if args.command == "examples":
            for case in catalog():
                print("%s\t%d nodes\t%d edges\t%s" % (
                    case["id"], case["n_nodes"], case["n_edges"], case["family"]
                ))
            status("success")
            return 0
        if args.command == "paper":
            result = load_case(args.case_id, recompute=args.recompute)
            if args.save or args.recompute:
                folder = save_run(result, language=args.language)
                print("RESULT_DIR=" + str(folder))
                status("success", folder)
            else:
                status("success")
            print(json.dumps({
                "case": result["name"], "nodes": len(result["nodes"]),
                "edges": len(result["edges"]), "computed_now": result["computed_now"],
                "Eperc_a": result["readout"]["Eperc_a"],
                "Eperc_b": result["readout"]["Eperc_b"],
                "Eperc_c": result["readout"]["Eperc_c"],
                "winding_rank": result["readout"]["winding_vector_rank"],
                "candidates": len(result["recommendations"]),
            }, ensure_ascii=False, default=str))
            return 0
        if args.command == "analyze":
            result, folder = analyze_file(args.path, with_removal=not args.skip_removal, language=args.language)
        elif args.command == "structure":
            payload = project_from_structure(
                args.path, args.species, args.moving_atom_index,
                _triple(args.end_frac), _triple(args.shift, integer=True),
            )
            from .inputs import load_project
            from .paths import RUNS
            RUNS.mkdir(exist_ok=True, parents=True)
            prepared = RUNS / (args.path.stem + "_prepared_" + uuid.uuid4().hex[:8] + ".json")
            prepared.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            result, folder = analyze_file(prepared, language=args.language)
        elif args.command == "graph-csv":
            project = load_graph_csv(args.nodes, args.edges, args.barrier_column)
            result = analyze_project(project)
            folder = save_run(result, input_path=args.edges, language=args.language)
        elif args.command == "validate":
            from .model import FrozenBarrierModel
            from .paths import ensure_science_path
            ensure_science_path()
            from revision_core.periodic_graph import run_toy_validation
            model = FrozenBarrierModel()
            if not run_toy_validation()["passed"]:
                raise RuntimeError("原始周期图自检失败")
            cases = catalog()
            if len(cases) != 8:
                raise RuntimeError("论文图数量不符")
            print(json.dumps({"status": "PASS", "model_sha256": model.model_version, "paper_examples": len(cases)}, ensure_ascii=False))
            status("success")
            return 0
        else:
            parser.print_help()
            return 2
        print("RESULT_DIR=" + str(folder))
        status("success", folder)
        print(json.dumps({
            "graph_id": result["graph_id"], "edges": len(result["edges"]),
            "winding_rank": result["readout"]["winding_vector_rank"],
            "Eperc_a": result["readout"]["Eperc_a"],
            "Eperc_b": result["readout"]["Eperc_b"],
            "Eperc_c": result["readout"]["Eperc_c"],
        }, ensure_ascii=False))
        return 0
    except Exception as exc:
        print("ERROR: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        status("error", error="%s: %s" % (type(exc).__name__, exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
