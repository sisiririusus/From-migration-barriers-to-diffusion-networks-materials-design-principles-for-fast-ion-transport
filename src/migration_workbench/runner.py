"""Execute a user project and export traceable results to a new run directory."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import uuid
from collections import Counter

from .inputs import load_project
from .i18n import translate
from .model import FrozenBarrierModel, sha256_file
from .network import periodic_analysis, rank_candidates
from .paths import RUNS
from .paths import ensure_science_path


def analyze_project(project, with_removal=True):
    result = dict(project)
    result.pop("structure", None)
    if project["kind"] == "path_project":
        ensure_science_path()
        from revision_core.data_governance import reduced_formula
        if project["formula"]:
            source_formula = project["formula"]
        else:
            amounts = Counter(project["structure"]["elements"])
            source_formula = "".join(element + str(count) for element, count in amounts.items())
        group = reduced_formula(source_formula) + "|" + project["migrating_species"]
        result["formula_species_group"] = group
        model = FrozenBarrierModel()
        result["edges"], graph_ood = model.predict(
            project["structure"], project["migrating_species"], project["edges"], group
        )
        result["model_sha256"] = model.model_version
        result["graph_applicability"] = graph_ood
        result["barrier_interpretation"] = "完整训练集模型对指定预 NEB 路径的预测；非实测势垒"
    else:
        result["edges"] = project["edges"]
        result["model_sha256"] = None
        result["graph_applicability"] = None
        result["barrier_interpretation"] = "用户给定的边势垒；未运行机器学习模型"
    for edge in result["edges"]:
        edge["graph_id"] = project["graph_id"]
    result["readout"] = periodic_analysis(
        project["nodes"], result["edges"], with_removal=with_removal
    )
    result["recommendations"] = rank_candidates(result["edges"], result["readout"])
    result["computed_now"] = True
    result["interpretation"] = "周期贯通、关键边和阈值只描述当前静态网络；不能换算为实验电导率"
    return result


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Cannot serialize %s" % type(value).__name__)


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value
                for key, value in row.items()
            })


def _report_html(result, language="zh"):
    readout = result["readout"]
    esc = lambda value: html.escape(str(value))
    t = lambda value: translate(value, language)
    def threshold(axis):
        value = readout.get("Eperc_%s" % axis)
        return t("不可达") if value is None or value == "" else "%.6f eV" % float(value)
    edge_rows = "\n".join(
        "<tr><td>%s</td><td>%s → %s</td><td>%s</td><td>%.6f</td><td>%s</td></tr>" % (
            esc(row["edge_uid"]), esc(row["node_i"]), esc(row["node_j"]),
            esc(row["image_shift"]), float(row["predicted_barrier_eV"]),
            esc(row.get("applicability_score", "—")),
        )
        for row in result["edges"][:200]
    )
    return """<!doctype html><html lang="%s"><head><meta charset="utf-8"><title>%s</title>
<style>body{font:16px/1.65 system-ui,Arial,sans-serif;max-width:1050px;margin:40px auto;padding:0 24px;color:#1e293b}h1{font-size:28px}h2{margin-top:34px}small{color:#64748b}.cards{display:flex;gap:14px;flex-wrap:wrap}.card{border:1px solid #cbd5e1;border-radius:8px;padding:14px;min-width:150px}table{border-collapse:collapse;width:100%%;font-size:13px}td,th{border-bottom:1px solid #ddd;text-align:left;padding:7px;vertical-align:top}th{background:#f1f5f9}.note{background:#fff7ed;border-left:4px solid #f59e0b;padding:12px}</style>
</head><body><h1>%s</h1><p><small>graph_id: %s · %d %s · %d %s</small></p>
<div class="cards"><div class="card">a %s<br><strong>%s</strong></div><div class="card">b %s<br><strong>%s</strong></div><div class="card">c %s<br><strong>%s</strong></div><div class="card">%s<br><strong>%s</strong></div></div>
<p class="note">%s%s%s</p><h2>%s</h2><table><thead><tr><th>edge_uid</th><th>%s</th><th>%s</th><th>%s</th><th>%s</th></tr></thead><tbody>%s</tbody></table>
<p><small>%s</small></p>
</body></html>""" % (
        language, esc(t("迁移网络报告")), esc(result["name"]), esc(result["graph_id"]),
        len(result["nodes"]), t("个节点"), len(result["edges"]), t("条边"),
        t("方向"), threshold("a"), t("方向"), threshold("b"), t("方向"), threshold("c"),
        t("周期维数"), esc(readout.get("winding_vector_rank", "—")),
        esc(t(result.get("barrier_interpretation", "论文分组 OOF 预测"))), ". " if language == "en" else "。", esc(t(result["interpretation"])),
        t("边与来源"), t("端点"), t("周期位移"), t("势垒 (eV)"), t("适用域"), edge_rows,
        t("若超过 200 条边，本页只显示前 200 条。完整数据见 edges.csv 和 result.json。"),
    )


def save_run(result, input_path=None, language="zh"):
    RUNS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS / (stamp + "_" + uuid.uuid4().hex[:8])
    run_dir.mkdir()
    _write_json(run_dir / "result.json", result)
    _write_csv(run_dir / "edges.csv", result["edges"])
    _write_csv(run_dir / "recommendations.csv", result.get("recommendations", []))
    (run_dir / "report.html").write_text(_report_html(result, language), encoding="utf-8")
    source = Path(input_path) if input_path else None
    provenance = {
        "run_id": run_dir.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(source.resolve()) if source and source.is_file() else None,
        "input_sha256": sha256_file(source) if source and source.is_file() else None,
        "model_sha256": result.get("model_sha256"),
        "calculation_role": result["kind"],
        "scientific_scope": result.get("interpretation"),
        "outputs": {name: sha256_file(run_dir / name) for name in ["result.json", "edges.csv", "recommendations.csv", "report.html"]},
    }
    _write_json(run_dir / "manifest.json", provenance)
    return run_dir


def analyze_file(path, with_removal=True, language="zh"):
    project = load_project(path)
    result = analyze_project(project, with_removal=with_removal)
    return result, save_run(result, input_path=path, language=language)
