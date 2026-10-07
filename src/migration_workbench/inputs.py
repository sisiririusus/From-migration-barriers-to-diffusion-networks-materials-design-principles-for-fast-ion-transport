"""Validate user graph/path inputs before scientific calculations."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


def _vector(value, name, length=3):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError("%s 必须有 %d 个值" % (name, length))
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError("%s 包含非有限数值" % name)
    return result


def _shift(value):
    vector = _vector(value, "image_shift")
    if any(value != round(value) for value in vector):
        raise ValueError("image_shift 必须是整数晶格位移")
    return [int(item) for item in vector]


def _distance_frac(frac_a, frac_b, lattice):
    delta = [float(frac_a[i]) - float(frac_b[i]) for i in range(3)]
    wrapped = [item - round(item) for item in delta]
    cart = [sum(wrapped[row] * lattice[row][col] for row in range(3)) for col in range(3)]
    return math.sqrt(sum(item * item for item in cart))


def validate_structure(structure, species):
    import numpy as np

    if not isinstance(structure, dict):
        raise ValueError("structure 必须是对象")
    lattice = [_vector(row, "lattice_mat[%d]" % i) for i, row in enumerate(structure.get("lattice_mat", []))]
    if len(lattice) != 3 or abs(float(np.linalg.det(np.asarray(lattice)))) < 1e-8:
        raise ValueError("晶格必须是非退化的 3×3 Å 矩阵")
    elements = structure.get("elements", [])
    coords = structure.get("coords", [])
    if not elements or len(elements) != len(coords):
        raise ValueError("elements 与 coords 数量不一致或为空")
    if species not in elements:
        raise ValueError("初始结构中没有指定的迁移离子 %s" % species)
    if structure.get("cartesian") is True:
        raise ValueError("coords 必须是分数坐标；请先转换笛卡尔坐标")
    normalized = {
        "lattice_mat": lattice,
        "elements": [str(value) for value in elements],
        "coords": [_vector(value, "coords[%d]" % i) for i, value in enumerate(coords)],
        "cartesian": False,
    }
    return normalized


def validate_nodes(nodes):
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("nodes 必须是非空列表")
    result = []
    seen = set()
    for item in nodes:
        node_id = str(item.get("id", "")).strip()
        if not node_id or node_id in seen:
            raise ValueError("节点 id 为空或重复：%s" % node_id)
        seen.add(node_id)
        result.append({"id": node_id, "frac_coords": _vector(item.get("frac_coords"), "node %s" % node_id)})
    return result


def _validate_edges(edges, nodes, lattice=None, structure=None, species=None, predict=False):
    if not isinstance(edges, list) or not edges:
        raise ValueError("edges 必须是非空列表")
    node_by_id = {node["id"]: node for node in nodes}
    result = []
    seen = set()
    for index, item in enumerate(edges):
        uid = str(item.get("edge_uid") or "edge_%04d" % (index + 1)).strip()
        node_i = str(item.get("node_i", ""))
        node_j = str(item.get("node_j", ""))
        if uid in seen or node_i not in node_by_id or node_j not in node_by_id:
            raise ValueError("边 id 重复或端点不存在：%s" % uid)
        seen.add(uid)
        shift = _shift(item.get("image_shift", [0, 0, 0]))
        if node_i == node_j and shift == [0, 0, 0]:
            raise ValueError("零位移自边不是有效跃迁：%s" % uid)
        edge = {
            "edge_uid": uid,
            "node_i": node_i,
            "node_j": node_j,
            "image_shift": shift,
            "start_frac_coords": node_by_id[node_i]["frac_coords"],
            "end_frac_coords": node_by_id[node_j]["frac_coords"],
            "is_labeled": bool(item.get("is_labeled", False)),
        }
        if predict:
            if "moving_atom_index" not in item:
                raise ValueError("路径 %s 缺少 moving_atom_index" % uid)
            mover = int(item["moving_atom_index"])
            if not 0 <= mover < len(structure["elements"]) or structure["elements"][mover] != species:
                raise ValueError("路径 %s 的 moving_atom_index 不是 %s 原子" % (uid, species))
            start_error = _distance_frac(structure["coords"][mover], edge["start_frac_coords"], lattice)
            if start_error > 0.01:
                raise ValueError("路径 %s 起点与 moving_atom_index 相距 %.4f Å" % (uid, start_error))
            occupied = [
                i for i, atom in enumerate(structure["elements"])
                if i != mover and atom == species
                and _distance_frac(structure["coords"][i], edge["end_frac_coords"], lattice) < 0.01
            ]
            if occupied:
                raise ValueError("路径 %s 终点被同种离子占据；请提供空位或独立机制" % uid)
            edge["moving_atom_index"] = mover
        else:
            if "barrier_eV" not in item:
                raise ValueError("网络边 %s 缺少 barrier_eV" % uid)
            barrier = float(item["barrier_eV"])
            if not math.isfinite(barrier) or barrier < 0:
                raise ValueError("网络边 %s 的 barrier_eV 无效" % uid)
            edge["predicted_barrier_eV"] = barrier
            edge["barrier_source"] = str(item.get("barrier_source", "user_supplied"))
        result.append(edge)
    return result


def load_project(path):
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("项目 JSON 需要 schema_version=1")
    kind = payload.get("kind")
    if kind not in ("path_project", "periodic_graph"):
        raise ValueError("kind 应为 path_project 或 periodic_graph")
    nodes = validate_nodes(payload.get("nodes"))
    name = str(payload.get("name") or source.stem)
    graph_id = str(payload.get("graph_id") or source.stem)
    species = str(payload.get("migrating_species") or "")
    if kind == "path_project":
        structure = validate_structure(payload.get("structure"), species)
        edges = _validate_edges(
            payload.get("edges"), nodes, lattice=structure["lattice_mat"],
            structure=structure, species=species, predict=True,
        )
    else:
        structure = None
        edges = _validate_edges(payload.get("edges"), nodes)
    result = {
        "kind": kind,
        "name": name,
        "graph_id": graph_id,
        "formula": str(payload.get("formula") or ""),
        "migrating_species": species,
        "structure": structure,
        "nodes": nodes,
        "edges": edges,
        "input_path": str(source),
    }
    return result


def structure_from_file(path):
    """Read CIF or POSCAR with ASE; atom positions are returned as fractional coordinates."""
    from ase.io import read

    source = Path(path).resolve()
    atoms = read(str(source), index=0)
    if not atoms.pbc.all():
        raise ValueError("需要三维周期结构及完整晶格")
    return {
        "lattice_mat": atoms.get_cell().array.tolist(),
        "elements": atoms.get_chemical_symbols(),
        "coords": atoms.get_scaled_positions(wrap=False).tolist(),
        "cartesian": False,
    }


def project_from_structure(path, species, moving_atom_index, end_frac_coords, image_shift=None):
    structure = structure_from_file(path)
    index = int(moving_atom_index)
    if not 0 <= index < len(structure["elements"]):
        raise ValueError("初始原子索引超出结构范围")
    start = structure["coords"][index]
    payload = {
        "schema_version": 1,
        "kind": "path_project",
        "name": Path(path).stem,
        "formula": "",
        "migrating_species": species,
        "structure": structure,
        "nodes": [{"id": "start", "frac_coords": start}, {"id": "end", "frac_coords": _vector(end_frac_coords, "终点分数坐标")}],
        "edges": [{"edge_uid": "path_1", "node_i": "start", "node_j": "end", "image_shift": image_shift or [0, 0, 0], "moving_atom_index": index}],
    }
    return payload


def load_graph_csv(nodes_path, edges_path, barrier_column="barrier_eV"):
    with Path(nodes_path).open(encoding="utf-8-sig", newline="") as stream:
        nodes_rows = list(csv.DictReader(stream))
    nodes = [
        {"id": row.get("node_id") or row.get("id"), "frac_coords": json.loads(row["frac_coords"])}
        for row in nodes_rows
    ]
    with Path(edges_path).open(encoding="utf-8-sig", newline="") as stream:
        edge_rows = list(csv.DictReader(stream))
    node_graphs = {row["graph_id"] for row in nodes_rows if row.get("graph_id")}
    edge_graphs = {row["graph_id"] for row in edge_rows if row.get("graph_id")}
    if len(node_graphs) > 1 or len(edge_graphs) > 1:
        raise ValueError("CSV 中有多个 graph_id；请先选择同一结构与迁移离子的单张图")
    if node_graphs and edge_graphs and node_graphs != edge_graphs:
        raise ValueError("节点和边 CSV 的 graph_id 不一致")
    graph_id = next(iter(node_graphs or edge_graphs), "imported_graph")
    edges = []
    for row in edge_rows:
        edges.append({
            "edge_uid": row["edge_uid"],
            "node_i": row["node_i"],
            "node_j": row["node_j"],
            "image_shift": [row["image_shift_a"], row["image_shift_b"], row["image_shift_c"]],
            "barrier_eV": row[barrier_column],
            "barrier_source": "csv:%s" % barrier_column,
        })
    validated_nodes = validate_nodes(nodes)
    return {
        "kind": "periodic_graph",
        "name": Path(edges_path).stem,
        "graph_id": graph_id,
        "formula": "",
        "migrating_species": "",
        "structure": None,
        "nodes": validated_nodes,
        "edges": _validate_edges(edges, validated_nodes),
        "input_path": str(Path(edges_path).resolve()),
    }
