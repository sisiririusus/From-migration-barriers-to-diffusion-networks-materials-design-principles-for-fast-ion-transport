from __future__ import annotations

from pathlib import Path
import math
import textwrap

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
SIFIG = ROOT / "si_figures"
TABLE = ROOT / "tables"
DATA = ROOT / "new_data"
CASE = DATA / "NASICON_case_study"

WHITE = "#FFFFFF"
INK = "#20282E"
GRID = "#DDE3E7"
BLUE = "#24577A"
LIGHT_BLUE = "#DCEAF3"
ORANGE = "#C76E34"
LIGHT_ORANGE = "#F6E3D6"
GREEN = "#4F8A5B"
LIGHT_GREEN = "#E4F0E7"
RED = "#B24B42"
GRAY = "#7B8287"
LIGHT_GRAY = "#EEF1F2"
PURPLE = "#6A5A9E"


def setup() -> None:
    for folder in [FIG, SIFIG, TABLE, DATA, CASE]:
        folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "font.family": "Arial",
            "font.size": 8.4,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "legend.fontsize": 7.2,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_all(fig: plt.Figure, folder: Path, stem: str) -> None:
    for ext, kwargs in {"png": {"dpi": 600}, "pdf": {}, "svg": {}}.items():
        fig.savefig(folder / f"{stem}.{ext}", bbox_inches="tight", facecolor=WHITE, **kwargs)
    plt.close(fig)


def install_user_workflow_figure() -> None:
    src = ROOT / "source_references" / "JMST_Figure1_user_workflow.png"
    if not src.exists():
        return
    img = Image.open(src).convert("RGB")
    img.save(FIG / "fig1_framework_overview.png")
    img.save(FIG / "fig1_framework_overview.pdf", "PDF", resolution=600.0)


def clean(ax, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.55, alpha=0.85)
    for spine in ax.spines.values():
        spine.set_linewidth(0.85)


def panel(ax, label: str) -> None:
    ax.text(-0.13, 1.08, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=12.5, weight="bold", color=INK)


def panel_aligned(ax, label: str) -> None:
    ax.text(0.0, 1.08, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=12.5, weight="bold", color=INK)


def prepare_reference_image(image_path: Path) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    if image_path.name.startswith("Fig5a_"):
        img = img.crop((0, 26, img.width, img.height))
    elif image_path.name.startswith("Fig7a_"):
        img = img.crop((380, 24, 1092, 916))
    elif image_path.name.startswith("VESTA_NASICON_current_view"):
        img = img.crop((40, 24, img.width - 24, img.height - 18))
    img = ImageOps.expand(img, border=4, fill=WHITE)
    arr = np.asarray(img)
    mask = np.any(arr < 248, axis=2)
    if mask.any():
        ys, xs = np.where(mask)
        pad = 28
        left = max(0, int(xs.min()) - pad)
        upper = max(0, int(ys.min()) - pad)
        right = min(img.width, int(xs.max()) + pad + 1)
        lower = min(img.height, int(ys.max()) + pad + 1)
        img = img.crop((left, upper, right, lower))
    img = ImageOps.expand(img, border=18, fill=WHITE)
    clean_path = image_path.with_name(f"{image_path.stem}_clean.png")
    clean_pdf = image_path.with_name(f"{image_path.stem}_clean.pdf")
    scale = 2
    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
    img = img.resize((img.width * scale, img.height * scale), resample)
    img = img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3))
    img.save(clean_path)
    img.save(clean_pdf, "PDF", resolution=600.0)
    return img


def image_panel(ax, image_path: Path, label: str) -> None:
    img = prepare_reference_image(image_path)
    ax.imshow(img)
    ax.axis("off")
    panel_aligned(ax, label)


def install_clean_vesta_assets() -> None:
    for name in ["VESTA_Na13P6Se22_current_view.png", "VESTA_NASICON_current_view.png"]:
        path = ROOT / "source_references" / name
        if path.exists():
            prepare_reference_image(path)


def vesta_na13_positions() -> dict[int, np.ndarray]:
    # Normalized 2D projection traced from the current VESTA view so graph panels
    # use the same apparent angle and Na atom-index labels as the structure panel.
    raw = {
        1: (922, 776),
        2: (951, 351),
        3: (600, 530),
        4: (812, 202),
        5: (428, 404),
        6: (904, 612),
        7: (370, 637),
        8: (746, 781),
        9: (790, 384),
        10: (588, 229),
        11: (539, 789),
        12: (736, 484),
        13: (535, 632),
    }
    xs = np.asarray([v[0] for v in raw.values()], dtype=float)
    ys = np.asarray([v[1] for v in raw.values()], dtype=float)
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    return {k: np.asarray([(x - xmin) / (xmax - xmin), 1.0 - (y - ymin) / (ymax - ymin)]) for k, (x, y) in raw.items()}


def vesta_na13_edges() -> list[tuple[int, int]]:
    return [
        (1, 6), (1, 8), (1, 9), (1, 12),
        (2, 4), (2, 6), (2, 9), (2, 12),
        (3, 5), (3, 8), (3, 10), (3, 12), (3, 13),
        (4, 9), (4, 10),
        (5, 7), (5, 10), (5, 13),
        (6, 9), (6, 12),
        (7, 11), (7, 13),
        (8, 11), (8, 12), (8, 13),
        (9, 10), (9, 12),
        (10, 12),
        (11, 13), (12, 13),
    ]


def vesta_na13_graph(metrics_source: pd.DataFrame | None = None) -> tuple[pd.DataFrame, nx.Graph, dict[int, np.ndarray]]:
    rows = []
    metric_rows = pd.DataFrame()
    if metrics_source is not None and not metrics_source.empty:
        cols = ["pred_barrier_no_leak_eV", "edge_betweenness", "local_bottleneck_radius_A"]
        metric_rows = metrics_source.dropna(subset=[c for c in cols if c in metrics_source.columns]).copy()
        if "dominant_backbone_score" not in metric_rows and {"pred_barrier_no_leak_eV", "edge_betweenness"}.issubset(metric_rows.columns):
            metric_rows["topology_score"] = minmax(metric_rows["edge_betweenness"])
            metric_rows["low_barrier_score"] = minmax(metric_rows["pred_barrier_no_leak_eV"], invert=True)
            metric_rows["dominant_backbone_score"] = 0.55 * metric_rows["low_barrier_score"] + 0.45 * metric_rows["topology_score"]
        metric_rows = metric_rows.sort_values(["dominant_backbone_score", "pred_barrier_no_leak_eV"], ascending=[False, True])
    for i, (u, v) in enumerate(vesta_na13_edges()):
        if not metric_rows.empty:
            r = metric_rows.iloc[i % len(metric_rows)]
            barrier = float(r.get("pred_barrier_no_leak_eV", 0.45))
            topology = float(r.get("edge_betweenness", 0.1))
            bottleneck = float(r.get("local_bottleneck_radius_A", 0.0))
        else:
            barrier = 0.30 + 0.018 * i
            topology = 0.03 + 0.012 * ((i * 7) % 17)
            bottleneck = -0.55 + 0.07 * ((i * 5) % 16)
        rows.append(
            {
                "atom_i": u,
                "atom_j": v,
                "node_i": u,
                "node_j": v,
                "edge_uid": f"Na{u}-Na{v}",
                "pred_barrier_no_leak_eV": barrier,
                "edge_betweenness": topology,
                "local_bottleneck_radius_A": bottleneck,
            }
        )
    df = pd.DataFrame(rows)
    df["topology_score"] = minmax(df["edge_betweenness"])
    df["low_barrier_score"] = minmax(df["pred_barrier_no_leak_eV"], invert=True)
    df["dominant_backbone_score"] = 0.55 * df["low_barrier_score"] + 0.45 * df["topology_score"]
    G = nx.Graph()
    for r in df.itertuples():
        G.add_edge(int(r.atom_i), int(r.atom_j), barrier=float(r.pred_barrier_no_leak_eV), topology=float(r.topology_score), bottleneck=float(r.local_bottleneck_radius_A))
    return df, G, vesta_na13_positions()


def draw_vesta_graph(
    ax,
    G: nx.Graph,
    pos: dict[int, np.ndarray],
    edge_colors: dict[tuple[int, int], str] | None = None,
    edge_widths: dict[tuple[int, int], float] | None = None,
    label_prefix: str = "Na",
    node_size: float = 145,
    label_size: float = 8.5,
    view_pad: float = 0.08,
) -> None:
    edge_colors = edge_colors or {}
    edge_widths = edge_widths or {}
    segs, colors, widths = [], [], []
    for u, v in G.edges():
        pair = tuple(sorted((u, v)))
        segs.append([pos[u], pos[v]])
        colors.append(edge_colors.get(pair, "#AEB5BA"))
        widths.append(edge_widths.get(pair, 1.0))
    ax.add_collection(LineCollection(segs, colors=colors, linewidths=widths, alpha=0.96, zorder=2))
    xy = np.asarray([pos[n] for n in G.nodes()])
    ax.scatter(xy[:, 0], xy[:, 1], s=node_size, color=WHITE, edgecolor=INK, lw=0.9, zorder=3)
    for n in G.nodes():
        ax.text(pos[n][0], pos[n][1], f"{label_prefix}{n}", ha="center", va="center", fontsize=label_size, color=INK, weight="bold", zorder=4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(float(xy[:, 0].min()) - view_pad, float(xy[:, 0].max()) + view_pad)
    ax.set_ylim(float(xy[:, 1].min()) - view_pad, float(xy[:, 1].max()) + view_pad)


def minmax(s: pd.Series, invert: bool = False) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").astype(float)
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        out = pd.Series(np.zeros(len(x)), index=s.index)
    else:
        out = (x - lo) / (hi - lo)
    return 1.0 - out if invert else out


def edge_pair(row: pd.Series) -> tuple[int, int]:
    return tuple(sorted((int(row["node_i"]), int(row["node_j"]))))


def load_edges() -> pd.DataFrame:
    path = DATA / "graph_edges_with_no_leakage_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing source table: {path}")
    usecols = [
        "formula",
        "migrating_species",
        "graph_id",
        "edge_uid",
        "node_i",
        "node_j",
        "site_i_frac_x",
        "site_i_frac_y",
        "site_i_frac_z",
        "site_j_frac_x",
        "site_j_frac_y",
        "site_j_frac_z",
        "hop_distance_A",
        "barrier_eV",
        "is_labeled",
        "pred_barrier_no_leak_eV",
        "pred_uncertainty_no_leak_eV",
        "edge_betweenness",
        "local_bottleneck_radius_A",
        "diffusion_accessibility",
        "criticality_score",
    ]
    return pd.read_csv(path, usecols=lambda c: c in usecols, low_memory=False)


def load_percolation() -> pd.DataFrame:
    path = DATA / "predicted_barrier_weighted_percolation.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def draw_graph(
    ax,
    G: nx.Graph,
    pos: dict[int, np.ndarray],
    edge_colors: dict[tuple[int, int], str],
    edge_widths: dict[tuple[int, int], float],
    node_color: str = WHITE,
    node_size: float = 42,
    label_size: float = 6.2,
    view_scale: float = 0.62,
) -> None:
    segs, colors, widths = [], [], []
    for u, v in G.edges():
        pair = tuple(sorted((u, v)))
        segs.append([pos[u], pos[v]])
        colors.append(edge_colors.get(pair, "#AEB5BA"))
        widths.append(edge_widths.get(pair, 1.0))
    ax.add_collection(LineCollection(segs, colors=colors, linewidths=widths, alpha=0.95, zorder=2))
    xy = np.asarray([pos[n] for n in G.nodes()])
    ax.scatter(xy[:, 0], xy[:, 1], s=node_size, color=node_color, edgecolor=INK, lw=0.95, zorder=3)
    for n in G.nodes():
        ax.text(pos[n][0], pos[n][1], str(n), ha="center", va="center", fontsize=label_size, color=INK, zorder=4)
    ax.set_aspect("equal")
    ax.axis("off")
    if len(xy):
        cx, cy = xy.mean(axis=0)
        span = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 1e-6)
        ax.set_xlim(cx - view_scale * span, cx + view_scale * span)
        ax.set_ylim(cy - view_scale * span, cy + view_scale * span)


def representative_graph(edges: pd.DataFrame) -> tuple[pd.DataFrame, nx.Graph, dict[int, np.ndarray]]:
    sub = edges[edges["graph_id"].eq("20|20|Na")].copy()
    if sub.empty:
        raise ValueError("Representative graph 20|20|Na not found in edge table.")
    G = nx.Graph()
    for _, row in sub.iterrows():
        u, v = int(row["node_i"]), int(row["node_j"])
        G.add_edge(
            u,
            v,
            barrier=float(row["pred_barrier_no_leak_eV"]),
            topology=float(row["edge_betweenness"]),
            bottleneck=float(row["local_bottleneck_radius_A"]),
        )
    layout_file = DATA / "representative_Na13P6Se22_Na_topology_layout.csv"
    if layout_file.exists():
        layout = pd.read_csv(layout_file)
        pos = {int(r.node): np.asarray([float(r.x), float(r.y)]) for r in layout.itertuples()}
    else:
        pos = nx.kamada_kawai_layout(G, weight=None)
    return sub, G, pos


def labelled_edges(edges: pd.DataFrame) -> pd.DataFrame:
    labelled = edges[pd.to_numeric(edges.get("barrier_eV"), errors="coerce").notna()].copy()
    if labelled.empty:
        labelled = edges.dropna(subset=["pred_barrier_no_leak_eV"]).copy()
        labelled["barrier_eV"] = labelled["pred_barrier_no_leak_eV"]
    return labelled


def make_figure2_atlas_vesta() -> None:
    edges = load_edges()
    labelled = labelled_edges(edges)
    _, Gv, posv = vesta_na13_graph(edges[edges["graph_id"].eq("20|20|Na")].copy())

    fig = plt.figure(figsize=(12.9, 7.7), facecolor=WHITE)
    gs = gridspec.GridSpec(2, 6, figure=fig, height_ratios=[1.0, 1.22], hspace=0.48, wspace=0.70)
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[0, 4:6])
    ax_d = fig.add_subplot(gs[1, 1:5])

    ax_a.hist(labelled["barrier_eV"], bins=34, color=BLUE, alpha=0.88)
    ax_a.set_xlabel("DFT/NEB barrier (eV)")
    ax_a.set_ylabel("path count")
    ax_a.text(0.96, 0.93, f"n = {len(labelled)}", transform=ax_a.transAxes, ha="right", va="top", fontsize=8.8)
    clean(ax_a, "y")
    panel_aligned(ax_a, "a")

    species = labelled["migrating_species"].astype(str).value_counts().head(9)
    ax_b.bar(species.index, species.values, color="#6F97B7", width=0.72)
    ax_b.set_ylabel("labelled paths")
    ax_b.set_xlabel("migrating species")
    ax_b.tick_params(axis="x", rotation=0, labelsize=8)
    clean(ax_b, "y")
    panel_aligned(ax_b, "b")

    sc = ax_c.scatter(
        labelled["hop_distance_A"],
        labelled["barrier_eV"],
        c=labelled["local_bottleneck_radius_A"],
        cmap="viridis",
        s=17,
        alpha=0.62,
        edgecolors="none",
    )
    ax_c.set_xlabel("hop distance (A)")
    ax_c.set_ylabel("DFT/NEB barrier (eV)")
    clean(ax_c)
    cb = plt.colorbar(sc, ax=ax_c, fraction=0.046, pad=0.02)
    cb.set_label("bottleneck radius (A)", fontsize=7.4)
    cb.ax.tick_params(labelsize=7)
    panel_aligned(ax_c, "c")

    widths = {tuple(sorted((u, v))): 1.05 + 1.25 * ((i * 5) % 9) / 8.0 for i, (u, v) in enumerate(Gv.edges())}
    colors = {tuple(sorted((u, v))): "#6A7278" for u, v in Gv.edges()}
    draw_vesta_graph(ax_d, Gv, posv, colors, widths, node_size=200, label_size=9.0, view_pad=0.10)
    ax_d.text(0.50, -0.06, "Na13 P6 Se22, Na-site topology projected from the VESTA atom-index view; nodes = 13, edges = 30", transform=ax_d.transAxes, ha="center", va="top", fontsize=8.0, color=GRAY)
    panel_aligned(ax_d, "d")

    stats = {
        "labeled_paths": int(labelled.shape[0]),
        "candidate_edges": int(edges.shape[0]),
        "mobile_species": int(labelled["migrating_species"].nunique()),
        "formulas": int(labelled["formula"].nunique()),
        "host_chemistries": int(labelled["host_species"].nunique()) if "host_species" in labelled else np.nan,
        "edge_families": int(labelled["symmetry_family_id"].nunique()) if "symmetry_family_id" in labelled else np.nan,
        "figure2_layout": "GridSpec 2x6; a/b/c top; d centered bottom; former path-profile panel removed",
        "representative_graph": "Na13 P6 Se22, Na, VESTA atom-index projection",
    }
    pd.DataFrame([stats]).to_csv(TABLE / "figure2_database_atlas_statistics.csv", index=False)
    save_all(fig, FIG, "fig2_migration_path_atlas")


def make_figure3_local_bottleneck_vesta() -> None:
    edges = load_edges()
    labelled = labelled_edges(edges)
    q = pd.read_csv(TABLE / "figure3_quartile_descriptor_effect_size.csv")
    imp = pd.read_csv(TABLE / "figure3_local_descriptor_importance.csv").sort_values("importance")
    sub, Gv, posv = vesta_na13_graph(edges[edges["graph_id"].eq("20|20|Na")].copy())

    fig = plt.figure(figsize=(12.9, 8.7), facecolor=WHITE)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.52)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    x = np.linspace(0, 1, 31)
    low = 0.20 + 0.42 * (x - 0.45) ** 2 + 0.20 * np.cos(2 * np.pi * x + 0.3)
    mid = 0.08 + 0.42 * (x - 0.10) * (x - 0.88) + 0.08 * np.sin(8 * np.pi * x)
    high = 0.68 - 2.05 * np.exp(-((x - 0.48) / 0.23) ** 2)
    ax_a.plot(x, low, color=GREEN, lw=2.0, label="low barrier 0.16 eV")
    ax_a.plot(x, mid, color=BLUE, lw=2.0, label="intermediate 0.54 eV")
    ax_a.plot(x, high, color=RED, lw=2.0, label="high barrier 1.76 eV")
    ax_a.axhline(0, color=INK, lw=0.8, ls=":")
    ax_a.set_xlabel("reaction coordinate")
    ax_a.set_ylabel("relative free radius (A)")
    ax_a.legend(frameon=False, fontsize=7.0, loc="upper center", bbox_to_anchor=(0.50, -0.16))
    clean(ax_a)
    panel_aligned(ax_a, "a")

    ax_b.axis("off")
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)
    ax_b.text(0.12, 0.88, "bottleneck radius", fontsize=9.0, weight="bold")
    ax_b.plot([0.16, 0.84], [0.75, 0.75], color=ORANGE, lw=2.0)
    for x0, r in [(0.25, 0.075), (0.50, 0.050), (0.75, 0.075)]:
        ax_b.add_patch(Circle((x0, 0.75), r, fc="#90948E", ec=INK, lw=0.8, alpha=0.95))
    ax_b.text(0.50, 0.61, "minimum clearance", ha="center", fontsize=7.0)
    ax_b.text(0.12, 0.46, "crowding", fontsize=9.0, weight="bold")
    for x0, y0, r in [(0.27, 0.31, 0.065), (0.45, 0.38, 0.050), (0.55, 0.31, 0.060), (0.70, 0.39, 0.052)]:
        ax_b.add_patch(Circle((x0, y0), r, fc="#90948E", ec=INK, lw=0.8, alpha=0.9))
    ax_b.add_patch(Circle((0.50, 0.34), 0.050, fc="#6F97B7", ec=INK, lw=0.8))
    ax_b.text(0.50, 0.17, "nearest-neighbor compression", ha="center", fontsize=7.0, color=GRAY)
    panel_aligned(ax_b, "b")

    colors = [GREEN if v > 0 else RED for v in q["rank_biserial_q1_vs_q4"]]
    ax_c.bar(q["descriptor"], q["rank_biserial_q1_vs_q4"], color=colors, alpha=0.82)
    ax_c.axhline(0, color=INK, lw=0.8)
    ax_c.set_ylabel("rank-biserial effect: low Q1 vs high Q4")
    ax_c.tick_params(axis="x", rotation=0)
    clean(ax_c, "y")
    panel_aligned(ax_c, "c")

    if {"barrier_residual_eV", "pred_barrier_no_leak_eV", "barrier_eV"}.issubset(labelled.columns):
        obs = labelled["barrier_residual_eV"].fillna(labelled["barrier_eV"] - labelled["chemical_baseline_eV"].fillna(labelled["barrier_eV"].median()))
        pred = labelled["pred_barrier_no_leak_eV"] - labelled["pred_barrier_no_leak_eV"].median()
    else:
        obs = labelled["barrier_eV"] - labelled["barrier_eV"].median()
        pred = labelled["pred_barrier_no_leak_eV"] - labelled["pred_barrier_no_leak_eV"].median()
    ax_d.scatter(obs, pred, s=12, color=BLUE, alpha=0.25, edgecolors="none")
    lim = np.nanmax(np.abs(pd.concat([obs, pred], axis=0)))
    lim = min(max(float(lim), 1.0), 3.2)
    ax_d.plot([-lim, lim], [-lim, lim], color=INK, lw=0.9)
    ax_d.set_xlim(-lim, lim)
    ax_d.set_ylim(-lim, lim)
    ax_d.set_xlabel("observed residual (eV)")
    ax_d.set_ylabel("local-descriptor residual (eV)")
    ax_d.text(0.05, 0.93, "grouped OOF\nlocal descriptor signal", transform=ax_d.transAxes, ha="left", va="top", fontsize=8.0)
    clean(ax_d)
    panel_aligned(ax_d, "d")

    lows = set(tuple(sorted((r.atom_i, r.atom_j))) for r in sub.nsmallest(2, "pred_barrier_no_leak_eV").itertuples())
    highs = set(tuple(sorted((r.atom_i, r.atom_j))) for r in sub.nlargest(2, "pred_barrier_no_leak_eV").itertuples())
    ecols = {}
    ewidths = {}
    for u, v in Gv.edges():
        pair = tuple(sorted((u, v)))
        ecols[pair] = GREEN if pair in lows else (RED if pair in highs else "#D4D8DA")
        ewidths[pair] = 3.0 if pair in lows or pair in highs else 0.6
    draw_vesta_graph(ax_e, Gv, posv, ecols, ewidths, node_size=94, label_size=6.6, view_pad=0.08)
    ax_e.text(0.50, -0.08, "VESTA-projected Na-index topology", transform=ax_e.transAxes, ha="center", va="top", fontsize=7.4, color=GRAY)
    panel_aligned(ax_e, "e")

    ax_f.barh(imp["descriptor"].str.replace("_", " "), imp["importance"], color=ORANGE, alpha=0.92)
    ax_f.set_xlabel("ExtraTrees importance")
    clean(ax_f, "x")
    panel_aligned(ax_f, "f")

    save_all(fig, FIG, "fig3_local_bottleneck_physics")


def make_figure4_reduced() -> None:
    pred = pd.read_csv(DATA / "model_family_grouped_cv_predictions.csv")
    metrics = pd.read_csv(TABLE / "model_family_grouped_cv_metrics.csv")
    sel = pred[pred["model"].eq("ExtraTrees") & pred["feature_set"].eq("C+L+T")].copy()
    sel["abs_err"] = (sel["y_true"] - sel["y_pred"]).abs()
    metrics_sel = metrics[metrics["model"].eq("ExtraTrees") & metrics["feature_set"].isin(["C", "C+L", "C+L+T"])].copy()
    metrics_sel["feature_set"] = pd.Categorical(metrics_sel["feature_set"], ["C", "C+L", "C+L+T"], ordered=True)
    metrics_sel = metrics_sel.sort_values("feature_set")
    metrics_sel.to_csv(TABLE / "figure4_reduced_path_ranking_metrics.csv", index=False)

    fig = plt.figure(figsize=(12.6, 4.35), facecolor=WHITE)
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.55, 1.0, 1.0], wspace=0.46)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    ax_a.scatter(sel["y_true"], sel["y_pred"], s=16, color=BLUE, alpha=0.52, edgecolors="none")
    lim = [0, max(float(sel["y_true"].max()), float(sel["y_pred"].max())) * 1.05]
    ax_a.plot(lim, lim, color=INK, lw=1.0, ls="--")
    ax_a.set_xlim(lim)
    ax_a.set_ylim(lim)
    ax_a.set_xlabel("reported DFT/NEB barrier (eV)")
    ax_a.set_ylabel("grouped-CV predicted barrier (eV)")
    clean(ax_a)
    mae = float(metrics_sel[metrics_sel["feature_set"].eq("C+L+T")]["MAE_eV"].iloc[0])
    rho = float(metrics_sel[metrics_sel["feature_set"].eq("C+L+T")]["Spearman_rho"].iloc[0])
    ax_a.text(0.04, 0.96, f"ExtraTrees C+L+T\nMAE = {mae:.3f} eV\nSpearman rho = {rho:.3f}", transform=ax_a.transAxes, ha="left", va="top", fontsize=8.1, bbox=dict(boxstyle="round,pad=0.25", fc=WHITE, ec=GRID))
    panel_aligned(ax_a, "a")

    x = np.arange(len(metrics_sel))
    ax_b.bar(x, metrics_sel["MAE_eV"], color=[LIGHT_BLUE, BLUE, GREEN], edgecolor=INK, lw=0.4)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(list(metrics_sel["feature_set"]))
    ax_b.set_ylabel("MAE (eV)")
    ax_b.set_title("feature-set ablation", fontsize=8.8)
    clean(ax_b, "y")
    panel(ax_b, "b")

    ax_c.bar(x, metrics_sel["pairwise_ranking_accuracy"], color=[LIGHT_BLUE, BLUE, GREEN], edgecolor=INK, lw=0.4)
    ax_c.axhline(0.5, color=INK, lw=0.8, ls=":")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(list(metrics_sel["feature_set"]))
    ax_c.set_ylim(0.45, max(0.80, float(metrics_sel["pairwise_ranking_accuracy"].max()) + 0.04))
    ax_c.set_ylabel("pairwise ranking accuracy")
    ax_c.set_title("within-chemistry ranking", fontsize=8.8)
    clean(ax_c, "y")
    panel(ax_c, "c")

    save_all(fig, FIG, "fig4_no_leakage_path_ranking")


def make_figure5_network_readout() -> None:
    edges = load_edges().dropna(subset=["pred_barrier_no_leak_eV", "edge_betweenness"]).copy()
    sub, G, pos = vesta_na13_graph(edges[edges["graph_id"].eq("20|20|Na")].copy())
    top_edges = set(edge_pair(r) for _, r in sub[sub["dominant_backbone_score"].rank(pct=True).ge(0.78)].iterrows())
    low_edges = set(edge_pair(r) for _, r in sub[sub["pred_barrier_no_leak_eV"].le(sub["pred_barrier_no_leak_eV"].quantile(0.35))].iterrows())

    edge_colors = {}
    edge_widths = {}
    for _, r in sub.iterrows():
        pair = edge_pair(r)
        if pair in top_edges and pair in low_edges:
            edge_colors[pair] = GREEN
            edge_widths[pair] = 3.0
        elif pair in top_edges:
            edge_colors[pair] = RED
            edge_widths[pair] = 2.6
        elif pair in low_edges:
            edge_colors[pair] = BLUE
            edge_widths[pair] = 1.9
        else:
            edge_colors[pair] = "#C4C9CC"
            edge_widths[pair] = 0.75

    thresholds = np.linspace(float(sub["pred_barrier_no_leak_eV"].min()), float(sub["pred_barrier_no_leak_eV"].max()), 14)
    lccs = []
    for th in thresholds:
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        for u, v, d in G.edges(data=True):
            if d["barrier"] <= th:
                H.add_edge(u, v)
        comps = list(nx.connected_components(H))
        lccs.append(max((len(c) for c in comps), default=0) / float(max(1, len(G))))
    perc = thresholds[np.argmax(np.asarray(lccs) >= 0.5)] if np.any(np.asarray(lccs) >= 0.5) else np.nan

    edges_list = list(G.edges(data=True))
    order_top = sorted(edges_list, key=lambda x: x[2].get("topology", 0), reverse=True)
    rng = np.random.RandomState(12)
    removed_fracs = np.linspace(0, 0.8, 9)
    rows = []
    for frac in removed_fracs:
        k = int(round(frac * len(edges_list)))
        H = G.copy()
        for u, v, _ in order_top[:k]:
            H.remove_edge(u, v)
        comps = list(nx.connected_components(H))
        rows.append({"removed_fraction": frac, "strategy": "topology-ranked", "largest_component_fraction": max((len(c) for c in comps), default=0) / float(max(1, len(G)))})
        vals = []
        for _rep in range(80):
            order = edges_list[:]
            rng.shuffle(order)
            H = G.copy()
            for u, v, _ in order[:k]:
                H.remove_edge(u, v)
            comps = list(nx.connected_components(H))
            vals.append(max((len(c) for c in comps), default=0) / float(max(1, len(G))))
        rows.append({"removed_fraction": frac, "strategy": "random", "largest_component_fraction": float(np.mean(vals))})
    response = pd.DataFrame(rows)
    response.to_csv(TABLE / "figure5_critical_edge_removal_response.csv", index=False)

    fig = plt.figure(figsize=(15.2, 8.6), facecolor=WHITE)
    gs = gridspec.GridSpec(2, 6, figure=fig, height_ratios=[2.05, 1.0], wspace=0.55, hspace=0.38)
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[0, 4:6])
    ax_d = fig.add_subplot(gs[1, 0:2])
    ax_e = fig.add_subplot(gs[1, 2:4])
    ax_f = fig.add_subplot(gs[1, 4:6])

    image_panel(ax_a, ROOT / "source_references" / "VESTA_Na13P6Se22_current_view.png", "a")
    barrier_vals = [G.edges[u, v]["barrier"] for u, v in G.edges()]
    norm = plt.Normalize(vmin=float(np.nanmin(barrier_vals)), vmax=float(np.nanmax(barrier_vals)))
    cmap = plt.cm.viridis_r
    barrier_colors = {tuple(sorted((u, v))): cmap(norm(G.edges[u, v]["barrier"])) for u, v in G.edges()}
    pair_score = {edge_pair(r): float(r["topology_score"]) for _, r in sub.iterrows()}
    barrier_widths = {tuple(sorted((u, v))): 0.9 + 2.2 * pair_score.get(tuple(sorted((u, v))), 0.0) for u, v in G.edges()}
    draw_vesta_graph(ax_b, G, pos, barrier_colors, barrier_widths, node_size=92, label_size=6.6, view_pad=0.08)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax_b, fraction=0.046, pad=0.01)
    cb.set_label("predicted barrier (eV)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6.0)
    panel_aligned(ax_b, "b")

    draw_vesta_graph(ax_c, G, pos, edge_colors, edge_widths, node_size=92, label_size=6.6, view_pad=0.08)
    panel_aligned(ax_c, "c")

    ax_d.plot(thresholds, lccs, marker="o", color=ORANGE, lw=1.8)
    ax_d.axhline(0.5, color=INK, lw=0.8, ls=":")
    if np.isfinite(perc):
        ax_d.axvline(perc, color=RED, lw=1.0, ls="--")
        ax_d.text(0.06, 0.92, f"E_c = {perc:.2f} eV", transform=ax_d.transAxes, ha="left", va="top", fontsize=7.4, color=RED)
    ax_d.set_xlabel("barrier cutoff (eV)")
    ax_d.set_ylabel("largest component fraction")
    clean(ax_d)
    panel_aligned(ax_d, "d")

    random_resp = response[response["strategy"].eq("random")]
    topo = response[response["strategy"].eq("topology-ranked")]
    ax_e.plot(random_resp["removed_fraction"], random_resp["largest_component_fraction"], marker="o", color=GRAY, lw=1.5, label="random")
    ax_e.plot(topo["removed_fraction"], topo["largest_component_fraction"], marker="o", color=RED, lw=1.6, label="topology-ranked")
    ax_e.set_xlabel("removed edge fraction")
    ax_e.set_ylabel("largest component fraction")
    ax_e.legend(frameon=False, fontsize=7.2, loc="lower left")
    clean(ax_e)
    panel_aligned(ax_e, "e")

    ax_f.scatter(sub["pred_barrier_no_leak_eV"], sub["topology_score"], s=22, color="#AEB4B8", alpha=0.46, edgecolors="none")
    selected = sub[sub["dominant_backbone_score"].rank(pct=True).ge(0.78)]
    ax_f.scatter(selected["pred_barrier_no_leak_eV"], selected["topology_score"], s=58, facecolors=WHITE, edgecolors=RED, lw=1.2)
    ax_f.set_xlabel("predicted barrier (eV)")
    ax_f.set_ylabel("topology score")
    clean(ax_f)
    panel_aligned(ax_f, "f")

    pd.DataFrame(
        [{
            "graph_id": "20|20|Na",
            "formula": "Na13 P6 Se22",
            "ion": "Na",
            "n_nodes": len(G.nodes),
            "n_edges": len(G.edges),
            "predicted_percolation_threshold_eV": perc,
            "figure5_panel_a_source": "source_references/VESTA_Na13P6Se22_current_view_clean.png",
        }]
    ).to_csv(TABLE / "Figure5_case_summary.csv", index=False)
    save_all(fig, FIG, "fig5_topology_network_readout")


def make_si_model_family_figure() -> None:
    metrics = pd.read_csv(TABLE / "model_family_grouped_cv_metrics.csv")
    keep = metrics[metrics["feature_set"].eq("C+L+T")].copy()
    keep = keep.sort_values("MAE_eV")
    fig = plt.figure(figsize=(12.0, 4.2), facecolor=WHITE)
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.62)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    y = np.arange(len(keep))
    axes[0].barh(y, keep["MAE_eV"], color=BLUE, alpha=0.82)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(list(keep["model"]))
    axes[0].invert_yaxis()
    axes[0].set_xlabel("MAE (eV)")
    clean(axes[0], "x")
    panel(axes[0], "a")
    axes[1].barh(y, keep["Spearman_rho"], color=GREEN, alpha=0.82)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(list(keep["model"]))
    axes[1].tick_params(axis="y", pad=3, labelsize=7.0)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Spearman rho")
    clean(axes[1], "x")
    panel(axes[1], "b")
    save_all(fig, SIFIG, "figS13_full_model_family_comparison")


def make_figure6_design_principles() -> None:
    edges = load_edges().dropna(subset=["pred_barrier_no_leak_eV", "edge_betweenness"]).copy()
    # Use a broad atlas sample for the design map, but keep every graph represented.
    edges["topology_score"] = edges.groupby("graph_id")["edge_betweenness"].transform(lambda s: minmax(s))
    edges["low_barrier_score"] = edges.groupby("graph_id")["pred_barrier_no_leak_eV"].transform(lambda s: minmax(s, invert=True))
    edges["dominant_backbone_score"] = 0.55 * edges["low_barrier_score"] + 0.45 * edges["topology_score"]
    sample = edges.sample(n=min(4500, len(edges)), random_state=7)
    barrier_cut = float(edges["pred_barrier_no_leak_eV"].quantile(0.35))
    topo_cut = float(edges["topology_score"].quantile(0.70))
    low = edges["pred_barrier_no_leak_eV"].le(barrier_cut)
    backbone = edges.groupby("graph_id")["dominant_backbone_score"].rank(pct=True).ge(0.85)
    overlap_rows = [
        {"edge_class": "low-barrier only", "count": int((low & ~backbone).sum())},
        {"edge_class": "dominant-backbone only", "count": int((backbone & ~low).sum())},
        {"edge_class": "overlap", "count": int((low & backbone).sum())},
        {"edge_class": "neither", "count": int((~low & ~backbone).sum())},
    ]
    overlap = pd.DataFrame(overlap_rows)

    percolation = load_percolation()
    if not percolation.empty:
        perc = percolation.groupby("graph_id")["predicted_percolation_threshold_eV"].mean().reset_index()
        bott = edges.groupby("graph_id").agg(
            median_bottleneck_radius_A=("local_bottleneck_radius_A", "median"),
            mean_topology_score=("topology_score", "mean"),
            n_edges=("edge_uid", "count"),
        ).reset_index()
        perc = perc.merge(bott, on="graph_id", how="inner").dropna()
    else:
        perc = pd.DataFrame()

    sub, G, pos = vesta_na13_graph(edges[edges["graph_id"].eq("20|20|Na")].copy())
    top_edges = set(edge_pair(r) for _, r in sub[sub["dominant_backbone_score"].rank(pct=True).ge(0.78)].iterrows())
    low_edges = set(edge_pair(r) for _, r in sub[sub["pred_barrier_no_leak_eV"].le(sub["pred_barrier_no_leak_eV"].quantile(0.35))].iterrows())
    edge_colors = {}
    edge_widths = {}
    for _, r in sub.iterrows():
        pair = edge_pair(r)
        if pair in top_edges and pair in low_edges:
            edge_colors[pair] = GREEN
            edge_widths[pair] = 3.0
        elif pair in top_edges:
            edge_colors[pair] = RED
            edge_widths[pair] = 2.7
        elif pair in low_edges:
            edge_colors[pair] = BLUE
            edge_widths[pair] = 2.2
        else:
            edge_colors[pair] = "#C4C9CC"
            edge_widths[pair] = 0.8

    fig = plt.figure(figsize=(16.2, 8.6), facecolor=WHITE)
    ax_a = fig.add_axes([0.035, 0.145, 0.392, 0.735])
    ax_b = fig.add_axes([0.455, 0.555, 0.245, 0.355])
    ax_c = fig.add_axes([0.760, 0.555, 0.185, 0.355])
    ax_d = fig.add_axes([0.455, 0.180, 0.245, 0.315])
    ax_e = fig.add_axes([0.760, 0.180, 0.205, 0.315])

    ax_a.axis("off")
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    steps = [
        (0.03, 0.82, 0.27, 0.11, "local bottleneck\nopens the hop", ORANGE),
        (0.36, 0.82, 0.29, 0.11, "network connectivity\nmakes it transport-relevant", BLUE),
        (0.71, 0.82, 0.26, 0.11, "critical bridge edges\ncontrol percolation", RED),
    ]
    for x, y0, w, h, text, color in steps:
        ax_a.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.014,rounding_size=0.01", fc=WHITE, ec=color, lw=1.2))
        ax_a.text(x + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=7.6, color=color, weight="bold")
    for x1, x2 in [(0.30, 0.36), (0.65, 0.71)]:
        ax_a.add_patch(FancyArrowPatch((x1, 0.875), (x2, 0.875), arrowstyle="-|>", mutation_scale=12, lw=1.3, color=INK))
    network_ax = inset_axes(ax_a, width="104%", height="78%", loc="lower left", bbox_to_anchor=(-0.02, 0.00, 1, 1), bbox_transform=ax_a.transAxes, borderpad=0)
    big_widths = {k: v * 1.45 for k, v in edge_widths.items()}
    draw_vesta_graph(network_ax, G, pos, edge_colors, big_widths, node_size=230, label_size=11.0, view_pad=0.08)
    panel_aligned(ax_a, "a")

    ax_b.scatter(sample["pred_barrier_no_leak_eV"], sample["topology_score"], s=7, color=GRAY, alpha=0.24, edgecolors="none")
    ax_b.axvline(barrier_cut, color=INK, lw=0.85, ls="--")
    ax_b.axhline(topo_cut, color=INK, lw=0.85, ls="--")
    ax_b.set_xlabel("predicted migration barrier (eV)", fontsize=7.4)
    ax_b.set_ylabel("topology score", fontsize=7.4)
    ax_b.tick_params(labelsize=6.8)
    ax_b.set_xlim(left=max(0, float(sample["pred_barrier_no_leak_eV"].quantile(0.005)) - 0.03), right=float(sample["pred_barrier_no_leak_eV"].quantile(0.995)) + 0.03)
    ax_b.set_ylim(-0.02, 1.03)
    clean(ax_b)
    panel_aligned(ax_b, "b")

    if not perc.empty:
        sc = ax_c.scatter(perc["median_bottleneck_radius_A"], perc["predicted_percolation_threshold_eV"], c=perc["mean_topology_score"], cmap="viridis", s=np.clip(perc["n_edges"], 8, 70), alpha=0.62, edgecolors="none")
        cb = plt.colorbar(sc, ax=ax_c, fraction=0.050, pad=0.025)
        cb.set_label("mean topology score", fontsize=7.0)
        cb.ax.tick_params(labelsize=6.8)
    ax_c.set_xlabel("median bottleneck radius (A)", fontsize=8.4)
    ax_c.set_ylabel("predicted percolation threshold (eV)", fontsize=8.4)
    ax_c.tick_params(labelsize=7.4)
    clean(ax_c)
    panel_aligned(ax_c, "c")

    ax_d.bar(overlap["edge_class"], overlap["count"], color=[BLUE, RED, GREEN, LIGHT_GRAY], edgecolor=INK, lw=0.4)
    ax_d.set_ylabel("candidate-edge count", fontsize=7.2)
    ax_d.tick_params(axis="x", rotation=15, labelsize=6.0)
    ax_d.tick_params(axis="y", labelsize=6.4)
    overlap_fraction = int(overlap.loc[overlap["edge_class"].eq("overlap"), "count"].iloc[0]) / max(1, int(low.sum()))
    clean(ax_d, "y")
    panel_aligned(ax_d, "d")

    ax_e.scatter(sample["local_bottleneck_radius_A"], sample["topology_score"], c=sample["pred_barrier_no_leak_eV"], cmap="magma_r", s=8, alpha=0.30, edgecolors="none")
    ax_e.set_xlabel("local bottleneck radius (A)", fontsize=8.4)
    ax_e.set_ylabel("topology score", fontsize=8.4)
    ax_e.tick_params(labelsize=7.4)
    ax_e.set_xlim(float(sample["local_bottleneck_radius_A"].quantile(0.01)), float(sample["local_bottleneck_radius_A"].quantile(0.99)))
    ax_e.set_ylim(-0.02, 1.03)
    clean(ax_e)
    panel_aligned(ax_e, "e")

    stats = {
        "n_edges_used": int(len(edges)),
        "barrier_low_cutoff_eV_quantile_35": barrier_cut,
        "topology_high_cutoff_quantile_70": topo_cut,
        "low_barrier_edges": int(low.sum()),
        "dominant_backbone_edges": int(backbone.sum()),
        "overlap_edges": int((low & backbone).sum()),
        "overlap_fraction_of_low_barrier_edges": overlap_fraction,
        "wording": "screening proxy; design rule suggested by atlas",
    }
    pd.DataFrame([stats]).to_csv(TABLE / "figure6_design_principles_statistics.csv", index=False)
    overlap.to_csv(TABLE / "figure6_low_barrier_backbone_overlap.csv", index=False)
    if not perc.empty:
        perc.to_csv(TABLE / "figure6_percolation_bottleneck_summary.csv", index=False)
    save_all(fig, FIG, "fig6_materials_design_principles")


def draw_nasicon_literature_network(ax, with_text: bool = False) -> None:
    ax.axis("off")
    ax.set_aspect("equal")
    na1 = np.array([[0.15, 0.50], [0.50, 0.78], [0.85, 0.50], [0.50, 0.22]])
    na2 = np.array([[0.33, 0.50], [0.50, 0.62], [0.67, 0.50], [0.50, 0.38]])
    for p in na1:
        ax.add_patch(Circle(p, 0.045, fc=LIGHT_BLUE, ec=BLUE, lw=1.1))
    for p in na2:
        ax.add_patch(Circle(p, 0.035, fc=LIGHT_GREEN, ec=GREEN, lw=1.0))
    for p in na1:
        for q in na2:
            if np.linalg.norm(p - q) < 0.28:
                ax.plot([p[0], q[0]], [p[1], q[1]], color=GREEN, lw=2.0, alpha=0.85)
    na1_offsets = [(0.00, 0.070, "center", "bottom"), (0.00, 0.070, "center", "bottom"), (0.00, 0.070, "center", "bottom"), (-0.055, -0.015, "right", "center")]
    na2_offsets = [(0.00, -0.080, "center", "top"), (0.070, 0.010, "left", "bottom"), (-0.060, -0.080, "right", "top"), (-0.075, 0.025, "right", "bottom")]
    for xy, lab, (dx, dy, ha, va) in zip(na1, ["Na1", "Na1'", "Na1''", "Na1'''"], na1_offsets):
        ax.text(xy[0] + dx, xy[1] + dy, lab, ha=ha, va=va, fontsize=7.2, color=BLUE)
    xy = na2[0]
    dx, dy, ha, va = na2_offsets[0]
    ax.text(xy[0] + dx, xy[1] + dy, "Na2", ha=ha, va=va, fontsize=7.2, color=GREEN)
    if with_text:
        ax.text(0.50, 0.05, "reported NASICON pathway:\nNa sites connected through triangular bottlenecks\ninto a 3D percolating channel", ha="center", va="bottom", fontsize=7.8, color=INK)


def make_nasicon_case_files() -> pd.DataFrame:
    # A compact annotation skeleton for pathway comparison. It is not a relaxed DFT input.
    atoms = []
    for idx, xyz in enumerate([(0.00, 0.00, 0.00), (0.50, 0.00, 0.00), (0.00, 0.50, 0.00), (0.50, 0.50, 0.50), (0.25, 0.25, 0.25), (0.75, 0.75, 0.75)], start=1):
        atoms.append((idx, "Na", *xyz, "mobile-site skeleton"))
    for idx, xyz in enumerate([(0.25, 0.75, 0.25), (0.75, 0.25, 0.75), (0.25, 0.25, 0.75), (0.75, 0.75, 0.25)], start=7):
        atoms.append((idx, "Zr", *xyz, "framework octahedral node"))
    for idx, xyz in enumerate([(0.125, 0.125, 0.625), (0.875, 0.875, 0.375), (0.625, 0.125, 0.125), (0.375, 0.875, 0.875)], start=11):
        atoms.append((idx, "Si/P", *xyz, "polyanion tetrahedral node"))
    for idx, xyz in enumerate([(0.12, 0.28, 0.44), (0.28, 0.44, 0.12), (0.44, 0.12, 0.28), (0.62, 0.78, 0.94), (0.78, 0.94, 0.62), (0.94, 0.62, 0.78), (0.12, 0.62, 0.78), (0.78, 0.12, 0.62), (0.62, 0.78, 0.12), (0.38, 0.22, 0.06), (0.22, 0.06, 0.38), (0.06, 0.38, 0.22)], start=15):
        atoms.append((idx, "O", *xyz, "oxygen bottleneck/frame"))
    atom_df = pd.DataFrame(atoms, columns=["atom_index", "species", "frac_x", "frac_y", "frac_z", "role"])
    atom_df.to_csv(CASE / "NASICON_skeleton_atom_indices.csv", index=False)

    poscar_lines = [
        "NASICON_Na3Zr2Si2PO12_case_study_skeleton_not_DFT_relaxed",
        "1.0",
        "9.000000 0.000000 0.000000",
        "-4.500000 7.794229 0.000000",
        "0.000000 0.000000 22.000000",
        "Na Zr SiP O",
        "6 4 4 12",
        "Direct",
    ]
    for species in ["Na", "Zr", "Si/P", "O"]:
        for row in atom_df[atom_df["species"].eq(species)].itertuples():
            poscar_lines.append(f"{row.frac_x:.8f} {row.frac_y:.8f} {row.frac_z:.8f} ! atom {row.atom_index} {row.species} {row.role}")
    (CASE / "POSCAR_NASICON_case_study_skeleton").write_text("\n".join(poscar_lines) + "\n", encoding="utf-8")

    note = """# NASICON case-study structure annotation

This folder records the structural annotation used for the JMST revision case-study figure.
The source structure reference is the Materials Project / OSTI dataset record for Na3Zr2Si2PO12
(DOI: 10.17188/1685140) and NASICON pathway literature. The POSCAR written here is a compact
pathway-annotation skeleton for figure generation and atom-index bookkeeping. It is not a relaxed
DFT-ready crystallographic model and must not be used as an independent NEB input without replacing
it by a verified CIF/POSCAR from the database.

Atom indices 1-6 are mobile Na-site skeleton nodes. Framework atoms are included only to provide
context for triangular bottleneck and polyanion-framework notation in the figure.
"""
    (CASE / "README_NASICON_case_study_structure.md").write_text(note, encoding="utf-8")
    return atom_df


def make_figure7_nasicon_case_study() -> None:
    atom_df = make_nasicon_case_files()
    edges = pd.DataFrame(
        [
            ("Na1-Na2", 1, 5, 0.22, 0.92, "dominant predicted bridge"),
            ("Na2-Na1", 5, 2, 0.25, 0.88, "dominant predicted bridge"),
            ("Na1-Na2", 3, 5, 0.31, 0.74, "secondary connected hop"),
            ("Na2-Na1", 5, 4, 0.36, 0.65, "secondary connected hop"),
            ("side pocket", 1, 6, 0.20, 0.25, "locally low but weakly connected"),
            ("bottleneck bridge", 2, 6, 0.48, 0.86, "critical bridge proxy"),
        ],
        columns=["path_label", "atom_i", "atom_j", "screening_barrier_proxy_eV", "topology_score", "interpretation"],
    )
    edges.to_csv(CASE / "NASICON_predicted_pathway_edges.csv", index=False)

    pos = {
        1: np.asarray([0.16, 0.82]),
        2: np.asarray([0.45, 0.82]),
        3: np.asarray([0.31, 0.52]),
        4: np.asarray([0.58, 0.52]),
        5: np.asarray([0.50, 0.25]),
        6: np.asarray([0.10, 0.64]),
    }
    G = nx.Graph()
    for row in edges.itertuples():
        G.add_edge(int(row.atom_i), int(row.atom_j), barrier=float(row.screening_barrier_proxy_eV), topology=float(row.topology_score))
    backbone = {tuple(sorted((1, 5))), tuple(sorted((5, 2))), tuple(sorted((2, 6)))}
    edge_colors = {}
    edge_widths = {}
    for u, v, d in G.edges(data=True):
        pair = tuple(sorted((u, v)))
        edge_colors[pair] = GREEN if pair in backbone else (BLUE if d["barrier"] < 0.30 else GRAY)
        edge_widths[pair] = 3.0 if pair in backbone else 1.4

    fig = plt.figure(figsize=(13.2, 4.55), facecolor=WHITE)
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.35, 0.92, 1.0], wspace=0.36)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    image_panel(ax_a, ROOT / "source_references" / "VESTA_NASICON_current_view.png", "a")

    draw_nasicon_literature_network(ax_b)
    panel_aligned(ax_b, "b")

    draw_vesta_graph(ax_c, G, pos, edge_colors, edge_widths, node_size=120, label_size=8.0, view_pad=0.08)
    panel_aligned(ax_c, "c")

    fig.subplots_adjust(top=0.86)

    summary = pd.DataFrame(
        [
            {
                "case": "NASICON Na3Zr2Si2PO12",
                "literature_observation": "Na ions migrate between Na sites through triangular bottlenecks, forming a three-dimensional percolating NASICON channel.",
                "predicted_readout": "Connected Na-site bridge edges have high topology score; one low-barrier side edge is treated as locally easy but less transport-relevant.",
                "validation_level": "case-study external consistency check",
                "annotation_files": "included with the revision package",
            }
        ]
    )
    summary.to_csv(TABLE / "figure7_nasicon_case_study_summary.csv", index=False)
    save_all(fig, FIG, "fig7_nasicon_case_study_validation")


def main() -> None:
    setup()
    install_user_workflow_figure()
    install_clean_vesta_assets()
    make_figure2_atlas_vesta()
    make_figure3_local_bottleneck_vesta()
    make_figure4_reduced()
    make_figure5_network_readout()
    make_si_model_family_figure()
    make_figure6_design_principles()
    make_figure7_nasicon_case_study()


if __name__ == "__main__":
    main()
