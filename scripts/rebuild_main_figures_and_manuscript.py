from __future__ import annotations

import math
import shutil
import sys
import textwrap
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / "manuscript_diffusion_network_v1"
FIG = ROOT / "figures"
TABLE = ROOT / "tables"
SRC = ROOT / "scripts"

sys.path.insert(0, str(BASE / "paper_figures_pub_v2" / "scripts"))
from shared_plot_utils import (  # noqa: E402
    EDGES,
    MAIN_DB,
    SELECTED_PROFILES,
    PAL,
    build_graph,
    center_graph_view,
    minmax,
    panel,
    path_profile_table,
)


WHITE = "#FFFFFF"
INK = "#1E262B"
GRID = "#DDE1E5"
BLUE = "#174A7C"
BLUE2 = "#6F97B7"
ORANGE = "#D66E2A"
GREEN = "#4D8E57"
RED = "#B94D3F"
PURPLE = "#6E5AA6"
LIGHT_BLUE = "#EAF2F8"
LIGHT_GREEN = "#EAF5EE"
LIGHT_ORANGE = "#FCEBDD"


def setup_white() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    SRC.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_all(fig: plt.Figure, stem: str) -> None:
    for ext, kwargs in {
        "png": dict(dpi=600),
        "pdf": {},
        "svg": {},
    }.items():
        fig.savefig(FIG / f"{stem}.{ext}", bbox_inches="tight", facecolor=WHITE, **kwargs)
    plt.close(fig)


def clean_grid(ax, axis: str = "both") -> None:
    ax.grid(color=GRID, lw=0.65, alpha=0.8, axis=axis)


def compact_formula(text: str) -> str:
    return str(text).replace(" ", "")


def draw_mini_network(ax, x, y, w, h, color=BLUE, highlight=ORANGE):
    pts = np.array(
        [
            [0.15, 0.20],
            [0.30, 0.75],
            [0.55, 0.55],
            [0.80, 0.82],
            [0.72, 0.25],
            [0.42, 0.28],
        ]
    )
    pts[:, 0] = x + pts[:, 0] * w
    pts[:, 1] = y + pts[:, 1] * h
    edges = [(0, 1), (1, 2), (2, 3), (2, 4), (4, 5), (5, 0), (2, 5)]
    for i, j in edges:
        lw = 2.1 if (i, j) in [(1, 2), (2, 4)] else 1.0
        ec = highlight if (i, j) in [(1, 2), (2, 4)] else color
        ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]], color=ec, lw=lw, alpha=0.95)
    ax.scatter(pts[:, 0], pts[:, 1], s=34, color=WHITE, edgecolor=color, lw=1.1, zorder=3)


def draw_matrix(ax, x, y, w, h):
    rows, cols = 5, 7
    colors = [WHITE, LIGHT_BLUE, LIGHT_GREEN, LIGHT_ORANGE]
    for i in range(rows):
        for j in range(cols):
            fc = colors[(i * 2 + j) % len(colors)]
            ax.add_patch(Rectangle((x + j * w / cols, y + i * h / rows), w / cols, h / rows, fc=fc, ec="#C5CDD3", lw=0.45))
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec="#77828A", lw=0.8))


def figure1_framework_rebuilt() -> None:
    fig, ax = plt.subplots(figsize=(16.4, 8.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def wrap_lines(text, width):
        return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))

    def box(x, y, w, h, title, bullets, fc=WHITE, ec=BLUE, dashed=False, icon=None, title_size=10.5, bullet_size=6.9, wrap=23):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            fc=fc,
            ec=ec,
            lw=1.25,
            linestyle="--" if dashed else "-",
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h - 0.048, wrap_lines(title, max(14, wrap)), ha="center", va="top", color=ec, fontsize=title_size, weight="bold", linespacing=1.02)
        if icon == "network":
            draw_mini_network(ax, x + 0.028, y + 0.070, w * 0.30, h * 0.36)
            tx = x + w * 0.42
        elif icon == "matrix":
            draw_matrix(ax, x + 0.026, y + 0.060, w * 0.30, h * 0.42)
            tx = x + w * 0.42
        elif icon == "target":
            cx, cy = x + w * 0.22, y + h * 0.48
            for rr in [0.070, 0.044, 0.017]:
                ax.add_patch(Circle((cx, cy), rr, fc=BLUE if rr == 0.017 else "none", ec=BLUE, lw=1.0))
            tx = x + w * 0.48
        else:
            tx = x + w * 0.14
        yy = y + h - 0.135
        for b in bullets:
            wrapped = wrap_lines(b, wrap)
            ax.text(tx, yy, u"\u2022 " + wrapped, ha="left", va="top", color=INK, fontsize=bullet_size, linespacing=1.08)
            yy -= 0.030 * (wrapped.count("\n") + 1)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=17, lw=1.8, color=BLUE))

    ax.text(
        0.018,
        0.94,
        "1. Atlas construction",
        color=WHITE,
        fontsize=15,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.32,rounding_size=0.12", fc=BLUE, ec=BLUE),
    )

    box(
        0.02,
        0.54,
        0.18,
        0.34,
        "Source data",
        ["barrier database", "crystal structures", "mobile species / vacancies"],
        fc=WHITE,
        icon="network",
        wrap=25,
    )

    box(
        0.235,
        0.54,
        0.20,
        0.34,
        "Candidate migration-path atlas",
        ["identify mobile sites", "enumerate neighboring hops", "merge equivalent edges"],
        fc=WHITE,
        icon="network",
        wrap=25,
    )

    ax.add_patch(
        FancyBboxPatch(
            (0.47, 0.525),
            0.51,
            0.375,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            fc=WHITE,
            ec=BLUE,
            lw=1.2,
            linestyle="--",
        )
    )
    ax.text(
        0.725,
        0.875,
        "Integrated path characterization on the candidate migration graph",
        ha="center",
        va="center",
        color=BLUE,
        fontsize=12.0,
        weight="bold",
        style="italic",
    )
    box(0.49, 0.555, 0.17, 0.27, "Local bottleneck descriptors", ["path profile", "free/bottleneck radius", "local coordination"], fc=WHITE, ec=ORANGE, bullet_size=6.6, wrap=17)
    ax.plot([0.515, 0.590], [0.655, 0.655], color=INK, lw=1.0)
    ax.add_patch(Circle((0.515, 0.655), 0.013, fc=LIGHT_BLUE, ec=INK, lw=0.8))
    ax.add_patch(Circle((0.590, 0.655), 0.018, fc=LIGHT_ORANGE, ec=INK, lw=0.8))
    for r in [0.025, 0.045, 0.065]:
        ax.add_patch(Circle((0.558, 0.655), r, fc="none", ec="#9AA1A6", lw=0.6, ls="--"))

    box(0.68, 0.555, 0.16, 0.27, "Graph topology", ["node degree", "edge betweenness", "critical edges"], fc=WHITE, ec=GREEN, bullet_size=6.6, wrap=16)
    draw_mini_network(ax, 0.695, 0.585, 0.075, 0.13, color=GREEN, highlight=GREEN)

    box(0.855, 0.555, 0.115, 0.27, "Physics-informed kinetic prior", ["Arrhenius weighting", "random-walk proxy", "percolation"], fc=WHITE, ec=BLUE, title_size=8.7, bullet_size=6.2, wrap=14)
    xs = np.linspace(0.882, 0.925, 5)
    ys = np.linspace(0.705, 0.60, 5)
    ax.plot(xs, ys, color=BLUE, lw=1.1, ls="--")
    ax.scatter(xs, ys, s=16, color=LIGHT_BLUE, edgecolor=BLUE, lw=0.8)
    ax.plot([0.875, 0.875], [0.585, 0.735], color=INK, lw=0.8)
    ax.plot([0.875, 0.940], [0.585, 0.585], color=INK, lw=0.8)

    arrow(0.205, 0.695, 0.230, 0.695)
    arrow(0.440, 0.695, 0.465, 0.695)

    ax.text(
        0.018,
        0.455,
        "2. Validation and network readout",
        color=WHITE,
        fontsize=15,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.32,rounding_size=0.12", fc=BLUE, ec=BLUE),
    )
    box(
        0.02,
        0.08,
        0.23,
        0.32,
        "Feature matrix",
        ["chemical descriptors", "local bottleneck descriptors", "topology descriptors"],
        fc=WHITE,
        icon="matrix",
        wrap=24,
    )

    box(0.285, 0.08, 0.21, 0.32, "Grouped validation", ["leave-formula-out", "leave-chemistry-out", "within-chemistry split"], fc=WHITE, wrap=22)
    box(0.53, 0.08, 0.21, 0.32, "Model objectives", ["absolute error", "rank correlation", "top-k enrichment"], fc=WHITE, icon="target", wrap=22)
    box(0.78, 0.08, 0.20, 0.32, "Network readout", ["critical diffusion edges", "pathway accessibility", "next NEB priority"], fc=WHITE, icon="network", wrap=22)
    arrow(0.255, 0.235, 0.280, 0.235)
    arrow(0.500, 0.235, 0.525, 0.235)
    arrow(0.745, 0.235, 0.775, 0.235)
    ax.add_patch(FancyArrowPatch((0.455, 0.505), (0.125, 0.430), arrowstyle="-|>", mutation_scale=14, lw=1.3, color=BLUE, connectionstyle="angle3"))

    save_all(fig, "fig1_framework_overview")


def upright_triangle_pos(G: nx.Graph) -> dict[int, np.ndarray]:
    pos = {
        5: (-0.20, 0.96),
        11: (0.30, 0.88),
        0: (-0.58, 0.36),
        3: (-0.08, 0.51),
        2: (0.70, 0.56),
        9: (-0.68, -0.08),
        6: (-0.25, 0.23),
        8: (0.17, 0.15),
        12: (0.53, 0.14),
        13: (-0.44, -0.54),
        7: (-0.11, -0.22),
        10: (0.35, -0.10),
        4: (0.01, -0.77),
        1: (0.55, -0.58),
    }
    if set(pos) == set(G.nodes()):
        return {n: np.asarray(pos[n], dtype=float) for n in G.nodes()}
    raw = nx.kamada_kawai_layout(G, weight=None)
    xy = np.asarray([raw[n] for n in G.nodes()], dtype=float)
    xy -= xy.mean(axis=0)
    span = np.maximum(np.ptp(xy, axis=0), 1e-12)
    xy /= span.max()
    if np.ptp(xy[:, 1]) < np.ptp(xy[:, 0]):
        xy = np.column_stack([-xy[:, 1], xy[:, 0]])
    return {n: xy[i] for i, n in enumerate(G.nodes())}


def draw_upright_graph(ax, edges: pd.DataFrame):
    graph_id = "20|20|Na"
    G, sub = build_graph(edges, graph_id)
    pos = upright_triangle_pos(G)
    edge_list = list(G.edges(data=True))
    segs = [[pos[u], pos[v]] for u, v, _ in edge_list]
    widths = 0.75 + 1.85 * minmax(pd.Series([float(d.get("betweenness", 0.0)) for _, _, d in edge_list])).to_numpy()
    ax.add_collection(LineCollection(segs, colors="#5E6870", linewidths=widths, alpha=0.88, zorder=2))
    xy = np.asarray([pos[n] for n in G.nodes()])
    ax.scatter(xy[:, 0], xy[:, 1], s=27, color=WHITE, edgecolor="#66717A", lw=0.55, zorder=3)
    ax.set_aspect("equal")
    center_graph_view(ax, pos, pad=0.11)
    ax.axis("off")
    return G, sub


def figure2_rebuilt() -> None:
    train = pd.read_csv(MAIN_DB)
    edges = pd.read_csv(EDGES)
    profiles = pd.read_csv(SELECTED_PROFILES)
    labelled = train[train["barrier_eV"].notna()].copy()

    fig = plt.figure(figsize=(12.9, 8.2), facecolor=WHITE)
    gs = gridspec.GridSpec(
        2,
        6,
        figure=fig,
        height_ratios=[1.0, 1.08],
        width_ratios=[1, 1, 1, 1, 1, 1],
        hspace=0.46,
        wspace=0.70,
    )
    ax_a = fig.add_subplot(gs[0, 0:2])
    ax_b = fig.add_subplot(gs[0, 2:4])
    ax_c = fig.add_subplot(gs[0, 4:6])
    ax_d = fig.add_subplot(gs[1, 0:3])
    ax_e = fig.add_subplot(gs[1, 3:6])

    ax_a.hist(labelled["barrier_eV"], bins=34, color=BLUE, alpha=0.88)
    ax_a.set_xlabel("DFT/NEB barrier (eV)")
    ax_a.set_ylabel("path count")
    ax_a.text(0.96, 0.93, f"n = {len(labelled)}", transform=ax_a.transAxes, ha="right", va="top", fontsize=8.8)
    clean_grid(ax_a, axis="y")
    panel(ax_a, "a")

    species = labelled["migrating_species"].astype(str).value_counts().head(9)
    ax_b.bar(species.index, species.values, color=BLUE2, width=0.72)
    ax_b.set_ylabel("labeled paths")
    ax_b.set_xlabel("migrating species")
    ax_b.tick_params(axis="x", rotation=0, labelsize=8)
    clean_grid(ax_b, axis="y")
    panel(ax_b, "b")

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
    clean_grid(ax_c)
    cb = plt.colorbar(sc, ax=ax_c, fraction=0.046, pad=0.02)
    cb.set_label("bottleneck radius (A)", fontsize=7.4)
    cb.ax.tick_params(labelsize=7)
    panel(ax_c, "c")

    G, sub = draw_upright_graph(ax_d, edges)
    ax_d.text(0.50, -0.08, "Na13 P6 Se22, Na; nodes = %d, edges = %d" % (len(G.nodes), len(G.edges)), transform=ax_d.transAxes, ha="center", va="top", fontsize=8.0, color="#6F777D")
    panel(ax_d, "d")

    meta = path_profile_table(profiles, edges).sort_values("display_barrier_eV").dropna(subset=["display_barrier_eV"])
    if len(meta) >= 3:
        chosen = [meta.iloc[0]["edge_uid"], meta.iloc[len(meta) // 2]["edge_uid"], meta.iloc[-1]["edge_uid"]]
    else:
        chosen = list(meta["edge_uid"])
    labels = ["low", "intermediate", "high"]
    colors = [GREEN, BLUE, RED]
    for uid, lab, color in zip(chosen, labels, colors):
        p = profiles[profiles["edge_uid"].eq(uid)].copy().sort_values("s")
        row = meta[meta["edge_uid"].eq(uid)].iloc[0]
        ax_e.plot(p["s"], p["r_free_A"], lw=2.0, color=color, label=f"{lab}, E = {float(row['display_barrier_eV']):.2f} eV")
    ax_e.axhline(0, color=INK, lw=0.8, ls=":")
    ax_e.set_xlabel("reaction coordinate")
    ax_e.set_ylabel("free radius (A)")
    ax_e.legend(frameon=False, fontsize=7.2, loc="upper center", bbox_to_anchor=(0.50, -0.17), ncol=1)
    clean_grid(ax_e)
    panel(ax_e, "e")

    stats = {
        "labeled_paths": int(labelled.shape[0]),
        "candidate_edges": int(edges.shape[0]),
        "mobile_species": int(labelled["migrating_species"].nunique()),
        "formulas": int(labelled["formula"].nunique()),
        "host_chemistries": int(labelled["host_species"].nunique()),
        "edge_families": int(labelled["symmetry_family_id"].nunique()) if "symmetry_family_id" in labelled else np.nan,
        "figure2_layout": "GridSpec 2x6; a/b/c top; d/e centered bottom",
        "representative_graph": "Na13 P6 Se22, Na, graph_id=20|20|Na",
    }
    pd.DataFrame([stats]).to_csv(TABLE / "figure2_database_atlas_statistics.csv", index=False)
    save_all(fig, "fig2_migration_path_atlas")


def copy_external_validation_to_main() -> None:
    src = BASE / "SI_external_database_validation_20260608" / "figures" / "SI_external_database_validation.pdf"
    if src.exists():
        shutil.copy2(src, FIG / "fig7_external_database_validation.pdf")
    src_png = BASE / "SI_external_database_validation_20260608" / "figures" / "SI_external_database_validation.png"
    if src_png.exists():
        shutil.copy2(src_png, FIG / "fig7_external_database_validation.png")


def write_figure_description_file() -> None:
    text = """# Main figure descriptions and source files

Figure 1. Overall diffusion-network workflow. This figure is a schematic redrawn in the manuscript project from the user-specified atlas-construction workflow. It shows how source migration-barrier data and crystal structures are converted into a candidate migration-path atlas, represented by local bottleneck descriptors, graph topology terms and physics-informed network readouts, and finally evaluated by grouped validation and active NEB selection. Source script: scripts/rebuild_main_figures_and_manuscript.py.

Figure 2. Migration-path atlas and representative edge-level path examples. Panel a uses outputs_v17_polished_final/cache/01_labeled_barrier_database_clean.csv to show the distribution of DFT/NEB-labelled barriers. Panel b uses the same cleaned labelled database to summarize migrating-species coverage. Panel c uses the same database to compare hop distance, DFT/NEB barrier and local bottleneck radius. Panel d uses npj_no_leakage_diffusion_network_20260604/results/tables/graph_edges_with_no_leakage_predictions.csv only to show the representative Na13 P6 Se22, Na candidate graph topology in grayscale; no predicted-barrier color scale is shown in this atlas panel so that barrier-weighted interpretation is reserved for Figure 5. The layout is a fixed upright topology layout, not a crystallographic projection. Panel e uses outputs_v17_polished_final/cache/selected_path_profiles_v17.csv and graph_edges_with_no_leakage_predictions.csv to show low, intermediate and high barrier local free-radius profiles. Source script: scripts/rebuild_main_figures_and_manuscript.py.

Figure 3. Local bottleneck descriptors. Source figure copied from the publication-style figure workflow in paper_figures_pub_v2 and derived from the cleaned labelled barrier database, selected path profiles and no-leakage edge predictions. It explains free-radius profiles, bottleneck radius, crowding, coordination, within-chemistry residual prediction and descriptor importance.

Figure 4. No-leakage prediction and model comparison. Source figure copied from paper_figures_pub_v2 and derived from grouped-CV prediction/model tables in npj_no_leakage_diffusion_network_20260604/results/tables. It compares tabular model families and feature groups under grouped validation, with random-walk/barrier-weighted readouts excluded from model inputs.

Figure 5. Predicted barrier-weighted diffusion network readout. The updated layout enlarges panels a and b in the top row: panel a shows the predicted barrier-weighted representative Na13 P6 Se22, Na migration graph, and panel b shows the sparse dominant backbone extracted from the same graph. Panels c-e are aligned in the bottom row to explain the network readout: predicted percolation threshold, critical-edge removal response and barrier-topology trade-off. Source script: scripts/finalize_white_figures_and_fixed_workflow.py.

Figure 6. Active-learning NEB path selection. Source figure copied from paper_figures_pub_v2 and derived from edge_scores_no_leakage_active_neb.csv and top50_active_neb_candidates_no_leakage.csv. It compares low-barrier, topology and DiffPathScore-driven selection policies and reports candidate paths for targeted NEB validation.

Figure 7. External database consistency checks. Source figure copied from SI_external_database_validation_20260608/figures/SI_external_database_validation.pdf. It summarizes OBELiX composition/family-level comparisons, LiTraj/BVEL network barrier-scale comparisons, anti-perovskite descriptor-barrier trends and overlap between current predicted Li networks and external chemical systems. These checks are used as consistency evidence, not as one-to-one validation of every predicted candidate edge.
"""
    (ROOT / "FIGURE_DESCRIPTIONS.md").write_text(text, encoding="utf-8")


def main() -> None:
    setup_white()
    figure1_framework_rebuilt()
    figure2_rebuilt()
    copy_external_validation_to_main()
    write_figure_description_file()


if __name__ == "__main__":
    main()
