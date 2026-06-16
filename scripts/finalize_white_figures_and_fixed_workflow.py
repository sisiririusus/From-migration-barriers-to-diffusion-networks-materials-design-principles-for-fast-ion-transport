from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from PIL import Image


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / "manuscript_diffusion_network_v1"
FIG = ROOT / "figures"
SIFIG = ROOT / "si_figures"
SRC_WORKFLOW = BASE / "source_references" / "JMST_Figure1_user_workflow.png"

EXT = BASE / "SI_external_database_validation_20260608"
DATA = EXT / "external_data"
TAB = EXT / "tables"
PAPER_CACHE = BASE / "paper_figures_pub_v2" / "data_cache"
NOLEAK_TABLES = BASE / "npj_no_leakage_diffusion_network_20260604" / "results" / "tables"

sys.path.insert(0, str(BASE / "paper_figures_pub_v2" / "scripts"))
from shared_plot_utils import (  # noqa: E402
    representative_na_graph,
    draw_barrier_colored_na_graph,
    draw_schematic_graph,
    edge_pair,
    minmax,
)

WHITE = "#FFFFFF"
INK = "#20292f"
GRID = "#DDE1E5"
BLUE = "#35617d"
BLUE2 = "#7299b8"
ORANGE = "#c87438"
GRAY = "#7d817c"


def png_to_raster_pdf(png_path: Path, pdf_path: Path) -> None:
    """Embed the PNG as a full-page raster PDF without redrawing its contents."""
    img = Image.open(png_path).convert("RGB")
    img.save(pdf_path, "PDF", resolution=600.0)


def install_fixed_workflow() -> None:
    if not SRC_WORKFLOW.exists():
        raise FileNotFoundError(f"Missing user workflow image: {SRC_WORKFLOW}")
    FIG.mkdir(parents=True, exist_ok=True)
    img = Image.open(SRC_WORKFLOW).convert("RGB")
    out_png = FIG / "fig1_framework_overview.png"
    img.save(out_png)
    png_to_raster_pdf(out_png, FIG / "fig1_framework_overview.pdf")
    # SVG cannot faithfully preserve the user-provided raster without embedding;
    # keep the authoritative PNG/PDF pair instead.
    svg = FIG / "fig1_framework_overview.svg"
    if svg.exists():
        svg.unlink()


def whiten_png(path: Path) -> None:
    img = Image.open(path).convert("RGBA")
    arr = np.asarray(img).copy()
    rgb = arr[..., :3].astype(np.int16)
    # Replace only near-neutral, very light off-white backgrounds.
    light = (rgb[..., 0] > 235) & (rgb[..., 1] > 235) & (rgb[..., 2] > 230)
    neutral = (rgb.max(axis=2) - rgb.min(axis=2)) < 14
    mask = light & neutral
    arr[..., :3][mask] = 255
    Image.fromarray(arr).save(path)


def regenerate_pdf_from_png(path: Path) -> None:
    pdf = path.with_suffix(".pdf")
    if path.exists():
        png_to_raster_pdf(path, pdf)


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_grid(ax, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, lw=0.55, alpha=0.78)
    for sp in ax.spines.values():
        sp.set_linewidth(0.9)


def panel(ax, letter: str) -> None:
    ax.text(-0.13, 1.08, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=12.5, weight="bold", color=INK)


def regenerate_fig7_without_panel_a() -> None:
    kim_file = DATA / "Kim_Siegel_antiperovskite_SI_1.xlsx"
    bvel_file = DATA / "LiTraj_BVEL13k" / "BVEL13k" / "BVEL13k_index.csv"
    current_perc_file = BASE / "npj_no_leakage_diffusion_network_20260604" / "results" / "tables" / "predicted_barrier_weighted_percolation.csv"
    if not (kim_file.exists() and bvel_file.exists() and current_perc_file.exists()):
        raise FileNotFoundError("External validation source files are incomplete; cannot regenerate Figure 7.")

    frames = []
    for sheet, mech in [("Vacancy data", "vacancy"), ("Dumbell data", "dumbbell")]:
        d = pd.read_excel(kim_file, sheet_name=sheet)
        d["mechanism"] = mech
        frames.append(d)
    kim = pd.concat(frames, ignore_index=True)
    kim["barrier_eV"] = pd.to_numeric(kim["Energy barrier (target)"], errors="coerce") / 1000.0
    kim_plot = kim.dropna(subset=["Di", "PW", "barrier_eV"]).copy()

    bvel = pd.read_csv(bvel_file)
    for c in ["E_1D", "E_2D", "E_3D"]:
        bvel[c] = pd.to_numeric(bvel[c], errors="coerce")
    vals_b = bvel[["E_1D", "E_2D", "E_3D"]].min(axis=1).replace([np.inf, -np.inf], np.nan).dropna()

    cur = pd.read_csv(current_perc_file)
    vals_c = pd.to_numeric(cur["predicted_percolation_threshold_eV"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.55), facecolor=WHITE)
    colors = {"vacancy": BLUE2, "dumbbell": ORANGE}
    for mech, g in kim_plot.groupby("mechanism"):
        axes[0].scatter(g["Di"], g["barrier_eV"], s=14, alpha=0.48, edgecolors="none", color=colors.get(mech, GRAY), label=mech)
    axes[0].set_xlabel("hopping distance Di")
    axes[0].set_ylabel("DFT-NEB barrier (eV)")
    axes[0].legend(frameon=False, loc="upper left")
    clean_grid(axes[0])
    panel(axes[0], "a")

    for mech, g in kim_plot.groupby("mechanism"):
        axes[1].scatter(g["PW"], g["barrier_eV"], s=14, alpha=0.48, edgecolors="none", color=colors.get(mech, GRAY), label=mech)
    axes[1].set_xlabel("path width PW")
    axes[1].set_ylabel("DFT-NEB barrier (eV)")
    clean_grid(axes[1])
    panel(axes[1], "b")

    upper = min(2.5, float(np.nanpercentile(pd.concat([vals_b, vals_c]), 98)))
    bins = np.linspace(0, upper, 36)
    axes[2].hist(vals_b.clip(upper=bins[-1]), bins=bins, color=GRAY, alpha=0.35, density=True, label="LiTraj-BVEL")
    axes[2].hist(vals_c.clip(upper=bins[-1]), bins=bins, color=BLUE, alpha=0.58, density=True, label="current predicted")
    axes[2].set_xlabel("percolation barrier / threshold (eV)")
    axes[2].set_ylabel("density")
    axes[2].legend(frameon=False, loc="upper right")
    clean_grid(axes[2], axis="y")
    panel(axes[2], "c")

    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.22, top=0.91, wspace=0.34)
    out_png = FIG / "fig7_external_database_validation.png"
    out_pdf = FIG / "fig7_external_database_validation.pdf"
    fig.savefig(out_png, dpi=600, facecolor=WHITE)
    fig.savefig(out_pdf, facecolor=WHITE)
    fig.savefig(FIG / "fig7_external_database_validation.svg", facecolor=WHITE)
    plt.close(fig)


def regenerate_fig5_emphasis_layout() -> None:
    edges_file = NOLEAK_TABLES / "graph_edges_with_no_leakage_predictions.csv"
    if not edges_file.exists():
        raise FileNotFoundError(f"Missing graph edge table: {edges_file}")
    edges = pd.read_csv(edges_file, low_memory=False)
    graph_id, G, sub, pos = representative_na_graph(edges)
    edge_df = sub.copy()
    edge_df["topology_score"] = minmax(edge_df["edge_betweenness"])
    edge_df["low_barrier_score"] = minmax(edge_df["pred_barrier_no_leak_eV"], invert=True)
    edge_df["DiffPathScore"] = 0.55 * edge_df["low_barrier_score"] + 0.45 * edge_df["topology_score"]

    topcrit = edge_df.sort_values("DiffPathScore", ascending=False).head(max(6, min(8, len(edge_df) // 4)))
    crit_pairs = {edge_pair(r) for _, r in topcrit.iterrows()}
    low_cut = edge_df["pred_barrier_no_leak_eV"].quantile(0.30)
    low_pairs = {edge_pair(r) for _, r in edge_df[edge_df["pred_barrier_no_leak_eV"].le(low_cut)].iterrows()}
    Hlow = nx.Graph()
    for _, r in edge_df[edge_df["pred_barrier_no_leak_eV"].le(low_cut)].iterrows():
        Hlow.add_edge(int(r["node_i"]), int(r["node_j"]))
    backbone_edges = set(low_pairs) | set(crit_pairs)
    if len(Hlow):
        for comp in nx.connected_components(Hlow):
            if len(comp) > 2:
                Hsub = G.subgraph(comp).copy()
                for u, v in nx.minimum_spanning_tree(Hsub, weight="barrier").edges():
                    backbone_edges.add(tuple(sorted((u, v))))

    edges_list = list(G.edges(data=True))
    order_top = sorted(edges_list, key=lambda x: x[2].get("betweenness", 0), reverse=True)
    rng = np.random.RandomState(12)
    removed_fracs = np.linspace(0, 0.8, 9)
    rows = []
    for frac in removed_fracs:
        k = int(round(frac * len(edges_list)))
        H = G.copy()
        for u, v, _ in order_top[:k]:
            if H.has_edge(u, v):
                H.remove_edge(u, v)
        lcc = len(max(nx.connected_components(H), key=len)) / float(len(G)) if len(H) else 0
        rows.append({"removed_fraction": frac, "strategy": "topology-ranked", "replicate": 0, "largest_component_fraction": lcc})
        for rep in range(80):
            order = edges_list[:]
            rng.shuffle(order)
            H = G.copy()
            for u, v, _ in order[:k]:
                if H.has_edge(u, v):
                    H.remove_edge(u, v)
            lcc = len(max(nx.connected_components(H), key=len)) / float(len(G)) if len(H) else 0
            rows.append({"removed_fraction": frac, "strategy": "random", "replicate": rep, "largest_component_fraction": lcc})
    response = pd.DataFrame(rows)
    response.to_csv(PAPER_CACHE / "Figure5_critical_edge_removal_response.csv", index=False)

    def zoom_graph_axis(ax, pos_dict, pad=0.025):
        xy = np.asarray([pos_dict[n] for n in pos_dict], dtype=float)
        cx, cy = xy.mean(axis=0)
        span = max(xy[:, 0].max() - xy[:, 0].min(), xy[:, 1].max() - xy[:, 1].min())
        half = 0.5 * span * (1.0 + 2.0 * pad)
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)

    fig = plt.figure(figsize=(13.2, 8.9), facecolor=WHITE)
    gs = gridspec.GridSpec(
        2,
        6,
        figure=fig,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        height_ratios=[2.15, 1.0],
        wspace=0.55,
        hspace=0.34,
    )
    ax_a = fig.add_subplot(gs[0, 0:3])
    ax_b = fig.add_subplot(gs[0, 3:6])
    ax_c = fig.add_subplot(gs[1, 0:2])
    ax_d = fig.add_subplot(gs[1, 2:4])
    ax_e = fig.add_subplot(gs[1, 4:6])

    draw_barrier_colored_na_graph(ax_a, G, pos, colorbar=True, annotate_layout=False)
    zoom_graph_axis(ax_a, pos, pad=0.035)
    ax_a.text(0.50, -0.045, f"Na13 P6 Se22, Na; nodes = {len(G.nodes)}, edges = {len(G.edges)}",
              transform=ax_a.transAxes, ha="center", va="top", fontsize=8.0, color="#6F777D")
    panel(ax_a, "a")

    draw_schematic_graph(ax_b, G, pos, edge_df, highlight_edges=backbone_edges,
                         critical_edges=crit_pairs, backbone_only=True,
                         node_size=42, bg_alpha=0.0)
    zoom_graph_axis(ax_b, pos, pad=0.035)
    panel(ax_b, "b")

    bins = np.linspace(edge_df["pred_barrier_no_leak_eV"].min(), edge_df["pred_barrier_no_leak_eV"].max(), 14)
    lccs = []
    for th in bins:
        H = nx.Graph()
        for u, v, d0 in G.edges(data=True):
            if d0["barrier"] <= th:
                H.add_edge(u, v)
        lccs.append(len(max(nx.connected_components(H), key=len)) / float(len(G)) if len(H) else 0)
    perc = bins[np.argmax(np.asarray(lccs) >= 0.5)] if np.any(np.asarray(lccs) >= 0.5) else np.nan
    ax_c.plot(bins, lccs, marker="o", color=ORANGE, lw=1.8)
    ax_c.axhline(0.5, color=INK, lw=0.8, ls=":")
    if np.isfinite(perc):
        ax_c.axvline(perc, color="#B94D3F", lw=1.0, ls="--")
        ax_c.text(0.06, 0.92, f"E_c = {perc:.2f} eV", transform=ax_c.transAxes,
                  ha="left", va="top", fontsize=7.6, color="#B94D3F")
    ax_c.set_xlabel("barrier cutoff (eV)")
    ax_c.set_ylabel("largest component fraction")
    clean_grid(ax_c)
    panel(ax_c, "c")

    random_resp = response[response["strategy"].eq("random")]
    rand = random_resp.groupby("removed_fraction")["largest_component_fraction"].agg(["mean", "std"]).reset_index()
    topo = response[response["strategy"].eq("topology-ranked")]
    ax_d.fill_between(rand["removed_fraction"], rand["mean"] - rand["std"], rand["mean"] + rand["std"],
                      color="#C4C9C4", alpha=0.38, lw=0)
    ax_d.plot(rand["removed_fraction"], rand["mean"], marker="o", color=GRAY, lw=1.5, label="random")
    ax_d.plot(topo["removed_fraction"], topo["largest_component_fraction"], marker="o", color="#B94D3F", lw=1.6, label="topology-ranked")
    ax_d.set_xlabel("removed edge fraction")
    ax_d.set_ylabel("largest component fraction")
    ax_d.legend(frameon=False, fontsize=7.2, loc="lower left")
    clean_grid(ax_d)
    panel(ax_d, "d")

    bx = edge_df["pred_barrier_no_leak_eV"].quantile(0.50)
    ty = edge_df["topology_score"].quantile(0.65)
    ax_e.scatter(edge_df["pred_barrier_no_leak_eV"], edge_df["topology_score"], s=22,
                 color="#AEB4B8", alpha=0.46, edgecolors="none", label="all edges")
    ax_e.scatter(topcrit["pred_barrier_no_leak_eV"], topcrit["topology_score"], s=58,
                 facecolors=WHITE, edgecolors="#B94D3F", lw=1.2, label="selected")
    ax_e.axvline(bx, color=GRID, lw=0.9, ls=":")
    ax_e.axhline(ty, color=GRID, lw=0.9, ls=":")
    ax_e.set_xlabel("predicted barrier (eV)")
    ax_e.set_ylabel("topology score")
    ax_e.legend(frameon=False, fontsize=7.2, loc="lower left")
    clean_grid(ax_e)
    panel(ax_e, "e")

    topcrit.to_csv(PAPER_CACHE / "Figure5_top_dominant_edges.csv", index=False)
    pd.DataFrame(
        {
            "graph_id": [graph_id],
            "formula": [sub["formula"].iloc[0]],
            "ion": [sub["migrating_species"].iloc[0]],
            "n_nodes": [len(G.nodes)],
            "n_edges": [len(G.edges)],
            "predicted_percolation_threshold_eV": [perc],
            "layout": ["top row enlarged panels a-b plus bottom row network readout panels c-e"],
        }
    ).to_csv(PAPER_CACHE / "Figure5_case_summary.csv", index=False)

    fig.subplots_adjust(left=0.060, right=0.975, bottom=0.085, top=0.965)
    out_png = FIG / "fig5_topology_network_readout.png"
    out_pdf = FIG / "fig5_topology_network_readout.pdf"
    fig.savefig(out_png, dpi=600, facecolor=WHITE, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_pdf, facecolor=WHITE, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(FIG / "fig5_topology_network_readout.svg", facecolor=WHITE, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def whiten_existing_pngs() -> None:
    for folder in [FIG, SIFIG]:
        for path in folder.glob("*.png"):
            # Figure 1 is the user-provided fixed workflow; do not alter its pixels.
            if path.name == "fig1_framework_overview.png":
                continue
            whiten_png(path)
            regenerate_pdf_from_png(path)


def main() -> None:
    style()
    install_fixed_workflow()
    regenerate_fig5_emphasis_layout()
    regenerate_fig7_without_panel_a()
    whiten_existing_pngs()


if __name__ == "__main__":
    main()
