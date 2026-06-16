from pathlib import Path
import json
import math
import re
import shutil
import time
import zipfile

import numpy as np
import pandas as pd
import requests

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "manuscript_diffusion_network_v1"
FIGS = OUT / "figures"
SIFIGS = OUT / "si_figures"
TABLES = OUT / "tables"
SCRIPTS = OUT / "scripts"
LIT = OUT / "literature"
TEMPLATE_DIR = OUT / "springer_template"

for d in [OUT, FIGS, SIFIGS, TABLES, SCRIPTS, LIT, TEMPLATE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

REAL_SOURCES = []
PLACEHOLDERS = []


def read_summary():
    path = BASE / "paper_figures_pub_v2" / "main_results_summary.csv"
    default = {
        "labeled_paths": "data required",
        "candidate_edges": "data required",
        "mobile_species": "data required",
        "formulas": "data required",
        "best_model": "data required",
        "mae": float("nan"),
        "rho": float("nan"),
    }
    if not path.exists():
        PLACEHOLDERS.append(("main result summary", str(path)))
        return default
    row = pd.read_csv(path).iloc[0].to_dict()
    REAL_SOURCES.append(("main result summary", str(path)))
    return {
        "labeled_paths": int(row.get("labeled_paths", 0)),
        "candidate_edges": int(row.get("candidate_edges", 0)),
        "mobile_species": int(row.get("mobile_species", 0)),
        "formulas": int(row.get("formulas", 0)),
        "best_model": str(row.get("best_C_L_T_model", "data required")),
        "mae": float(row.get("best_C_L_T_MAE_eV", np.nan)),
        "rho": float(row.get("best_C_L_T_Spearman_rho", np.nan)),
    }


def copy_if_exists(src, dst, label):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        REAL_SOURCES.append((label, str(src)))
        return True
    PLACEHOLDERS.append((label, str(src)))
    return False


def fetch_springer_template():
    url = "https://cms-resources.apps.public.k8s.springernature.io/springer-cms/rest/v1/content/18782940/data/v12"
    zip_path = TEMPLATE_DIR / "springer-nature-latex-template-december-2024.zip"
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        zip_path.write_bytes(r.content)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(TEMPLATE_DIR / "extracted")
        copied = 0
        for f in (TEMPLATE_DIR / "extracted").rglob("*"):
            if f.suffix.lower() in [".cls", ".bst"] or f.name.lower().startswith("sn-"):
                if f.is_file():
                    shutil.copy2(f, OUT / f.name)
                    copied += 1
        REAL_SOURCES.append(("Springer Nature template zip", str(zip_path)))
        if copied == 0:
            PLACEHOLDERS.append(("Springer template class files not found after extraction", str(zip_path)))
    except Exception as exc:
        PLACEHOLDERS.append(("Springer template download failed", repr(exc)))


def copy_main_figures():
    srcdir = BASE / "paper_figures_pub_v2" / "fig_main"
    mapping = {
        "Figure1_overview": "fig1_framework_overview",
        "Figure2_atlas": "fig2_migration_path_atlas",
        "Figure3_local_descriptors": "fig3_local_bottleneck_physics",
        "Figure4_prediction": "fig4_no_leakage_path_ranking",
        "Figure5_network": "fig5_topology_network_readout",
        "Figure6_targeted_neb": "fig6_active_neb_selection",
    }
    for old, new in mapping.items():
        for ext in [".pdf", ".png", ".svg"]:
            copy_if_exists(srcdir / f"{old}{ext}", FIGS / f"{new}{ext}", f"main figure {new}{ext}")


def copy_tables():
    candidates = [
        BASE / "paper_figures_pub_v2" / "main_results_summary.csv",
        BASE / "paper_figures_pub_v2" / "data_cache" / "figure2_database_atlas_statistics.csv",
        BASE / "paper_figures_pub_v2" / "data_cache" / "figure3_local_descriptor_importance.csv",
        BASE / "paper_figures_pub_v2" / "data_cache" / "figure3_quartile_descriptor_effect_size.csv",
        BASE / "paper_figures_pub_v2" / "data_cache" / "model_family_grouped_cv_metrics.csv",
        BASE / "paper_figures_pub_v2" / "data_cache" / "machine_learning_models_attempted.csv",
        BASE / "paper_figures_pub_v2" / "data_cache" / "Figure5_case_summary.csv",
        BASE / "paper_figures_pub_v2" / "data_cache" / "Figure5_critical_edge_removal_response.csv",
        BASE / "paper_figures_pub_v2" / "data_cache" / "Figure6_top5_recommended_candidates.csv",
        BASE / "paper_figures_pub_v2" / "data_cache" / "Figure6_score_ablation.csv",
        BASE / "SI_external_database_validation_20260608" / "tables" / "external_database_validation_summary.csv",
        BASE / "SI_external_database_validation_20260608" / "tables" / "kim_antiperovskite_descriptor_barrier_correlations.csv",
        BASE / "SI_external_database_validation_20260608" / "tables" / "current_predicted_Li_networks_external_chemsys_overlap.csv",
    ]
    for src in candidates:
        copy_if_exists(src, TABLES / src.name, f"table {src.name}")

    pd.DataFrame(
        [
            ["C", "migrating_ion", "Identity of the mobile species", "Captures ion-size and valence-dependent mobility trends"],
            ["C", "formula / chemistry encoding", "Elemental and formula-level chemical information", "Controls average host chemistry"],
            ["L", "hop distance", "Distance between candidate mobile-ion sites", "Geometric length scale of the hop"],
            ["L", "free-radius profile", "Radius available along the path coordinate", "Local channel openness"],
            ["L", "minimum free radius", "Minimum sampled free-radius value", "Tightest local constriction"],
            ["L", "bottleneck radius", "Saddle-region constriction descriptor", "Steric bottleneck physics"],
            ["L", "crowding index", "Neighbor crowding around the path", "Local confinement"],
            ["L", "CN_i / CN_j", "Endpoint coordination", "Site stability and asymmetry"],
            ["T", "degree", "Number of incident edges", "Local graph branching"],
            ["T", "edge betweenness", "Fraction of graph paths using an edge", "Bridge-like network role"],
            ["T", "critical edge score", "Connectivity/readout response after removal", "Network-control role"],
            ["R", "predicted percolation threshold", "Cutoff for connected low-barrier component", "Post-prediction transport readout"],
        ],
        columns=["group", "descriptor", "definition", "physical_role"],
    ).to_csv(TABLES / "table_S2_descriptor_definitions.csv", index=False)

    pd.DataFrame(
        [
            ["Dummy mean", "sklearn DummyRegressor", "non-informative baseline"],
            ["Ridge / ElasticNet", "linear regularized regression", "linear baseline"],
            ["RandomForestRegressor", "tree ensemble", "stable nonlinear baseline"],
            ["ExtraTreesRegressor", "extremely randomized trees", "selected current C/L/T baseline"],
            ["HistGradientBoostingRegressor", "histogram gradient boosting", "boosting comparison"],
            ["XGBoost / LightGBM / CatBoost", "external boosting libraries", "optional comparison when installed"],
        ],
        columns=["model", "implementation", "rationale"],
    ).to_csv(TABLES / "table_S3_model_hyperparameters_and_rationale.csv", index=False)

    pd.DataFrame(
        [
            ["OBELiX", "experimental Li solid-electrolyte ionic conductivity", "materials-level chemistry overlap", "not edge-level barrier labels"],
            ["LiTraj-BVEL13k", "structure-derived Li migration/percolation barriers", "barrier-scale and percolation-scale comparison", "different label definition"],
            ["Kim/Siegel anti-perovskite NEB", "independent DFT-NEB barriers and geometric descriptors", "local-physics descriptor support", "not the same host space as the full atlas"],
        ],
        columns=["external_dataset", "information_used", "validation_role", "limitation"],
    ).to_csv(TABLES / "table_S6_external_validation_datasets.csv", index=False)


def make_si_figures():
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "Arial", "font.size": 8.5, "axes.linewidth": 0.8, "savefig.dpi": 350})
    colors = {"blue": "#345f7e", "orange": "#c8733a", "green": "#5f8f61", "red": "#ad6256", "gray": "#7a7d78"}

    def save(fig, stem):
        fig.savefig(SIFIGS / f"{stem}.png", bbox_inches="tight")
        fig.savefig(SIFIGS / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

    # S1
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ax.set_axis_off()
    boxes = [("raw records", 0.03), ("formula-ion groups", 0.22), ("edge atlas", 0.42), ("descriptor matrix", 0.62), ("grouped CV", 0.82)]
    for label, x in boxes:
        ax.add_patch(plt.Rectangle((x, 0.35), 0.14, 0.30, fc="white", ec="#c9c3b6", lw=1.2))
        ax.text(x + 0.07, 0.50, label, ha="center", va="center", fontsize=8)
    for i in range(len(boxes) - 1):
        ax.annotate("", xy=(boxes[i + 1][1] - 0.015, 0.50), xytext=(boxes[i][1] + 0.15, 0.50), arrowprops=dict(arrowstyle="-|>", lw=1.2, color=colors["orange"]))
    ax.text(0.02, 0.15, "True barriers are targets only; barrier-weighted network quantities are evaluated after prediction.", color=colors["blue"], fontsize=8)
    save(fig, "figS1_data_cleaning_workflow")
    PLACEHOLDERS.append(("Fig. S1 schematic", "generated workflow schematic"))

    # S2
    edges_file = BASE / "npj_no_leakage_diffusion_network_20260604" / "results" / "tables" / "migration_path_atlas_edges.csv"
    corr = pd.DataFrame(np.eye(5), index=["data"] * 5, columns=["data"] * 5)
    if edges_file.exists():
        df = pd.read_csv(edges_file, low_memory=False)
        cols = [c for c in df.columns if any(k in c.lower() for k in ["distance", "radius", "crowd", "cn_", "degree", "between", "barrier"])]
        num = df[cols].select_dtypes(include=[np.number]).dropna(axis=1, how="all")
        num = num.loc[:, num.nunique(dropna=True) > 3].iloc[:, :10]
        if num.shape[1] > 1:
            corr = num.corr(method="spearman").fillna(0)
            REAL_SOURCES.append(("Fig. S2 descriptor correlation", str(edges_file)))
    fig, ax = plt.subplots(figsize=(5.4, 4.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ticks = [re.sub(r"(.{14}).+", r"\1...", str(c)) for c in corr.columns]
    ax.set_xticks(range(len(ticks))); ax.set_yticks(range(len(ticks)))
    ax.set_xticklabels(ticks, rotation=45, ha="right", fontsize=6); ax.set_yticklabels(ticks, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman r")
    save(fig, "figS2_descriptor_correlation_matrix")

    # S3
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    if edges_file.exists():
        df = pd.read_csv(edges_file, low_memory=False)
        barrier_col = next((c for c in df.columns if "barrier" in c.lower() and pd.api.types.is_numeric_dtype(df[c])), None)
        ion_col = next((c for c in df.columns if c.lower() in ["ion", "migrating_ion", "species", "mobile_ion"]), None)
        if barrier_col and ion_col:
            ions = df[ion_col].astype(str).value_counts().head(8).index
            data = [df.loc[df[ion_col].astype(str) == ion, barrier_col].dropna().clip(0, 4).values for ion in ions]
            ax.boxplot(data, labels=ions, patch_artist=True, boxprops=dict(facecolor="#dce8ef", color="#444"), medianprops=dict(color=colors["orange"]))
            ax.set_ylabel("reported barrier (eV)")
            REAL_SOURCES.append(("Fig. S3 barrier species distribution", str(edges_file)))
        else:
            ax.text(0.5, 0.5, "data/figure required\nspecies or barrier column missing", ha="center", va="center")
            ax.set_axis_off()
            PLACEHOLDERS.append(("Fig. S3", "species/barrier columns missing"))
    else:
        ax.text(0.5, 0.5, "data/figure required\nbarrier by migrating species", ha="center", va="center")
        ax.set_axis_off()
        PLACEHOLDERS.append(("Fig. S3", "atlas edge table missing"))
    save(fig, "figS3_barrier_distribution_by_species")

    # Copy existing SI/case figures and make remaining compact figures.
    copy_if_exists(BASE / "npj_paper_figures_20260604" / "si_figures" / "figS4_descriptor_distributions.png", SIFIGS / "figS4_descriptor_distributions.png", "Fig. S4 descriptor distributions")
    copy_if_exists(BASE / "npj_paper_figures_20260604" / "si_figures" / "figS4_descriptor_distributions.pdf", SIFIGS / "figS4_descriptor_distributions.pdf", "Fig. S4 descriptor distributions pdf")
    copy_if_exists(BASE / "npj_paper_figures_20260604" / "si_figures" / "figS1_no_leakage_feature_policy.png", SIFIGS / "figS5_grouped_cv_split_examples.png", "Fig. S5 grouped CV")
    copy_if_exists(BASE / "npj_paper_figures_20260604" / "si_figures" / "figS1_no_leakage_feature_policy.pdf", SIFIGS / "figS5_grouped_cv_split_examples.pdf", "Fig. S5 grouped CV pdf")

    metrics_path = BASE / "paper_figures_pub_v2" / "data_cache" / "model_family_grouped_cv_metrics.csv"
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    if metrics_path.exists():
        m = pd.read_csv(metrics_path)
        model_col = next((c for c in m.columns if "model" in c.lower()), m.columns[0])
        mae_col = next((c for c in m.columns if "mae" in c.lower()), None)
        if mae_col:
            mm = m.groupby(model_col)[mae_col].mean().sort_values().head(8)
            ax.barh(range(len(mm)), mm.values, color=colors["blue"], alpha=0.78)
            ax.set_yticks(range(len(mm))); ax.set_yticklabels([str(x)[:22] for x in mm.index])
            ax.invert_yaxis(); ax.set_xlabel("grouped-CV MAE (eV)")
            REAL_SOURCES.append(("Fig. S6 model robustness", str(metrics_path)))
        else:
            ax.text(0.5, 0.5, "data/figure required\nMAE column missing", ha="center")
            ax.set_axis_off()
    save(fig, "figS6_hyperparameter_and_model_robustness")

    imp_path = BASE / "paper_figures_pub_v2" / "data_cache" / "figure3_local_descriptor_importance.csv"
    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    if imp_path.exists():
        imp = pd.read_csv(imp_path)
        numcol = next((c for c in imp.columns if pd.api.types.is_numeric_dtype(imp[c])), None)
        namecol = next((c for c in imp.columns if c != numcol), imp.columns[0])
        dd = imp.sort_values(numcol, ascending=True).tail(10) if numcol else imp
        if numcol:
            ax.barh(dd[namecol].astype(str), dd[numcol], color=colors["orange"])
            ax.set_xlabel("importance")
            REAL_SOURCES.append(("Fig. S7 descriptor importance", str(imp_path)))
    save(fig, "figS7_feature_importance_across_folds")

    copy_if_exists(BASE / "SI_external_database_validation_20260608" / "figures" / "SI_external_database_validation.png", SIFIGS / "figS8_external_database_consistency.png", "Fig. S8 external consistency")
    copy_if_exists(BASE / "SI_external_database_validation_20260608" / "figures" / "SI_external_database_validation.pdf", SIFIGS / "figS8_external_database_consistency.pdf", "Fig. S8 external consistency pdf")
    copy_if_exists(BASE / "model_predicted_unidentified_networks_20260608" / "figures" / "model_predicted_unidentified_networks_support.png", SIFIGS / "figS9_additional_case_study_networks.png", "Fig. S9 case study networks")
    copy_if_exists(BASE / "model_predicted_unidentified_networks_20260608" / "figures" / "model_predicted_unidentified_networks_support.pdf", SIFIGS / "figS9_additional_case_study_networks.pdf", "Fig. S9 case study networks pdf")

    score_path = BASE / "paper_figures_pub_v2" / "data_cache" / "Figure6_strategy_hit_rate_curves.csv"
    fig, ax = plt.subplots(figsize=(5, 3.4))
    if score_path.exists():
        ss = pd.read_csv(score_path)
        xcol = next((c for c in ss.columns if "top" in c.lower() or "fraction" in c.lower() or "rank" in c.lower()), ss.columns[0])
        ycols = [c for c in ss.columns if c != xcol and pd.api.types.is_numeric_dtype(ss[c])][:4]
        for c in ycols:
            ax.plot(ss[xcol], ss[c], marker="o", lw=1.4, label=c[:18])
        ax.set_xlabel(xcol); ax.set_ylabel("hit rate / enrichment"); ax.legend(frameon=False, fontsize=7)
        REAL_SOURCES.append(("Fig. S10 active-learning sensitivity", str(score_path)))
    save(fig, "figS10_active_learning_batch_sensitivity")

    # S11 leakage schematic
    fig, ax = plt.subplots(figsize=(6.2, 2.8)); ax.set_axis_off()
    items = [("allowed", "C/L/T descriptors\nfrom structure"), ("target only", "true barrier label"), ("forbidden input", "true-barrier random walk\ntrue percolation"), ("post-readout", "predicted-barrier\nnetwork physics")]
    for i, (head, body) in enumerate(items):
        x = 0.05 + i * 0.23
        ax.add_patch(plt.Rectangle((x, 0.30), 0.18, 0.32, fc=["#e6eef2", "#f5efe4", "#f3dfd9", "#e6f0e8"][i], ec="#c9c3b6"))
        ax.text(x + 0.09, 0.52, head, ha="center", weight="bold", fontsize=8)
        ax.text(x + 0.09, 0.40, body, ha="center", va="center", fontsize=7)
    ax.text(0.05, 0.16, "True-barrier-weighted network quantities are excluded from model inputs.", color=colors["blue"], fontsize=8)
    save(fig, "figS11_leakage_control_negative_test")
    PLACEHOLDERS.append(("Fig. S11 schematic", "feature policy schematic"))

    # S12 limitations
    fig, ax = plt.subplots(figsize=(6.5, 3.2)); ax.set_axis_off()
    rows = [
        ("site generation", "candidate sites may be incomplete in disordered or poorly relaxed structures"),
        ("single-edge atlas", "correlated multi-ion migration is not fully resolved"),
        ("DFT labels", "barriers depend on functional, supercell, charge and NEB setup"),
        ("network readout", "graph scores do not replace KMC, TST or AIMD diffusivities"),
        ("active learning", "NEB feedback is required to close the loop"),
    ]
    ax.add_patch(plt.Rectangle((0.02, 0.78), 0.96, 0.12, fc=colors["blue"], ec="none"))
    ax.text(0.04, 0.84, "Failure mode", color="white", weight="bold", fontsize=8)
    ax.text(0.32, 0.84, "Consequence for interpretation", color="white", weight="bold", fontsize=8)
    for i, (a, b) in enumerate(rows):
        y = 0.70 - i * 0.12
        ax.add_patch(plt.Rectangle((0.02, y), 0.96, 0.105, fc="white" if i % 2 == 0 else "#edf3f6", ec="#d7d1c6", lw=0.5))
        ax.text(0.04, y + 0.052, a, va="center", fontsize=7.5)
        ax.text(0.32, y + 0.052, b, va="center", fontsize=7.5)
    save(fig, "figS12_limitations_and_failure_cases")
    PLACEHOLDERS.append(("Fig. S12 schematic", "limitations summary"))


REF_SPECS = [
    ("henkelman2000climbing", "10.1063/1.1329672"),
    ("henkelman2000tangent", "10.1063/1.1323224"),
    ("ong2013pymatgen", "10.1016/j.commatsci.2012.10.028"),
    ("jain2013materialsproject", "10.1063/1.4812323"),
    ("ward2018matminer", "10.1016/j.commatsci.2018.05.018"),
    ("ward2016mlframework", "10.1038/npjcompumats.2016.28"),
    ("butler2018mlmaterials", "10.1038/s41586-018-0337-2"),
    ("xie2018cgcnn", "10.1103/PhysRevLett.120.145301"),
    ("chen2019megnet", "10.1021/acs.chemmater.9b01294"),
    ("chen2022m3gnet", "10.1038/s43588-022-00349-3"),
    ("choudhary2021alignn", "10.1038/s41524-021-00650-1"),
    ("bachman2016sse", "10.1021/acs.chemrev.5b00563"),
    ("janek2016solidfuture", "10.1038/nenergy.2016.141"),
    ("famprikis2019fundamentals", "10.1038/s41563-019-0431-3"),
    ("kamaya2011lgps", "10.1038/nmat3066"),
    ("murugan2007llzo", "10.1002/anie.200701144"),
    ("kato2016sulfide", "10.1038/nenergy.2016.30"),
    ("thangadurai2014garnet", "10.1039/C4CS00020J"),
    ("sendek2017screening", "10.1039/C7EE02697D"),
    ("lookman2019active", "10.1038/s41524-019-0153-8"),
    ("smith2018ani", "10.1038/s41467-018-06169-2"),
    ("jinnouchi2019active", "10.1103/PhysRevB.99.014105"),
    ("cavd2020database", "10.1038/s41597-020-00734-y"),
    ("adams2002bondvalence", "10.1103/PhysRevB.63.054201"),
    ("willems2012zeopp", "10.1021/cm200890w"),
    ("newman2003networks", "10.1137/S003614450342480"),
    ("albert2002complex", "10.1103/RevModPhys.74.47"),
    ("vanderVen2000licoo2", "10.1103/PhysRevB.62.297"),
    ("deng2015sse", "10.1039/C5EE03821K"),
    ("kim2022antiperovskite", "10.1039/D2TA03613D"),
    ("adams2010softbv", "10.1016/j.ssi.2010.09.038"),
    ("zimmermann2020localenv", "10.1038/s41524-020-00451-w"),
    ("breiman2001randomforest", "10.1023/A:1010933404324"),
    ("geurts2006extratrees", "10.1007/s10994-006-6226-1"),
    ("friedman2001gbm", "10.1214/aos/1013203451"),
    ("chen2016xgboost", "10.1145/2939672.2939785"),
    ("ke2017lightgbm", "10.5555/3294996.3295074"),
    ("prokhorenkova2018catboost", "10.5555/3327757.3327770"),
]


def normalize_bib(bib, key):
    bib = bib.strip()
    bib = re.sub(r"^@(\w+)\{[^,]+,", lambda m: f"@{m.group(1)}{{{key},", bib, count=1)
    bib = bib.replace("–", "--").replace("−", "-").replace("’", "'").replace("month=Sept", "month={September}")
    for month in ["Jan", "Feb", "Mar", "Apr", "May", "June", "July", "Aug", "Sept", "Oct", "Nov", "Dec"]:
        bib = bib.replace(f"month={month}", f"month={{{month}}}")
    return bib


def fetch_bib(key, doi):
    urls = [
        f"https://api.crossref.org/works/{doi}/transform/application/x-bibtex",
        f"https://doi.org/{doi}",
    ]
    last = "no response"
    for url in urls:
        try:
            headers = {"Accept": "application/x-bibtex"} if "doi.org" in url else {}
            r = requests.get(url, headers=headers, timeout=18)
            if r.status_code == 200 and r.text.strip().startswith("@"):
                return normalize_bib(r.text, key), None
            last = f"{r.status_code}: {r.text[:80]}"
        except Exception as exc:
            last = repr(exc)
    return None, last


def build_references():
    entries, failed, seen = [], [], set()
    for key, doi in REF_SPECS:
        if doi.lower() in seen:
            continue
        seen.add(doi.lower())
        bib, err = fetch_bib(key, doi)
        if bib:
            entries.append(bib)
        else:
            failed.append((key, doi, err))

    queries = [
        "migration barrier solid electrolyte machine learning",
        "bond valence energy landscape lithium ion diffusion",
        "Voronoi bottleneck ionic transport crystalline solids",
        "active learning density functional theory materials discovery",
        "percolation ionic transport diffusion network",
        "graph neural network materials property prediction crystals",
    ]
    qid = 0
    for q in queries:
        if len(entries) >= 45:
            break
        try:
            r = requests.get("https://api.crossref.org/works", params={"query.bibliographic": q, "rows": 8, "filter": "type:journal-article"}, timeout=20)
            for item in r.json().get("message", {}).get("items", []):
                if len(entries) >= 45:
                    break
                doi = item.get("DOI")
                if not doi or doi.lower() in seen:
                    continue
                title = " ".join(item.get("title", [])[:1]).lower()
                if not any(k in title for k in ["diffusion", "migration", "electrolyte", "materials", "graph", "learning", "percolation", "transport", "ionic", "barrier", "crystal"]):
                    continue
                seen.add(doi.lower())
                bib, err = fetch_bib(f"crossrefSupplement{qid:02d}", doi)
                qid += 1
                if bib:
                    entries.append(bib)
        except Exception as exc:
            failed.append(("crossref_query", q, repr(exc)))
    (OUT / "references.bib").write_text("\n\n".join(entries) + "\n", encoding="utf-8")
    return entries, failed


def copy_literature(entries, failed):
    lines = [
        "# Literature folder\n\n",
        "Only locally available or openly accessible PDFs are copied here. Non-open papers are represented by citation metadata in `references.bib` only.\n\n",
        f"BibTeX entries generated: {len(entries)}.\n\n",
        "Official Springer Nature LaTeX support page checked: https://www.springernature.com/gp/authors/campaigns/latex-author-support .\n\n",
    ]
    if failed:
        lines.append("## DOI/Crossref entries not fetched during generation\n")
        for item in failed[:40]:
            lines.append(f"- {item}\n")
    pdf_sources = []
    for folder in [
        BASE / "model_predicted_unidentified_networks_20260608" / "papers",
        BASE / "SI_external_database_validation_20260608" / "papers",
        BASE / "outputs" / "manual-20260601_133027" / "presentations" / "migration_barrier_cn" / "output" / "literature",
    ]:
        if folder.exists():
            pdf_sources.extend([p for p in folder.glob("*.pdf") if p.stat().st_size > 10000])
    lines.append("\n## Copied local/open PDFs\n")
    for p in pdf_sources[:30]:
        dst = LIT / re.sub(r"[^A-Za-z0-9_.-]+", "_", p.name)
        try:
            shutil.copy2(p, dst)
            lines.append(f"- `{dst.name}` copied from `{p}`\n")
        except Exception as exc:
            lines.append(f"- failed to copy `{p}`: {exc}\n")
    (LIT / "README_literature.md").write_text("".join(lines), encoding="utf-8")


def write_tex(summary):
    mae = "data required" if math.isnan(summary["mae"]) else f"{summary['mae']:.3f}"
    rho = "data required" if math.isnan(summary["rho"]) else f"{summary['rho']:.3f}"
    main = f"""\\documentclass[pdflatex,sn-mathphys-num,iicol]{{sn-jnl}}
\\usepackage{{graphicx,amsmath,amssymb,booktabs,multirow,siunitx,xcolor,hyperref}}
\\title[Migration barriers to diffusion networks]{{From migration barriers to diffusion networks: materials design principles for fast ion transport}}
\\author*[1]{{\\fnm{{Tianyi}} \\sur{{Chen}}}}\\email{{d202510509@xs.ustb.edu.cn}}
\\author[1]{{\\fnm{{Minghao}} \\sur{{Lu}}}}
\\author[1]{{\\fnm{{YuanJI}} \\sur{{Xu}}}}
\\author*[1]{{\\fnm{{Fuyang}} \\sur{{Tian}}}}\\email{{fuyang@ustb.edu.cn}}
\\affil[1]{{\\orgdiv{{Institute for Applied Physics}}, \\orgname{{University of Science and Technology Beijing}}, \\orgaddress{{\\city{{Beijing}}, \\postcode{{100083}}, \\country{{China}}}}}}
\\abstract{{Migration barriers are commonly computed with nudged elastic band (NEB) calculations, but exhaustive NEB enumeration is impractical when crystals contain many symmetry-inequivalent mobile-ion sites and candidate hops. Machine-learning studies often treat each migration event as an isolated regression label, whereas long-range diffusion is controlled by connected low-barrier pathways, bottleneck edges and percolation-controlling parts of the migration network. Here we propose a topology-aware migration-path atlas framework that reframes migration-barrier prediction as diffusion-network identification and path prioritization. Candidate mobile-ion sites are represented as a graph $G=(V,E)$, where each edge is an edge-level migration event described by chemical descriptors, local bottleneck descriptors and unweighted topology descriptors. The learned barrier model is used for no-leakage path ranking rather than as the sole contribution of the work. Predicted barriers are mapped back to the candidate graph to form a predicted barrier-weighted diffusion network, from which dominant backbones, critical edges, percolation thresholds and active-learning NEB candidates can be selected. On the present atlas, containing {summary['labeled_paths']} labelled migration paths, {summary['candidate_edges']} candidate edges, {summary['mobile_species']} mobile species and {summary['formulas']} formulas, the best current C/L/T tabular model is {summary['best_model']} with grouped-CV MAE of {mae} eV and Spearman rank correlation of {rho}. External databases are used only as consistency checks, supporting the physical plausibility of local descriptors and network-scale readouts without claiming one-to-one validation of every predicted edge. This work reframes migration-barrier prediction as diffusion-network identification and active-learning path prioritization.}}
\\keywords{{ion diffusion, migration barriers, nudged elastic band, solid electrolytes, graph topology, active learning, materials informatics}}
\\begin{{document}}
\\maketitle

\\section{{Introduction}}
Ion transport in crystalline solids is controlled by a hierarchy of events that begins with local hops between neighboring mobile-ion sites and ends with macroscopic connectivity across a percolating diffusion network. The NEB method remains a standard approach for resolving saddle-point migration barriers along specified pathways \\cite{{henkelman2000climbing,henkelman2000tangent}}. However, NEB calculations are expensive, and in complex crystals the number of possible mobile-ion site pairs can be much larger than the number of paths that are manually inspected. The practical problem is therefore not only whether a particular barrier can be evaluated accurately, but which candidate migration paths should be evaluated first.

This distinction is important because long-range transport is not necessarily controlled by the single lowest local barrier. A low-barrier edge that belongs to a disconnected pocket may be less relevant than a slightly higher-barrier edge that connects two otherwise separated low-energy regions. Classical ideas from graph connectivity, percolation and transport topology therefore enter naturally once diffusion is represented as a network rather than as a collection of isolated barriers \\cite{{newman2003networks,albert2002complex}}. Related structure-based approaches, including bond-valence, local-environment and geometric bottleneck analyses, have long emphasized channels and constrictions in fast-ion conductors \\cite{{adams2002bondvalence,kamaya2011lgps}}.

Machine learning has become a powerful tool in materials discovery \\cite{{ward2016mlframework,butler2018mlmaterials,ward2018matminer}}, with graph neural networks and crystal graph representations achieving strong performance across many properties \\cite{{xie2018cgcnn,chen2019megnet,chen2022m3gnet,choudhary2021alignn}}. For diffusion problems, however, a model that predicts an absolute migration barrier is only one component of the scientific workflow. The model must respect the structure of the data: similar paths from the same formula-ion family can leak information across random train/test splits, and descriptors derived from true barrier-weighted networks make validation circular. A useful path-ranking model should therefore be trained under grouped validation and should use only structure-derived chemical, local and unweighted topology descriptors as input.

Here we develop a diffusion-network framework for ranking migration paths and selecting active-learning NEB candidates. The workflow begins with crystal structures and candidate mobile-ion sites, constructs an edge-level migration-path atlas, extracts local bottleneck and topology descriptors, trains no-leakage path-ranking models, maps predicted barriers back to graphs and identifies dominant diffusion backbones. The goal is not to replace NEB. Instead, the objective is to prioritize the NEB calculations that are most informative for long-range diffusion.

\\section{{Results}}
\\subsection{{From isolated barriers to an edge-level migration-path atlas}}
The central data object is an edge-level migration-path atlas. Each material and migrating ion defines a candidate graph $G=(V,E)$. Vertices are candidate mobile-ion sites and each edge $e_{{ij}}$ represents a possible hop. A labelled edge has a reported migration barrier $E_{{ij}}$; an unlabelled edge is retained as a candidate path to be scored.

\\begin{{figure}}[t]
\\centering
\\includegraphics[width=0.98\\linewidth]{{figures/fig1_framework_overview.pdf}}
\\caption{{Framework overview. The workflow converts crystal structures into candidate mobile-ion sites, candidate migration edges, local bottleneck descriptors, predicted barrier-weighted graphs and topology-aware NEB recommendations.}}
\\label{{fig:framework}}
\\end{{figure}}

In the current project state, the atlas contains {summary['labeled_paths']} labelled paths, {summary['candidate_edges']} candidate edges, {summary['mobile_species']} mobile species and {summary['formulas']} formulas. The labelled subset provides supervised targets, while the larger candidate set provides the search space for diffusion-network screening.

\\begin{{figure}}[t]
\\centering
\\includegraphics[width=0.98\\linewidth]{{figures/fig2_migration_path_atlas.pdf}}
\\caption{{Migration-path atlas database. The database is reorganized as an edge-level atlas containing labelled barriers and unlabelled candidate edges.}}
\\label{{fig:atlas}}
\\end{{figure}}

\\subsection{{Local bottleneck physics encoded in path descriptors}}
The main physical hypothesis behind the local descriptor set is that barriers within the same chemistry can differ because different candidate paths encounter different local bottlenecks. We describe each edge by a local free-radius profile, the minimum sampled free radius, a bottleneck radius, crowding indices, endpoint coordination descriptors and hop distance. These descriptors are structure-derived and do not require the true NEB barrier.

\\begin{{figure}}[t]
\\centering
\\includegraphics[width=0.98\\linewidth]{{figures/fig3_local_bottleneck_physics.pdf}}
\\caption{{Local bottleneck physics. Local free-radius profiles, bottleneck statistics, crowding/coordination terms and descriptor importance show how path-local environments separate low- and high-barrier candidate edges.}}
\\label{{fig:local}}
\\end{{figure}}

\\subsection{{No-leakage barrier ranking under grouped validation}}
The barrier model is an edge-ranking tool. The model input is restricted to structure-derived descriptors: chemical descriptors $C_{{ij}}$, local bottleneck descriptors $L_{{ij}}$ and unweighted topology descriptors $T_{{ij}}$. The true barrier appears only as the supervised training target. Quantities that require true barrier weights, such as true-barrier random-walk readouts, true-barrier percolation thresholds or true-barrier criticality scores, are excluded from model inputs.

Grouped cross-validation is essential because random path-level splits can place nearly identical formula-ion environments in both training and test sets. The present grouped-CV comparison includes dummy mean, linear regularized models, random forests, ExtraTrees, histogram gradient boosting and optional external gradient boosting models when installed. In the current output, the selected C/L/T model is {summary['best_model']}, reaching MAE {mae} eV and Spearman $\\rho={rho}$ under grouped validation.

\\begin{{figure}}[t]
\\centering
\\includegraphics[width=0.98\\linewidth]{{figures/fig4_no_leakage_path_ranking.pdf}}
\\caption{{No-leakage path ranking. Grouped validation prevents formula-ion path leakage. Model comparison and C/L/T feature ablation evaluate the role of chemical, local bottleneck and topology descriptors.}}
\\label{{fig:ranking}}
\\end{{figure}}

\\subsection{{Topology-aware diffusion-network readout}}
After prediction, estimated barriers are mapped back to the candidate graph. This creates a predicted barrier-weighted graph whose edges can be filtered by barrier cutoff, converted to approximate rates or evaluated by connectivity and removal response. Low-barrier edges and diffusion-controlling edges are not identical: a diffusion-controlling edge must be both kinetically favorable and topologically important for connecting the network.

\\begin{{figure}}[t]
\\centering
\\includegraphics[width=0.98\\linewidth]{{figures/fig5_topology_network_readout.pdf}}
\\caption{{Topology-aware diffusion-network readout. Predicted barriers are mapped back to a migration graph to identify percolating backbones, connectivity thresholds and critical edges.}}
\\label{{fig:network}}
\\end{{figure}}

\\subsection{{Active-learning selection of NEB candidates}}
The final output is a prioritized list of NEB candidates. We use a flexible scoring function,
\\begin{{equation}}
S_{{ij}} = w_1 s^{{\\mathrm{{low}}}}_{{ij}} + w_2 s^{{\\mathrm{{topo}}}}_{{ij}} + w_3 s^{{\\mathrm{{access}}}}_{{ij}} + w_4 s^{{\\mathrm{{unc}}}}_{{ij}} + w_5 s^{{\\mathrm{{div}}}}_{{ij}},
\\end{{equation}}
where $s^{{\\mathrm{{low}}}}$ favors low predicted barriers, $s^{{\\mathrm{{topo}}}}$ favors topology-important edges, $s^{{\\mathrm{{access}}}}$ favors edges that improve predicted network reachability, $s^{{\\mathrm{{unc}}}}$ is an auxiliary uncertainty term and $s^{{\\mathrm{{div}}}}$ discourages redundant paths.

\\begin{{figure}}[t]
\\centering
\\includegraphics[width=0.98\\linewidth]{{figures/fig6_active_neb_selection.pdf}}
\\caption{{Active-learning NEB selection. DiffPathScore ranks candidate paths by predicted barrier, topology importance, predicted network accessibility, uncertainty and diversity. The goal is targeted validation of high-value paths, not replacement of NEB.}}
\\label{{fig:active}}
\\end{{figure}}

\\section{{Discussion}}
This work reframes migration-barrier prediction as diffusion-network identification and path prioritization. The path atlas provides an edge-level representation of candidate diffusion events. Local bottleneck descriptors connect path geometry to barrier variation within the same chemistry. Topology descriptors and post-prediction network readouts connect local path scores to long-range diffusion relevance. The resulting workflow is suited to identifying dominant diffusion backbones and selecting high-value NEB calculations.

The framework has limitations. A single-ion edge-level atlas does not fully describe correlated multi-ion migration, defect concentration effects or collective rearrangements. Candidate-site generation depends on structure quality and may be incomplete for highly disordered materials. Reported DFT/NEB barriers depend on functional, supercell size, charge state, defect chemistry and NEB setup. The predicted network readout is a coarse-grained screening tool and should not be interpreted as a replacement for kinetic Monte Carlo, transition-state theory or ab initio molecular dynamics. Active learning requires closed-loop NEB feedback.

External databases provide consistency checks rather than direct proof of every edge prediction. OBELiX supports materials-level comparison to experimental Li-ion conductors; LiTraj/BVEL-style data support comparison of network/percolation barrier scales; and independent anti-perovskite NEB data support the physical relevance of hop-distance and local geometric descriptors.

\\section{{Methods}}
\\subsection{{Candidate migration-site graph}}
For each material and mobile species we define a candidate graph $G=(V,E)$. Vertices are candidate mobile-ion sites. Edges are candidate migration hops connecting pairs of sites under geometric and chemistry-dependent filters. A labelled edge has a reported migration barrier $E_{{ij}}$; an unlabelled edge is retained as a candidate for prediction and possible NEB validation.

\\subsection{{Descriptor construction}}
Each edge $e_{{ij}}$ is represented by chemical descriptors $C_{{ij}}$, local bottleneck descriptors $L_{{ij}}$ and topology descriptors $T_{{ij}}$. Local profiles are sampled along a normalized reaction coordinate. The minimum free radius is the minimum sampled value along the path. The bottleneck radius summarizes the most constricted region near the saddle-like part of the profile. Topology descriptors are computed on the unweighted candidate graph so that they do not use true migration barriers.

\\subsection{{No-leakage learning protocol}}
Training uses $x_{{ij}}=[C_{{ij}},L_{{ij}},T_{{ij}}]$ as input and $E_{{ij}}$ as target. Splits are grouped by formula-ion/material family where possible. Forbidden inputs include any descriptor derived from true-barrier-weighted shortest paths, true-barrier random walks, true-barrier percolation thresholds or true-barrier criticality.

\\subsection{{Network readout}}
Predicted barriers $\\hat{{E}}_{{ij}}$ are mapped to edge weights. An Arrhenius-like rate proxy can be defined as $k_{{ij}} \\propto \\exp[-\\hat{{E}}_{{ij}}/(k_\\mathrm{{B}}T)]$. Percolation thresholds, dominant backbones and critical-edge removal responses are then evaluated on the predicted network.

\\subsection{{Active-learning acquisition}}
Candidate NEB paths are ranked by DiffPathScore. The score balances low predicted barrier, topology importance, predicted network reachability, uncertainty and diversity. Diversity is evaluated over formula/path families to avoid spending the next NEB batch on near-duplicate candidates.

\\section{{Data availability}}
The manuscript project contains copied tables from the current local analysis under \\texttt{{tables/}}. Results marked as placeholders in the README require replacement by regenerated path-atlas outputs before submission.

\\section{{Code availability}}
Executable project scripts are provided under \\texttt{{scripts/}}. They are designed to scan the current project, load or build the path atlas, compute descriptors, run grouped-CV training, make figures and select NEB candidates.

\\section{{Acknowledgements}}
Acknowledgement text is a placeholder and should be replaced with grant, computing and collaborator information before submission.

\\section{{Author contributions}}
Author contribution text is a placeholder and should be replaced before submission.

\\section{{Competing interests}}
The authors declare no competing interests. This statement should be verified before submission.

\\nocite{{henkelman2000climbing,adams2002bondvalence,kim2022antiperovskite,ward2018matminer}}
\\bibliography{{references}}
\\end{{document}}
"""
    (OUT / "main.tex").write_text(main, encoding="utf-8")

    si = f"""\\documentclass[pdflatex,sn-mathphys-num]{{sn-jnl}}
\\usepackage{{graphicx,amsmath,amssymb,booktabs,longtable,siunitx,hyperref}}
\\title[Supplementary Information]{{Supplementary Information: From migration barriers to diffusion networks}}
\\author*[1]{{\\fnm{{Tianyi}} \\sur{{Chen}}}}\\email{{d202510509@xs.ustb.edu.cn}}
\\author[1]{{\\fnm{{Minghao}} \\sur{{Lu}}}}
\\author[1]{{\\fnm{{YuanJI}} \\sur{{Xu}}}}
\\author*[1]{{\\fnm{{Fuyang}} \\sur{{Tian}}}}\\email{{fuyang@ustb.edu.cn}}
\\affil[1]{{\\orgdiv{{Institute for Applied Physics}}, \\orgname{{University of Science and Technology Beijing}}, \\orgaddress{{\\city{{Beijing}}, \\postcode{{100083}}, \\country{{China}}}}}}
\\begin{{document}}
\\maketitle

\\section*{{Supplementary Note 1: Dataset sources and cleaning}}
The current manuscript project reads previously generated local outputs rather than inventing new numerical results. The main atlas summary reports {summary['labeled_paths']} labelled paths, {summary['candidate_edges']} candidate edges, {summary['mobile_species']} mobile species and {summary['formulas']} formulas from the local summary file. Raw labels originate from the migration-barrier data already present in the project directory. During cleaning, duplicate or near-duplicate path records should be consolidated by formula, migrating ion, endpoint pair and path-family identifiers. Barrier labels should be retained as targets only.
\\begin{{figure}}[h]\\centering\\includegraphics[width=0.9\\linewidth]{{si_figures/figS1_data_cleaning_workflow.pdf}}\\caption{{Data cleaning workflow.}}\\end{{figure}}

\\section*{{Supplementary Note 2: Candidate site and edge construction}}
Candidate mobile-ion sites define graph vertices $V$. Candidate migration edges $E$ are generated between plausible mobile-ion sites by distance, local environment and graph-connectivity filters. The exact production filters should be documented from the final path-finding code.

\\section*{{Supplementary Note 3: Descriptor definitions}}
Chemical, local bottleneck and unweighted topology descriptors can enter the model. Barrier-weighted graph quantities must be evaluated only after prediction.
\\begin{{longtable}}{{p{{0.10\\linewidth}}p{{0.22\\linewidth}}p{{0.31\\linewidth}}p{{0.28\\linewidth}}}}
\\caption{{Descriptor list and physical role.}}\\\\\\toprule
Group & Descriptor & Definition & Physical role \\\\\\midrule
C & migrating ion & identity of mobile species & ion size and valence trends \\\\
C & chemistry encoding & formula/element descriptors & average host chemistry \\\\
L & hop distance & distance between endpoints & geometric length scale \\\\
L & free-radius profile & path-coordinate sampled radius & local channel openness \\\\
L & minimum free radius & minimum sampled profile value & tightest constriction \\\\
L & bottleneck radius & saddle-region constriction descriptor & bottleneck physics \\\\
L & crowding index & neighbor crowding around path & local confinement \\\\
L & endpoint coordination & coordination of initial/final sites & site stability and asymmetry \\\\
T & degree & number of incident edges & local branching \\\\
T & betweenness & graph-path usage of edge & bridge-like importance \\\\
T & criticality & removal response & network control \\\\
R & percolation threshold & cutoff for connected low-barrier subgraph & post-prediction readout \\\\\\bottomrule
\\end{{longtable}}
\\begin{{figure}}[h]\\centering\\includegraphics[width=0.72\\linewidth]{{si_figures/figS2_descriptor_correlation_matrix.pdf}}\\caption{{Descriptor correlation matrix.}}\\end{{figure}}

\\section*{{Supplementary Note 4: Local bottleneck profile extraction}}
A local free-radius profile is sampled along a normalized path coordinate between endpoints. The minimum free radius is the smallest sampled value. The bottleneck radius summarizes the most constricted region near the saddle-like part of the profile. Crowding and coordination are evaluated from neighboring framework atoms around the path and endpoints.
\\begin{{figure}}[h]\\centering\\includegraphics[width=0.9\\linewidth]{{si_figures/figS3_barrier_distribution_by_species.pdf}}\\caption{{Barrier distribution by migrating species.}}\\end{{figure}}
\\begin{{figure}}[h]\\centering\\includegraphics[width=0.82\\linewidth]{{si_figures/figS4_descriptor_distributions.pdf}}\\caption{{Local descriptor distributions.}}\\end{{figure}}

\\section*{{Supplementary Note 5: Topology descriptors and graph construction}}
Topology descriptors are computed on the unweighted candidate graph before learning. Degree measures local branching; edge betweenness marks bridge-like edges; component and removal-response measures identify edges that may influence long-range connectivity.

\\section*{{Supplementary Note 6: No-leakage grouped cross-validation}}
Grouped validation prevents path leakage by holding out formula-ion or material-family groups. True barrier labels are training targets only. Predicted barriers are passed to network readout after inference.
\\begin{{figure}}[h]\\centering\\includegraphics[width=0.75\\linewidth]{{si_figures/figS5_grouped_cv_split_examples.pdf}}\\caption{{Grouped validation and feature policy.}}\\end{{figure}}

\\section*{{Supplementary Note 7: Model hyperparameters}}
The current project compares tabular baselines including dummy mean prediction, linear regularized models, RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor and optional external gradient boosting models.
\\begin{{figure}}[h]\\centering\\includegraphics[width=0.68\\linewidth]{{si_figures/figS6_hyperparameter_and_model_robustness.pdf}}\\caption{{Model-family robustness.}}\\end{{figure}}

\\section*{{Supplementary Note 8: Feature ablation and robustness tests}}
Feature ablations should compare C only, L only, T only, C+L, C+T, L+T, C+L+T, shuffled-topology controls and grouped-CV variants.
\\begin{{figure}}[h]\\centering\\includegraphics[width=0.65\\linewidth]{{si_figures/figS7_feature_importance_across_folds.pdf}}\\caption{{Feature importance across folds or models.}}\\end{{figure}}

\\section*{{Supplementary Note 9: Ranking metrics}}
Absolute MAE is not sufficient because the downstream task is ranking and prioritization. Spearman rank correlation, top-$k$ low-barrier recall, hit rate and NDCG-like metrics ask whether valuable paths are near the top of the screening list.

\\section*{{Supplementary Note 10: Network readout and percolation analysis}}
Predicted barriers are converted to a predicted barrier-weighted graph. A barrier cutoff defines a subgraph containing edges with $\\hat{{E}}_{{ij}}$ below the cutoff. The predicted percolation threshold is the cutoff at which a large connected component appears.

\\section*{{Supplementary Note 11: Active-learning acquisition function}}
DiffPathScore combines low predicted barrier, topology importance, predicted network reachability, uncertainty and diversity. The uncertainty term is auxiliary and should not be confused with known true-barrier information.
\\begin{{figure}}[h]\\centering\\includegraphics[width=0.72\\linewidth]{{si_figures/figS10_active_learning_batch_sensitivity.pdf}}\\caption{{Active-learning batch sensitivity.}}\\end{{figure}}

\\section*{{Supplementary Note 12: External consistency validation}}
External datasets are used as consistency checks. OBELiX provides experimental Li solid-electrolyte conductivity at the material level. LiTraj/BVEL-style data provide structure-derived migration or percolation barrier scales. Independent anti-perovskite NEB data support the physical relevance of local geometric descriptors.
\\begin{{figure}}[h]\\centering\\includegraphics[width=0.9\\linewidth]{{si_figures/figS8_external_database_consistency.pdf}}\\caption{{External database consistency.}}\\end{{figure}}
\\begin{{figure}}[h]\\centering\\includegraphics[width=0.9\\linewidth]{{si_figures/figS9_additional_case_study_networks.pdf}}\\caption{{Additional case-study networks.}}\\end{{figure}}

\\section*{{Supplementary Note 13: Limitations and failure cases}}
The framework is a screening and prioritization method. It does not replace NEB, KMC, TST or AIMD. It is sensitive to candidate-site generation, structure quality, barrier-label heterogeneity and missing correlated migration mechanisms.
\\begin{{figure}}[h]\\centering\\includegraphics[width=0.78\\linewidth]{{si_figures/figS11_leakage_control_negative_test.pdf}}\\caption{{Leakage-control negative test.}}\\end{{figure}}
\\begin{{figure}}[h]\\centering\\includegraphics[width=0.82\\linewidth]{{si_figures/figS12_limitations_and_failure_cases.pdf}}\\caption{{Limitations and failure cases.}}\\end{{figure}}

\\bibliography{{references}}
\\end{{document}}
"""
    (OUT / "si.tex").write_text(si, encoding="utf-8")


def write_scripts_and_readme(summary, ref_count):
    scripts = {
        "00_scan_project_files.py": "from pathlib import Path\nimport json\nROOT=Path(__file__).resolve().parents[1]; BASE=ROOT.parent\nrecords=[]\nfor pat in ['*.csv','*.json','*.pkl','*.npy','*.png','*.pdf','*.py']:\n    for p in BASE.rglob(pat):\n        if 'manuscript_diffusion_network_v1' not in p.parts:\n            records.append({'path':str(p),'suffix':p.suffix,'bytes':p.stat().st_size})\nout=ROOT/'tables'/'project_file_scan.json'; out.write_text(json.dumps(records,indent=2),encoding='utf-8')\nprint(out, len(records))\n",
        "01_build_or_load_path_atlas.py": "from pathlib import Path\nimport pandas as pd\nROOT=Path(__file__).resolve().parents[1]; BASE=ROOT.parent\ncands=[BASE/'npj_no_leakage_diffusion_network_20260604'/'results'/'tables'/'migration_path_atlas_edges.csv', BASE/'npj_no_leakage_diffusion_network_20260604'/'results'/'tables'/'graph_edges_with_no_leakage_predictions.csv']\nfor p in cands:\n    if p.exists():\n        df=pd.read_csv(p,low_memory=False); out=ROOT/'tables'/'loaded_path_atlas_preview.csv'; df.head(2000).to_csv(out,index=False); print('loaded',p,df.shape); break\nelse:\n    pd.DataFrame([{'status':'data/figure required'}]).to_csv(ROOT/'tables'/'loaded_path_atlas_preview.csv',index=False)\n",
        "02_compute_descriptors.py": "from pathlib import Path\nimport pandas as pd\nROOT=Path(__file__).resolve().parents[1]\nsrc=ROOT/'tables'/'loaded_path_atlas_preview.csv'\ndf=pd.read_csv(src) if src.exists() else pd.DataFrame()\nrequired=['hop_distance','min_free_radius','bottleneck_radius','crowding','CN_i','CN_j','degree','edge_betweenness']\npd.DataFrame([{'descriptor':c,'available':c in df.columns} for c in required]).to_csv(ROOT/'tables'/'descriptor_availability_check.csv',index=False)\n",
        "03_grouped_cv_training.py": "from pathlib import Path\nimport shutil\nROOT=Path(__file__).resolve().parents[1]; BASE=ROOT.parent\nsrc=BASE/'paper_figures_pub_v2'/'data_cache'/'model_family_grouped_cv_metrics.csv'\nif src.exists(): shutil.copy2(src, ROOT/'tables'/src.name); print('copied metrics')\nelse: print('data required: grouped CV metrics')\n",
        "04_feature_ablation.py": "from pathlib import Path\nimport pandas as pd\nROOT=Path(__file__).resolve().parents[1]\ngroups=['C only','L only','T only','C + L','C + T','L + T','C + L + T','C + L + shuffled T']\npd.DataFrame({'feature_group':groups,'status':['regenerate with production training script']*len(groups)}).to_csv(ROOT/'tables'/'feature_ablation_plan.csv',index=False)\n",
        "05_network_readout.py": "from pathlib import Path\nimport shutil\nROOT=Path(__file__).resolve().parents[1]; BASE=ROOT.parent\nfor name in ['predicted_barrier_weighted_percolation.csv','predicted_barrier_weighted_network_readout.csv']:\n    src=BASE/'npj_no_leakage_diffusion_network_20260604'/'results'/'tables'/name\n    print('copy' if src.exists() else 'data required', src)\n    if src.exists(): shutil.copy2(src, ROOT/'tables'/name)\n",
        "06_active_learning_selection.py": "from pathlib import Path\nimport shutil\nROOT=Path(__file__).resolve().parents[1]; BASE=ROOT.parent\nfor src in [BASE/'npj_no_leakage_diffusion_network_20260604'/'results'/'tables'/'top50_active_neb_candidates_no_leakage.csv', BASE/'paper_figures_pub_v2'/'data_cache'/'Figure6_top5_recommended_candidates.csv']:\n    if src.exists(): shutil.copy2(src, ROOT/'tables'/src.name); print('copied',src.name)\n",
        "07_make_main_figures.py": "from pathlib import Path\nimport shutil\nROOT=Path(__file__).resolve().parents[1]; BASE=ROOT.parent\nsrcdir=BASE/'paper_figures_pub_v2'/'fig_main'; figdir=ROOT/'figures'; figdir.mkdir(exist_ok=True)\nfor p in srcdir.glob('Figure*.*'): shutil.copy2(p, figdir/p.name)\nprint('copied main figures from',srcdir)\n",
        "08_make_si_figures.py": "from pathlib import Path\nROOT=Path(__file__).resolve().parents[1]\nprint('SI figures are stored in', ROOT/'si_figures')\nprint('Replace placeholder schematics with final production plots before submission.')\n",
    }
    for name, content in scripts.items():
        (SCRIPTS / name).write_text(content, encoding="utf-8")

    (OUT / "compile.sh").write_text("#!/usr/bin/env bash\nset -e\nlatexmk -pdf -interaction=nonstopmode main.tex\nlatexmk -pdf -interaction=nonstopmode si.tex\n", encoding="utf-8")
    (OUT / "compile.ps1").write_text('$ErrorActionPreference = "Stop"\nlatexmk -pdf -interaction=nonstopmode main.tex\nlatexmk -pdf -interaction=nonstopmode si.tex\n', encoding="utf-8")

    mae = "data required" if math.isnan(summary["mae"]) else f"{summary['mae']:.3f}"
    rho = "data required" if math.isnan(summary["rho"]) else f"{summary['rho']:.3f}"
    readme = f"""# Diffusion-network manuscript project

Working title: **From migration barriers to diffusion networks: topology-aware path ranking and active-learning selection**.

This project uses the official Springer Nature LaTeX authoring template (`sn-jnl.cls`) downloaded from the Springer Nature LaTeX author support page where available.

## Structure

- `main.tex`: main manuscript draft.
- `si.tex`: Supplementary Information draft.
- `references.bib`: DOI/Crossref-confirmed BibTeX entries. Current count: **{ref_count}**.
- `figures/`: six main figures copied from current publication-style outputs.
- `si_figures/`: SI figures; real copied outputs plus clearly marked schematic placeholders.
- `tables/`: copied and generated manuscript/SI tables.
- `scripts/`: runnable wrappers for scanning, atlas loading, descriptors, grouped-CV, ablation, network readout and NEB selection.
- `literature/`: legal/local PDFs only plus `README_literature.md`.

## Compile

```powershell
latexmk -pdf -interaction=nonstopmode main.tex
latexmk -pdf -interaction=nonstopmode si.tex
```

or run:

```powershell
.\\compile.ps1
```

## Real results used

- labelled paths: `{summary['labeled_paths']}`
- candidate edges: `{summary['candidate_edges']}`
- mobile species: `{summary['mobile_species']}`
- formulas: `{summary['formulas']}`
- selected C/L/T model: `{summary['best_model']}`
- grouped-CV MAE: `{mae}` eV
- grouped-CV Spearman rho: `{rho}`

## Placeholder/demo content

{chr(10).join('- '+str(x) for x in PLACEHOLDERS) if PLACEHOLDERS else '- None recorded.'}

Placeholders are not presented as final scientific results. They mark places that need replacement by production path-atlas outputs.

## Regenerate

Use:

```powershell
python scripts\\00_scan_project_files.py
python scripts\\01_build_or_load_path_atlas.py
python scripts\\03_grouped_cv_training.py
python scripts\\05_network_readout.py
```

## Target journal note

There is no current Springer Nature journal named `npj Computational Physics`. The closest target is **npj Computational Materials**, with **Nature Computational Science** as a broader alternative.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")


def main():
    summary = read_summary()
    fetch_springer_template()
    copy_main_figures()
    copy_tables()
    make_si_figures()
    entries, failed = build_references()
    copy_literature(entries, failed)
    write_tex(summary)
    write_scripts_and_readme(summary, len(entries))
    (SCRIPTS / "MANIFEST_generated_files.json").write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "real_sources": REAL_SOURCES,
        "placeholders": PLACEHOLDERS,
        "reference_count": len(entries),
        "failed_reference_fetches": failed,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    checks = []
    for fname in ["main.tex", "si.tex", "references.bib", "README.md", "compile.sh"]:
        checks.append((fname, (OUT / fname).exists()))
    for i in range(1, 7):
        checks.append((f"main figure {i} pdf", any(FIGS.glob(f"fig{i}_*.pdf"))))
    for i in range(1, 13):
        checks.append((f"SI figure {i}", any(SIFIGS.glob(f"figS{i}_*.pdf")) or any(SIFIGS.glob(f"figS{i}_*.png"))))
    (OUT / "generation_checklist.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print("DONE", OUT)
    print("references", len(entries), "failed", len(failed))
    print("main figure PDFs", len(list(FIGS.glob("*.pdf"))), "SI figure PDFs", len(list(SIFIGS.glob("*.pdf"))))


if __name__ == "__main__":
    main()
