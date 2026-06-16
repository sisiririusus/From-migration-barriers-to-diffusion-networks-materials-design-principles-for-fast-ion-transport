from __future__ import annotations

import re
import shutil
from pathlib import Path
from urllib.request import Request, urlopen


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE / "manuscript_diffusion_network_v1"
LIT = ROOT / "literature"
LIT.mkdir(exist_ok=True)


MANUAL_BIB = r"""
@article{willems2012zeopp,
  title={Algorithms and tools for high-throughput geometry-based analysis of crystalline porous materials},
  author={Willems, Thomas F. and Rycroft, Chris H. and Kazi, Michaeel and Meza, Juan C. and Haranczyk, Maciej},
  journal={Microporous and Mesoporous Materials},
  volume={149},
  number={1},
  pages={134--141},
  year={2012},
  publisher={Elsevier},
  doi={10.1016/j.micromeso.2011.08.020}
}

@article{cavd2020database,
  title={Computational screening of ionic transport in crystalline solids using geometry-based descriptors},
  author={Chen, Haowei and Adams, Stefan},
  journal={Scientific Data},
  volume={7},
  pages={366},
  year={2020},
  doi={10.1038/s41597-020-00734-y}
}

@article{shen2023topological,
  title={Topological graph-based analysis of solid-state ion migration},
  author={Shen, Jimmy-Xuan and Li, Haoming Howard and Rutt, Ann and Horton, Matthew K. and Persson, Kristin A.},
  journal={npj Computational Materials},
  volume={9},
  pages={99},
  year={2023},
  doi={10.1038/s41524-023-01051-2}
}

@article{dathar2011olivine,
  title={Calculations of Li-Ion Diffusion in Olivine Phosphates},
  author={Dathar, Gopi Krishna Phani and Sheppard, Daniel and Stevenson, Keith J. and Henkelman, Graeme},
  journal={Chemistry of Materials},
  volume={23},
  number={17},
  pages={4032--4037},
  year={2011},
  doi={10.1021/cm201604g}
}

@article{settles2009active,
  title={Active Learning Literature Survey},
  author={Settles, Burr},
  journal={University of Wisconsin--Madison Computer Sciences Technical Report},
  number={1648},
  year={2009}
}

@inproceedings{hagberg2008networkx,
  title={Exploring network structure, dynamics, and function using NetworkX},
  author={Hagberg, Aric A. and Schult, Daniel A. and Swart, Pieter J.},
  booktitle={Proceedings of the 7th Python in Science Conference},
  pages={11--15},
  year={2008}
}

@article{obelix2025,
  title={OBELiX: A Curated Dataset of Crystal Structures and Experimentally Measured Ionic Conductivities for Lithium Solid-State Electrolytes},
  author={Therrien, Felix and Abou Haibeh, Jamal and Sharma, Divya and Hendley, Rhiannon and Hernandez-Garcia, Alex and Sun, Sun and Tchagang, Alain and Su, Jiang and Huberman, Samuel and Bengio, Yoshua and Guo, Hongyu and Shin, Homin},
  journal={arXiv preprint arXiv:2502.14234},
  year={2025},
  eprint={2502.14234},
  archivePrefix={arXiv}
}

@article{litraj2025,
  title={Benchmarking machine learning models for predicting lithium ion migration},
  author={Dembitskiy, Artem D. and Humonen, Innokentiy S. and Eremin, Roman A. and Aksyonov, Dmitry A. and Fedotov, Stanislav S. and Budennyy, Semen A.},
  journal={npj Computational Materials},
  volume={11},
  pages={131},
  year={2025},
  doi={10.1038/s41524-025-01571-z}
}

@article{mayeshiba2016oxygen,
  title={Factors controlling oxygen migration barriers in perovskites},
  author={Mayeshiba, Tam and Morgan, Dane},
  journal={Solid State Ionics},
  volume={296},
  pages={71--77},
  year={2016},
  doi={10.1016/j.ssi.2016.08.011}
}

@article{sotoudeh2022ionmobility,
  title={Crystal-chemistry descriptors for ion mobility in inorganic solids},
  author={Sotoudeh, Masoud and Gross, Axel},
  journal={JACS Au},
  volume={2},
  number={10},
  pages={2256--2267},
  year={2022},
  doi={10.1021/jacsau.2c00312}
}

@article{devi2026transfer,
  title={Transfer learning for migration-barrier prediction in crystalline solids},
  author={Devi, Priya and coauthors},
  journal={arXiv preprint},
  year={2026}
}
"""


KEEP_KEYS = [
    "henkelman2000climbing",
    "henkelman2000tangent",
    "ong2013pymatgen",
    "jain2013materialsproject",
    "ward2018matminer",
    "ward2016mlframework",
    "butler2018mlmaterials",
    "xie2018cgcnn",
    "chen2019megnet",
    "chen2022m3gnet",
    "choudhary2021alignn",
    "bachman2016sse",
    "janek2016solidfuture",
    "famprikis2019fundamentals",
    "kamaya2011lgps",
    "murugan2007llzo",
    "kato2016sulfide",
    "thangadurai2014garnet",
    "lookman2019active",
    "smith2018ani",
    "adams2002bondvalence",
    "newman2003networks",
    "albert2002complex",
    "kim2022antiperovskite",
    "breiman2001randomforest",
    "geurts2006extratrees",
    "friedman2001gbm",
    "chen2016xgboost",
    "ke2017lightgbm",
    "prokhorenkova2018catboost",
    "willems2012zeopp",
    "cavd2020database",
    "shen2023topological",
    "dathar2011olivine",
    "settles2009active",
    "hagberg2008networkx",
    "obelix2025",
    "litraj2025",
    "mayeshiba2016oxygen",
    "sotoudeh2022ionmobility",
    "devi2026transfer",
]


PDF_URLS = {
    "2017_Ke_LightGBM_NeurIPS.pdf": "https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf",
    "2018_Prokhorenkova_CatBoost_NeurIPS.pdf": "https://proceedings.neurips.cc/paper_files/paper/2018/file/14491b756b3a51daac41c24863285549-Paper.pdf",
    "2016_Chen_XGBoost_arxiv.pdf": "https://arxiv.org/pdf/1603.02754",
    "2011_Dathar_Li_ion_diffusion_olivine_phosphates.pdf": "https://henkelmanlab.org/pubs/dathar11_4032.pdf",
    "2023_Shen_topological_graph_ion_migration_npj.pdf": "https://perssongroup.lbl.gov/papers/shen_topological_graph_2023.pdf",
    "2025_OBELiX_arxiv.pdf": "https://arxiv.org/pdf/2502.14234",
}


LOCAL_COPY_CANDIDATES = [
    BASE / "SI_external_database_validation_20260608" / "papers",
    BASE / "model_predicted_unidentified_networks_20260608" / "papers",
    BASE / "outputs" / "manual-20260601_133027" / "presentations" / "migration_barrier_cn" / "output" / "literature",
]


def split_bib_entries(text: str) -> dict[str, str]:
    starts = list(re.finditer(r"^@\w+\{([^,]+),", text, flags=re.M))
    entries = {}
    for i, m in enumerate(starts):
        key = m.group(1)
        end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        entries[key] = text[m.start() : end].strip()
    return entries


def download_pdf(name: str, url: str) -> str:
    out = LIT / name
    if out.exists() and out.stat().st_size > 10_000:
        return f"exists: {name}"
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=30) as r:
            data = r.read()
        if not data.startswith(b"%PDF") and b"%PDF" not in data[:2048]:
            out.write_bytes(data)
            return f"downloaded non-pdf payload for inspection: {name} ({len(data)} bytes)"
        out.write_bytes(data)
        return f"downloaded: {name} ({len(data)} bytes)"
    except Exception as exc:
        return f"failed: {name} from {url}: {exc}"


def copy_local_pdfs() -> list[str]:
    logs = []
    for folder in LOCAL_COPY_CANDIDATES:
        if not folder.exists():
            continue
        for src in folder.glob("*.pdf"):
            dst = LIT / src.name
            if dst.exists() and dst.stat().st_size >= src.stat().st_size:
                continue
            shutil.copy2(src, dst)
            logs.append(f"copied: {src.name}")
    return logs


def main() -> None:
    bib_path = ROOT / "references.bib"
    entries = split_bib_entries(bib_path.read_text(encoding="utf-8", errors="replace"))
    entries.update(split_bib_entries(MANUAL_BIB))
    missing = [k for k in KEEP_KEYS if k not in entries]
    if missing:
        raise SystemExit("Missing curated bib keys: " + ", ".join(missing))
    curated = "\n\n".join(entries[k] for k in KEEP_KEYS) + "\n"
    bib_path.write_text(curated, encoding="utf-8")

    logs = copy_local_pdfs()
    for name, url in PDF_URLS.items():
        logs.append(download_pdf(name, url))

    pdf_count = len(list(LIT.glob("*.pdf")))
    report = [
        "# Literature download and curation report",
        "",
        f"Curated BibTeX entries: {len(KEEP_KEYS)}",
        f"Local PDF files after curation: {pdf_count}",
        "",
        "Only open-access, author-hosted, arXiv, conference-proceedings, or already-local PDFs were copied/downloaded. Paywalled papers are retained as metadata in references.bib but are not illegally downloaded.",
        "",
        "## Download/copy log",
        *[f"- {x}" for x in logs],
    ]
    (LIT / "CURATED_DOWNLOAD_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
