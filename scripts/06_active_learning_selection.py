from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT.parent
for src in [BASE/'npj_no_leakage_diffusion_network_20260604'/'results'/'tables'/'top50_active_neb_candidates_no_leakage.csv', BASE/'paper_figures_pub_v2'/'data_cache'/'Figure6_top5_recommended_candidates.csv']:
    if src.exists(): shutil.copy2(src, ROOT/'tables'/src.name); print('copied',src.name)
