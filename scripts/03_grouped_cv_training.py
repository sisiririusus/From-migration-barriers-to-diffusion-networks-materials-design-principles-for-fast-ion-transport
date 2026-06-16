from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT.parent
src=BASE/'paper_figures_pub_v2'/'data_cache'/'model_family_grouped_cv_metrics.csv'
if src.exists(): shutil.copy2(src, ROOT/'tables'/src.name); print('copied metrics')
else: print('data required: grouped CV metrics')
