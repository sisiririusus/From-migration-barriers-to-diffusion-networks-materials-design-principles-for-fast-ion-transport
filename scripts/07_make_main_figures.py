from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT.parent
srcdir=BASE/'paper_figures_pub_v2'/'fig_main'; figdir=ROOT/'figures'; figdir.mkdir(exist_ok=True)
for p in srcdir.glob('Figure*.*'): shutil.copy2(p, figdir/p.name)
print('copied main figures from',srcdir)
