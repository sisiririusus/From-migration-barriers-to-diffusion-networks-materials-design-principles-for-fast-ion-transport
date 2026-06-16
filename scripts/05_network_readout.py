from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT.parent
for name in ['predicted_barrier_weighted_percolation.csv','predicted_barrier_weighted_network_readout.csv']:
    src=BASE/'npj_no_leakage_diffusion_network_20260604'/'results'/'tables'/name
    print('copy' if src.exists() else 'data required', src)
    if src.exists(): shutil.copy2(src, ROOT/'tables'/name)
