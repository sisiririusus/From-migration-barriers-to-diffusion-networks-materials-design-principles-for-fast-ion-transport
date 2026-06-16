from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT.parent
cands=[BASE/'npj_no_leakage_diffusion_network_20260604'/'results'/'tables'/'migration_path_atlas_edges.csv', BASE/'npj_no_leakage_diffusion_network_20260604'/'results'/'tables'/'graph_edges_with_no_leakage_predictions.csv']
for p in cands:
    if p.exists():
        df=pd.read_csv(p,low_memory=False); out=ROOT/'tables'/'loaded_path_atlas_preview.csv'; df.head(2000).to_csv(out,index=False); print('loaded',p,df.shape); break
else:
    pd.DataFrame([{'status':'data/figure required'}]).to_csv(ROOT/'tables'/'loaded_path_atlas_preview.csv',index=False)
