from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
src=ROOT/'tables'/'loaded_path_atlas_preview.csv'
df=pd.read_csv(src) if src.exists() else pd.DataFrame()
required=['hop_distance','min_free_radius','bottleneck_radius','crowding','CN_i','CN_j','degree','edge_betweenness']
pd.DataFrame([{'descriptor':c,'available':c in df.columns} for c in required]).to_csv(ROOT/'tables'/'descriptor_availability_check.csv',index=False)
