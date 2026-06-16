from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
groups=['C only','L only','T only','C + L','C + T','L + T','C + L + T','C + L + shuffled T']
pd.DataFrame({'feature_group':groups,'status':['regenerate with production training script']*len(groups)}).to_csv(ROOT/'tables'/'feature_ablation_plan.csv',index=False)
