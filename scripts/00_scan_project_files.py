from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT.parent
records=[]
for pat in ['*.csv','*.json','*.pkl','*.npy','*.png','*.pdf','*.py']:
    for p in BASE.rglob(pat):
        if 'manuscript_diffusion_network_v1' not in p.parts:
            records.append({'path':str(p),'suffix':p.suffix,'bytes':p.stat().st_size})
out=ROOT/'tables'/'project_file_scan.json'; out.write_text(json.dumps(records,indent=2),encoding='utf-8')
print(out, len(records))
