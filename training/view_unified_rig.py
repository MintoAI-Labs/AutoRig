"""Generate a private mesh comparison using only the unified checkpoint at inference."""
import argparse
import json
from pathlib import Path
import tempfile
import shutil

import numpy as np
import torch

from unified_rig import UnifiedRig
from unified_sources import load_records,to_device
from training_progress import StopFlag

ROOT=Path(__file__).resolve().parent.parent
HTML='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Unified rig mesh comparison</title><style>body{font:16px system-ui;background:#101916;color:#def3e8;margin:20px}select,input{margin:12px}main{display:grid;grid-template-columns:1fr 1fr;gap:15px}canvas{width:100%;height:550px;background:#172a20}pre{white-space:pre-wrap}@media(max-width:650px){main{grid-template-columns:1fr}}</style>
<h1>Unified rig</h1><p>Original rig and the unified student's predicted mesh. Shared coordinates and scale.</p>
<select id="parameter"></select><input id="pose" type="range"><span id="value"></span>
<main><section>Original rig<canvas id="truth"></canvas></section><section>AI rig<canvas id="prediction"></canvas></section></main><pre id="stats"></pre>
<script>const data=__DATA__,$=id=>document.getElementById(id);
data.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent=r.param;$('parameter').append(o)});
let lo=[Infinity,Infinity],hi=[-Infinity,-Infinity];
for(const r of data)for(const series of [r.truth,r.prediction])for(const pose of series)for(const p of pose)for(let j=0;j<2;j++){lo[j]=Math.min(lo[j],p[j]);hi[j]=Math.max(hi[j],p[j])}
function draw(){const r=data[+$('parameter').value],i=+$('pose').value;$('value').textContent=r.values[i];
let max=0;for(let n=0;n<r.truth[i].length;n++)for(let j=0;j<2;j++)max=Math.max(max,Math.abs(r.truth[i][n][j]-r.prediction[i][n][j]));
$('stats').textContent='Maximum coordinate error: '+max.toExponential(6);
for(const [id,series,color]of [['truth',r.truth,'#bce9db'],['prediction',r.prediction,'#f6d5e2']]){
const c=$(id),w=c.clientWidth,h=550,d=devicePixelRatio||1;c.width=w*d;c.height=h*d;const g=c.getContext('2d');g.scale(d,d);
const s=Math.min((w-30)/Math.max(hi[0]-lo[0],1e-9),(h-30)/Math.max(hi[1]-lo[1],1e-9));
const xy=p=>[w/2+(p[0]-(hi[0]+lo[0])/2)*s,h/2-(p[1]-(hi[1]+lo[1])/2)*s];
g.strokeStyle=color;g.lineWidth=.5;for(const tri of r.triangles){g.beginPath();tri.forEach((n,k)=>{const p=xy(series[i][n]);k?g.lineTo(...p):g.moveTo(...p)});g.closePath();g.stroke()}}}
function select(){$('pose').min=0;$('pose').max=data[+$('parameter').value].values.length-1;$('pose').value=Math.floor(+$('pose').max/2);draw()}
$('parameter').onchange=select;$('pose').oninput=draw;window.onresize=draw;select();</script>'''


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/unified_rig_v06.best.pt')
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--models',type=Path,default=ROOT/'models')
    p.add_argument('--model',default='zundamon')
    p.add_argument('--output',type=Path,default=ROOT/'dataset/unified_comparison.html')
    args=p.parse_args()
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if ck.get('architecture')!='unified_visual_v06':
        p.error('Expected the unified checkpoint')
    dtype=getattr(torch,ck.get('precision',ck.get('config',{}).get('precision','float64')))
    net=UnifiedRig(ck['width']).to(dtype=dtype)
    net.load_state_dict(ck['model_state'])
    net.rule_mode=ck.get('rule_mode','affine')
    net.eval()
    selected=[]
    triangles_by_param={}
    values_by_param={}
    for path in args.data.glob('*/*.npz'):
        with np.load(path,allow_pickle=False) as z:
            if Path(str(z['source'].item())).stem.casefold()==args.model.casefold():
                param=str(z['parameter_id'].item())
                if param in triangles_by_param:
                    p.error('Duplicate parameter record')
                triangles_by_param[param]=z['triangles'].tolist()
                values_by_param[param]=z['parameter_values'].tolist()
                selected.append(path)
    if not selected or len({path.parent for path in selected})!=1:
        p.error('Choose one model group with extracted NPZs')
    rows=[]
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        dest=root/'data'/selected[0].parent.name
        dest.mkdir(parents=True)
        shutil.copy2(selected[0].parent/'source.json',dest/'source.json')
        for path in selected:
            shutil.copy2(path,dest/path.name)
        records=load_records(root/'data',args.models,root/'audit','inference',StopFlag())
        for record in records:
            r=to_device(record,torch.device('cpu'),dtype)
            r['motion_scale']=ck['scales'][r['param']]
            indices=np.unique(np.linspace(0,len(r['t'])-1,21).round().astype(int)).tolist()
            prediction,_,_=net.pose_batch(r,r['t'][indices],hard_gate=True)
            # The first two input features store the exact rest coordinates used by the network.
            rest=r['feat'][:,:2].numpy()
            pred=prediction.numpy()*ck['scales'][r['param']]
            triangles=triangles_by_param[r['param']]
            rows.append(dict(param=r['param'],triangles=triangles,
                             values=[values_by_param[r['param']][i] for i in indices],truth=(rest+record['delta'][indices]).tolist(),
                             prediction=(rest+pred).tolist()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(rows,separators=(',',':'),allow_nan=False).replace('<',chr(92)+'u003c')
    args.output.write_text(HTML.replace('__DATA__',payload),encoding='utf-8')
    print('Saved private comparison:',args.output,flush=True)


if __name__=='__main__':
    main()
