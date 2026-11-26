"""MintoAI AutoRig: local mesh comparison and leakage diagnostics.
SPDX-License-Identifier: AGPL-3.0-only
Keep generated HTML local: it contains derived geometry of authorized models.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from train_real_rig_v01 import ROOT, load_record, target
from train_real_rig_v02 import GatedRig

HTML = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>MintoAI V0.2 geometry comparison</title>
<style>body{font:15px system-ui;background:#101820;color:#e0f2ef;margin:24px}h1{font-size:24px}label{display:inline-block;margin:8px}select,input{accent-color:#86d6ba} .panes{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:12px}canvas{width:100%;height:440px;background:#17232c;border:1px solid #38505c}small{color:#9cb7ba}button{padding:6px}pre{white-space:pre-wrap}</style>
<h1>MintoAI V0.2 | Local geometry comparison</h1><p>Private derived geometry. Do not publish without authorization. Mesh-only visualization, no textures, clipping, opacity or draw order.</p>
<label>Parameter <select id="param"></select></label><label>Pose <input type="range" id="pose" min="0" value="0"> <span id="val"></span></label><label>Display <select id="mode"><option value="all">All meshes</option><option value="moving">Moving vertices' meshes</option><option value="static">Static meshes only</option></select></label>
<div class="panes"><section>Rest / zero baseline<canvas id="rest"></canvas></section><section>Actual rig<canvas id="truth"></canvas></section><section>AI prediction<canvas id="pred"></canvas></section></div><pre id="stats"></pre><small>Moving = maximum displacement greater than 1e-5 normalized units over the parameter sweep. The filter uses ground truth for diagnostics only, never as AI input. All three canvases use identical coordinates and scale.</small>
<script>const records=__DATA__;
const $=id=>document.getElementById(id), sel=$('param'), slider=$('pose');
records.forEach((r,i)=>{let o=document.createElement('option');o.value=i;o.textContent=r.param;sel.append(o)});
function setParam(){let r=records[+sel.value];slider.max=r.values.length-1;slider.value=Math.floor((r.values.length-1)/2);draw()}
function draw(){const r=records[+sel.value], i=+slider.value; $('val').textContent=r.values[i].toFixed(3);let mode=$('mode').value;
let all=[r.rest,r.truth[i],r.pred[i]], xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity;
for(let pts of all)for(let p of pts){xmin=Math.min(xmin,p[0]);xmax=Math.max(xmax,p[0]);ymin=Math.min(ymin,p[1]);ymax=Math.max(ymax,p[1])}
for(let [j,id] of ['rest','truth','pred'].entries()){let c=$(id),dpr=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*dpr;c.height=h*dpr;let g=c.getContext('2d');g.scale(dpr,dpr);let s=Math.min((w-24)/Math.max(xmax-xmin,1e-6),(h-24)/Math.max(ymax-ymin,1e-6));let xy=p=>[w/2+(p[0]-(xmin+xmax)/2)*s,h/2-(p[1]-(ymin+ymax)/2)*s];g.lineWidth=.55;let pts=all[j];for(let tri of r.tri){let mesh=r.mesh[tri[0]],active=r.active[mesh];if(mode==='moving'&&!active||mode==='static'&&active)continue;g.strokeStyle=active?'#83dfbe':'#536b7b';g.beginPath();for(let k=0;k<3;k++){let [x,y]=xy(pts[tri[k]]);if(k===0)g.moveTo(x,y);else g.lineTo(x,y)}g.closePath();g.stroke()}}
let q=r.stats[i];$('stats').textContent='Model: '+r.source+' | '+r.param+' | pose '+(i+1)+'/'+r.values.length+'\\nMean error (normalized units): all '+q[0].toFixed(7)+' | moving '+q[1].toFixed(7)+' | static '+q[2].toFixed(7)+'\\nMean zero-baseline error: all '+q[3].toFixed(7)+' | moving '+q[4].toFixed(7)+' | static '+q[5].toFixed(7)+'\\nGate active-mesh mean: '+r.gateActive.toFixed(3)+' | gate static-mesh mean: '+r.gateStatic.toFixed(3)}
sel.onchange=setParam;slider.oninput=draw;$('mode').onchange=draw;window.onresize=draw;setParam();</script></html>'''

def avg(x, mask=None):
    if mask is not None: x=x[mask]
    return float(x.mean()) if x.size else 0.0

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'dataset/processed_v02')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/real_rig_v02.pt')
    p.add_argument('--model',default='Ren');p.add_argument('--poses',type=int,default=21)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--output',type=Path);p.add_argument('--confirm-authorized',action='store_true')
    a=p.parse_args()
    if not a.confirm_authorized: p.error('Confirm authorization for local derived geometry.')
    if a.poses<2: p.error('--poses must be >=2')
    device=torch.device(a.device)
    if device.type=='cpu': torch.set_num_threads(min(4,torch.get_num_threads()))
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if ck['architecture']!='gated_geometry_v02': raise ValueError('Expected V0.2 checkpoint')
    model=GatedRig(ck['width']).to(device);model.load_state_dict(ck['model_state']);model.eval()
    rows=[]
    for path in sorted(a.data.rglob('*.npz')):
        with np.load(path,allow_pickle=False) as z:
            if Path(str(z['source'].item())).stem.casefold()!=a.model.casefold(): continue
            rest=z['rest'].astype(np.float32);tri=z['triangles'].astype(int);values=z['parameter_values']
        r=load_record(path,a.data,device); mesh=r['mesh'].cpu().numpy(); moving=r['moving'].cpu().numpy();active=np.bincount(mesh,weights=moving,minlength=r['nm'])>0
        indices=np.unique(np.linspace(0,len(values)-1,a.poses).round().astype(int))
        default_idx=int(r['t'].abs().argmin().item());indices=np.unique(np.append(indices,default_idx))
        truths=[];preds=[];stats=[]
        for i in indices:
            y=target(r,i,device).cpu().numpy();out,logits=model(r,r['t'][i]);pred=(out*ck['scales'][r['param']]).cpu().numpy();gate=logits.sigmoid().cpu().numpy()
            err=np.linalg.norm(pred-y,axis=1); zero=np.linalg.norm(y,axis=1)
            stats.append([avg(err),avg(err,moving),avg(err,~moving),avg(zero),avg(zero,moving),avg(zero,~moving)])
            truths.append((rest+y).round(7).tolist());preds.append((rest+pred).round(7).tolist())
        ga=avg(gate,active);gs=avg(gate,~active);mean=np.mean(stats,axis=0)
        print(f"{r['param']:17s} static error {mean[2]:.8f} | moving {mean[1]:.8f}/{mean[4]:.8f} | gates active/static {ga:.3f}/{gs:.3f}")
        rows.append(dict(source=r['source'],param=r['param'],rest=rest.round(7).tolist(),tri=tri.tolist(),mesh=mesh.tolist(),active=active.tolist(),values=values[indices].tolist(),truth=truths,pred=preds,stats=stats,gateActive=ga,gateStatic=gs))
    if not rows: raise SystemExit('No matching records')
    if len({r['param'] for r in rows})!=len(rows): raise SystemExit('Duplicate parameter records; resolve dataset duplicates first')
    out=a.output or ROOT/'dataset'/f'real_rig_v02_{a.model}_viewer.html'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(HTML.replace('__DATA__',json.dumps(rows,separators=(',',':'),allow_nan=False).replace('<','\\u003c')),encoding='utf-8')
    print('Saved private viewer:',out)

if __name__=='__main__': main()
