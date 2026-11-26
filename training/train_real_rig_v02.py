"""MintoAI AutoRig V0.2: gated geometry-only GNN experiment.
SPDX-License-Identifier: AGPL-3.0-only
Requires train_real_rig_v01.py alongside this file. Local authorized data only.
"""
import argparse
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from train_real_rig_v01 import PARAMS, ROOT, load_record, mean_by, target


class GatedRig(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(15, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(width*4, width), nn.SiLU(), nn.Linear(width,width), nn.LayerNorm(width)) for _ in range(3)])
        self.gate = nn.Sequential(nn.Linear(width*2,width), nn.SiLU(), nn.Linear(width,1))
        self.head = nn.Sequential(nn.Linear(width+3,width), nn.SiLU(), nn.Linear(width,2))

    def forward(self, r, t):
        n = len(r['feat']); src,dst = r['edge']; mesh = r['mesh']; nm = r['nm']
        h = self.embed(torch.cat([r['feat'],r['onehot'].expand(n,-1)],1))
        for block in self.blocks:
            neigh = mean_by(h[src],dst,n)
            mh = mean_by(h,mesh,nm)
            glob = mh.mean(0,keepdim=True).expand(n,-1)
            h = h + block(torch.cat([h,neigh,mh[mesh],glob],1))
        mh = mean_by(h,mesh,nm)
        logits = self.gate(torch.cat([mh,mh.mean(0,keepdim=True).expand(nm,-1)],1)).squeeze(-1)
        tfeat = torch.stack([t,t.abs(),t.square()]).expand(n,-1)
        raw = self.head(torch.cat([h,tfeat],1)) * t
        return raw * logits.sigmoid()[mesh,None], logits


def balanced_mean(x, mask):
    # Average positive and negative groups separately, without empty-group NaNs.
    return ((x*mask).sum()/mask.sum().clamp_min(1) + (x*(1-mask)).sum()/(1-mask).sum().clamp_min(1)) / ((mask.sum()>0).float()+((1-mask).sum()>0).float()).clamp_min(1)


def prepare(records):
    for r in records:
        m = torch.zeros(r['nm'],device=r['mesh'].device)
        m.index_add_(0,r['mesh'],r['moving'].float())
        r['mesh_active'] = (m>0).float()


@torch.no_grad()
def evaluate(model, records, scales, device, poses=11):
    model.eval(); rows=[]; by={p:[] for p in PARAMS}
    for r in records:
        for i in np.unique(np.linspace(0,len(r['t'])-1,poses).round().astype(int)):
            y=target(r,i,device); out,_=model(r,r['t'][i]); pred=out*scales[r['param']]
            e=(pred-y).norm(dim=-1); z=y.norm(dim=-1); m=r['moving']
            row=[e.mean().item(),z.mean().item(),e[m].mean().item() if m.any() else np.nan,z[m].mean().item() if m.any() else np.nan]
            rows.append(row); by[r['param']].append(row)
    return np.nanmean(rows,axis=0), {p:np.nanmean(v,axis=0) for p,v in by.items() if v}


def score(v):
    return float(np.mean([v[0]/max(v[1],1e-10),v[2]/max(v[3],1e-10)]))


def report(label,v,by):
    print(label,flush=True)
    print(f' All vertices: model={v[0]:.8f} | zero={v[1]:.8f}',flush=True)
    print(f' Moving vertices: model={v[2]:.8f} | zero={v[3]:.8f}',flush=True)
    for p,a in by.items():
        print(f' {p:17s} all {a[0]:.8f} / {a[1]:.8f} | moving {a[2]:.8f} / {a[3]:.8f}',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,default=ROOT/'dataset/processed_v02')
    ap.add_argument('--validation',default='Hiyori'); ap.add_argument('--holdout',default='Ren')
    ap.add_argument('--steps',type=int,default=4000); ap.add_argument('--eval-every',type=int,default=500)
    ap.add_argument('--width',type=int,default=64); ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    ap.add_argument('--output',type=Path,default=ROOT/'checkpoints/real_rig_v02.pt')
    ap.add_argument('--confirm-authorized',action='store_true')
    a=ap.parse_args()
    if not a.confirm_authorized: ap.error('Confirm actual dataset and SDK authorization first.')
    if min(a.steps,a.eval_every)<1 or a.width<8 or a.validation.casefold()==a.holdout.casefold(): ap.error('Invalid configuration')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device)
    if device.type=='cuda':
        if not torch.cuda.is_available(): raise SystemExit('CUDA unavailable')
        torch.cuda.manual_seed_all(a.seed); torch.cuda.reset_peak_memory_stats()
    else: torch.set_num_threads(min(4,torch.get_num_threads()))
    rec=[load_record(p,a.data,device) for p in sorted(a.data.rglob('*.npz'))]
    def group(name): return {r['group'] for r in rec if Path(r['source']).stem.casefold()==name.casefold()}
    vg,hg=group(a.validation),group(a.holdout)
    if not vg or not hg or vg&hg: raise SystemExit('Missing or overlapping validation/holdout models')
    val=[r for r in rec if r['group'] in vg]; held=[r for r in rec if r['group'] in hg]
    train=[r for r in rec if r['group'] not in vg|hg]
    if not train: raise SystemExit('No training data')
    for name,rs in [('Train',train),('Validation',val),('Holdout',held)]:
        print(name,len(rs),'files',sorted({r['source'] for r in rs}),flush=True)
    if {r['param'] for r in val+held}-{r['param'] for r in train}: raise SystemExit('Missing training parameter')
    prepare(rec)
    # Training-only per-parameter scaling; no target data from held-out models.
    scales={}
    for p in PARAMS:
        mag=[np.linalg.norm(r['delta'][:,r['moving'].cpu().numpy(),:],axis=-1).max() for r in train if r['param']==p and r['moving'].any()]
        scales[p]=max(float(np.median(mag)) if mag else 0.,1e-4)
    model=GatedRig(a.width).to(device); opt=torch.optim.AdamW(model.parameters(),lr=5e-4)
    start=time.perf_counter(); best=float('inf'); best_state=None; best_step=0; running=0.; count=0
    for step in range(1,a.steps+1):
        model.train(); r=random.choice(train); i=random.randrange(len(r['t'])); y=target(r,i,device)/scales[r['param']]
        pred,logits=model(r,r['t'][i]); m=r['moving'].float(); active=r['mesh_active']
        geom=balanced_mean((pred-y).square().sum(-1),m)
        gate=balanced_mean(F.binary_cross_entropy_with_logits(logits,active,reduction='none'),active)
        loss=geom+0.05*gate
        if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        running+=loss.item(); count+=1
        if step%100==0 or step==a.steps:
            print(f'Step {step}/{a.steps} | combined loss {running/count:.8f}',flush=True); running=0.; count=0
        if step%a.eval_every==0 or step==a.steps:
            v,_=evaluate(model,val,scales,device); s=score(v)
            print(f'Validation step {step} | all {v[0]:.8f}/{v[1]:.8f} | moving {v[2]:.8f}/{v[3]:.8f} | relative score {s:.4f}',flush=True)
            if np.isfinite(s) and s<best:
                best=s; best_step=step; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if best_state is None: raise RuntimeError('No finite validation checkpoint')
    model.load_state_dict(best_state)
    v,vby=evaluate(model,val,scales,device); h,hby=evaluate(model,held,scales,device)
    print('Best validation step:',best_step,flush=True); report('Validation:',v,vby); report('Holdout:',h,hby)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model_state=best_state,architecture='gated_geometry_v02',width=a.width,parameters=PARAMS,scales=scales,seed=a.seed,best_step=best_step,validation_metrics=v.tolist(),held_metrics=h.tolist(),train_sources=sorted({r['source'] for r in train}),validation_sources=sorted({r['source'] for r in val}),held_sources=sorted({r['source'] for r in held})),a.output)
    print('Saved:',a.output,flush=True); print(f'Elapsed {time.perf_counter()-start:.2f}s',flush=True)
    if device.type=='cuda': print(f'Peak allocated GPU: {torch.cuda.max_memory_allocated()/2**20:.1f} MiB',flush=True)

if __name__=='__main__': main()
