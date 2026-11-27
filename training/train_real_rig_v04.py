"""MintoAI AutoRig V0.4: parameter-specific visual mesh rigging.
SPDX-License-Identifier: AGPL-3.0-only
Requires V0.1-V0.3 scripts. Authorized local data only. Geometry experiment.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import os
import random
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from train_real_rig_v01 import ROOT, PARAMS, mean_by, target
from train_real_rig_v02 import balanced_mean, report
from train_real_rig_v03 import VisualRig, load_visual, masked_mean

THRESHOLDS=(0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95)

class ParameterRig(VisualRig):
    def __init__(self,width=64):
        super().__init__(width)
        del self.gate, self.head
        self.gates=nn.ModuleDict({p:nn.Sequential(nn.Linear(width*2,width),nn.SiLU(),nn.Linear(width,1)) for p in PARAMS})
        self.heads=nn.ModuleDict({p:nn.Sequential(nn.Linear(width+3,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,2)) for p in PARAMS})
    def encode(self,r):
        n=len(r['feat']); src,dst=r['edge']; mesh=r['mesh']; nm=r['nm']
        vis=self.visual(r['visual'])
        h=self.embed(torch.cat([r['feat'],r['onehot'].expand(n,-1),vis[mesh]],1))
        for block in self.blocks:
            mh=mean_by(h,mesh,nm); glob=mh.mean(0,keepdim=True).expand(n,-1)
            h=h+block(torch.cat([h,mean_by(h[src],dst,n),mh[mesh],glob],1))
        mh=mean_by(h,mesh,nm)
        return h,self.gates[r['param']](torch.cat([mh,mh.mean(0,keepdim=True).expand(nm,-1)],1)).squeeze(-1)
    def deform(self,h,t,param):
        return self.heads[param](torch.cat([h,torch.stack([t,t.abs(),t.square()]).expand(len(h),-1)],1))*t
    def forward(self,r,t,threshold):
        h,logits=self.encode(r); raw=self.deform(h,t,r['param'])
        return raw*(logits.sigmoid()>=threshold)[r['mesh'],None],logits,raw

def train_scales(train):
    # Training-only characteristic RMS displacement on moving vertices.
    out={}
    for p in PARAMS:
        vals=[]
        for r in train:
            if r['param']==p and r['moving'].any():
                d=r['delta'][:,r['moving'].cpu().numpy(),:]
                vals.append(float(np.sqrt(np.mean(np.sum(d*d,axis=-1)))))
        out[p]=max(float(np.median(vals)) if vals else 0.,1e-5)
    return out

def loss_for(net,r,i,scales,device):
    y=target(r,i,device).to(dtype=next(net.parameters()).dtype)/scales[r['param']]
    h,logits=net.encode(r); raw=net.deform(h,r['t'][i],r['param'])
    active=r['mesh_active']; m=r['moving'].float(); static=1-m
    # Train the decoder directly, not through the possibly closed gate.
    err=(raw-y).square().sum(-1)
    geom=masked_mean(err,m)
    still=masked_mean(err,static)
    # Relative edge deformation preserves local coherence rather than forcing rigid meshes.
    src,dst=r['edge']; edgeerr=((raw[src]-raw[dst])-(y[src]-y[dst])).square().sum(-1)
    edge_mask=torch.maximum(m[src],m[dst]); edge=masked_mean(edgeerr,edge_mask)
    gate=balanced_mean(F.binary_cross_entropy_with_logits(logits,active,reduction='none'),active)
    # Gate gradients are independent of the decoder loss. Shared encoder still learns both.
    return geom+0.15*still+0.1*edge+0.5*gate, (geom.detach(),gate.detach())

@torch.no_grad()
def evaluate(net,records,scales,device,threshold_map=None,poses=11):
    net.eval(); metrics={}; counts={}
    for r in records:
        p=r['param']; ts=THRESHOLDS if threshold_map is None else (threshold_map[p],)
        h,logits=net.encode(r); prob=logits.sigmoid(); active=r['mesh_active'].bool(); moving=r['moving']; mesh=r['mesh']
        ix=np.unique(np.linspace(0,len(r['t'])-1,poses).round().astype(int))
        masks={t:prob>=t for t in ts}
        for t,mask in masks.items():
            key=(p,t); counts.setdefault(key,np.zeros(4,dtype=np.int64))
            counts[key]+=np.array([int((mask&active).sum()),int((mask&~active).sum()),int((~mask&active).sum()),int((~mask&~active).sum())])
        for i in ix:
            y=target(r,i,device); z=y.norm(dim=-1); raw=net.deform(h,r['t'][i],p)*scales[p]
            for t,mask in masks.items():
                pred=raw*mask[mesh,None]; e=(pred-y).norm(dim=-1)
                metrics.setdefault((p,t),[]).append([float(e.mean()),float(z.mean()),float(e[moving].mean()) if moving.any() else np.nan,float(z[moving].mean()) if moving.any() else np.nan])
    return {k:(np.nanmean(v,axis=0),counts[k]) for k,v in metrics.items()}

def quality(row):
    a,c=row; tp,fp,fn,tn=map(int,c)
    f1=2*tp/max(2*tp+fp+fn,1) if tp+fn else float(fp==0)
    # Penalize both false positives and false negatives. No credit for merely doing nothing.
    terms=[a[0]/max(a[1],1e-8)]
    if np.isfinite(a[2:]).all(): terms.append(a[2]/max(a[3],1e-8))
    return float(np.mean(terms)+0.5*(1-f1)),f1

def choose(results):
    thresholds={}; scores=[]
    for p in PARAMS:
        opts=[(t,row) for (q,t),row in results.items() if q==p]
        if not opts: raise ValueError(f'Missing validation parameter: {p}')
        t,row=min(opts,key=lambda x:(quality(x[1])[0],-quality(x[1])[1]))
        thresholds[p]=t; scores.append(quality(row)[0])
    return thresholds,float(np.mean(scores))

def summarize(label,results,thresholds):
    by={p:results[p,t][0] for p,t in thresholds.items() if (p,t) in results}
    report(label,np.nanmean(list(by.values()),axis=0),by)
    rows={}
    for p,t in thresholds.items():
        if (p,t) not in results: continue
        a,c=results[p,t]; tp,fp,fn,tn=map(int,c); score,f1=quality((a,c))
        print(f' {p:17s} threshold {t:.2f} | TP {tp} FP {fp} FN {fn} TN {tn} | F1 {f1:.3f} | recall {tp/max(tp+fn,1):.3f}',flush=True)
        rows[p]=dict(errors=a.tolist(),counts=[tp,fp,fn,tn],f1=f1,score=score)
    return rows

def run(a,rec,device):
    def groups(name): return {r['group'] for r in rec if Path(r['source']).stem.casefold()==name.casefold()}
    vg,hg=groups(a.validation),groups(a.holdout)
    if not vg or not hg or vg&hg: raise ValueError('Missing/overlapping splits')
    train=[r for r in rec if r['group'] not in vg|hg]; val=[r for r in rec if r['group'] in vg]; held=[r for r in rec if r['group'] in hg]
    pool={p:[r for r in train if r['param']==p] for p in PARAMS}
    if not all(pool.values()) or {r['param'] for r in val}!=set(PARAMS): raise ValueError('Need all seven parameters in training and validation')
    for name,rs in [('Train',train),('Validation',val),('Holdout',held)]: print(name,len(rs),sorted({r['source'] for r in rs}),flush=True)
    scales=train_scales(train); print('Train-only RMS scales:',scales,flush=True)
    net=ParameterRig(a.width).to(device); opt=torch.optim.AdamW(net.parameters(),lr=a.lr)
    rng=random.Random(a.seed); order=[]; state=None; best=float('inf'); running=[]; start=time.perf_counter()
    for step in range(1,a.steps+1):
        net.train()
        if not order: order=list(PARAMS); rng.shuffle(order)
        p=order.pop(); r=rng.choice(pool[p]); i=rng.randrange(len(r['t']))
        loss,parts=loss_for(net,r,i,scales,device)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.); opt.step()
        running.append([float(loss.detach()),float(parts[0]),float(parts[1])])
        if step%100==0 or step==a.steps:
            l,g,b=np.mean(running,axis=0); print(f'Step {step}/{a.steps} loss {l:.8f} | geometry {g:.8f} | gate {b:.8f}',flush=True); running=[]
        if step%a.eval_every==0 or step==a.steps:
            res=evaluate(net,val,scales,device); th,s=choose(res)
            print(f'Validation {step} score {s:.5f} | thresholds {th}',flush=True)
            if np.isfinite(s) and s<best:
                best=s; best_step=step; best_th=th; state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    if state is None: raise RuntimeError('No valid checkpoint')
    net.load_state_dict(state); print(f'Best step {best_step} | score {best:.5f}',flush=True)
    v=summarize('Validation',evaluate(net,val,scales,device,best_th),best_th)
    h=summarize('Holdout (not used for threshold/step selection)',evaluate(net,held,scales,device,best_th),best_th)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    ck=dict(architecture='parameter_visual_v04',model_state=state,width=a.width,parameters=PARAMS,scales=scales,thresholds=best_th,best_step=best_step,validation_score=best,metrics={'validation':v,'holdout':h},seed=a.seed,steps=a.steps,train_sources=sorted({r['source'] for r in train}),validation_sources=sorted({r['source'] for r in val}),held_sources=sorted({r['source'] for r in held}))
    with a.output.open('xb') as f: torch.save(ck,f)
    print('Saved:',a.output,flush=True); print(f'Elapsed {time.perf_counter()-start:.2f}s',flush=True)
    if device.type=='cuda': print(f'Peak allocated GPU: {torch.cuda.max_memory_allocated()/2**20:.1f} MiB',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'dataset/processed_v02'); p.add_argument('--models',type=Path,default=ROOT/'models')
    p.add_argument('--steps',type=int,default=4000); p.add_argument('--eval-every',type=int,default=500); p.add_argument('--width',type=int,default=64); p.add_argument('--lr',type=float,default=5e-4)
    p.add_argument('--validation',default='Hiyori'); p.add_argument('--holdout',default='Ren'); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda'); p.add_argument('--output',type=Path); p.add_argument('--deterministic',action='store_true'); p.add_argument('--confirm-authorized',action='store_true'); a=p.parse_args()
    if not a.confirm_authorized: p.error('Confirm permission covers models, texture processing and AI training.')
    if min(a.steps,a.eval_every)<1 or a.width<8 or not np.isfinite(a.lr) or a.lr<=0: p.error('Invalid configuration')
    if a.output is None: a.output=ROOT/'checkpoints'/('real_rig_v04_seed'+str(a.seed)+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.pt')
    if a.output.exists(): p.error('Output exists; choose a new checkpoint path.')
    if a.deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8'); torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device=torch.device(a.device)
    if device.type=='cuda':
        if not torch.cuda.is_available(): p.error('CUDA unavailable')
        torch.cuda.manual_seed_all(a.seed); torch.cuda.reset_peak_memory_stats()
    else: torch.set_num_threads(min(4,torch.get_num_threads()))
    audit=ROOT/'dataset/visual_audit_v03'; audit.mkdir(parents=True,exist_ok=True)
    rec=load_visual(a.data,a.models,device,audit)
    run(a,rec,device)

if __name__=='__main__': main()