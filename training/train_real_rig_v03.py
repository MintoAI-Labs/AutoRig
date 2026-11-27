"""MintoAI AutoRig V0.3: visual-conditioned sparse rigging experiment.
SPDX-License-Identifier: AGPL-3.0-only
Requires train_real_rig_v01.py and train_real_rig_v02.py, numpy, torch, Pillow.
All texture crops and checkpoints stay local. No anatomical labels are assumed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image, ImageDraw
from train_real_rig_v01 import ROOT, PARAMS, load_record, mean_by, target
from train_real_rig_v02 import prepare, balanced_mean, report

SIZE = 48
CHECK_STOP = lambda: None

def local_path(base, rel, root):
    p = (base / rel).resolve()
    if not p.is_relative_to(root.resolve()) or not p.is_file():
        raise ValueError(f'Missing or outside models folder: {p}')
    return p

def resolve_textures(group, model_root):
    meta = json.loads((group/'source.json').read_text(encoding='utf-8-sig'))
    moc = local_path(model_root, meta['model_path'], model_root)
    digest = hashlib.sha256(moc.read_bytes()).hexdigest()
    if digest != group.name or digest != meta['sha256']:
        raise ValueError(f'MOC3 hash mismatch: {moc.name}')
    candidates=[]
    for p in moc.parent.glob('*.model3.json'):
        refs=json.loads(p.read_text(encoding='utf-8-sig'))['FileReferences']
        if (p.parent/refs['Moc']).resolve()==moc: candidates.append((p,refs))
    if len(candidates)!=1: raise ValueError(f'Need exactly one matching model3.json for {moc}')
    p,refs=candidates[0]
    return [local_path(p.parent,t,model_root) for t in refs['Textures']]

def crop_mesh(texture, uv, triangles):
    # Cubism Web shader uses (u, 1-v), with ordinary non-flipped PNG upload.
    w,h=texture.size
    xy=np.column_stack([uv[:,0]*(w-1),(1-uv[:,1])*(h-1)])
    if not np.isfinite(xy).all(): raise ValueError('Invalid UV coordinates')
    x0,y0=np.floor(xy.min(0)).astype(int); x1,y1=np.ceil(xy.max(0)).astype(int)+1
    x0=max(0,x0); y0=max(0,y0); x1=min(w,x1); y1=min(h,y1)
    canvas=Image.new('RGBA',(SIZE,SIZE))
    if x1<=x0 or y1<=y0 or not len(triangles): return np.asarray(canvas).copy()
    crop=texture.crop((x0,y0,x1,y1)); mask=Image.new('L',crop.size); draw=ImageDraw.Draw(mask)
    pts=xy-[x0,y0]
    for t in triangles: draw.polygon([tuple(pts[i]) for i in t],fill=255)
    a=np.asarray(crop).copy(); a[:,:,3]=(a[:,:,3].astype(np.uint16)*np.asarray(mask).astype(np.uint16)//255).astype(np.uint8)
    a[a[:,:,3]==0,:3]=0
    crop=Image.fromarray(a); crop.thumbnail((SIZE-4,SIZE-4),Image.Resampling.LANCZOS)
    canvas.paste(crop,((SIZE-crop.width)//2,(SIZE-crop.height)//2))
    return np.asarray(canvas).copy()

def mesh_patches(path, model_root):
    textures=[]
    for p in resolve_textures(path.parent,model_root):
        with Image.open(p) as im: textures.append(im.convert('RGBA'))
    with np.load(path,allow_pickle=False) as z:
        uv=z['uv'].reshape(-1,2); tri=z['triangles']; off=z['mesh_offsets']; tx=z['texture_indices']; ids=z['mesh_ids']
    if len(tx)!=len(off)-1 or len(ids)!=len(tx) or np.any(tx<0) or np.any(tx>=len(textures)):
        raise ValueError(f'Invalid texture mapping: {path.name}')
    patches=[]
    for i,(a,b) in enumerate(zip(off[:-1],off[1:])):
        CHECK_STOP()
        tr=tri[(tri[:,0]>=a)&(tri[:,0]<b)]-a
        if np.any(tr<0) or np.any(tr>=b-a): raise ValueError('Invalid local triangle')
        patches.append(crop_mesh(textures[int(tx[i])],uv[a:b],tr))
    return np.stack(patches),ids

def sheet(patches, ids, out):
    cols=10; cell=88; rows=(len(patches)+cols-1)//cols
    im=Image.new('RGB',(cols*cell,rows*80),(34,44,53)); d=ImageDraw.Draw(im)
    for i,arr in enumerate(patches):
        x=i%cols*cell; y=i//cols*80
        tile=Image.fromarray(arr).resize((60,60),Image.Resampling.NEAREST)
        im.paste(tile,(x+14,y),tile); d.text((x+4,y+61),str(i),fill='white')
    im.save(out)
    out.with_suffix('.json').write_text(json.dumps([str(i) for i in ids],ensure_ascii=False),encoding='utf-8')

def load_visual(data, model_root, device, audit_dir):
    groups={}; records=[]
    for p in sorted(data.rglob('*.npz')):
        CHECK_STOP()
        if p.parent.parent.resolve()!=data.resolve(): raise ValueError(f'Unexpected dataset layout: {p}')
        with np.load(p,allow_pickle=False) as z:
            # Compare rest-input metadata across parameter sweeps, not target displacements.
            signature=hashlib.sha256()
            for k in ('rest','uv','triangles','mesh_offsets','mesh_ids','texture_indices'):
                a=np.ascontiguousarray(z[k]); signature.update(a.tobytes())
            sig=signature.hexdigest(); source=str(z['source'].item())
        key=p.parent.name
        if key not in groups:
            patches,ids=mesh_patches(p,model_root)
            count=int(np.any(patches[:,:,:,3]>0,axis=(1,2)).sum())
            if not count: raise ValueError(f'All texture patches empty: {source}')
            print(f'{source}: {count}/{len(patches)} nonempty texture patches',flush=True)
            sheet(patches,ids,audit_dir/f'{Path(source).stem}_{key[:8]}.png')
            x=torch.from_numpy(patches).to(device=device,dtype=torch.float32).permute(0,3,1,2)/255
            x=torch.cat([x[:,:3]*x[:,3:4],x[:,3:4]],1).contiguous()
            groups[key]=(sig,x)
        if groups[key][0]!=sig: raise ValueError(f'Rest/UV metadata differs across poses: {p}')
        r=load_record(p,data,device); r['visual']=groups[key][1]; records.append(r)
    if not records: raise ValueError('No NPZ files found')
    if len({(r['group'],r['param']) for r in records})!=len(records): raise ValueError('Duplicate group/parameter records')
    prepare(records)
    return records

class VisualRig(nn.Module):
    def __init__(self,width=64):
        super().__init__()
        self.visual=nn.Sequential(nn.Conv2d(4,16,5,stride=2,padding=2),nn.SiLU(),nn.Conv2d(16,32,3,stride=2,padding=1),nn.SiLU(),nn.Conv2d(32,32,3,stride=2,padding=1),nn.SiLU(),nn.AdaptiveAvgPool2d((2,2)),nn.Flatten(),nn.Linear(128,32),nn.SiLU())
        self.embed=nn.Sequential(nn.Linear(47,width),nn.SiLU(),nn.Linear(width,width))
        self.blocks=nn.ModuleList([nn.Sequential(nn.Linear(width*4,width),nn.SiLU(),nn.Linear(width,width),nn.LayerNorm(width)) for _ in range(3)])
        self.gate=nn.Sequential(nn.Linear(width*2,width),nn.SiLU(),nn.Linear(width,1))
        self.head=nn.Sequential(nn.Linear(width+3,width),nn.SiLU(),nn.Linear(width,2))
    def encode(self,r):
        n=len(r['feat']); src,dst=r['edge']; mesh=r['mesh']; nm=r['nm']
        vis=self.visual(r['visual'])
        h=self.embed(torch.cat([r['feat'],r['onehot'].expand(n,-1),vis[mesh]],1))
        for block in self.blocks:
            mh=mean_by(h,mesh,nm); glob=mh.mean(0,keepdim=True).expand(n,-1)
            h=h+block(torch.cat([h,mean_by(h[src],dst,n),mh[mesh],glob],1))
        mh=mean_by(h,mesh,nm)
        logits=self.gate(torch.cat([mh,mh.mean(0,keepdim=True).expand(nm,-1)],1)).squeeze(-1)
        return h,logits
    def deform(self,h,t):
        return self.head(torch.cat([h,torch.stack([t,t.abs(),t.square()]).expand(len(h),-1)],1))*t
    def forward(self,r,t,threshold=None):
        h,logits=self.encode(r); gate=logits.sigmoid()
        raw=self.deform(h,t)
        weight=gate if threshold is None else (gate>=threshold).float()*gate
        return raw*weight[r['mesh'],None],logits,raw

def masked_mean(x,m):
    return (x*m).sum()/m.sum().clamp_min(1)

@torch.no_grad()
def evaluate(net,records,scales,device,thresholds,poses=11):
    net.eval(); rows={t:[] for t in thresholds}; by={t:{} for t in thresholds}; leak={t:[] for t in thresholds}
    for r in records:
        h,logits=net.encode(r); prob=logits.sigmoid(); m=r['moving']; active=r['mesh_active'].bool(); static=~active
        for th in thresholds:
            chosen=prob>=th; leak[th].append([int((chosen&static).sum()),int(static.sum()),int((chosen&active).sum()),int(active.sum())])
        for i in np.unique(np.linspace(0,len(r['t'])-1,poses).round().astype(int)):
            y=target(r,i,device); raw=net.deform(h,r['t'][i])*scales[r['param']]; z=y.norm(dim=-1)
            for th in thresholds:
                pred=raw*(prob*(prob>=th))[r['mesh'],None]; e=(pred-y).norm(dim=-1)
                row=[e.mean().item(),z.mean().item(),e[m].mean().item() if m.any() else np.nan,z[m].mean().item() if m.any() else np.nan]
                rows[th].append(row); by[th].setdefault(r['param'],[]).append(row)
    return {t:(np.nanmean(rows[t],axis=0),{p:np.nanmean(v,axis=0) for p,v in by[t].items()},np.sum(leak[t],axis=0)) for t in thresholds}

def select_score(v,by):
    # Macro average across parameters, so high-displacement head angles cannot hide mouth failures.
    return float(np.mean([np.mean([a[0]/max(a[1],1e-9),a[2]/max(a[3],1e-9)]) for a in by.values() if np.isfinite(a).all()]))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'dataset/processed_v02'); p.add_argument('--models',type=Path,default=ROOT/'models')
    p.add_argument('--audit-dir',type=Path,default=ROOT/'dataset/visual_audit_v03'); p.add_argument('--audit-only',action='store_true')
    p.add_argument('--steps',type=int,default=4000); p.add_argument('--eval-every',type=int,default=500); p.add_argument('--width',type=int,default=64)
    p.add_argument('--validation',default='Hiyori'); p.add_argument('--holdout',default='Ren'); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda'); p.add_argument('--output',type=Path,default=ROOT/'checkpoints/real_rig_v03.pt')
    p.add_argument('--confirm-authorized',action='store_true'); a=p.parse_args()
    if not a.confirm_authorized: p.error('Confirm permission covers texture processing, derived data and AI training.')
    if min(a.steps,a.eval_every)<1 or a.width<8: p.error('Invalid steps or width')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device=torch.device('cpu' if a.audit_only else a.device)
    if device.type=='cuda':
        if not torch.cuda.is_available(): raise SystemExit('CUDA unavailable')
        torch.cuda.manual_seed_all(a.seed); torch.cuda.reset_peak_memory_stats()
    else: torch.set_num_threads(min(4,torch.get_num_threads()))
    a.audit_dir.mkdir(parents=True,exist_ok=True)
    rec=load_visual(a.data,a.models,device,a.audit_dir)
    print('Private contact sheets:',a.audit_dir,flush=True)
    if a.audit_only: return
    def groups(name): return {r['group'] for r in rec if Path(r['source']).stem.casefold()==name.casefold()}
    vg,hg=groups(a.validation),groups(a.holdout)
    if not vg or not hg or vg&hg: raise ValueError('Missing or overlapping splits')
    train=[r for r in rec if r['group'] not in vg|hg]; val=[r for r in rec if r['group'] in vg]; held=[r for r in rec if r['group'] in hg]
    if not train or {r['param'] for r in val+held}-{r['param'] for r in train}: raise ValueError('Missing training data or parameter')
    for label,rs in [('Train',train),('Validation',val),('Holdout',held)]: print(label,len(rs),sorted({r['source'] for r in rs}),flush=True)
    scales={}
    for param in PARAMS:
        mag=[np.linalg.norm(r['delta'][:,r['moving'].cpu().numpy(),:],axis=-1).max() for r in train if r['param']==param and r['moving'].any()]
        scales[param]=max(float(np.median(mag)) if mag else 0.,1e-4)
    net=VisualRig(a.width).to(device); opt=torch.optim.AdamW(net.parameters(),lr=5e-4); best=float('inf'); state=None; running=[]; start=time.perf_counter()
    thresholds=[0.,.35,.5,.65,.8,.9]
    for step in range(1,a.steps+1):
        net.train(); r=random.choice(train); i=random.randrange(len(r['t'])); y=target(r,i,device)/scales[r['param']]
        pred,logits,raw=net(r,r['t'][i]); m=r['moving'].float(); active=r['mesh_active']; static=1-active[r['mesh']]
        geom=balanced_mean((pred-y).square().sum(-1),m)
        direct=masked_mean((raw-y).square().sum(-1),m)
        still=masked_mean(pred.square().sum(-1),static)
        gate=balanced_mean(F.binary_cross_entropy_with_logits(logits,active,reduction='none'),active)
        loss=geom+.2*direct+.5*still+.1*gate
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(),1.); opt.step(); running.append(loss.item())
        if step%100==0 or step==a.steps: print(f'Step {step}/{a.steps} loss {np.mean(running):.8f}',flush=True); running=[]
        if step%a.eval_every==0 or step==a.steps:
            results=evaluate(net,val,scales,device,thresholds); th=min(thresholds,key=lambda t:select_score(*results[t][:2])); v,by,l=results[th]; s=select_score(v,by)
            print(f'Validation {step} threshold {th:.2f} macro ratio {s:.4f} | all {v[0]:.8f}/{v[1]:.8f} | moving {v[2]:.8f}/{v[3]:.8f} | static gates {l[0]}/{l[1]} active gates {l[2]}/{l[3]}',flush=True)
            if np.isfinite(s) and s<best: best=s; best_th=th; best_step=step; state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    if state is None: raise RuntimeError('No valid checkpoint')
    net.load_state_dict(state)
    print(f'Best step {best_step} | threshold {best_th}',flush=True)
    final={}
    for label,rs in [('Validation',val),('Holdout',held)]:
        v,by,l=evaluate(net,rs,scales,device,[best_th])[best_th]; report(label,v,by); print(f' static gates {l[0]}/{l[1]} | active gates {l[2]}/{l[3]}',flush=True); final[label]=v.tolist()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(architecture='visual_sparse_v03',model_state=state,width=a.width,scales=scales,threshold=best_th,best_step=best_step,parameters=PARAMS,metrics=final,seed=a.seed,train_sources=sorted({r['source'] for r in train}),validation_sources=sorted({r['source'] for r in val}),held_sources=sorted({r['source'] for r in held})),a.output)
    print('Saved:',a.output,flush=True); print(f'Elapsed {time.perf_counter()-start:.2f}s',flush=True)
    if device.type=='cuda': print(f'Peak allocated GPU: {torch.cuda.max_memory_allocated()/2**20:.1f} MiB',flush=True)

if __name__=='__main__': main()