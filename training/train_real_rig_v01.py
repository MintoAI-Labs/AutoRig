"""MintoAI AutoRig: experimental geometry-only graph rigging baseline.
SPDX-License-Identifier: AGPL-3.0-only
Run locally with authorized NPZ data produced by Batch Extractor V0.2.
"""
import argparse
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn

PARAMS = ['ParamAngleX','ParamAngleY','ParamAngleZ','ParamEyeLOpen',
          'ParamEyeROpen','ParamMouthForm','ParamMouthOpenY']
ROOT = Path(__file__).resolve().parent.parent


def mean_by(x, idx, n):
    out = x.new_zeros((n, x.shape[1]))
    out.index_add_(0, idx, x)
    count = x.new_zeros((n, 1))
    count.index_add_(0, idx, x.new_ones((len(idx), 1)))
    return out / count.clamp_min(1)


class GraphRig(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        # rest xy, mesh center xy, mesh extent xy, local xy, 7 parameter IDs, signed value
        self.embed = nn.Sequential(nn.Linear(16, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(width*3, width), nn.SiLU(), nn.Linear(width,width),nn.LayerNorm(width))
            for _ in range(3)])
        self.head = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width,2))

    def forward(self, r, t):
        n = len(r['feat']); src, dst = r['edge']
        onehot = r['onehot'].expand(n,-1)
        h = self.embed(torch.cat([r['feat'], onehot, t.expand(n,1)],dim=1))
        for block in self.blocks:
            neigh = mean_by(h[src], dst, n)
            mesh = mean_by(h, r['mesh'], r['nm'])[r['mesh']]
            h = h + block(torch.cat([h,neigh,mesh],dim=1))
        # Exactly zero displacement at the default parameter value.
        return self.head(h)*t


def load_record(path, root, device):
    with np.load(path,allow_pickle=False) as z:
        p = str(z['parameter_id'].item()); source = str(z['source'].item())
        if p not in PARAMS: raise ValueError(f'Unsupported parameter {p}')
        if not str(z['permission_note'].item()).strip(): raise ValueError('Missing permission reference')
        rest = z['rest'].astype(np.float32); delta = z['delta'].astype(np.float32)
        tri = z['triangles'].astype(np.int64); off = z['mesh_offsets'].astype(np.int64)
        values = z['parameter_values'].astype(np.float32); lo,hi,default = z['parameter_range']
    n=len(rest)
    if rest.shape != (n,2) or delta.shape != (len(values),n,2) or not np.isfinite(rest).all() or not np.isfinite(delta).all():
        raise ValueError(f'Invalid geometry: {path}')
    if not np.isfinite(values).all() or not np.isfinite([lo,hi,default]).all() or not lo<=default<=hi or lo>=hi:
        raise ValueError(f'Invalid parameter: {path}')
    if off[0]!=0 or off[-1]!=n or np.any(np.diff(off)<=0) or tri.ndim!=2 or tri.shape[1]!=3 or np.any(tri<0) or np.any(tri>=n):
        raise ValueError(f'Invalid topology: {path}')
    mesh=np.repeat(np.arange(len(off)-1),np.diff(off))
    if len(tri) and np.any(mesh[tri]!=mesh[tri[:,0,None]]): raise ValueError('Cross-mesh triangle')
    feat=np.empty((n,8),np.float32)
    for a,b in zip(off[:-1],off[1:]):
        xy=rest[a:b]; center=xy.mean(0); extent=np.ptp(xy,axis=0)
        feat[a:b]=np.concatenate([xy,np.broadcast_to(center,xy.shape),np.broadcast_to(extent,xy.shape),(xy-center)/max(float(extent.max()),1e-6)],axis=1)
    e=np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]],axis=0)
    e=np.unique(np.concatenate([e,e[:,::-1]],axis=0),axis=0).T.copy()
    t=(values-default)/np.where(values>=default,max(float(hi-default),1e-6),max(float(default-lo),1e-6))
    peak=np.linalg.norm(delta,axis=2).max(axis=0)
    def tensor(x): return torch.as_tensor(x,device=device)
    # Targets remain on CPU until a pose is selected; no target-derived model inputs.
    return dict(source=source, group=path.relative_to(root).parts[0], param=p,
                feat=tensor(feat), edge=tensor(e), mesh=tensor(mesh), nm=len(off)-1,
                onehot=tensor(np.eye(len(PARAMS),dtype=np.float32)[PARAMS.index(p)][None]),
                t=tensor(t.astype(np.float32)), delta=delta,
                moving=tensor(peak>1e-5))


def target(r, i, device):
    return torch.as_tensor(r['delta'][i],device=device)


@torch.no_grad()
def evaluate(model, records, device, poses=11):
    model.eval(); rows=[]
    for r in records:
        for i in np.unique(np.linspace(0,len(r['t'])-1,poses).round().astype(int)):
            y=target(r,i,device); pred=model(r,r['t'][i]); moving=r['moving']
            err=(pred-y).norm(dim=-1); zero=y.norm(dim=-1)
            rows.append([err.mean().item(),zero.mean().item(),err[moving].mean().item() if moving.any() else np.nan,zero[moving].mean().item() if moving.any() else np.nan])
    return np.nanmean(rows,axis=0)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,default=ROOT/'dataset/processed_v02')
    ap.add_argument('--holdout',default='Hiyori',help='Source filename stem excluded entirely from training')
    ap.add_argument('--steps',type=int,default=2000)
    ap.add_argument('--width',type=int,default=64)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    ap.add_argument('--output',type=Path,default=ROOT/'checkpoints/real_rig_v01.pt')
    ap.add_argument('--confirm-authorized',action='store_true',help='Confirm actual permission covers all selected NPZ files and training')
    a=ap.parse_args()
    if not a.confirm_authorized: ap.error('Confirm actual dataset/SDK authorization before training.')
    if a.steps<1 or a.width<8: ap.error('Invalid steps/width')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device)
    if device.type=='cuda':
        if not torch.cuda.is_available(): raise SystemExit('CUDA unavailable')
        torch.cuda.manual_seed_all(a.seed); torch.cuda.reset_peak_memory_stats()
    else: torch.set_num_threads(min(4,torch.get_num_threads()))
    print('Device:',device,torch.cuda.get_device_name() if device.type=='cuda' else '',flush=True)
    paths=sorted(a.data.rglob('*.npz')); records=[load_record(p,a.data,device) for p in paths]
    # Exclude all records for the held-out model; no random split over neighboring poses.
    held=[r for r in records if Path(r['source']).stem.casefold()==a.holdout.casefold()]
    groups={r['group'] for r in held}
    train=[r for r in records if r['group'] not in groups and Path(r['source']).stem.casefold()!=a.holdout.casefold()]
    if not train or not held: raise SystemExit('Need nonempty training and held-out groups; check --data and --holdout')
    train_names=sorted({r['source'] for r in train}); held_names=sorted({r['source'] for r in held})
    print('Train:',len(train),'files',train_names,flush=True)
    print('Held-out:',len(held),'files',held_names,flush=True)
    missing={r['param'] for r in held}-{r['param'] for r in train}
    if missing: raise SystemExit(f'Held-out parameters missing from training: {missing}')
    model=GraphRig(a.width).to(device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    start=time.perf_counter(); running=0.
    print('Training geometry only; no opacity, draw-order, physics, or Cubism project export.',flush=True)
    for step in range(1,a.steps+1):
        model.train(); r=random.choice(train); i=random.randrange(len(r['t']))
        y=target(r,i,device); pred=model(r,r['t'][i]); per=(pred-y).square().sum(-1)
        w=1+4*r['moving'].float(); loss=(per*w).sum()/w.sum()
        if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        running+=loss.item()
        if step%100==0 or step==a.steps:
            print(f'Step {step}/{a.steps} | train weighted MSE {running/(100 if step%100==0 else step%100):.8f}',flush=True); running=0.
    # Fixed holdout is not used for optimization or checkpoint selection.
    v=evaluate(model,held,device)
    print('Held-out mean vertex distance (normalized model units):',flush=True)
    print(f' All vertices: model={v[0]:.8f} | zero baseline={v[1]:.8f}',flush=True)
    print(f' Moving vertices: model={v[2]:.8f} | zero baseline={v[3]:.8f}',flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model_state=model.state_dict(),width=a.width,parameters=PARAMS,seed=a.seed,steps=a.steps,train_sources=train_names,held_sources=held_names,held_metrics=v.tolist()),a.output)
    print('Saved:',a.output,flush=True)
    print(f'Elapsed {time.perf_counter()-start:.2f}s',flush=True)
    if device.type=='cuda': print(f'Peak allocated GPU: {torch.cuda.max_memory_allocated()/2**20:.1f} MiB',flush=True)

if __name__=='__main__': main()
