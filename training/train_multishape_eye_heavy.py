"""MintoAI V0.3 Heavy: synthetic paired-lid deformation, sampled on demand.
No artist data, SDK, or external downloads. Python 3 + PyTorch + NumPy.
"""
import argparse, math, time, json
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
DESCRIPTORS = ['width','upper_height','lower_height','tilt','skew','roundness','bend','offset','upper_skew','lower_skew','x_warp']
LO = [.55,.12,.07,-.30,-.50,.33,-.15,-.12,-.30,-.30,-.20]
HI = [1.5,.72,.55,.30,.50,.95,.15,.12,.30,.30,.20]
FEATURES = DESCRIPTORS + ['local_x','eye_open','rest_x','rest_upper_y','rest_lower_y']


def make_bank(n, seed, device):
    rng = np.random.default_rng(seed)
    a = rng.uniform(LO,HI,size=(n,len(LO))).astype(np.float32)
    return torch.from_numpy(a).to(device)


def geometry(d, s, p):
    w,u,l,tilt,skew,power,bend,offset,us,ls,warp = d.unbind(-1)
    a = (1-s.square()).clamp_min(0).pow(power)
    x = w*(s+warp*s*(1-s.square()))
    mid = offset+tilt*s+bend*s.square()
    profile = a*(1+skew*s)
    up = mid+u*profile*(1+us*s)
    low = mid-l*profile*(1+ls*s)
    # Authored synthetic target, not learned from artists.
    ratio = (.27+.12*skew+.12*bend+.08*s).clamp(.1,.45)
    seam = low+ratio*(up-low)
    gu = 1.15+.9*u+.28*skew+.2*us+.12*s
    gl = .75+.65*l+.22*ls-.08*s
    xu = x+w*a*(1-p)*(.025*skew-.035*s)
    yu = seam+p.pow(gu)*(up-seam)
    yl = seam+p.pow(gl)*(low-seam)
    rest = torch.stack((x,up,x,low),-1)
    target = torch.stack((xu,yu,xu,yl),-1)
    closed = torch.stack((x+w*a*(.025*skew-.035*s),seam,x+w*a*(.025*skew-.035*s),seam),-1)
    feat = torch.cat((d,s[:,None],p[:,None],x[:,None],up[:,None],low[:,None]),-1)
    return feat,rest,target,closed


class Net(nn.Module):
    def __init__(self,width,depth):
        super().__init__()
        layers=[]
        for i in range(depth):
            layers.extend((nn.Linear(len(FEATURES) if i==0 else width,width), nn.SiLU()))
        layers.append(nn.Linear(width,6))
        self.layers=nn.Sequential(*layers)
    def forward(self,feat,rest):
        z=self.layers(feat).float()
        p=feat[:,12:13]
        common=z[:,:2].repeat(1,2)
        # Paired prediction: exactly coincident lids at p=0, exact rest at p=1.
        return p*rest+(1-p)*common+p*(1-p)*z[:,2:]


def sample(bank,ntrain,batch,poses,points):
    dev=bank.device
    i=torch.randint(ntrain,(batch,),device=dev)
    s=torch.randint(points,(batch,),device=dev).float()*(2/(points-1))-1
    p=torch.randint(poses,(batch,),device=dev).float()/(poses-1)
    return geometry(bank[i],s,p)


def grid_batches(bank, ids, pose_count, points, chunk):
    # Flattened indexing avoids materializing the full grid.
    n=len(ids)*pose_count*points
    for start in range(0,n,chunk):
        ix=torch.arange(start,min(start+chunk,n),device=bank.device)
        s=(ix%points).float()*(2/(points-1))-1
        p=((ix//points)%pose_count).float()/(pose_count-1)
        shape=ids[ix//(points*pose_count)]
        yield geometry(bank[shape],s,p),p


@torch.no_grad()
def evaluate(net,bank,ids,poses,points,chunk):
    net.eval(); total=0.; errsum=0.; basesum=0.; maxerr=0.; count=0; crossing=0; gap=0.; gapcount=0
    for (x,r,y,c),p in grid_batches(bank,ids,poses,points,chunk):
        pred=net(x,r)
        linear=c+(r-c)*p[:,None]
        e=(pred-y).reshape(-1,2,2).norm(dim=-1)
        b=(linear-y).reshape(-1,2,2).norm(dim=-1)
        total+=(pred-y).square().sum().item(); errsum+=e.sum().item(); basesum+=b.sum().item()
        maxerr=max(maxerr,e.max().item()); count+=len(x)
        crossing+=((pred[:,1]<pred[:,3]-1e-6)&(p>0)).sum().item()
        mask=p==0
        gap+=(pred[mask,:2]-pred[mask,2:]).norm(dim=-1).sum().item(); gapcount+=mask.sum().item()
    return {'mse':total/(4*count),'mean_vertex_error':errsum/(2*count),'linear_oracle_error':basesum/(2*count),'max_vertex_error':maxerr,'closed_gap':gap/max(1,gapcount),'crossed_pairs':crossing,'pairs':count}


def state_cpu(net):
    return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}


def save_atomic(obj,path):
    tmp=path.with_suffix(path.suffix+'.tmp'); torch.save(obj,tmp); tmp.replace(path)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key,default in [('shapes',100000),('poses',101),('points',65),('steps',50000),('batch',8192),('width',384),('depth',5),('eval-every',1000),('val-shapes',256),('test-shapes',1024),('eval-poses',21),('test-poses',51),('eval-batch',16384),('save-every',500),('preview-shapes',24),('seed',2026)]:
        ap.add_argument('--'+key,type=int,default=default)
    ap.add_argument('--output-dir',type=Path,default=ROOT/'checkpoints'/'heavy_v03')
    ap.add_argument('--resume',action='store_true'); ap.add_argument('--no-amp',action='store_true'); ap.add_argument('--cpu',action='store_true')
    a=ap.parse_args()
    if min(a.shapes-9,a.poses-1,a.points-1,a.steps,a.batch,a.width,a.depth,a.eval_every,a.val_shapes,a.test_shapes,a.eval_poses-1,a.test_poses-1,a.eval_batch,a.save_every,a.preview_shapes)<1: ap.error('Invalid dimensions or counts (at least 10 shapes, 2 grid points/poses).')
    dev=torch.device('cuda' if torch.cuda.is_available() and not a.cpu else 'cpu')
    if dev.type=='cpu': torch.set_num_threads(min(4,torch.get_num_threads()))
    amp=dev.type=='cuda' and not a.no_amp
    out=a.output_dir; out.mkdir(parents=True,exist_ok=True)
    last=out/'last.pt'; bestpath=out/'best.pt'
    if last.exists() and not a.resume: ap.error('Checkpoint exists; use --resume or a different --output-dir.')
    if a.resume and not last.exists(): ap.error('No last.pt to resume.')
    torch.manual_seed(a.seed); bank=make_bank(a.shapes,a.seed,dev)
    nt=int(a.shapes*.70); nv=int(a.shapes*.15); ntest=a.shapes-nt-nv
    rng=np.random.default_rng(a.seed+1)
    vi=torch.tensor(rng.choice(np.arange(nt,nt+nv),min(nv,a.val_shapes),replace=False),device=dev)
    ti=torch.tensor(rng.choice(np.arange(nt+nv,a.shapes),min(ntest,a.test_shapes),replace=False),device=dev)
    net=Net(a.width,a.depth).to(dev); opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=1e-5)
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    # Architecture + data generator configuration must match for resume. Batch may be changed.
    cfg={k:getattr(a,k) for k in ['shapes','poses','points','width','depth','seed','val_shapes','eval_poses']}
    start=0; best=float('inf'); bestweights=None
    if a.resume:
        ck=torch.load(last,map_location='cpu',weights_only=True)
        if ck['config']!=cfg: ap.error('Resume configuration mismatch; keep original shape/network settings.')
        net.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); scaler.load_state_dict(ck['scaler'])
        start=ck['step']; best=ck['best_val']; bestweights=ck['best_weights']
        torch.set_rng_state(ck['cpu_rng'])
        if dev.type=='cuda' and ck.get('cuda_rng') is not None: torch.cuda.set_rng_state(ck['cuda_rng'])
    def checkpoint(step):
        save_atomic({'config':cfg,'step':step,'model':state_cpu(net),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'best_val':best,'best_weights':bestweights,'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state() if dev.type=='cuda' else None,'source':'synthetic'},last)
    print('MintoAI V0.3 HEAVY - SYNTHETIC ONLY',flush=True)
    print('Device:',torch.cuda.get_device_name(0) if dev.type=='cuda' else 'CPU','| AMP FP16:',amp,flush=True)
    print(f'Shapes: {a.shapes:,} | Train/Val/Test: {nt:,}/{nv:,}/{ntest:,}',flush=True)
    print(f'Pose grid: {a.poses} | Vertices/eye: {a.points*2} | Potential vertex samples: {a.shapes*a.poses*a.points*2:,}',flush=True)
    print(f'Network parameters: {sum(p.numel() for p in net.parameters()):,} | Batch: {a.batch:,} pairs ({a.batch*2:,} vertices) | Steps: {a.steps:,}',flush=True)
    print(f'Evaluation: {len(vi)} validation eyes, {len(ti)} test eyes (fixed subsets, disjoint from train)',flush=True)
    t=time.perf_counter(); batch=a.batch
    try:
        for step in range(start+1,a.steps+1):
            net.train(); opt.zero_grad(set_to_none=True)
            try:
                x,r,y,_=sample(bank,nt,batch,a.poses,a.points)
                with torch.autocast(device_type=dev.type,dtype=torch.float16,enabled=amp): pred=net(x,r)
                loss=(pred-y).square().mean()
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(net.parameters(),1.)
                scaler.step(opt); scaler.update()
            except torch.OutOfMemoryError:
                print('GPU OOM: restart with --resume --batch',max(128,batch//2),'--eval-batch',max(128,a.eval_batch//2),'(or omit --resume if no checkpoint exists).',flush=True)
                raise
            if step==1 or step%a.eval_every==0 or step==a.steps:
                v=evaluate(net,bank,vi,a.eval_poses,a.points,a.eval_batch)
                if v['mse']<best:
                    best=v['mse']; bestweights=state_cpu(net)
                    save_atomic({'state_dict':bestweights,'config':cfg,'features':FEATURES,'best_val_mse':best,'source':'synthetic','production_rig':False},bestpath)
                msg=f'Step {step:6d} | Batch MSE {loss.item():.8f} | Val MSE {v["mse"]:.8f} | Elapsed {time.perf_counter()-t:.1f}s'
                if dev.type=='cuda': msg+=f' | Peak allocated/reserved {torch.cuda.max_memory_allocated()/2**20:.0f}/{torch.cuda.max_memory_reserved()/2**20:.0f} MiB'
                print(msg,flush=True)
            if step==1 or step%a.save_every==0 or step==a.steps: checkpoint(step)
    except KeyboardInterrupt:
        print('\nInterrupted. Resume from last completed saved checkpoint using --resume; unsaved steps are lost.',flush=True); return
    if bestweights is None: raise RuntimeError('No validation checkpoint. Increase steps or restart.')
    net.load_state_dict(bestweights); net.eval()
    result=evaluate(net,bank,ti,a.test_poses,a.points,a.eval_batch)
    print('FINAL TEST (unseen synthetic shapes)',flush=True)
    for k,v in result.items(): print(f'{k}: {v}',flush=True)
    (out/'test_report.json').write_text(json.dumps({'config':cfg,'test_shapes':len(ti),'test_poses':a.test_poses,'metrics':result},indent=2),encoding='utf-8')
    # Save a bounded preview of prediction and target for later visualization.
    ids=ti[:min(len(ti),a.preview_shapes)]; accum=[[],[],[],[]]
    with torch.no_grad():
        for (x,r,y,c),p in grid_batches(bank,ids,a.poses,a.points,a.eval_batch):
            for dst,src in zip(accum,(r,y,net(x,r),c+(r-c)*p[:,None])): dst.append(src.cpu().numpy())
    def convert(chunks):
        z=np.concatenate(chunks).reshape(len(ids),a.poses,a.points,2,2)
        return z.transpose(0,1,3,2,4).reshape(len(ids),a.poses,2*a.points,2)
    r,y,pred,lin=map(convert,accum)
    tri=np.asarray([(i,i+1,i+a.points) for i in range(a.points-1)]+[(i+1,i+a.points+1,i+a.points) for i in range(a.points-1)],dtype=np.int32)
    np.savez_compressed(out/'predictions.npz',parameter_values=np.linspace(0,1,a.poses,dtype=np.float32),vertices_rest=r[:,0],vertices_target=y,vertices_predicted=pred,vertices_linear=lin,triangles=tri,test_shape_ids=ids.cpu().numpy())
    print('Saved:',out,flush=True)
    print('Synthetic benchmark only. No real artist model used. Not a production rig.',flush=True)

if __name__=='__main__': main()
