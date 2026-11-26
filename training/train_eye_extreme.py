"""MintoAI V0.4 Extreme: streaming synthetic eye deformation experiment.
No artist data. Python + PyTorch + NumPy. Run with --help for options.
"""
import argparse, json, math, subprocess, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parent.parent
# Shape descriptors are known synthetic inputs, not inferred from artwork.
NAMES=['width','upper','lower','tilt','skew','power','bend','offset','us','ls','warp','corner','crease','style']
LO=[.45,.08,.04,-.45,-.55,.32,-.22,-.2,-.35,-.35,-.23,-.15,-.15,0]
HI=[1.65,.85,.68,.45,.55,1.35,.22,.2,.35,.35,.23,.15,.15,1]
FEATURES=NAMES+['local_x','open','rest_x','rest_upper','rest_lower']

def shapes(n,dev,g):
    lo=torch.tensor(LO,device=dev); hi=torch.tensor(HI,device=dev)
    d=lo+(hi-lo)*torch.rand((n,len(LO)),device=dev,generator=g)
    # Four procedural style families; all are synthetic, not artist examples.
    f=torch.randint(4,(n,),device=dev,generator=g)
    d[:,13]=f.float()/3
    d[:,1]*=torch.tensor([.65,1.,1.2,.85],device=dev)[f]
    d[:,2]*=torch.tensor([.65,1.,1.15,.8],device=dev)[f]
    return d

def geometry(d,s,p):
    w,u,l,tilt,skew,power,bend,offset,us,ls,warp,corner,crease,style=d.unbind(-1)
    a=(1-s.square()).clamp_min(0).pow(power)
    x=w*(s+warp*s*(1-s.square()))
    mid=offset+tilt*s+bend*s.square()+corner*s.pow(3)
    profile=a*(1+skew*s)
    up=mid+u*profile*(1+us*s)
    low=mid-l*profile*(1+ls*s)
    ratio=(.22+.14*style+.14*skew+.15*bend+.12*s+.15*crease*torch.cos(math.pi*s)).clamp(.08,.65)
    seam=low+ratio*(up-low)
    gu=(1.0+1.1*u+.35*skew+.25*us+.22*s+.45*style).clamp(.5,3.)
    gl=(.65+.85*l+.25*ls-.15*s+.35*style).clamp(.45,2.5)
    close_x=x+w*a*(.055*skew-.065*s+.08*crease)
    px=x+(1-p)*(close_x-x)+p*(1-p)*w*a*(.08*corner+.035*s)
    yu=seam+p.pow(gu)*(up-seam)
    yl=seam+p.pow(gl)*(low-seam)
    r=torch.stack((x,up,x,low),-1)
    y=torch.stack((px,yu,px,yl),-1)
    c=torch.stack((close_x,seam,close_x,seam),-1)
    feat=torch.cat((d,s[:,None],p[:,None],x[:,None],up[:,None],low[:,None]),-1)
    return feat,r,y,c

class Net(nn.Module):
    def __init__(self,width,depth):
        super().__init__()
        self.layers=nn.Sequential(*[z for i in range(depth) for z in (nn.Linear(len(FEATURES) if i==0 else width,width),nn.SiLU())],nn.Linear(width,5))
    def forward(self,x,r):
        z=self.layers(x).float(); s=x[:,14]; p=x[:,15]; a=(1-s.square()).clamp_min(0)
        # Structural constraints: preserve corners, exact open rest, closed lids meet,
        # and no crossing of paired lids. Not evidence of learned artist technique.
        rx=r[:,0]; mid=(r[:,1]+r[:,3])*.5; gap=(r[:,1]-r[:,3]).clamp_min(0)
        px=rx+(1-p)*a.sqrt()*(z[:,0]+p*z[:,1])
        m=mid+(1-p)*a.sqrt()*(z[:,2]+p*z[:,3])
        g=p*gap*torch.exp(((1-p)*z[:,4]).clamp(-5,5))
        return torch.stack((px,m+g*.5,px,m-g*.5),-1)

def batch_data(n,dev,g):
    # Multiple vertices/poses per fresh shape; no persistent training bank.
    k=max(1,(n+31)//32)
    d=shapes(k,dev,g).repeat_interleave(32,dim=0)[:n]
    s=2*torch.rand(n,device=dev,generator=g)-1
    p=torch.rand(n,device=dev,generator=g)
    q=torch.rand(n,device=dev,generator=g)
    p=torch.where(q<.05,0.,torch.where(q>.95,1.,p))
    s=torch.where(q<.02,-1.,torch.where(q>.98,1.,s))
    return geometry(d,s,p)

def grid(bank,poses,points,chunk):
    n=len(bank)*poses*points
    for i in range(0,n,chunk):
        idx=torch.arange(i,min(n,i+chunk),device=bank.device)
        s=(idx%points).float()*2/(points-1)-1
        p=((idx//points)%poses).float()/(poses-1)
        yield geometry(bank[idx//(points*poses)],s,p),p

@torch.no_grad()
def evaluate(net,bank,poses,points,chunk):
    net.eval(); sums=torch.zeros(5,device=bank.device); maxerr=0.; n=0
    for (x,r,y,c),p in grid(bank,poses,points,chunk):
        pred=net(x,r); linear=c+(r-c)*p[:,None]
        err=(pred-y).reshape(-1,2,2).norm(dim=-1)
        base=(linear-y).reshape(-1,2,2).norm(dim=-1)
        sums+=torch.stack(((pred-y).square().sum(),err.sum(),base.sum(),(pred[:,1]<pred[:,3]-1e-6).sum().float(),((pred[:,:2]-pred[:,2:]).norm(dim=-1)*(p==0)).sum()))
        maxerr=max(maxerr,float(err.max())); n+=len(x)
    a,b,c,d,e=sums.tolist()
    return dict(mse=a/(4*n),mean_vertex_error=b/(2*n),linear_oracle_error=c/(2*n),max_vertex_error=maxerr,crossed_pairs=int(d),closed_gap=e/(len(bank)*points),pairs=n)

def atomic(obj,path):
    tmp=path.with_suffix('.tmp'); torch.save(obj,tmp); tmp.replace(path)

def weights(net): return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}

def telemetry():
    try:
        r=subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
        if r.returncode==0:
            t,u,m,mt,p=[x.strip() for x in r.stdout.splitlines()[0].split(',')]
            return f'GPU {t} C | Util {u}% | Total VRAM {m}/{mt} MiB | Power {p} W'
    except (OSError,subprocess.SubprocessError,ValueError,IndexError): pass
    return 'GPU telemetry unavailable'

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k,v in [('steps',1000000),('batch',0),('max-batch',32768),('width',512),('depth',8),('seed',2026),('val-shapes',256),('test-shapes',1024),('val-poses',21),('test-poses',51),('points',65),('eval-batch',8192),('eval-every',2000),('save-every',1000),('log-every',200),('patience',50),('preview-shapes',12)]: ap.add_argument('--'+k,type=int,default=v)
    ap.add_argument('--output-dir',type=Path,default=ROOT/'checkpoints'/'extreme_v04')
    ap.add_argument('--resume',action='store_true'); ap.add_argument('--cpu',action='store_true'); ap.add_argument('--no-amp',action='store_true')
    a=ap.parse_args()
    if min(a.steps,a.max_batch,a.width,a.depth,a.val_shapes,a.test_shapes,a.val_poses-1,a.test_poses-1,a.points-1,a.eval_batch,a.eval_every,a.save_every,a.log_every,a.preview_shapes)<1 or a.batch<0 or a.patience<0: ap.error('Invalid dimensions')
    dev=torch.device('cuda' if torch.cuda.is_available() and not a.cpu else 'cpu')
    if dev.type=='cpu': torch.set_num_threads(min(4,torch.get_num_threads()))
    amp=dev.type=='cuda' and not a.no_amp
    torch.manual_seed(a.seed)
    def generator(seed,device=dev): return torch.Generator(device=device).manual_seed(seed)
    g=generator(a.seed+1)
    val=shapes(a.val_shapes,dev,generator(a.seed+2)); test=shapes(a.test_shapes,dev,generator(a.seed+3))
    out=a.output_dir; out.mkdir(parents=True,exist_ok=True); last=out/'last.pt'; bestfile=out/'best.pt'
    if last.exists()!=a.resume: ap.error('Use --resume for existing last.pt; otherwise use a fresh output directory.')
    cfg={k:getattr(a,k) for k in ['width','depth','seed','val_shapes','test_shapes','val_poses','test_poses','points']}
    def make():
        n=Net(a.width,a.depth).to(dev); o=torch.optim.AdamW(n.parameters(),lr=.0004,weight_decay=1e-5); sc=torch.amp.GradScaler('cuda',enabled=amp); return n,o,sc
    def update(n,o,sc,rng,b):
        n.train(); o.zero_grad(set_to_none=True); x,r,y,_=batch_data(b,dev,rng)
        with torch.autocast(device_type=dev.type,dtype=torch.float16,enabled=amp): pred=n(x,r)
        loss=(pred-y).square().mean()
        sc.scale(loss).backward(); sc.unscale_(o); nn.utils.clip_grad_norm_(n.parameters(),1.)
        sc.step(o); sc.update(); return loss.detach()
    def probe(b):
        try:
            n,o,sc=make(); rng=generator(a.seed+99)
            for _ in range(2): update(n,o,sc,rng,b)
            torch.cuda.synchronize(); t=time.perf_counter()
            for _ in range(3): update(n,o,sc,rng,b)
            torch.cuda.synchronize()
            return b*3/(time.perf_counter()-t),torch.cuda.max_memory_reserved()
        except torch.OutOfMemoryError: return None
    b=a.batch
    if b==0 and dev.type=='cuda':
        free,total=torch.cuda.mem_get_info(); budget=int(free*.75); candidates=sorted(set([min(a.max_batch,x) for x in [2048,4096,8192,16384,32768]])); results=[]
        for cand in candidates:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); result=probe(cand)
            torch.cuda.empty_cache()
            if result is None: print(f'Probe {cand}: OOM',flush=True); break
            rate,mem=result; print(f'Probe {cand}: {rate:,.0f} pairs/s, {mem/2**20:.0f} MiB reserved',flush=True)
            if mem>budget: break
            results.append((rate,cand))
        if not results: ap.error('No safe GPU batch found; try --batch 256 or close GPU apps.')
        b=max(results)[1]; torch.cuda.reset_peak_memory_stats()
    elif b==0: b=min(1024,a.max_batch)
    torch.manual_seed(a.seed); net,opt,scaler=make(); start=0; best=float('inf'); stale=0; bestweights=None; seen=0; pairs=0
    if a.resume:
        ck=torch.load(last,map_location='cpu',weights_only=True)
        if ck['config']!=cfg: ap.error('Resume configuration mismatch. Keep architecture and eval settings.')
        net.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); scaler.load_state_dict(ck['scaler']); g.set_state(ck['rng']); start=ck['step']; best=ck['best']; stale=ck['stale']; bestweights=ck['bestweights']; seen=ck['shape_draws']; pairs=ck['pairs']
    def save(step):
        atomic(dict(config=cfg,model=weights(net),optimizer=opt.state_dict(),scaler=scaler.state_dict(),rng=g.get_state(),step=step,best=best,stale=stale,bestweights=bestweights,shape_draws=seen,pairs=pairs),last)
    print('MintoAI V0.4 EXTREME | SYNTHETIC ONLY',flush=True)
    print(f'Device: {torch.cuda.get_device_name(0) if dev.type=="cuda" else "CPU"} | AMP: {amp} | Parameters: {sum(p.numel() for p in net.parameters()):,}',flush=True)
    print(f'Batch: {b:,} pairs | Steps target: {a.steps:,} | Resume step: {start:,} | Streaming procedural shapes',flush=True)
    t=time.perf_counter(); done=start
    try:
        for step in range(start+1,a.steps+1):
            loss=update(net,opt,scaler,g,b); done=step; seen+=(b+31)//32; pairs+=b
            if step==1 or step%a.log_every==0:
                elapsed=time.perf_counter()-t; rate=(step-start)/max(elapsed,1e-9)
                print(f'Step {step:,}/{a.steps:,} | Batch MSE {float(loss):.8f} | Shape draws {seen:,} | Pairs {pairs:,} | {rate:.2f} steps/s | ETA at current avg {(a.steps-step)/max(rate,1e-9)/3600:.2f} h',flush=True)
                if dev.type=='cuda': print(f'Peak tensor/reserved: {torch.cuda.max_memory_allocated()/2**20:.0f}/{torch.cuda.max_memory_reserved()/2**20:.0f} MiB | '+telemetry(),flush=True)
            stop=False
            if step==1 or step%a.eval_every==0 or step==a.steps:
                v=evaluate(net,val,a.val_poses,a.points,a.eval_batch)
                improved=v['mse']<best*(1-1e-4)
                if v['mse']<best:
                    best=v['mse']; bestweights=weights(net); atomic(dict(state_dict=bestweights,config=cfg,features=FEATURES,best_val_mse=best,source='synthetic'),bestfile)
                stale=0 if improved else stale+1
                print(f'VAL {step:,}: MSE {v["mse"]:.9f} | Mean error {v["mean_vertex_error"]:.6f} | Crossed {v["crossed_pairs"]} | Best {best:.9f}',flush=True)
                stop=bool(a.patience and stale>=a.patience)
            if step==1 or step%a.save_every==0 or step==a.steps or stop: save(step)
            if stop: print('Early stop: validation plateau.',flush=True); break
    except KeyboardInterrupt:
        print('\nStopped. Resume from last saved checkpoint; unsaved steps are discarded.',flush=True); return
    except torch.OutOfMemoryError:
        print(f'CUDA OOM. Resume from last checkpoint with --batch {max(128,b//2)} --eval-batch {max(128,a.eval_batch//2)} (omit --resume if no checkpoint).',flush=True); raise
    if bestweights is None: raise RuntimeError('No validated checkpoint.')
    net.load_state_dict(bestweights)
    result=evaluate(net,test,a.test_poses,a.points,a.eval_batch)
    report=dict(config=cfg,step=done,shape_draws=seen,pairs=pairs,test=result,source='synthetic',production_rig=False)
    (out/'test_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('FINAL TEST:',json.dumps(result,indent=2),flush=True)
    # Bounded preview in same array convention as V0.3.
    bank=test[:a.preview_shapes]; acc=[[],[],[],[]]
    with torch.no_grad():
        for (x,r,y,c),p in grid(bank,a.test_poses,a.points,a.eval_batch):
            for dst,z in zip(acc,(r,y,net(x,r),c+(r-c)*p[:,None])): dst.append(z.cpu().numpy())
    def convert(q): return np.concatenate(q).reshape(len(bank),a.test_poses,a.points,2,2).transpose(0,1,3,2,4).reshape(len(bank),a.test_poses,a.points*2,2)
    r,y,pred,lin=map(convert,acc)
    tri=np.asarray([(i,i+1,i+a.points) for i in range(a.points-1)]+[(i+1,i+a.points+1,i+a.points) for i in range(a.points-1)],dtype=np.int32)
    np.savez_compressed(out/'predictions.npz',parameter_values=np.linspace(0,1,a.test_poses,dtype=np.float32),vertices_rest=r[:,0],vertices_target=y,vertices_predicted=pred,vertices_linear=lin,triangles=tri)
    print('Saved:',out,'| Synthetic only, not a production Live2D rig.',flush=True)

if __name__=='__main__': main()
