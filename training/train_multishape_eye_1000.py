"""MintoAI V0.2-1000: synthetic, shape-disjoint eye deformation experiment."""
from pathlib import Path
import argparse
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'checkpoints'
FEATURES = ['rest_x','rest_y','upper_lid','eye_open','half_width',
            'upper_height','lower_height','tilt','asymmetry','local_x']

N_SHAPES = 1000
N_TRAIN = 700
N_VAL = 150

def make_data():
    rng = np.random.default_rng(2026)
    p = np.linspace(0,1,21,dtype=np.float32)
    local = np.linspace(-1,1,17,dtype=np.float32)
    arch = np.maximum(np.sqrt(np.maximum(0,1-local**2)),.04)
    features, rests, targets = [], [], []
    for _ in range(N_SHAPES):
        w = rng.uniform(.65,1.25)
        u = rng.uniform(.20,.58)
        l = rng.uniform(.12,.43)
        tilt = rng.uniform(-.18,.18)
        skew = rng.uniform(-.35,.35)
        profile = arch*(1+skew*local)
        mid = tilt*local
        upper = mid+u*profile
        lower = mid-l*profile
        seam = mid-(.24*u+.12*l)*profile
        # Nonlinear closure: deliberately authored synthetic teaching signal.
        # No real artist data or Live2D SDK involved.
        gamma = 1.25 + .65*u + .40*skew
        q = p[:,None]**gamma
        up = seam[None,:]+q*(upper-seam)[None,:]
        lo = seam[None,:]+q*(lower-seam)[None,:]
        x = np.tile(local*w,2)
        rest = np.stack([x,np.concatenate([upper,lower])],axis=-1)
        target = np.stack([np.broadcast_to(x,(len(p),len(x))),np.concatenate([up,lo],axis=1)],axis=-1)
        shape = np.column_stack([rest,np.repeat([1.,0.],17),np.zeros(34),
                   np.tile([w,u,l,tilt,skew],(34,1)),np.tile(local,2)])
        f = np.broadcast_to(shape,(len(p),*shape.shape)).copy()
        f[:,:,3] = p[:,None]
        features.append(f); rests.append(rest); targets.append(target)
    return p,np.asarray(features,np.float32),np.asarray(rests,np.float32),np.asarray(targets,np.float32)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(10,96),nn.SiLU(),nn.Linear(96,96),nn.SiLU(),nn.Linear(96,2))
    def forward(self,x):
        return self.layers(x)*(1-x[...,3:4])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--steps",type=int,default=4000,help="Number of optimizer updates")
    parser.add_argument("--output-dir",type=Path,default=OUT)
    args=parser.parse_args()
    if args.steps < 1: parser.error("--steps must be positive")
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    p,f,rest,target=make_data()
    delta=target-rest[:,None,:,:]
    # Whole shapes separated; no vertex/pose of a test eye enters training.
    train=slice(0,N_TRAIN); val=slice(N_TRAIN,N_TRAIN+N_VAL); test=slice(N_TRAIN+N_VAL,N_SHAPES)
    torch.manual_seed(42)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cpu': torch.set_num_threads(1)
    X=torch.from_numpy(f).to(device); Y=torch.from_numpy(delta).to(device)
    xt=X[train].reshape(-1,10); yt=Y[train].reshape(-1,2)
    xv=X[val]; yv=Y[val]
    net=Net().to(device); opt=torch.optim.AdamW(net.parameters(),lr=.002)
    best=float('inf'); weights=None
    print('MintoAI V0.2-1000 - SYNTHETIC ONLY',flush=True)
    print('Device:',torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU',flush=True)
    print(f'Train eyes: {N_TRAIN} | Validation eyes: {N_VAL} | Test eyes: {N_SHAPES-N_TRAIN-N_VAL}',flush=True)
    print(f'Shapes: {N_SHAPES} | Poses per shape: {len(p)} | Total poses: {N_SHAPES*len(p)}',flush=True)
    print(f'Optimizer steps: {args.steps} | Batch: 2048 vertices',flush=True)
    for step in range(1,args.steps+1):
        net.train(); ids=torch.randint(len(xt),(2048,),device=device)
        loss=(net(xt[ids])-yt[ids]).square().mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step==1 or step%200==0 or step==args.steps:
            net.eval()
            with torch.no_grad():
                total=0.0
                for i in range(0,len(xv),16):
                    total+=(net(xv[i:i+16])-yv[i:i+16]).square().sum().item()
                vl=total/yv.numel()
            if vl<best:
                best=vl; weights={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
            print(f'Step {step:4d} | Batch MSE: {loss.item():.8f} | Val MSE: {vl:.8f}',flush=True)
    net.load_state_dict(weights); net.eval()
    with torch.no_grad():
        pred=np.concatenate([net(chunk).cpu().numpy() for chunk in X[test].split(16)])+rest[test,None,:,:]
    truth=target[test]
    # Linear interpolation is given both correct endpoints (oracle comparator).
    linear=truth[:,0:1]+p[None,:,None,None]*(truth[:,-1:]-truth[:,0:1])
    err=np.linalg.norm(pred-truth,axis=-1)
    base=np.linalg.norm(linear-truth,axis=-1)
    gap=np.linalg.norm(pred[:,0,:17]-pred[:,0,17:],axis=-1)
    mean=float(err.mean()); bm=float(base.mean())
    print(f'Test AI mean vertex error: {mean:.6f}')
    print(f'Test linear oracle mean error: {bm:.6f}')
    print(f'Test AI max vertex error: {err.max():.6f}')
    print(f'Closed-eye mean gap: {gap.mean():.6f}')
    print(f'Closed-eye max gap: {gap.max():.6f}')
    print('AI below linear oracle:',mean<bm)
    if device.type=='cuda': print(f'Peak allocated VRAM: {torch.cuda.max_memory_allocated()/1024**2:.2f} MiB')
    torch.save({'state_dict':weights,'features':FEATURES,'architecture':[10,96,96,2],
                'source':'synthetic','production_rig':False,'shape_counts':[N_TRAIN,N_VAL,N_SHAPES-N_TRAIN-N_VAL],
                'optimizer_steps':args.steps,'best_val_mse':best},out/'synthetic_eye_v02_1000.pt')
    np.savez_compressed(out/'synthetic_eye_v02_1000_predictions.npz',parameter_values=p,
        vertices_rest=rest[test],vertices_target=truth,vertices_predicted=pred,
        vertices_linear=linear,test_shape_ids=np.arange(N_TRAIN+N_VAL,N_SHAPES))
    print('Saved:',out/'synthetic_eye_v02_1000.pt')
    print('Saved:',out/'synthetic_eye_v02_1000_predictions.npz')
    print('This tests new synthetic shapes, not real character rigging.')

if __name__=='__main__': main()
