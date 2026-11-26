"""MintoAI V0.2: synthetic, character-disjoint eye deformation experiment."""
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'checkpoints'
FEATURES = ['rest_x','rest_y','upper_lid','eye_open','half_width',
            'upper_height','lower_height','tilt','asymmetry','local_x']

def make_data():
    rng = np.random.default_rng(2026)
    p = np.linspace(0,1,21,dtype=np.float32)
    local = np.linspace(-1,1,17,dtype=np.float32)
    arch = np.maximum(np.sqrt(np.maximum(0,1-local**2)),.04)
    features, rests, targets = [], [], []
    for _ in range(144):
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
    OUT.mkdir(parents=True,exist_ok=True)
    p,f,rest,target=make_data()
    delta=target-rest[:,None,:,:]
    # Whole shapes separated; no vertex/pose of a test eye enters training.
    train=slice(0,96); val=slice(96,120); test=slice(120,144)
    torch.manual_seed(42)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cpu': torch.set_num_threads(1)
    X=torch.from_numpy(f).to(device); Y=torch.from_numpy(delta).to(device)
    xt=X[train].reshape(-1,10); yt=Y[train].reshape(-1,2)
    xv=X[val]; yv=Y[val]
    net=Net().to(device); opt=torch.optim.AdamW(net.parameters(),lr=.002)
    best=float('inf'); weights=None
    print('MintoAI V0.2 - SYNTHETIC ONLY',flush=True)
    print('Device:',torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU',flush=True)
    print('Train eyes: 96 | Validation eyes: 24 | Test eyes: 24',flush=True)
    for step in range(1001):
        net.train(); ids=torch.randint(len(xt),(2048,),device=device)
        loss=(net(xt[ids])-yt[ids]).square().mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step%100==0:
            net.eval()
            with torch.no_grad(): vl=(net(xv)-yv).square().mean().item()
            if vl<best:
                best=vl; weights={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
            print(f'Step {step:4d} | Batch MSE: {loss.item():.8f} | Val MSE: {vl:.8f}',flush=True)
    net.load_state_dict(weights); net.eval()
    with torch.no_grad(): pred=net(X[test]).cpu().numpy()+rest[test,None,:,:]
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
                'source':'synthetic','production_rig':False},OUT/'synthetic_eye_v02.pt')
    np.savez_compressed(OUT/'synthetic_eye_v02_predictions.npz',parameter_values=p,
        vertices_rest=rest[test],vertices_target=truth,vertices_predicted=pred,
        vertices_linear=linear,test_shape_ids=np.arange(120,144))
    print('Saved: synthetic_eye_v02.pt and synthetic_eye_v02_predictions.npz')
    print('This tests new synthetic shapes, not real character rigging.')

if __name__=='__main__': main()
