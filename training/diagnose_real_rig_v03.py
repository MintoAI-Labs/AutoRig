"""MintoAI V0.3 gate/deformation diagnostics. Local authorized data only.
SPDX-License-Identifier: AGPL-3.0-only
No training or checkpoint modification. Oracle masks are diagnostic only.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from train_real_rig_v01 import ROOT, PARAMS, target
from train_real_rig_v03 import VisualRig, load_visual

THRESHOLDS=(0., .2, .35, .5, .65, .8, .9, .95, 1.01)

@torch.no_grad()
def inspect(net, record, scale, thresholds, device, poses=21):
    h, logits = net.encode(record)
    prob = logits.sigmoid()
    active = record['mesh_active'].bool()
    moving = record['moving'].bool()
    mesh = record['mesh']
    masks={t:prob>=t for t in thresholds}
    # Compare learned soft amplitude with oracle selection, and raw decoder with oracle.
    weights={t:(prob*masks[t])[mesh,None] for t in thresholds}
    weights['oracle_soft']=(prob*active)[mesh,None]
    weights['oracle_raw']=active[mesh,None].float()
    errors={k:[] for k in weights}; baseline=[]
    def metrics(d):
        return [float(d.mean()),float(d[moving].mean()) if moving.any() else float('nan')]
    for i in np.unique(np.linspace(0,len(record['t'])-1,poses).round().astype(int)):
        y=target(record,i,device)
        raw=net.deform(h,record['t'][i])*scale
        baseline.append(metrics(y.norm(dim=-1)))
        for k,w in weights.items():
            errors[k].append(metrics((raw*w-y).norm(dim=-1)))
    out={}
    for t,chosen in masks.items():
        tp=int((chosen&active).sum()); fp=int((chosen&~active).sum()); fn=int((~chosen&active).sum())
        out[t]={'tp':tp,'fp':fp,'fn':fn,'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'f1':2*tp/max(2*tp+fp+fn,1),'error':np.mean(errors[t],axis=0)}
    return out,np.mean(baseline,axis=0),{k:np.mean(errors[k],axis=0) for k in ('oracle_soft','oracle_raw')}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/real_rig_v03.pt')
    p.add_argument('--data',type=Path,default=ROOT/'dataset/processed_v02')
    p.add_argument('--models',type=Path,default=ROOT/'models')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--confirm-authorized',action='store_true')
    a=p.parse_args()
    if not a.confirm_authorized: p.error('Confirm permission covers this local analysis.')
    device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available(): p.error('CUDA unavailable')
    if device.type=='cpu': torch.set_num_threads(min(4,torch.get_num_threads()))
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if ck['architecture']!='visual_sparse_v03': p.error('Expected visual_sparse_v03 checkpoint')
    net=VisualRig(ck['width']).to(device); net.load_state_dict(ck['model_state']); net.eval()
    audit=ROOT/'dataset/visual_audit_v03'; audit.mkdir(parents=True,exist_ok=True)
    records=load_visual(a.data,a.models,device,audit)
    val=[r for r in records if r['source'] in ck['validation_sources']]
    if not val: p.error('Missing validation records')
    print(f"Checkpoint step={ck['best_step']} threshold={ck['threshold']} | validation only: {ck['validation_sources']}")
    ts=sorted(set(THRESHOLDS+(float(ck['threshold']),)))
    for r in sorted(val,key=lambda r:PARAMS.index(r['param'])):
        table,base,oracle=inspect(net,r,ck['scales'][r['param']],ts,device)
        print(f"\n{r['source']} {r['param']} | zero all/moving {base[0]:.8f} / {base[1]:.8f}")
        print(' threshold  TP  FP  FN  recall  precision  F1     error all   error moving')
        for t,q in table.items():
            e=q['error']; print(f" {t:9.2f} {q['tp']:3d} {q['fp']:3d} {q['fn']:3d} {q['recall']:7.3f} {q['precision']:10.3f} {q['f1']:6.3f} {e[0]:11.8f} {e[1]:13.8f}")
        best=max(table,key=lambda t:(table[t]['f1'],-table[t]['fp']))
        print(f' Diagnostic best F1 threshold={best:.2f} (not deployed)')
        for k,e in oracle.items(): print(f' {k}: all {e[0]:.8f} moving {e[1]:.8f}')
    print('\nOracle uses true mesh labels for diagnosis only. No checkpoint changed. Ren not evaluated or used for threshold selection.')

if __name__=='__main__': main()
