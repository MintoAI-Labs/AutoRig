"""Compare frozen and learned eye gates under the locked projected-replay schedule."""
import argparse
import copy
import json
from pathlib import Path

from codex_replay_gradient import (
    DTYPE, EYES, ROOT, ParameterRig, StopFlag, acceptance, evaluate, file_hash,
    load_lesson, passed, practice_only, prepare, step_gradient,
)
from codex_replay_lr_audit import MILESTONES
from train_real_rig_v01 import mean_by
from train_real_rig_v02 import balanced_mean
from training_progress import atomic_save
from unified_sources import family

import numpy as np
import torch
from torch.nn import functional as F

PARAM = EYES[1]


def drawable_mask(vertices, row):
    return mean_by(vertices.to(DTYPE)[:, None], row['mesh'], row['nm']).squeeze(1) > 0


def motion(row, pose_ids, cutoff=0.):
    peak = np.linalg.norm(row['delta'][pose_ids], axis=2).max(0)
    return torch.as_tensor(peak > cutoff, device=row['mesh'].device)


def gate_batch(batch):
    row, features, *_ = batch
    mesh_features = mean_by(features, row['mesh'], row['nm'])
    inputs = torch.cat((mesh_features, mesh_features.mean(0,keepdim=True).expand_as(mesh_features)), 1)
    return inputs, drawable_mask(row['moving'], row).to(DTYPE)


def gate_loss(gate, batch, _indices):
    inputs, labels = batch
    return .5 * balanced_mean(F.binary_cross_entropy_with_logits(
        gate(inputs).squeeze(-1), labels, reduction='none'), labels)


@torch.no_grad()
def apply_gate(batch, gate, inputs, threshold):
    row, features, _, poses, labels = batch
    mask = (gate(inputs).squeeze(-1).sigmoid() >= threshold)[row['mesh'], None]
    return row, features, mask, poses, labels


def confusion(pred, truth):
    return dict(tp=int((pred & truth).sum()), fp=int((pred & ~truth).sum()),
                fn=int((~pred & truth).sum()), tn=int((~pred & ~truth).sum()))


def rates(counts):
    tp, fp, fn = (counts[k] for k in ('tp','fp','fn'))
    return dict(counts, precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None)


def label_audit(row, practice_ids):
    full=list(range(len(row['delta'])))
    exact=motion(row,full)
    historical=motion(row,full,1e-5)
    practice=motion(row,practice_ids)
    practice_historical=motion(row,practice_ids,1e-5)
    labels=drawable_mask(exact,row)
    return dict(source=row['source'], vertices=len(exact), exact_motion=int(exact.sum()),
                historical_motion=int(historical.sum()),
                legacy_cutoff_changed_vertices=int((exact!=historical).sum()),
                practice_missing_vertices=int((exact & ~practice).sum()),
                practice_missing_drawables=int((labels & ~drawable_mask(practice,row)).sum()),
                historical_practice_missing_drawables=int((drawable_mask(historical,row) &
                                                            ~drawable_mask(practice_historical,row)).sum()),
                exact_active_drawables=int(labels.sum()),drawables=row['nm'],
                static_vertices_inside_active_drawables=int((labels[row['mesh']] & ~exact).sum()))


@torch.no_grad()
def spatial_metrics(net, batches, scale):
    sums={region:0. for region in ('moving','static','all')}
    counts={region:0 for region in sums}
    maxima={region:0. for region in sums}
    coverage={kind:dict(tp=0,fp=0,fn=0,tn=0) for kind in ('vertex','drawable')}
    for row,h,gate,t,target in batches:
        from codex_blink_lesson import geometry
        moving=row['moving']
        predicted=gate.squeeze(-1)
        drawable_pred=drawable_mask(predicted,row)
        for key,label_mask,pred in [('vertex',moving,predicted),
                                    ('drawable',drawable_mask(moving,row),drawable_pred)]:
            for k,v in confusion(pred,label_mask).items():
                coverage[key][k]+=v
        for lo in range(0,len(t),8):
            idx=slice(lo,lo+8)
            err=((geometry(net.heads[PARAM],h,t[idx])*gate-target[idx])*scale)
            for region,mask in [('moving',moving),('static',~moving),('all',torch.ones_like(moving))]:
                part=err[:,mask]
                if part.numel():
                    sums[region]+=float(part.square().sum())
                    counts[region]+=part.numel()
                    maxima[region]=max(maxima[region],float(part.abs().max()))
    return dict(coverage={k:rates(v) for k,v in coverage.items()},
                error_percent_model={r:dict(rms=100*(sums[r]/counts[r])**.5,max=100*maxima[r])
                                     if counts[r] else None for r in sums})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    root=ROOT/'checkpoints'
    best_path=root/'real_rig_curriculum.best.pt'
    historical=root/'codex_gradient_seed42_step50/projected.step050.candidate.pt'
    schedule_path=root/'codex_replay_trial_02/lesson.json'
    paths=[best_path,historical,schedule_path]
    hashes=[file_hash(p) for p in paths]
    prior=json.loads(schedule_path.read_text())
    assert hashes[0]==prior['checkpoint_hashes']['best']
    ck=torch.load(best_path,weights_only=True,map_location='cpu')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(prior['seed'])
    device=torch.device('cuda')
    stop=StopFlag()
    base=ParameterRig(ck['width']).to(device=device,dtype=DTYPE)
    base.load_state_dict(ck['model_state'])
    base.eval()
    practice,validation,excluded,digest=load_lesson(ck['curriculum_settings'],device)
    assert digest==prior['lesson_input_hash'] and excluded==prior['excluded_test_groups']
    old,=(r for r in practice if r['param']==PARAM and r['group']==ck['curriculum_settings']['groups'][0])
    old_ids,exam=prior['old_replay_pose_indices'],prior['old_exam_pose_indices']
    old_batch,=prepare(base,[practice_only(dict(old,exam=exam),old_ids)],ck,exam=False)
    new={r['group']:r for r in practice if r['param']==PARAM and family(r['source'])!=family(old['source'])}
    batches={}
    for key in dict.fromkeys(s['group'] for s in prior['schedule']):
        row=new[key]
        batches[key],=prepare(base,[practice_only(row,row['practice'])],ck,exam=False)
    old_g=gate_batch(old_batch)
    new_g={key:gate_batch(batch) for key,batch in batches.items()}
    old_eval=dict(old,exam=exam,moving=motion(old,list(range(len(old['t'])))))
    threshold=ck['thresholds'][PARAM]
    scale=ck['scales'][PARAM]
    audits=dict(old=label_audit(old,old_ids),train=[label_audit(r,r['practice']) for r in new.values()],
                hiyori=[label_audit(r,r['practice']) for r in validation if r['param']==PARAM])
    assert len(prior['schedule'])==100

    def measure(net):
        valid=prepare(net,validation,ck,exam=True)
        scores=evaluate(net,ck,valid,old,old_ids,exam,stop)
        calibrated=dict(ck,thresholds=dict(ck['thresholds'],**{PARAM:scores['old_gate']}))
        a=spatial_metrics(net,[b for b in valid if b[0]['param']==PARAM],scale)
        b=spatial_metrics(net,prepare(net,[old_eval],calibrated,exam=True),scale)
        return dict(primary=scores,hiyori_right_eye=a,old_eye=b)

    baseline=measure(base)
    print('Baseline',baseline['hiyori_right_eye'],audits['old'],flush=True)
    curves={}
    candidates=[]
    for arm in ('head_only','head_and_gate'):
        model=copy.deepcopy(base)
        head,gate=model.heads[PARAM],model.gates[PARAM]
        optimizer=torch.optim.AdamW(head.parameters(),lr=prior['lr'])
        gate_optimizer=torch.optim.AdamW(gate.parameters(),lr=prior['lr'])
        points=[]
        for step,item in enumerate(prior['schedule'],1):
            stop.check()
            key=item['group']
            old_idx=[old_ids.index(i) for i in item['replay_pose_indices']]
            new_idx=[new[key]['practice'].index(i) for i in item['new_pose_indices'][:2]]
            step_gradient(head,optimizer,
                          apply_gate(old_batch,gate,old_g[0],threshold),
                          apply_gate(batches[key],gate,new_g[key][0],threshold),
                          old_idx,new_idx,'projected')
            if arm=='head_and_gate':
                gate_step=step_gradient(gate,gate_optimizer,old_g,new_g[key],old_idx,new_idx,
                                        'projected',loss_function=gate_loss)
            if step not in MILESTONES:
                continue
            point=measure(model)
            checks=acceptance(point['primary'],baseline['primary'])
            point.update(step=step,checks=checks,accepted=passed(checks))
            if arm=='head_and_gate':
                point['gate_step']=gate_step
            points.append(point)
            print(arm,step,point['primary']['hiyori']['all']['rms'],
                  point['primary']['old_curriculum_max'],point['hiyori_right_eye']['coverage']['vertex'],
                  point['accepted'],flush=True)
            if point['accepted']:
                name=f'{arm}.step{step:03d}.candidate.pt'
                export={k:ck[k] for k in ('architecture','precision','width','scales','parameters',
                                           'thresholds','curriculum_settings')}
                export.update(model_state=model.state_dict(),head_optimizer=optimizer.state_dict(),
                              gate_optimizer=gate_optimizer.state_dict(),step=step,
                              rng=dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),
                              config=dict(arm=arm,seed=prior['seed'],lr=prior['lr'],schedule=prior['schedule'],
                                          source_hashes=hashes,lesson_input_hash=digest,
                                          gate_labels='train practice exact nonzero',gate_loss_weight=.5),
                              status='improvement_criteria_met_mastery_not_met',measurement=point)
                atomic_save(export,args.output/name)
                state=torch.load(args.output/name,map_location='cpu',weights_only=True)
                restored=copy.deepcopy(base)
                restored.load_state_dict(state['model_state'])
                verified=measure(restored)
                if any(verified[k]!=point[k] for k in verified):
                    raise RuntimeError('Candidate reload changed metrics')
                candidates.append(dict(arm=arm,step=step,path=name))
        curves[arm]=points
        prefixes=(f'heads.{PARAM}.', f'gates.{PARAM}.') if arm=='head_and_gate' else (f'heads.{PARAM}.',)
        for key,weight in model.state_dict().items():
            if not key.startswith(prefixes) and not torch.equal(weight,base.state_dict()[key]):
                raise RuntimeError(f'Unexpected parameter changed: {key}')
    assert [file_hash(p) for p in paths]==hashes
    report=dict(seed=prior['seed'],steps=100,validation_records=len(validation),excluded_test_groups=excluded,
                checkpoint_hashes=hashes,label_audit=audits,baseline=baseline,curves=curves,candidates=candidates)
    (args.output/'lesson.json').write_text(json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf8')


if __name__=='__main__':
    main()
