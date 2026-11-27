"""Joint training with two checkpoint teachers, PSD2Live pseudo-labels and experiment memory."""
import argparse
import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experience_memory import Experience, POLICIES
from algorithm_rules import inspect_algorithms
from training_progress import Progress, StopFlag, StopTraining, atomic_save, cpu_copy
from train_real_rig_v04 import ParameterRig
from unified_rig import UnifiedRig, CONTROLS
from unified_sources import (file_hash, tree_hash, family, load_records, synthetic_records,
                             make_ext_teacher, to_device, extreme_prediction, DeviceCache)

ROOT=Path(__file__).resolve().parent.parent


def masked_mean(values, mask):
    return (values*mask).sum()/mask.sum().clamp_min(1)


def objectives(net, r, indices, scale, policy, teachers, device, dtype):
    r['motion_scale']=scale
    values=r['t'][indices]
    target=torch.as_tensor(r['delta'][indices],device=device,dtype=dtype)/scale
    prediction,raw,logits=net.pose_batch(r,values)
    residual=prediction-target
    square=residual.square().sum(-1)
    active=r['moving'].to(dtype)[None].expand_as(square)
    moving=masked_mean(square,active)
    static=masked_mean(square,1-active)
    k=max(1,math.ceil(square.shape[-1]*.01))
    worst=square.topk(k,dim=-1).values.mean()
    src,dst=r['edge']
    edge=(residual[:,src]-residual[:,dst]).square().mean()
    gate=F.binary_cross_entropy_with_logits(logits,r['mesh_active'])
    weights=POLICIES[policy]
    # Raw decoder supervision prevents an initially closed gate from blocking geometry learning.
    loss=weights[0]*moving+weights[1]*worst+weights[2]*edge+weights[3]*static
    loss=loss+.2*(raw-target).square().mean()+.2*gate
    agreement=None
    with torch.no_grad():
        if r['teacher_kind']=='real':
            teacher=teachers['real']
            h,teacher_logits=teacher.encode(r)
            gate_mask=(teacher_logits.sigmoid()>=teachers['thresholds'].get(r['param'],.5))[r['mesh'],None]
            tfeatures=torch.stack([values,values.abs(),values.square()],-1)
            taught=teacher.heads[r['param']](torch.cat([
                h[None].expand(len(values),-1,-1),
                tfeatures[:,None].expand(-1,len(h),-1)],-1))*values[:,None,None]*gate_mask
        elif r['teacher_kind']=='extreme':
            taught=extreme_prediction(teachers['extreme'],r,indices,device,dtype)/scale
        else:
            taught=None
        if taught is not None:
            disagreement=(taught-target).square().mean()/target.square().mean().clamp_min(1e-12)
            agreement=float(disagreement.sqrt())
            teacher_error=(taught-target).square().sum(-1)
            student_error=(prediction.detach()-target).square().sum(-1)
            trust=.1*(teacher_error<student_error*.95).to(dtype)/(1+disagreement)
    if taught is not None:
        loss=loss+(trust*(prediction-taught).square().sum(-1)).mean()
    # PSD2Live output is only a low-weight proposal in a dedicated trial. Human validation judges it.
    if r['domain']=='psd2live':
        loss=loss*.02
    difficulty=square.detach().mean(-1).cpu().tolist()
    return loss,difficulty,dict(moving=float(moving.detach()),static=float(static.detach()),
                                top=float(worst.detach()),edge=float(edge.detach()),
                                teacher_disagreement=agreement)


@torch.no_grad()
def assess(net, records, scales, device, dtype, stop, pose_batch):
    net.eval()
    totals={}
    for record in records:
        stop.check()
        r=to_device(record,device,dtype)
        scale=scales[r['param']]
        r['motion_scale']=scale
        encoded=net.encode(r)
        indices=r['exam']
        domain=totals.setdefault(r['domain'],dict(squared=0.,count=0,max=0.,static=0.,static_count=0))
        for start in range(0,len(indices),pose_batch):
            stop.check()
            ids=indices[start:start+pose_batch]
            pred,_,_=net.pose_batch(r,r['t'][ids],encoded,hard_gate=True)
            target=torch.as_tensor(r['delta'][ids],device=device,dtype=dtype)/scale
            residual=pred-target
            if not torch.isfinite(residual).all():
                raise ValueError('Nonfinite validation residual')
            domain['squared']+=float(residual.square().sum())
            domain['count']+=residual.numel()
            domain['max']=max(domain['max'],float(residual.abs().amax()))
            mask=~r['moving']
            domain['static']+=float(residual[:,mask].square().sum())
            domain['static_count']+=residual[:,mask].numel()
    result={}
    for name,values in totals.items():
        result[name]=dict(rms=math.sqrt(values['squared']/max(values['count'],1)),
                          max=values['max'],
                          static=math.sqrt(values['static']/max(values['static_count'],1)))
        result[name]['quality']=.5*(result[name]['rms']+result[name]['max'])
    return result


def tune_batch(net, record, scales, device, dtype, stop, maximum):
    if device.type!='cuda':
        return min(8,maximum),[]
    r=to_device(record,device,dtype)
    r['motion_scale']=scales[r['param']]
    results=[]
    for batch in [v for v in (4,8,16,32,64,128,256) if v<=maximum]:
        stop.check()
        if batch>len(record['practice']):
            continue
        net.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        try:
            for _ in range(2):
                ids=(record['practice']*((batch//len(record['practice']))+1))[:batch]
                prediction,raw,logits=net.pose_batch(r,r['t'][ids])
                target=torch.as_tensor(r['delta'][ids],device=device,dtype=dtype)/scales[r['param']]
                loss=(prediction-target).square().mean()+raw.square().mean()*.01+logits.square().mean()*.001
                loss.backward()
                net.zero_grad(set_to_none=True)
                del prediction,raw,logits,loss,target
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-started
            free,total=torch.cuda.mem_get_info()
            peak=torch.cuda.max_memory_reserved()
            if free<512*1024**2 or peak>total*.8:
                break
            results.append(dict(batch=batch,poses_per_second=2*batch/max(elapsed,1e-9),
                                peak_reserved_mib=peak/1024**2))
            print('PROFILE',results[-1],flush=True)
        except torch.cuda.OutOfMemoryError:
            net.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            break
    if not results:
        return 1,[]
    return max(results,key=lambda row:row['poses_per_second'])['batch'],results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--psd2live',type=Path,help='Local source checkout for the algorithm curriculum')
    p.add_argument('--real-checkpoint',type=Path,default=ROOT/'checkpoints/real_rig_curriculum.pt')
    p.add_argument('--extreme-checkpoint',type=Path,default=ROOT/'checkpoints/fullbody_extreme_v04.pt')
    p.add_argument('--data',type=Path)
    p.add_argument('--models',type=Path,default=ROOT/'models')
    p.add_argument('--psd-index',type=Path,help='Optional candidate labels, disabled by default')
    p.add_argument('--output',type=Path,default=ROOT/'checkpoints/unified_rig_v06.pt')
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--precision',choices=('float32','float64'),default='float64')
    p.add_argument('--pose-batch',default='auto')
    p.add_argument('--cache-graphs',type=int,default=4)
    p.add_argument('--max-pose-batch',type=int,default=128)
    p.add_argument('--eval-every',type=int,default=500)
    p.add_argument('--steps',type=int,default=0)
    p.add_argument('--synthetic-shapes',type=int,default=16)
    p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--mastery',type=float,default=1e-9)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--confirm-authorized',action='store_true')
    args=p.parse_args()
    if not args.confirm_authorized:
        p.error('Confirm authorization for the datasets and both checkpoint teachers')
    if (args.steps<0 or min(args.eval_every,args.synthetic_shapes,args.max_pose_batch,args.cache_graphs)<1
            or not 0<=args.mastery<1 or not math.isfinite(args.lr) or args.lr<=0):
        p.error('Invalid training settings')
    if args.pose_batch!='auto' and (not args.pose_batch.isdigit() or int(args.pose_batch)<1):
        p.error('--pose-batch must be auto or a positive integer')
    if args.output.exists()!=args.resume:
        p.error('Choose a fresh output or pass --resume')
    saved=torch.load(args.output,map_location='cpu',weights_only=True) if args.resume else None
    if saved:
        for key in ('precision','seed','synthetic_shapes','mastery','eval_every','lr'):
            if key in saved['config']:
                setattr(args,key,saved['config'][key])
    stop=StopFlag()
    device=torch.device(args.device)
    dtype=getattr(torch,args.precision)
    if device.type=='cpu':
        torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng=random.Random(args.seed)
    real_ck=torch.load(args.real_checkpoint,map_location='cpu',weights_only=True)
    ext_ck=torch.load(args.extreme_checkpoint,map_location='cpu',weights_only=True)
    if real_ck.get('architecture')!='parameter_visual_v04':
        p.error('Real checkpoint must be from the ParameterRig V0.4/curriculum trainer')
    if args.data is None:
        cfg=real_ck.get('curriculum_settings',{})
        if 'data' not in cfg:
            p.error('--data is required when the real teacher checkpoint has no saved snapshot')
        args.data=Path(cfg['data'])
        args.models=Path(cfg['models'])
    recipe_seed=saved['recipe_seed'] if saved else args.seed+10000
    width=int(real_ck['width'])
    if width%4:
        p.error('Teacher width must be divisible by four for layer attention')
    teacher_real=ParameterRig(width).to(device=device,dtype=dtype)
    teacher_real.load_state_dict(real_ck['model_state'])
    teacher_real.eval().requires_grad_(False)
    teacher_ext=make_ext_teacher(ext_ck).to(device=device,dtype=dtype).requires_grad_(False)
    net=UnifiedRig(width).to(device=device,dtype=dtype)
    net.initialize_real(real_ck['model_state'])
    curriculum=(inspect_algorithms(args.psd2live) if args.psd2live
                else saved['config'].get('algorithm_curriculum') if saved else None)
    if curriculum:
        net.rule_fields.initialize_priors(curriculum['priors'])
    opt=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=1e-6)
    teachers=dict(real=teacher_real,extreme=teacher_ext,thresholds=real_ck.get('thresholds',{}))
    scales={key:float(real_ck['scales'].get(key,.1)) for key in CONTROLS}
    config=dict(architecture='unified_visual_v06',width=width,precision=args.precision,
                real_teacher=file_hash(args.real_checkpoint),extreme_teacher=file_hash(args.extreme_checkpoint),
                snapshot=tree_hash(args.data),seed=args.seed,synthetic_shapes=args.synthetic_shapes,
                mastery=args.mastery,eval_every=args.eval_every,lr=args.lr,controls=CONTROLS,
                data_recipe='body-v06-1',moving_labels='all_nonzero',criterion='human_anchor_v1',rule_library='rotation_tps_v1',
                algorithm_curriculum=curriculum,label_sources=dict(real='provided_rig',
                extreme='procedural_truth_and_checkpoint_teacher',psd2live='optional_candidate_only'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if curriculum:
        path=args.output.with_suffix('.algorithm_curriculum.json')
        path.write_text(json.dumps(curriculum,ensure_ascii=False,indent=2),encoding='utf-8')
    audit=args.output.parent/'unified_visual_audit'
    print('Loading real records on CPU; selected graphs move to the GPU for batched poses.',flush=True)
    records=load_records(args.data,args.models,audit/'real','real',stop)
    val_name=family(real_ck.get('curriculum_settings',{}).get('validation','Hiyori'))
    test_name=family(real_ck.get('curriculum_settings',{}).get('holdout','zundamon'))
    train=[r for r in records if family(r['source']) not in (val_name,test_name)]
    validation=[r for r in records if family(r['source'])==val_name]
    test=[r for r in records if family(r['source'])==test_name]
    if not train or not validation or not test or val_name==test_name:
        p.error('Real source must contain distinct training, validation and test character families')
    if not all(value>0 and math.isfinite(value) for value in scales.values()):
        p.error('Teacher scales must be finite and positive')
    fixed_real_count=len(train)
    train+=synthetic_records(recipe_seed,args.synthetic_shapes)
    validation+=synthetic_records(args.seed+20000,2,'validation')
    index_identity=None
    if args.psd_index is not None and args.psd_index.is_file():
        index=json.loads(args.psd_index.read_text(encoding='utf-8'))
        cases=index.get('cases',[])
        if not cases:
            p.error('PSD2Live index has no accepted cases')
        identities=[]
        for case in cases:
            if not case.get('complete'):
                continue
            data,models=Path(case['data']),Path(case['models'])
            identities.append((case['id'],tree_hash(data)))
            rows=load_records(data,models,audit/case['id'],'psd2live',stop)
            for r in rows:
                r['id']=case['id']+'/'+r['id']
            # These are candidate labels only; human rigs judge every experiment.
            train+=rows
        if not identities:
            p.error('PSD2Live index contains no complete usable cases')
        index_identity=identities
    elif args.psd_index is not None:
        p.error('The explicitly requested PSD2Live proposal index does not exist')
    else:
        print('PSD2Live output labels disabled. Human rigs remain the reference.',flush=True)
    config['psd_cases']=index_identity
    appearance_hash=hashlib.sha256()
    for record in [*train,*validation]:
        if record['domain']!='extreme':
            appearance_hash.update(record['id'].encode())
            appearance_hash.update(record['visual'].numpy().tobytes())
    config['appearance_hash']=appearance_hash.hexdigest()
    if saved and saved['config']!=config:
        p.error('Resume requires unchanged teacher files, data, precision and curriculum configuration')
    phase=saved.get('phase','fixed') if saved else 'fixed'
    step=0
    trial_number=0
    streak=0
    best=None
    hard={}
    latest=None
    if saved:
        net.load_state_dict(saved['model_state'])
        net.rule_mode=saved.get('rule_mode','affine')
        opt.load_state_dict(saved['optimizer'])
        rng.setstate(saved['rng'])
        step=saved['step']
        trial_number=saved['trial_number']
        streak=saved['streak']
        best=saved['best']
        hard=saved['hard']
        latest=saved.get('latest_metrics')
    memory=Experience(args.output.with_suffix('.experience.sqlite'),
                      saved.get('arms') if saved else None)
    memory.log(step,'resume' if saved else 'initialize',teacher_steps={
        'real':int(real_ck.get('steps',0)),'extreme':int(ext_ck.get('step',0))},
        teacher_hashes=[config['real_teacher'],config['extreme_teacher']],
        psd_cases=len(index_identity or []))
    progress=Progress(args.output,'unified')
    context=saved.get('context','moving_vertices') if saved else 'moving_vertices'
    policy=saved['policy'] if saved else memory.select(0,context,allow_psd=bool(index_identity))
    pose_batch=(saved['pose_batch'] if saved else None)
    if pose_batch is None:
        probe=max((r for r in train if r['domain']=='real'),key=lambda r:len(r['feat']))
        pose_batch,profile=(tune_batch(net,probe,scales,device,dtype,stop,args.max_pose_batch)
                            if args.pose_batch=='auto' else (int(args.pose_batch),[]))
        memory.log(step,'profile',device=args.device,precision=args.precision,
                   selected_pose_batch=pose_batch,results=profile)
    cache=DeviceCache(device,dtype,args.cache_graphs)
    domains=sorted({r['domain'] for r in train})
    by_domain={domain:[i for i,r in enumerate(train) if r['domain']==domain] for domain in domains}

    def save():
        atomic_save(dict(architecture='unified_visual_v06',width=width,controls=CONTROLS,
                         model_state=net.state_dict(),rule_mode=net.rule_mode,optimizer=opt.state_dict(),scales=scales,
                         rng=rng.getstate(),step=step,config=config,phase=phase,recipe_seed=recipe_seed,
                         trial_number=trial_number,streak=streak,best=best,hard=hard,
                         arms=memory.arms,context=context,policy=policy,pose_batch=pose_batch,
                         latest_metrics=latest),args.output)

    def evaluate():
        return assess(net,validation,scales,device,dtype,stop,min(pose_batch,16))

    def promote_best(metrics):
        return dict(model=cpu_copy(net.state_dict()),optimizer=cpu_copy(opt.state_dict()),
                    metrics=metrics,rule_mode=net.rule_mode,step=step)

    def export_best():
        atomic_save(dict(architecture='unified_visual_v06',width=width,controls=CONTROLS,
                         model_state=best['model'],rule_mode=best['rule_mode'],scales=scales,precision=args.precision,
                         metrics=best['metrics'],config=config,step=best['step']),
                    args.output.with_name(args.output.stem+'.best.pt'))

    def report(metrics,action,seconds,diagnosis,algorithm=None):
        flattened={f'{domain}/{key}':value for domain,row in metrics.items()
                   for key,value in row.items()}
        progress.write(step=step,round=1,current=0,total=1,task='joint_training',phase=phase,
                       value=metrics['real']['max'],
                       second=metrics['extreme']['max'],
                       best=best['metrics']['real']['max'],
                       details=flattened,limits=[args.mastery],passed=streak,passes=3,
                       lr=opt.param_groups[0]['lr'],action=action,attempts=trial_number,
                       eval_seconds=seconds,pose_batch=pose_batch,policy=policy,algorithm=algorithm or net.rule_mode,diagnosis=diagnosis)
        return progress.last

    end=step+args.steps if args.steps else None
    try:
        if best is None:
            started=time.monotonic()
            metrics=evaluate()
            best=promote_best(metrics)
            latest=report(metrics,'best',time.monotonic()-started,'baseline')
            export_best()
            save()
        while end is None or step<end:
            stop.check()
            # Stratification ensures every domain is learned; extra replay prioritizes practice errors.
            active_domains=[domain for domain in domains if domain!='psd2live' or policy=='psd_hint']
            domain=active_domains[step%len(active_domains)]
            replay=[(key,value) for key,value in hard.items()
                    if train[int(key.split(':')[0])]['domain']==domain]
            if replay and rng.random()<.4:
                key=rng.choices([x[0] for x in replay],weights=[x[1]+.01 for x in replay])[0]
                record_index,anchor=map(int,key.split(':'))
            else:
                record_index=rng.choice(by_domain[domain])
                anchor=None
            record=train[record_index]
            r=cache.get(record)
            ids=rng.sample(record['practice'],min(pose_batch,len(record['practice'])))
            if anchor is not None:
                ids[0]=anchor
            net.rule_mode={'rule_rotation':'rotation','rule_tps':'tps',
                           'rule_hybrid':'hybrid'}.get(policy,'affine')
            net.train()
            try:
                loss,difficulties,parts=objectives(net,r,ids,scales[r['param']],
                                                  policy,teachers,device,dtype)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),1.)
            except torch.cuda.OutOfMemoryError:
                if device.type!='cuda' or pose_batch==1:
                    raise
                opt.zero_grad(set_to_none=True)
                loss=None
                gc.collect()
                cache.clear()
                torch.cuda.empty_cache()
                pose_batch=max(1,pose_batch//2)
                memory.log(step,'oom_retry',pose_batch=pose_batch)
                continue
            opt.step()
            step+=1
            for pose,error in zip(ids,difficulties):
                key=f'{record_index}:{pose}'
                hard[key]=.8*hard.get(key,error)+.2*error
            hard=dict(sorted(hard.items(),key=lambda item:item[1],reverse=True)[:256])
            loss=None
            r=None
            if step%args.eval_every and step!=end:
                continue
            started=time.monotonic()
            metrics=evaluate()
            trial_algorithm=net.rule_mode
            previous_human=dict(best['metrics']['real'])
            old_quality=best['metrics']['real']['quality']
            quality=metrics['real']['quality']
            reward=(old_quality-quality)/max(old_quality,1e-12)
            memory.feedback(policy,reward,trial_number,context)
            protected=(metrics['real']['max']<=best['metrics']['real']['max']+1e-15
                       and all(metrics['real'][key]<=best['metrics']['real'][key]*1.001+1e-15
                           for key in ('rms','max','static'))
                       and metrics['extreme']['quality']<=best['metrics']['extreme']['quality']*1.02+1e-15)
            if reward>.001 and protected:
                best=promote_best(metrics)
                action='best'
                export_best()
            elif reward>=-1e-6 and protected:
                action='trial'
            else:
                net.load_state_dict(best['model'])
                net.rule_mode=best['rule_mode']
                opt.load_state_dict(best['optimizer'])
                for group in opt.param_groups:
                    group['lr']=max(1e-12,args.lr*.8**(trial_number//len(POLICIES))
                                    *(.5 if reward<0 else 1.))
                action='rollback'
            diagnosis=('static_drift' if max(row['static'] for row in metrics.values())>
                       .3*max(row['rms'] for row in metrics.values()) else 'moving_vertices')
            memory.log(step,'experiment',policy=policy,context=context,reward=reward,accepted=action=='best',
                       metrics=metrics,practice_components=parts,algorithm=trial_algorithm,diagnosis=diagnosis)
            streak=streak+1 if action!='rollback' and max(row['max'] for row in metrics.values())<=args.mastery else 0
            lesson=dict(step=step,hypothesis=dict(policy=policy,algorithm=trial_algorithm,context=context),
                        evidence=dict(human_before=previous_human,human_after=metrics['real'],
                                      extreme_after=metrics['extreme']),
                        decision=action,reward=reward)
            with args.output.with_suffix('.lessons.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(lesson,ensure_ascii=False,allow_nan=False)+'\n')
            latest=report(metrics,action,time.monotonic()-started,diagnosis,trial_algorithm)
            trial_number+=1
            context=diagnosis
            policy=memory.select(trial_number,context,allow_psd=bool(index_identity))
            if streak>=3:
                streak=0
                phase='random' if phase=='fixed' else 'fixed'
                recipe_seed=rng.randrange(1,2**31) if phase=='random' else args.seed+10000
                cache.clear()
                replacement=synthetic_records(recipe_seed,args.synthetic_shapes)
                train[fixed_real_count:fixed_real_count+len(replacement)]=replacement
                hard={k:v for k,v in hard.items() if train[int(k.split(':')[0])]['domain']!='extreme'}
                memory.log(step,'phase',phase=phase,recipe_seed=recipe_seed)
            save()
    except StopTraining:
        print('Stop requested; saving the latest completed update.',flush=True)
    finally:
        save()
        memory.close()
        print(f'Saved unified step {step}: {args.output}',flush=True)


if __name__=='__main__':
    main()
