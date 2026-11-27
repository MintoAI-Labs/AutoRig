"""Train on real Live2D snapshots with per-model progression."""
import argparse
import hashlib
import math
from pathlib import Path
import random
import signal
from train_thermal_supervisor import WorkerControl

import numpy as np
import torch
from PIL import Image
from train_real_rig_v03 import crop_mesh, load_record, prepare, resolve_textures, sheet

from train_real_rig_v04 import (
    PARAMS, ROOT, ParameterRig, choose, evaluate, loss_for,
    summarize, target, train_scales,
)


def mesh_patches_guarded(path, model_root, check):
    textures=[]
    for p in resolve_textures(path.parent,model_root):
        check()
        with Image.open(p) as im: textures.append(im.convert('RGBA'))
    with np.load(path,allow_pickle=False) as z:
        uv=z['uv'].reshape(-1,2); tri=z['triangles']; off=z['mesh_offsets']; tx=z['texture_indices']; ids=z['mesh_ids']
    if len(tx)!=len(off)-1 or len(ids)!=len(tx) or np.any(tx<0) or np.any(tx>=len(textures)):
        raise ValueError(f'Invalid texture mapping: {path.name}')
    patches=[]
    for i,(a,b) in enumerate(zip(off[:-1],off[1:])):
        if i % 8 == 0:
            check()
        tr=tri[(tri[:,0]>=a)&(tri[:,0]<b)]-a
        if np.any(tr<0) or np.any(tr>=b-a): raise ValueError('Invalid local triangle')
        patches.append(crop_mesh(textures[int(tx[i])],uv[a:b],tr))
    return np.stack(patches),ids

def load_visual_guarded(data, model_root, device, audit_dir, check):
    groups={}; records=[]
    for p in sorted(data.rglob('*.npz')):
        check()
        if p.parent.parent.resolve()!=data.resolve(): raise ValueError(f'Unexpected dataset layout: {p}')
        with np.load(p,allow_pickle=False) as z:
            # Compare rest-input metadata across parameter sweeps, not target displacements.
            signature=hashlib.sha256()
            for k in ('rest','uv','triangles','mesh_offsets','mesh_ids','texture_indices'):
                a=np.ascontiguousarray(z[k]); signature.update(a.tobytes())
            sig=signature.hexdigest(); source=str(z['source'].item())
        key=p.parent.name
        if key not in groups:
            patches,ids=mesh_patches_guarded(p,model_root,check)
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
    check()
    prepare(records)
    return records

def split_poses(record):
    # Check endpoints and alternating interior samples; train only the other poses.
    last = len(record['t']) - 1
    exam = sorted({0, last, *range(2, last, 2)})
    practice = list(range(1, last, 2))
    if not practice:
        raise ValueError(f"Need at least three poses: {record['source']} {record['param']}")
    return practice, exam


@torch.no_grad()
def exam_error(net, lessons, thresholds, scales, device):
    """Worst vertex displacement error, normalized by its parameter scale."""
    net.eval()
    worst = 0.0
    for record, _, exam in lessons:
        param = record['param']
        hidden, logits = net.encode(record)
        mask = (logits.sigmoid() >= thresholds[param])[record['mesh'], None]
        for index in exam:
            pred = net.deform(hidden, record['t'][index], param) * scales[param] * mask
            delta = pred - target(record, index, device)
            error = float(delta.abs().amax()) / scales[param]
            if not math.isfinite(error):
                return math.inf
            worst = max(worst, error)
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--models', type=Path, default=ROOT / 'models')
    parser.add_argument('--output', type=Path, default=ROOT / 'checkpoints/real_rig_curriculum.pt')
    parser.add_argument('--validation', default='Hiyori')
    parser.add_argument('--holdout', default='zundamon')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--mastery', type=float, default=.005,
                        help='Worst coordinate error relative to parameter scale; 0 requires exact equality')
    parser.add_argument('--round-factor', type=float, default=.8)
    parser.add_argument('--passes', type=int, default=3)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--steps', type=int, default=0, help='0 runs until interrupted')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--confirm-authorized', action='store_true')
    args = parser.parse_args()
    if not args.confirm_authorized:
        parser.error('Confirm authorization to use the model data for training')
    if (args.steps < 0 or min(args.passes, args.eval_every, args.width) < 1
            or not 0 <= args.mastery < 1 or not 0 < args.round_factor < 1
            or not math.isfinite(args.lr) or args.lr <= 0):
        parser.error('Invalid curriculum settings')
    if args.output.exists() != args.resume:
        parser.error('Choose a new output path or use --resume with an existing checkpoint')

    control = WorkerControl()
    stop = False

    def request_stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if not control.ready(lambda: stop):
        return
    device = torch.device(args.device)
    if device.type == 'cpu':
        torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    audit = args.output.parent / 'curriculum_visual_audit'
    audit.mkdir(parents=True, exist_ok=True)
    def check_startup():
        if not control.ready(lambda: stop, torch.cuda.empty_cache if device.type == 'cuda' else None):
            raise InterruptedError('Stopped during model preparation')

    try:
        records = load_visual_guarded(args.data, args.models, device, audit, check_startup)
    except InterruptedError as exc:
        print(exc, flush=True)
        return
    validation, held, groups = [], [], {}
    for record in records:
        name = Path(record['source']).stem.casefold()
        if name == args.validation.casefold():
            validation.append(record)
        elif name == args.holdout.casefold():
            held.append(record)
        else:
            groups.setdefault(record['group'], []).append(record)
    if ({r['param'] for r in validation} != set(PARAMS) or not held or not groups):
        parser.error('Need distinct validation and holdout groups, all seven validation parameters, and training data')
    order = sorted(groups, key=lambda key: (len(groups[key]), groups[key][0]['source'].casefold(), key))
    lessons = [[(r, *split_poses(r)) for r in groups[key]] for key in order]
    # Scales use practice poses only. They remain fixed as the curriculum tightens.
    practice_records = []
    for lesson in lessons:
        for record, practice, _ in lesson:
            practice_records.append(dict(record, delta=record['delta'][practice]))
    scales = train_scales(practice_records)
    settings = dict(validation=args.validation, holdout=args.holdout, width=args.width,
                    lr=args.lr, mastery=args.mastery, round_factor=args.round_factor,
                    passes=args.passes, eval_every=args.eval_every,
                    groups=order, data=str(args.data.resolve()),
                    models=str(args.models.resolve()))
    # Identify the actual snapshot contents, including parameters and source metadata.
    digest = hashlib.sha256()
    for path in sorted(args.data.glob('*/*')):
        if not control.ready(lambda: stop):
            print('Stopped during snapshot verification.', flush=True)
            return
        if path.is_file() and (path.suffix == '.npz' or path.name == 'source.json'):
            digest.update(path.relative_to(args.data).as_posix().encode())
            with path.open('rb') as stream:
                digest.update(hashlib.file_digest(stream, 'sha256').digest())
    settings['snapshot_sha256'] = digest.hexdigest()
    net = ParameterRig(args.width).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    step = current = streak = 0
    round_number, limit = 1, args.mastery
    thresholds = {p: .5 for p in PARAMS}
    if args.resume:
        state = torch.load(args.output, map_location='cpu', weights_only=True)
        if state.get('curriculum_settings') != settings:
            parser.error('Resume requires the same snapshot and curriculum settings')
        net.load_state_dict(state['model_state'])
        opt.load_state_dict(state['optimizer'])
        rng.setstate(state['rng'])
        step, current, streak = state['steps'], state['current'], state['streak']
        round_number, limit = state['round_number'], state['limit']
        thresholds = state['thresholds']

    def save():
        tmp = args.output.with_name(args.output.name + '.tmp')
        torch.save(dict(architecture='parameter_visual_v04', model_state=net.state_dict(),
                        optimizer=opt.state_dict(), rng=rng.getstate(), width=args.width,
                        scales=scales, parameters=PARAMS, thresholds=thresholds,
                        steps=step, current=current, streak=streak,
                        round_number=round_number, limit=limit,
                        curriculum_settings=settings), tmp)
        tmp.replace(args.output)
        print(f'Saved step {step}: {args.output}', flush=True)

    end = step + args.steps if args.steps else None
    while not stop and (end is None or step < end):
        if not control.ready(lambda: stop, torch.cuda.empty_cache if device.type == 'cuda' else None):
            break
        net.train()
        index = current
        if current and step % 4 == 3:
            index = rng.randrange(current)
        record, practice, _ = rng.choice(lessons[index])
        loss, _ = loss_for(net, record, rng.choice(practice), scales, device)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
        opt.step()
        step += 1
        if step % args.eval_every and step != end:
            continue
        thresholds, score = choose(evaluate(net, validation, scales, device))
        error = exam_error(net, lessons[current], thresholds, scales, device)
        streak = streak + 1 if error <= limit else 0
        name = lessons[current][0][0]['source']
        print(f'Step {step} | round {round_number} | model {current+1}/{len(lessons)} '
              f'{name} | exam {error:.8g} <= {limit:.8g} | pass {streak}/{args.passes} '
              f'| validation {score:.6f}', flush=True)
        if streak >= args.passes:
            current += 1
            streak = 0
            if current == len(lessons):
                summarize('Round holdout', evaluate(net, held, scales, device, thresholds), thresholds)
                current = 0
                round_number += 1
                limit *= args.round_factor
                print(f'ROUND {round_number} | first model | new limit {limit:.8g}', flush=True)
        save()
    save()


if __name__ == '__main__':
    main()