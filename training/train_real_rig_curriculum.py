"""Train on real Live2D snapshots with per-model progression."""
import argparse
import hashlib
import math
from pathlib import Path
import random
import shutil
import signal

import time
import numpy as np
import torch
from training_progress import BestTrial, Progress, StopFlag, StopTraining, atomic_save

from train_real_rig_v04 import (
    PARAMS, ROOT, ParameterRig, choose, evaluate, load_visual, loss_for,
    summarize, target, train_scales,
)


def split_poses(record):
    # Check endpoints and alternating interior samples; train only the other poses.
    last = len(record['t']) - 1
    exam = sorted({0, last, *range(2, last, 2)})
    practice = list(range(1, last, 2))
    if not practice:
        raise ValueError(f"Need at least three poses: {record['source']} {record['param']}")
    return practice, exam


@torch.no_grad()
def assess(net, lessons, scales, device, stop):
    net.eval()
    scores, thresholds, hard = {}, {}, {}
    # Learn gate thresholds from practice-pose errors, including all prior skills.
    candidates = (.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95)
    costs = {}
    encoded = []
    for lesson_id, lesson in enumerate(lessons):
        for r, practice, exam in lesson:
            stop.check()
            h, logits = net.encode(r)
            probabilities = logits.sigmoid()
            masks = [(probabilities >= t)[r['mesh'], None] for t in candidates]
            param = r['param']
            totals = costs.setdefault(param, [0.] * len(candidates))
            worst_practice, worst_index = -1., practice[0]
            for i in practice:
                stop.check()
                raw = net.deform(h, r['t'][i], param) * scales[param]
                y = target(r, i, device)
                for j, mask in enumerate(masks):
                    totals[j] += float((raw * mask - y).square().mean()) / scales[param] ** 2
                difficulty = float((raw - y).square().mean())
                if difficulty > worst_practice:
                    worst_practice, worst_index = difficulty, i
            hard[f'{lesson_id}/{param}'] = worst_index
            encoded.append((lesson_id, r, exam, h, probabilities))
    for param, totals in costs.items():
        thresholds[param] = candidates[min(range(len(candidates)), key=lambda i: totals[i])]
    for param in PARAMS:
        thresholds.setdefault(param, .5)
    for lesson_id, r, exam, h, probabilities in encoded:
        param = r['param']
        mask = (probabilities >= thresholds[param])[r['mesh'], None]
        worst = 0.
        for i in exam:
            stop.check()
            pred = net.deform(h, r['t'][i], param) * scales[param] * mask
            error = float((pred - target(r, i, device)).abs().amax()) / scales[param]
            worst = max(worst, error) if math.isfinite(error) else math.inf
        scores[f'{lesson_id}/{param}'] = worst
    return scores, thresholds, hard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--models', type=Path, default=ROOT / 'models')
    parser.add_argument('--output', type=Path, default=ROOT / 'checkpoints/real_rig_curriculum.pt')
    parser.add_argument('--validation', default='Hiyori')
    parser.add_argument('--holdout', default='zundamon')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--precision', choices=('float32', 'float64'), default='float64')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--mastery', type=float, default=1e-9,
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

    resume_state = None
    if args.resume:
        resume_state = torch.load(args.output, map_location='cpu', weights_only=True)
        cfg = resume_state['curriculum_settings']
        for key in ('data', 'models', 'validation', 'holdout', 'width', 'lr',
                    'mastery', 'round_factor', 'passes', 'eval_every'):
            setattr(args, key, Path(cfg[key]) if key in ('data', 'models') else cfg[key])
        print(f'Resume snapshot: {args.data}', flush=True)

    if args.data is None:
        parser.error('--data is required for a new run; --resume restores the saved snapshot')
    stop = StopFlag()
    device = torch.device(args.device)
    if device.type == 'cpu':
        torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    audit = args.output.parent / 'curriculum_visual_audit'
    audit.mkdir(parents=True, exist_ok=True)
    # load_visual has a per-record interrupt checkpoint in the bundled dependency.
    import train_real_rig_v03
    train_real_rig_v03.CHECK_STOP = stop.check
    try:
        records = load_visual(args.data, args.models, device, audit)
        stop.check()
    except StopTraining:
        print('Stopped during data loading; existing checkpoint retained.', flush=True)
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
        stop.check()
        if path.is_file() and (path.suffix == '.npz' or path.name == 'source.json'):
            digest.update(path.relative_to(args.data).as_posix().encode())
            with path.open('rb') as stream:
                digest.update(hashlib.file_digest(stream, 'sha256').digest())
    settings['snapshot_sha256'] = digest.hexdigest()
    dtype = getattr(torch, args.precision)
    for record in records:
        for key in ('feat', 'onehot', 't', 'visual', 'mesh_active'):
            record[key] = record[key].to(dtype=dtype)
    net = ParameterRig(args.width).to(device=device, dtype=dtype)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    step = current = streak = 0
    round_number, limit = 1, args.mastery
    thresholds = {p: .5 for p in PARAMS}
    trial = BestTrial(args.lr)
    progress = Progress(args.output, 'real')
    last_metrics = None
    hard = {}
    if args.resume:
        state = resume_state
        if state.get('curriculum_settings') != settings:
            parser.error('Resume requires the same snapshot and curriculum settings')
        if 'metric_version' not in state:
            backup = args.output.with_name(args.output.stem + '.before_adaptive.pt')
            if not backup.exists():
                shutil.copy2(args.output, backup)
                print(f'Previous checkpoint backed up: {backup}', flush=True)
        net.load_state_dict(state['model_state'])
        opt.load_state_dict(state['optimizer'])
        rng.setstate(state['rng'])
        step, current, streak = state['steps'], state['current'], state['streak']
        round_number, limit = state['round_number'], state['limit']
        thresholds = state['thresholds']
        compatible = (state.get('metric_version') == 'practice_calibrated_max_v1'
                      and state.get('precision') == args.precision)
        if compatible:
            trial.load(state.get('adaptive'))
        else:
            current, streak, round_number = 0, 0, 1
            limit = min(args.mastery, 1e-9)
            args.mastery = limit
            settings['mastery'] = limit
            print('Preserved weights; requalifying from model 1 with the new metric and precision.', flush=True)
        hard = state.get('hard_examples', {})
        last_metrics = state.get('latest_metrics')
        del state, resume_state

    def save():
        atomic_save(dict(metric_version='practice_calibrated_max_v1', precision=args.precision, architecture='parameter_visual_v04', model_state=net.state_dict(),
                         optimizer=opt.state_dict(), rng=rng.getstate(), width=args.width,
                         scales=scales, parameters=PARAMS, thresholds=thresholds,
                         steps=step, current=current, streak=streak,
                         round_number=round_number, limit=limit, curriculum_settings=settings,
                         adaptive=trial.state(), hard_examples=hard, latest_metrics=last_metrics),
                    args.output)
        print(f'Saved step {step}: {args.output}', flush=True)

    best_path = args.output.with_name(args.output.stem + '.best.pt')

    def export_best():
        atomic_save(dict(metric_version='practice_calibrated_max_v1', precision=args.precision, architecture='parameter_visual_v04', model_state=trial.best['model'],
                         width=args.width, scales=scales, parameters=PARAMS,
                         thresholds=trial.best['extras']['thresholds'], metrics=trial.best['scores'],
                         current=current, round_number=round_number, curriculum_settings=settings),
                    best_path)

    def examine():
        return assess(net, lessons[:current + 1], scales, device, stop)

    def log(scores, action, seconds):
        current_prefix = str(current) + '/'
        own = [value for key, value in scores.items() if key.startswith(current_prefix)]
        historical = [value for key, value in scores.items() if not key.startswith(current_prefix)]
        best = trial.best['scores'] if trial.best else scores
        progress.write(step=step, round=round_number, current=current, total=len(lessons),
                       task=lessons[current][0][0]['source'], value=max(own),
                       second=max(historical, default=0.), best=max(best.values()),
                       details=scores, limits=[limit], passed=streak, passes=args.passes,
                       lr=opt.param_groups[0]['lr'], action=action, attempts=trial.attempt,
                       eval_seconds=seconds)
        return progress.last

    end = step + args.steps if args.steps else None
    try:
        if trial.best is None:
            start = time.monotonic()
            scores, thresholds, hard = examine()
            trial.observe(net, opt, scores, dict(thresholds=thresholds))
            last_metrics = log(scores, 'best', time.monotonic() - start)
            export_best()
            save()
        while end is None or step < end:
            stop.check()
            net.train()
            index = current
            if current and step % 4 == 3:
                index = step // 4 % current
            available = lessons[index]
            if trial.attempt and rng.random() < .7:
                weights = [1. + trial.best['scores'].get(f'{index}/{r["param"]}', 0.)
                           for r, _, _ in available]
                record, practice, _ = rng.choices(available, weights=weights)[0]
            else:
                record, practice, _ = rng.choice(available)
            key = str(index) + '/' + record['param']
            pose = hard.get(key, rng.choice(practice)) if rng.random() < .5 else rng.choice(practice)
            # Only train practice poses; held-out poses never become hard training examples.
            if pose not in practice:
                pose = rng.choice(practice)
            loss, _ = loss_for(net, record, pose, scales, device)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            opt.step()
            step += 1
            if step % args.eval_every and step != end:
                continue
            start = time.monotonic()
            scores, thresholds, hard = examine()
            action = trial.observe(net, opt, scores, dict(thresholds=thresholds))
            passed = action != 'rollback' and max(scores.values()) <= limit
            streak = streak + 1 if passed else 0
            last_metrics = log(scores, action, time.monotonic() - start)
            if action == 'best':
                export_best()
            elif action == 'rollback':
                thresholds = trial.best['extras']['thresholds']
            if streak >= args.passes:
                current += 1
                streak = 0
                if current == len(lessons):
                    current = 0
                    round_number += 1
                trial = BestTrial(args.lr)
                start = time.monotonic()
                scores, thresholds, hard = examine()
                trial.observe(net, opt, scores, dict(thresholds=thresholds))
                last_metrics = log(scores, 'new_exercise', time.monotonic() - start)
                export_best()
            save()
    except StopTraining:
        print('Stop requested. Saving completed updates.', flush=True)
    finally:
        save()


if __name__ == '__main__':
    main()
