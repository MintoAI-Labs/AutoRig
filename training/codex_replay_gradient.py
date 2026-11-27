"""Compare replay gradient balancing and old-task projection on a locked split."""
import argparse
import copy
import json
import random
from pathlib import Path

from codex_replay_lesson import (
    DTYPE, EYES, ROOT, ParameterRig, StopFlag, acceptance, evaluate, file_hash,
    load_lesson, passed, practice_loss, practice_only, prepare,
)
from codex_replay_lr_audit import MILESTONES, gradient_stats, old_practice_loss

from training_progress import atomic_save
from unified_sources import family

import torch

HEAD = EYES[1]
EPSILON = 1e-12
MIN_SCALE = 1e-6


def step_gradient(head, optimizer, old_batch, new_batch, old_ids, new_ids, method, loss_function=practice_loss):
    params = tuple(head.parameters())
    old_loss = loss_function(head, old_batch, old_ids)
    new_loss = loss_function(head, new_batch, new_ids)
    if not torch.isfinite(old_loss) or not torch.isfinite(new_loss):
        raise ValueError('Nonfinite practice loss')
    old = torch.cat([value.flatten() for value in torch.autograd.grad(
        old_loss, params, retain_graph=method == 'control')])
    new = torch.cat([value.flatten() for value in torch.autograd.grad(
        new_loss, params, retain_graph=method == 'control')])
    before = gradient_stats(old, new)
    if not torch.isfinite(old).all() or not torch.isfinite(new).all():
        raise ValueError('Nonfinite practice gradient')
    scale = 1.
    projection = 0.
    adjusted = new
    if method != 'control':
        # Only scale down the new gradient. Never amplify either task near zero.
        scale = max(MIN_SCALE, min(1., before['old_l2'] / (before['new_l2'] + EPSILON)))
        adjusted = new * scale
    balanced = gradient_stats(old, adjusted)
    if method == 'projected' and before['old_l2'] > EPSILON:
        dot = torch.dot(adjusted, old)
        if dot < 0:
            projection = float(dot / torch.dot(old, old))
            adjusted = adjusted - projection * old
    after = gradient_stats(old, adjusted)

    optimizer.zero_grad(set_to_none=True)
    if method == 'control':
        (.5 * (new_loss + old_loss)).backward()
    else:
        combined = .5 * (old + adjusted)
        offset = 0
        for param in params:
            param.grad = combined[offset:offset + param.numel()].reshape_as(param)
            offset += param.numel()
    preclip = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
    optimizer.step()
    return dict(old_loss=float(old_loss.detach()), new_loss=float(new_loss.detach()),
                raw=before, balanced=balanced, effective=after, new_scale=scale,
                projection_coefficient=projection, combined_preclip_l2=float(preclip))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='New local experiment directory')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, choices=MILESTONES, default=MILESTONES[-1],
                        help='Stop at this absolute milestone')
    parser.add_argument('--method', choices=('all', 'control', 'balanced', 'projected'), default='all')
    parser.add_argument('--resume', type=Path, help='Resume an eligible checkpoint into a new directory')
    args = parser.parse_args()
    resumed = torch.load(args.resume, map_location='cpu', weights_only=True) if args.resume else None
    if resumed:
        args.seed = resumed['config']['seed']
        args.method = resumed['config']['method']
        if args.steps <= resumed['step']:
            parser.error('--steps must be after the saved milestone')
    args.output.mkdir(parents=True, exist_ok=False)

    paths = {
        'best': ROOT / 'checkpoints/real_rig_curriculum.best.pt',
        'blink_trial': ROOT / 'checkpoints/codex_blink_trial_03/accepted.pt',
        'replay_schedule': ROOT / 'checkpoints/codex_replay_trial_02/lesson.json',
        'lr_audit': ROOT / 'checkpoints/codex_replay_lr_trial_01/lesson.json',
    }
    hashes = {name: file_hash(path) for name, path in paths.items()}
    prior = json.loads(paths['replay_schedule'].read_text())
    audit = json.loads(paths['lr_audit'].read_text())
    if any(hashes[key] != prior['checkpoint_hashes'][key] for key in ('best', 'blink_trial')):
        raise ValueError('Checkpoint differs from the replay comparison')
    if hashes['replay_schedule'] != audit['source_hashes']['previous_replay']:
        raise ValueError('Replay schedule differs from the LR audit')

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    stop = StopFlag()
    checkpoint = torch.load(paths['best'], weights_only=True, map_location='cpu')
    settings = checkpoint['curriculum_settings']
    base = ParameterRig(checkpoint['width']).to(device=device, dtype=DTYPE)
    base.load_state_dict(checkpoint['model_state'])
    base.eval()
    practice, validation, excluded, digest = load_lesson(settings, device)
    if (digest != prior['lesson_input_hash'] or excluded != prior['excluded_test_groups']
            or len(validation) != prior['validation_records'] or settings['mastery'] != prior['mastery']):
        raise ValueError('Validation, data split or mastery differs from earlier replay')

    old, = (r for r in practice if r['group'] == settings['groups'][0] and r['param'] == HEAD)
    old_ids, old_exam = prior['old_replay_pose_indices'], prior['old_exam_pose_indices']
    old_batch, = prepare(base, [practice_only(dict(old, exam=old_exam), old_ids)],
                         checkpoint, exam=False)
    validation_batches = prepare(base, validation, checkpoint, exam=True)
    old_family = family(old['source'])
    by_group = {r['group']: r for r in practice
                if r['param'] == HEAD and family(r['source']) != old_family}
    rng = random.Random(args.seed)
    raw_schedule = prior['schedule']
    if args.seed != prior['seed']:
        population = list(by_group)
        raw_schedule = []
        for _ in range(prior['steps_per_arm']):
            group = population[rng.randrange(len(population))]
            raw_schedule.append(dict(group=group,
                                     new_pose_indices=rng.sample(by_group[group]['practice'], 4),
                                     replay_pose_indices=rng.sample(old_ids, 2)))
    groups = list(dict.fromkeys(row['group'] for row in raw_schedule))
    new_rows = [practice_only(by_group[group], by_group[group]['practice']) for group in groups]
    new_batches = prepare(base, new_rows, checkpoint, exam=False)
    schedule = []
    for item in raw_schedule:
        index = groups.index(item['group'])
        new_ids = [new_rows[index]['practice'].index(i) for i in item['new_pose_indices'][:2]]
        replay_ids = [old_ids.index(i) for i in item['replay_pose_indices']]
        schedule.append((index, new_ids, replay_ids))
    if len(schedule) != MILESTONES[-1] or len(schedule) != prior['steps_per_arm']:
        raise ValueError('Expected a 100-step schedule')

    common_config = dict(
        schema=1, seed=args.seed, lr=prior['lr'], trainable_head=HEAD, optimizer='AdamW',
        clip_norm=1., epsilon=EPSILON, scale_limits=[MIN_SCALE, 1.],
        dtype='float64', device=str(device), deterministic=True, torch_version=str(torch.__version__),
        source_hashes=hashes, lesson_input_hash=digest, schedule=raw_schedule,
        old_practice_pose_indices=old_ids, old_exam_pose_indices=old_exam,
        validation_pose_indices='0,4,...', validation_records=len(validation),
        excluded_test_groups=excluded, milestones=MILESTONES, mastery=settings['mastery'],
    )

    def measure(model):
        metrics = evaluate(model, checkpoint, validation_batches, old, old_ids, old_exam, stop)
        return dict(primary=metrics, old_practice_loss=old_practice_loss(model.heads[HEAD], old_batch))

    baseline = measure(base)
    curves, gradients, candidates = {}, {}, []
    methods = ('control', 'balanced', 'projected') if args.method == 'all' else (args.method,)
    for method in methods:
        trial = copy.deepcopy(base)
        head = trial.heads[HEAD]
        optimizer = torch.optim.AdamW(head.parameters(), lr=prior['lr'])
        config = dict(common_config, method=method)
        start = 0
        if resumed:
            if resumed['config'] != config:
                raise ValueError('Resume requires the original schedule, sources and configuration')
            trial.load_state_dict(resumed['model_state'])
            optimizer.load_state_dict(resumed['optimizer'])
            torch.set_rng_state(resumed['rng']['torch_cpu'])
            torch.cuda.set_rng_state_all(resumed['rng']['torch_cuda'])
            rng.setstate(resumed['rng']['schedule'])
            start = resumed['step']
            candidates.append(dict(method=method, step=start, path=str(args.resume.resolve()),
                                   hiyori_rms=resumed['measurement']['primary']['hiyori']['all']['rms']))
        curves[method] = []
        gradients[method] = []
        for step in range(start + 1, args.steps + 1):
            stop.check()
            index, new_ids, replay_ids = schedule[step - 1]
            diagnostics = step_gradient(head, optimizer, old_batch, new_batches[index],
                                        replay_ids, new_ids, method)
            gradients[method].append(dict(step=step, **diagnostics))
            if step not in MILESTONES:
                continue
            point = measure(trial)
            checks = acceptance(point['primary'], baseline['primary'])
            eligible = passed(checks)
            scored = dict(step=step, checks=checks, eligible=eligible, **point)
            curves[method].append(scored)
            if eligible:
                candidate = args.output / f'{method}.step{step:03d}.candidate.pt'
                if candidate.exists():
                    raise FileExistsError(candidate)
                exported = {key: checkpoint[key] for key in (
                    'architecture', 'precision', 'width', 'scales', 'parameters', 'thresholds',
                    'curriculum_settings')}
                exported.update(
                    model_state=trial.state_dict(), optimizer=optimizer.state_dict(),
                    rng=dict(torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state_all(),
                             schedule=rng.getstate()),
                    step=step, config=config, measurement=scored, baseline=baseline,
                    status='improvement_criteria_met_mastery_not_met',
                )
                atomic_save(exported, candidate)
                reloaded = torch.load(candidate, map_location='cpu', weights_only=True)
                verified = copy.deepcopy(base)
                verified.load_state_dict(reloaded['model_state'])
                verified_optimizer = torch.optim.AdamW(verified.heads[HEAD].parameters(), lr=prior['lr'])
                verified_optimizer.load_state_dict(reloaded['optimizer'])
                if measure(verified) != point:
                    raise RuntimeError(f'Reloaded checkpoint scores differ at step {step}')
                candidates.append(dict(method=method, step=step, path=candidate.name,
                                       hiyori_rms=point['primary']['hiyori']['all']['rms'],
                                       reload_verified=True))
            print(method, step, point['old_practice_loss'], point['primary']['old_curriculum_max'],
                  point['primary']['hiyori']['all']['rms'], eligible, flush=True)
        for key, tensor in trial.state_dict().items():
            if not key.startswith(f'heads.{HEAD}.') and not torch.equal(tensor, base.state_dict()[key]):
                raise RuntimeError(f'Unexpected frozen tensor change: {key}')

    if hashes != {name: file_hash(path) for name, path in paths.items()}:
        raise RuntimeError('An input changed during the experiment')
    eligible = [[method, point['step']] for method, curve in curves.items()
                for point in curve if point['eligible']]
    report = dict(decision='retain_original_best', eligible_milestones=eligible,
                  lr=prior['lr'], seed=args.seed, steps_per_arm=args.steps,
                  candidates=candidates,
                  best_candidate=min(candidates, key=lambda row: row['hiyori_rms']) if candidates else None,
                  schedule=raw_schedule, resume_from=str(args.resume) if args.resume else None,
                  epsilon=EPSILON, scale_limits=[MIN_SCALE, 1.],
                  source_hashes=hashes, lesson_input_hash=digest,
                  mastery=settings['mastery'], validation_records=len(validation),
                  old_practice_pose_indices=old_ids, old_exam_pose_indices=old_exam,
                  excluded_test_groups=excluded, baseline=baseline, curves=curves, gradients=gradients)
    (args.output / 'lesson.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print('Eligible milestones:', eligible, flush=True)


if __name__ == '__main__':
    main()
