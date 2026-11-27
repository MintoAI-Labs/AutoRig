"""Trace replay forgetting at two learning rates on the locked Real Rig split."""
import argparse
import copy
import json
from pathlib import Path
from statistics import median

# The replay module configures deterministic cuBLAS before importing torch.
from codex_replay_lesson import (DTYPE, EYES, ROOT, ParameterRig, StopFlag, acceptance,
                                 assess, evaluate, file_hash, geometry, load_lesson,
                                 passed, practice_loss, practice_only, prepare)

import torch

HEAD = EYES[1]
MILESTONES = (1, 5, 10, 25, 50, 75, 100)


def gradient_vector(head, batch, indices):
    gradients = torch.autograd.grad(practice_loss(head, batch, indices), tuple(head.parameters()))
    return torch.cat([gradient.detach().flatten() for gradient in gradients])


def gradient_stats(old, new):
    old_norm, new_norm = old.norm(), new.norm()
    denominator = old_norm * new_norm
    return dict(old_l2=float(old_norm), new_l2=float(new_norm),
                cosine=float(torch.dot(old, new) / denominator) if denominator > 0 else None)


def gradient_probe(head, old, new, schedule):
    pairs = [(gradient_vector(head, old, old_ids),
              gradient_vector(head, new[index], new_ids))
             for index, new_ids, old_ids in schedule[:12]]
    per_batch = [gradient_stats(a, b) for a, b in pairs]
    mean_old = torch.stack([a for a, _ in pairs]).mean(0)
    mean_new = torch.stack([b for _, b in pairs]).mean(0)
    cosines = [r['cosine'] for r in per_batch if r['cosine'] is not None]
    return dict(mean_vectors=gradient_stats(mean_old, mean_new),
                median_batch_cosine=median(cosines) if cosines else None,
                negative_batch_cosines=sum(value < 0 for value in cosines), batches=per_batch)


@torch.no_grad()
def old_practice_loss(head, batch):
    count = len(batch[3])
    total = 0.
    for start in range(0, count, 8):
        indices = list(range(start, min(count, start + 8)))
        total += len(indices) * float(practice_loss(head, batch, indices))
    return total / count


def ungated(batches):
    return [(row, features, torch.ones_like(gate), poses, target)
            for row, features, gate, poses, target in batches]


@torch.no_grad()
def old_pose_errors(head, batch, indices, sampled):
    _, features, gate, poses, target = batch
    rows = []
    for start in range(0, len(poses), 8):
        sl = slice(start, start + 8)
        raw = geometry(head, features, poses[sl])
        for j, (before, after) in enumerate(zip(raw - target[sl], raw * gate - target[sl]), start):
            nearest = min((abs(float(poses[j]) - x) for x in sampled), default=None)
            rows.append(dict(index=indices[j], normalized_pose=float(poses[j]),
                             nearest_sampled_pose_distance=nearest, raw_max=float(before.abs().max()),
                             hard_max=float(after.abs().max())))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='New local experiment directory')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = {
        'best': ROOT / 'checkpoints/real_rig_curriculum.best.pt',
        'blink_trial': ROOT / 'checkpoints/codex_blink_trial_03/accepted.pt',
        'previous_replay': ROOT / 'checkpoints/codex_replay_trial_02/lesson.json',
    }
    hashes = {name: file_hash(path) for name, path in paths.items()}
    prior = json.loads(paths['previous_replay'].read_text())
    for name in ('best', 'blink_trial'):
        if hashes[name] != prior['checkpoint_hashes'][name]:
            raise ValueError(f'Checkpoint changed since replay comparison: {name}')

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(prior['seed'])
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
        raise ValueError('Validation, data split, or mastery differs from previous replay')
    old, = (r for r in practice if r['group'] == settings['groups'][0] and r['param'] == HEAD)
    old_ids = prior['old_replay_pose_indices']
    old_exam = prior['old_exam_pose_indices']
    old_batch, = prepare(base, [practice_only(dict(old, exam=old_exam), old_ids)],
                         checkpoint, exam=False)
    validation_batches = prepare(base, validation, checkpoint, exam=True)
    by_group = {r['group']: r for r in practice if r['param'] == HEAD and r['group'] != old['group']}
    groups = list(dict.fromkeys(row['group'] for row in prior['schedule']))
    new_rows = [practice_only(by_group[group], by_group[group]['practice']) for group in groups]
    new_batches = prepare(base, new_rows, checkpoint, exam=False)
    schedule = []
    for step in prior['schedule']:
        index = groups.index(step['group'])
        new_ids = [new_rows[index]['practice'].index(i) for i in step['new_pose_indices'][:2]]
        replay_ids = [old_ids.index(i) for i in step['replay_pose_indices']]
        schedule.append((index, new_ids, replay_ids))
    if len(schedule) != MILESTONES[-1] or len(schedule) != prior['steps_per_arm']:
        raise ValueError('Expected the original 100-step schedule')

    def measure(net, step):
        primary = evaluate(net, checkpoint, validation_batches, old, old_ids, old_exam, stop)
        threshold = primary['old_gate']
        exam_checkpoint = dict(checkpoint, thresholds=dict(checkpoint['thresholds'], **{HEAD: threshold}))
        exam_row = dict(old, exam=old_exam, moving=(old['delta'] != 0).any(axis=(0, 2)))
        exam_row['moving'] = torch.as_tensor(exam_row['moving'], device=device)
        exam_batch, = prepare(base, [exam_row], exam_checkpoint, exam=True)
        seen_ids = sorted({old_ids[i] for _, _, indices in schedule[:step] for i in indices})
        sampled = old['t'][seen_ids].tolist()
        old_pose = old_pose_errors(net.heads[HEAD], exam_batch, old_exam, sampled)
        return dict(step=step, primary=primary, replay_practice_loss=old_practice_loss(net.heads[HEAD], old_batch),
                    raw_old_exam=assess(net, ungated([exam_batch]))['all'],
                    raw_hiyori=assess(net, ungated(validation_batches)),
                    old_exam_gate_active_vertices=int(exam_batch[2].sum()),
                    replay_unique_pose_indices=seen_ids, old_exam_pose_errors=old_pose)

    initial = measure(base, 0)
    initial['gradient_probe'] = gradient_probe(base.heads[HEAD], old_batch, new_batches, schedule)
    print('Baseline loss and gradient', initial['replay_practice_loss'],
          initial['gradient_probe']['mean_vectors'], flush=True)
    rates = {'current_lr': prior['lr'], 'one_tenth_lr': prior['lr'] / 10}
    curves = {}
    for name, lr in rates.items():
        trial = copy.deepcopy(base)
        head = trial.heads[HEAD]
        optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
        curve = []
        for index, (record, new_ids, replay_ids) in enumerate(schedule, 1):
            stop.check()
            loss = .5 * (practice_loss(head, new_batches[record], new_ids) +
                         practice_loss(head, old_batch, replay_ids))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite practice loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
            optimizer.step()
            if index not in MILESTONES:
                continue
            point = measure(trial, index)
            point['checks'] = acceptance(point['primary'], initial['primary'])
            point['eligible'] = passed(point['checks'])
            if index == MILESTONES[-1]:
                point['gradient_probe'] = gradient_probe(head, old_batch, new_batches, schedule)
            curve.append(point)
            print(name, index, point['replay_practice_loss'],
                  point['primary']['old_curriculum_max'], point['primary']['hiyori']['all']['rms'],
                  point['eligible'], flush=True)
        curves[name] = curve
        for key, tensor in trial.state_dict().items():
            if not key.startswith(f'heads.{HEAD}.') and not torch.equal(tensor, base.state_dict()[key]):
                raise RuntimeError(f'Unexpected frozen tensor change: {key}')

    if hashes != {name: file_hash(path) for name, path in paths.items()}:
        raise RuntimeError('An input changed during the experiment')
    eligible = [(name, point['step']) for name, curve in curves.items() for point in curve if point['eligible']]
    coverage = {
        'available_practice': len(old_ids), 'exam': len(old_exam),
        'max_nearest_available_distance': max(
            min(abs(float(old['t'][i]) - float(old['t'][j])) for j in old_ids) for i in old_exam),
        'unique_replayed': len(curves['current_lr'][-1]['replay_unique_pose_indices']),
    }
    report = dict(decision='retain_original_best', eligible_milestones=eligible, rates=rates,
                  seed=prior['seed'], milestones=MILESTONES, source_hashes=hashes, lesson_input_hash=digest,
                  validation_records=len(validation), excluded_test_groups=excluded, mastery=settings['mastery'],
                  baseline=initial, curves=curves, old_pose_coverage=coverage)
    (args.output / 'lesson.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print('Eligible milestones', eligible, 'coverage', coverage, flush=True)


if __name__ == '__main__':
    main()
