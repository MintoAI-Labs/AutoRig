"""Paired, bounded anti-forgetting lesson starting from the original Real Rig best."""
import argparse
import copy
import json
import os
from pathlib import Path
import random

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch

from codex_blink_lesson import DTYPE, EYES, assess, geometry, improves, load_lesson, prepare
from train_real_rig_curriculum import assess as assess_curriculum, split_poses
from train_real_rig_v04 import ParameterRig, ROOT
from training_progress import StopFlag, atomic_save
from unified_sources import family, file_hash

METRICS = ('rms', 'max', 'static')


def practice_only(row, indices):
    """Derive moving/static labels from precisely the poses available to the loss."""
    if not indices or set(indices).intersection(row['exam']):
        raise ValueError('Practice must be nonempty and disjoint from exam')
    moving = torch.as_tensor(np.any(row['delta'][indices] != 0, axis=(0, 2)),
                             device=row['t'].device)
    return dict(row, practice=indices, moving=moving)


def practice_loss(head, batch, indices):
    row, features, gate, poses, target = batch
    t = poses[indices]
    labels = target[indices]
    raw = geometry(head, features, t)
    error = (raw - labels).square().sum(-1)
    moving = row['moving']
    moving_loss = error[:, moving].sum() / max(int(moving.sum()) * len(t), 1)
    static_loss = error[:, ~moving].sum() / max(int((~moving).sum()) * len(t), 1)
    return moving_loss + .15 * static_loss + .25 * (raw * gate - labels).square().mean()


def train(base, new_batches, old_batch, schedule, replay, stop):
    trial = copy.deepcopy(base)
    head = trial.heads[EYES[1]]
    optimizer = torch.optim.AdamW(head.parameters(), lr=5e-5)
    for row_id, new_ids, old_ids in schedule:
        stop.check()
        if replay:
            loss = .5 * (practice_loss(head, new_batches[row_id], new_ids[:2]) +
                         practice_loss(head, old_batch, old_ids))
        else:
            loss = practice_loss(head, new_batches[row_id], new_ids)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite practice loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
        optimizer.step()
    for key, value in trial.state_dict().items():
        if not key.startswith(f'heads.{EYES[1]}.') and not torch.equal(value, base.state_dict()[key]):
            raise RuntimeError(f'Unexpected frozen tensor change: {key}')
    return trial


def evaluate(trial, checkpoint, validation, old, old_ids, exam_ids, stop):
    scores, thresholds, _ = assess_curriculum(
        trial, [[(old, old_ids, exam_ids)]], checkpoint['scales'], old['t'].device, stop)
    param = old['param']
    threshold = thresholds[param]
    scoring_checkpoint = dict(checkpoint, thresholds=dict(checkpoint['thresholds'], **{param: threshold}))
    # Static labels are only used for reporting and selection; the exam tensors
    # never enter the training batches.
    scoring_row = dict(old, exam=exam_ids,
                       moving=torch.as_tensor(np.any(old['delta'] != 0, axis=(0, 2)),
                                              device=old['t'].device))
    old_metrics = assess(trial, prepare(trial, [scoring_row], scoring_checkpoint, exam=True))['all']
    return dict(hiyori=assess(trial, validation), old_exam=old_metrics,
                old_curriculum_max=scores[f'0/{param}'], old_gate=threshold)


def acceptance(candidate, reference):
    return dict(
        hiyori=improves(candidate['hiyori'], reference['hiyori']),
        old_exam={metric: candidate['old_exam'][metric] <= reference['old_exam'][metric]
                  for metric in METRICS},
        old_curriculum_max=candidate['old_curriculum_max'] <= reference['old_curriculum_max'],
    )


def passed(checks):
    return checks['hiyori'] and all(checks['old_exam'].values()) and checks['old_curriculum_max']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='New local experiment directory')
    parser.add_argument('--steps', type=int, default=100, choices=range(1, 201), metavar='1..200')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    stop = StopFlag()
    sources = {
        'best': ROOT / 'checkpoints/real_rig_curriculum.best.pt',
        'blink_trial': ROOT / 'checkpoints/codex_blink_trial_03/accepted.pt',
    }
    hashes = {name: file_hash(path) for name, path in sources.items()}
    checkpoint = torch.load(sources['best'], weights_only=True, map_location='cpu')
    settings = checkpoint['curriculum_settings']
    base = ParameterRig(checkpoint['width']).to(device=device, dtype=DTYPE)
    base.load_state_dict(checkpoint['model_state'])
    base.eval()
    practice, validation, excluded, data_hash = load_lesson(settings, device)
    previous = json.loads((sources['blink_trial'].parent / 'lesson.json').read_text())
    if (data_hash != previous['lesson_input_hash'] or excluded != previous['excluded_test_groups']
            or hashes['best'] != previous['checkpoint_hashes'][sources['best'].name]):
        raise ValueError('Checkpoint or data differs from the prior experiment')

    old, = (r for r in practice if r['group'] == settings['groups'][0] and r['param'] == EYES[1])
    old_ids, old_exam = split_poses(old)
    allowed = set(old['practice'])
    replay_ids = [i for i in old_ids if i in allowed]
    old_batch, = prepare(base, [practice_only(dict(old, exam=old_exam), replay_ids)],
                         checkpoint, exam=False)
    old_family = family(old['source'])
    new_rows = [practice_only(row, row['practice']) for row in practice
                if row['param'] == EYES[1] and family(row['source']) != old_family]
    new_batches = prepare(base, new_rows, checkpoint, exam=False)
    validation_batches = prepare(base, validation, checkpoint, exam=True)
    rng = random.Random(42)
    schedule = []
    for _ in range(args.steps):
        index = rng.randrange(len(new_batches))
        schedule.append((index, rng.sample(range(len(new_batches[index][3])), 4),
                         rng.sample(range(len(replay_ids)), 2)))

    metrics = {'best': evaluate(base, checkpoint, validation_batches, old, old_ids, old_exam, stop)}
    print('baseline', metrics['best']['hiyori']['all'], metrics['best']['old_exam'], flush=True)
    checks = {}
    for label, replay in (('control', False), ('replay', True)):
        trial = train(base, new_batches, old_batch, schedule, replay, stop)
        metrics[label] = evaluate(trial, checkpoint, validation_batches, old, old_ids, old_exam, stop)
        checks[label] = acceptance(metrics[label], metrics['best'])
        print(label, metrics[label]['hiyori']['all'], metrics[label]['old_exam'], checks[label], flush=True)

    if hashes != {name: file_hash(path) for name, path in sources.items()}:
        raise RuntimeError('Original checkpoint changed during experiment')
    accepted = passed(checks['replay'])
    report = dict(
        decision='accepted_experiment' if accepted else 'retain_original_best',
        seed=42, steps_per_arm=args.steps, pose_budget_per_step=4, lr=5e-5,
        device=str(device), precision='float64', mastery=settings['mastery'],
        hypothesis='Replacing half the new poses with old practice may reduce forgetting',
        trainable_head=EYES[1], checkpoint_hashes=hashes, lesson_input_hash=data_hash,
        old_source=old['source'], old_replay_pose_indices=replay_ids, old_exam_pose_indices=old_exam,
        new_practice_records=len(new_batches), validation_records=len(validation_batches),
        validation_pose_indices='0,4,...', excluded_test_groups=excluded, metrics=metrics, checks=checks,
        schedule=[dict(group=new_rows[i]['group'],
                       new_pose_indices=[new_rows[i]['practice'][j] for j in new],
                       replay_pose_indices=[replay_ids[j] for j in replay])
                  for i, new, replay in schedule],
    )
    if accepted:
        export = {key: checkpoint[key] for key in ('architecture', 'precision', 'width',
                  'scales', 'parameters', 'thresholds', 'curriculum_settings')}
        export.update(model_state=trial.state_dict(), validation_metrics=metrics['replay'],
                      experiment='codex_replay', source_checkpoint=hashes['best'])
        atomic_save(export, args.output / 'replay_experimental.pt')
    (args.output / 'lesson.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(report['decision'], flush=True)


if __name__ == '__main__':
    main()
