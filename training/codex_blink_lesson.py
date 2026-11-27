"""Run a bounded, paired blink lesson using locally authorized real rigs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from algorithm_rules import SOURCE_ROOT
from train_real_rig_v01 import ROOT, load_record
from train_real_rig_v03 import mesh_patches
from train_real_rig_v04 import ParameterRig
from training_progress import atomic_save
from unified_rig import UnifiedRig
from unified_sources import (extreme_prediction, family, file_hash,
                             make_ext_teacher, synthetic_records, to_device)

EYES = ('ParamEyeLOpen', 'ParamEyeROpen')
DTYPE = torch.float64


def load_lesson(settings, device):
    root, models = Path(settings['data']), Path(settings['models'])
    validation_family, test_family = (family(settings[key]) for key in ('validation', 'holdout'))
    if validation_family == test_family:
        raise ValueError('Validation and test families must differ')
    practice, validation = [], []
    skipped = []
    digest = hashlib.sha256()
    for manifest in sorted(root.glob('*/source.json')):
        source = Path(json.loads(manifest.read_text(encoding='utf-8-sig'))['model_path']).name
        group = family(source)
        if group == test_family:
            skipped.append(manifest.parent.name)
            continue
        visual = None
        for path in sorted(manifest.parent.glob('*.npz')):
            if group != validation_family and not path.name.startswith(EYES):
                continue
            row = load_record(path, root, torch.device('cpu'))
            if family(row['source']) != group:
                raise ValueError(f'Family mismatch in {path}')
            ids = list(range(0, len(row['t']), 4))
            train_ids = [i for i in range(len(row['t'])) if i % 4]
            if not ids or not train_ids:
                raise ValueError(f'Insufficient poses in {path}')
            if visual is None:
                patches, _ = mesh_patches(path, models)
                rgba = torch.from_numpy(patches).to(torch.float32).permute(0, 3, 1, 2) / 255
                visual = torch.cat((rgba[:, :3] * rgba[:, 3:4], rgba[:, 3:4]), 1).contiguous()
            row.update(visual=visual, practice=train_ids, exam=ids, domain='real',
                       semantic=torch.full((row['nm'],), 14, dtype=torch.long))
            # Validation labels define the static error metric only.
            targets = row['delta'] if group == validation_family else row['delta'][train_ids]
            row['moving'] = torch.from_numpy(np.any(targets != 0, axis=(0, 2)))
            row['mesh_active'] = torch.zeros(row['nm'])
            row = to_device(row, device, DTYPE)
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(bytes.fromhex(file_hash(path)))
            (validation if group == validation_family else practice).append(row)
    if not practice or not validation or not skipped:
        raise ValueError('Expected practice, validation and excluded test families')
    return practice, validation, skipped, digest.hexdigest()


def geometry(head, features, poses):
    conditioning = torch.stack((poses, poses.abs(), poses.square()), -1)
    inputs = torch.cat((features[None].expand(len(poses), -1, -1),
                        conditioning[:, None].expand(-1, len(features), -1)), -1)
    return head(inputs) * poses[:, None, None]


@torch.no_grad()
def prepare(net, rows, checkpoint, exam):
    result = []
    for row in rows:
        h, logits = net.encode(row)
        param = row['param']
        gate = (logits.sigmoid() >= checkpoint['thresholds'][param])[row['mesh'], None]
        ids = row['exam' if exam else 'practice']
        labels = torch.as_tensor(row['delta'][ids], device=h.device, dtype=h.dtype)
        result.append((row, h, gate, row['t'][ids], labels / checkpoint['scales'][param]))
    return result


@torch.no_grad()
def assess(net, rows):
    totals = {}
    for row, features, gate, poses, target in rows:
        param = row['param']
        pieces = []
        for start in range(0, len(poses), 8):
            pieces.append(geometry(net.heads[param], features, poses[start:start + 8]) * gate)
        residual = torch.cat(pieces) - target
        if not torch.isfinite(residual).all():
            raise ValueError('Nonfinite validation error')
        still = residual[:, ~row['moving']]
        for key in ('all', param):
            total = totals.setdefault(key, [0., 0, 0., 0., 0])
            total[0] += float(residual.square().sum())
            total[1] += residual.numel()
            total[2] = max(total[2], float(residual.abs().max()))
            total[3] += float(still.square().sum())
            total[4] += still.numel()
    return {key: dict(rms=(s / n)**.5, max=m, static=(z / max(k, 1))**.5)
            for key, (s, n, m, z, k) in totals.items()}


@torch.no_grad()
def audit_teachers(real, active_checkpoint, extreme_checkpoint, validation, device):
    active = ParameterRig(active_checkpoint['width']).to(device=device, dtype=DTYPE)
    active.load_state_dict(active_checkpoint['model_state'])
    student = UnifiedRig(active_checkpoint['width']).to(device=device, dtype=DTYPE)
    student.initialize_real(active_checkpoint['model_state'])
    row = next(r for r in validation if r['param'] == EYES[1])
    h, logits = active.encode(row)
    _, raw, student_logits = student.pose_batch(row, row['t'][row['exam'][:4]])
    original = geometry(active.heads[row['param']], h, row['t'][row['exam'][:4]])
    changed = [name for name, value in real.state_dict().items()
               if not torch.equal(value.cpu(), active_checkpoint['model_state'][name])]
    teacher = make_ext_teacher(extreme_checkpoint).to(device=device, dtype=DTYPE)
    evidence = {}
    for synthetic in synthetic_records(10042, 1):
        synthetic = to_device(synthetic, device, DTYPE)
        param = synthetic['param']
        poses = synthetic['practice']
        scale = active_checkpoint['scales'].get(param, .1)
        student_pred = student.pose_batch(synthetic, synthetic['t'][poses])[0]
        target = torch.as_tensor(synthetic['delta'][poses], device=device, dtype=DTYPE) / scale
        taught = extreme_prediction(teacher, synthetic, poses, device, DTYPE) / scale
        eligible = (taught-target).square().sum(-1) < .95*(student_pred-target).square().sum(-1)
        disagreement = (taught-target).square().mean() / target.square().mean().clamp_min(1e-12)
        loss = (.1 * eligible * (student_pred-taught).square().sum(-1) / (1+disagreement)).mean()
        evidence[param] = dict(eligible=float(eligible.double().mean()), added_loss=float(loss),
                               teacher_max=float((taught-target).abs().max()))
    return dict(real_raw_max_difference=float((raw-original).abs().max()),
                real_gate_logits_max_difference=float((student_logits-logits).abs().max()),
                real_hard_gate_agreement=float(((student_logits >= 0) ==
                    (logits.sigmoid() >= active_checkpoint['thresholds'][row['param']])).double().mean()),
                changed_real_state_keys=changed, extreme=evidence)


def train(net, practice, episodes, blink_weight):
    optimizer = torch.optim.AdamW(net.heads.parameters(), lr=5e-5)
    for record_id, indices in episodes:
        row, features, gate, poses, target = practice[record_id]
        indices = list(indices)
        t = poses[indices]
        head = net.heads[row['param']]
        raw = geometry(head, features, t)
        residual = raw - target[indices]
        error = residual.square().sum(-1)
        moving = row['moving']
        loss = error[:, moving].sum() / max(int(moving.sum()) * len(t), 1)
        loss = loss + .15 * error[:, ~moving].sum() / max(int((~moving).sum()) * len(t), 1)
        loss = loss + .25 * (raw * gate - target[indices]).square().mean()
        if blink_weight:
            # Two-keyform prior, applied within each side of the neutral pose.
            # Endpoint displacement is predicted. No validation target is used.
            linear = geometry(head, features, t.sign()) * t.abs()[:, None, None]
            loss = loss + blink_weight * (raw - linear).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.heads.parameters(), 1.)
        optimizer.step()


def improves(candidate, reference):
    new, old = candidate['all'], reference['all']
    return (new['rms'] < old['rms'] and new['max'] <= old['max']
            and new['static'] <= old['static'] * 1.001 + 1e-15
            and all(candidate[p][metric] <= reference[p][metric]
                    for p in EYES for metric in ('rms', 'max', 'static')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='New local output directory')
    parser.add_argument('--psd2live', type=Path, default=Path(r'C:\PSD2Live'))
    parser.add_argument('--steps', type=int, default=100, choices=range(1, 501), metavar='1..500')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    rng = random.Random(42)
    sources = [ROOT / 'checkpoints' / name for name in (
        'real_rig_curriculum.best.pt', 'real_rig_curriculum.pt', 'fullbody_extreme_v04.pt')]
    hashes = {p.name: file_hash(p) for p in sources}
    best, active, extreme = [torch.load(p, weights_only=True, map_location='cpu') for p in sources]
    settings = best['curriculum_settings']
    base = ParameterRig(best['width']).to(device=device, dtype=DTYPE)
    base.load_state_dict(best['model_state'])
    base.eval()
    practice, validation, excluded, data_hash = load_lesson(settings, device)
    audit = audit_teachers(base, active, extreme, validation, device)
    first = next(row for row in practice if row['group'] == settings['groups'][0])
    peak = np.linalg.norm(first['delta'], axis=-1).max(axis=0)
    audit['first_lesson'] = dict(source=first['source'], poses=len(first['t']),
                                 moving_old=int((peak > 1e-5).sum()), moving_nonzero=int((peak > 0).sum()),
                                 vertices=len(peak), meshes=first['nm'])
    practice = prepare(base, practice, best, exam=False)
    validation = prepare(base, validation, best, exam=True)
    episodes = []
    for _ in range(args.steps):
        index = rng.randrange(len(practice))
        episodes.append((index, rng.sample(range(len(practice[index][3])), 4)))
    scores = {'before': assess(base, validation)}
    print('Before', scores['before']['all'], flush=True)
    for label, weight in [('control', 0.), ('blink_prior', .5)]:
        trial = copy.deepcopy(base)
        train(trial, practice, episodes, weight)
        scores[label] = assess(trial, validation)
        print(label, scores[label]['all'], flush=True)
        if label == 'control':
            right_only = trial
    accepted = (improves(scores['blink_prior'], scores['before']) and
                improves(scores['blink_prior'], scores['control']))
    if {p.name: file_hash(p) for p in sources} != hashes:
        raise RuntimeError('Input checkpoint changed during experiment')
    # The two heads share no optimized parameters. Restore the left head from baseline
    # and verify the resulting checkpoint on the same validation, including static error.
    right_only.heads[EYES[0]].load_state_dict(base.heads[EYES[0]].state_dict())
    scores['right_only'] = assess(right_only, validation)
    right_accepted = improves(scores['right_only'], scores['before'])
    selected = 'blink_prior' if accepted else 'right_only' if right_accepted else None
    if selected is not None:
        model = trial if accepted else right_only
        export = {key: best[key] for key in ('architecture', 'precision', 'width',
                  'scales', 'parameters', 'thresholds', 'curriculum_settings')}
        export.update(model_state=model.state_dict(), validation_metrics=scores[selected],
                      experiment=f'codex_{selected}', source_checkpoint=hashes[sources[0].name])
        atomic_save(export, args.output / 'accepted.pt')
    source = args.psd2live / SOURCE_ROOT / 'EyeLegacyMotion.kt'
    report = dict(decision='accepted' if selected else 'rollback',
                  blink_prior_decision='accepted' if accepted else 'rollback',
                  hypothesis='A two-keyform prior reduces held-out blink geometry errors',
                  steps_per_arm=args.steps, seed=42, device=str(device), precision='float64',
                  mastery_limit=settings['mastery'], source_sha256=file_hash(source),
                  checkpoint_hashes=hashes, checkpoint_steps=dict(real=active['steps'], extreme=extreme['step']),
                  selected=selected, lesson_input_hash=data_hash, validation_records=len(validation),
                  validation_pose_indices='0,4,...', practice_records=len(practice),
                  excluded_test_groups=excluded, teacher_audit=audit, metrics=scores)
    (args.output / 'lesson.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(report['decision'], flush=True)
    print('Selected', selected, scores['right_only']['all'], flush=True)


if __name__ == '__main__':
    main()
