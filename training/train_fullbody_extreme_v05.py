"""Synthetic full-body deformation with progressively stricter curriculum rounds."""
import argparse
from pathlib import Path
import shutil
import signal
import subprocess
import time

import torch
from torch import nn

# Each point belongs to an explicitly identified procedural part.
PARTS = (
    (0, .73, .18, .21), (0, .71, .12, .06), (0, .82, .14, .04),
    (0, .30, .24, .37), (-.37, .21, .15, .30), (.37, .21, .15, .30),
    (-.57, -.04, .16, .18), (.57, -.04, .16, .18),
    (-.14, -.35, .115, .31), (.14, -.35, .115, .31),
    (-.14, -.77, .10, .24), (.14, -.77, .10, .24),
    (-.17, -.98, .13, .065), (.17, -.98, .13, .065),
)
TASKS = (
    ('head tilt', (0,)), ('one arm', (1,)), ('one leg', (2,)),
    ('facial expressions', (3, 4, 5)), ('both arms and torso', (1, 6, 7)),
    ('steps and balance', (2, 6, 8)), ('combined poses', tuple(range(9))),
)


def sample(n, device, rng, bank, stage, extreme=False):
    def rand(*dims):
        return torch.rand(dims, generator=rng, device=device)

    shape = bank[(rand(n) * len(bank)).long()]
    points = torch.tensor(PARTS, device=device)
    angle = rand(n, len(PARTS)) * (2 * torch.pi)
    radius = rand(n, len(PARTS)).sqrt()
    local = radius[..., None] * torch.stack((angle.cos(), angle.sin()), -1)
    x, y = ((points[None, :, :2] + local * points[None, :, 2:]) *
            shape[:, None, :2]).unbind(-1)
    pose = torch.zeros(n, 9, device=device)
    channels = TASKS[stage][1]
    pose[:, channels] = 2 * rand(n, len(channels)) - 1
    if extreme:
        values = pose[:, channels]
        pose[:, channels] = torch.where(values < 0, -.9 - .1 * values.abs(), .9 + .1 * values.abs())
    head, arm, leg, mouth, smile, brow, lean, arm_r, leg_r = pose.unbind(-1)
    delta = torch.zeros(n, len(PARTS), 2, device=device)

    def rotate(indices, motion, px, py, strength):
        cx, cy = px * shape[:, 0, None], py * shape[:, 1, None]
        xx, yy = x[:, indices] - cx, y[:, indices] - cy
        theta = motion[:, None] * strength * shape[:, 2, None]
        delta[:, indices, 0] += (theta.cos() - 1) * xx - theta.sin() * yy
        delta[:, indices, 1] += theta.sin() * xx + (theta.cos() - 1) * yy

    rotate([0, 1, 2], head, 0, .53, .42)
    rotate([4, 6], arm, -.22, .49, 1.2)
    rotate([5, 7], arm_r, .22, .49, -1.2)
    rotate([8, 10, 12], leg, -.14, -.11, .65)
    rotate([9, 11, 13], leg_r, .14, -.11, -.65)
    rotate(list(range(8)), lean, 0, -.12, .30)
    delta[:, 1, 1] += -.08 * mouth.clamp_min(0) * (1 - local[:, 1, 1]) + .055 * smile * local[:, 1, 0].abs()
    delta[:, 2, 1] += .045 * brow * local[:, 2, 0].abs()
    rest = torch.stack((x, y), -1)
    identity = torch.eye(len(PARTS), device=device).expand(n, -1, -1)
    inputs = torch.cat((rest, local, identity, shape[:, None].expand(-1, len(PARTS), -1),
                        pose[:, None].expand(-1, len(PARTS), -1)), -1)
    return inputs, delta


class GpuGuard:
    def __init__(self, args):
        self.args = args
        self.smi = shutil.which('nvidia-smi')
        if self.smi is None:
            raise SystemExit('nvidia-smi was not found. Add the NVIDIA driver tools to PATH.')
        self.next_check = 0.0

    def wait(self, before_resume=None, stopped=lambda: False):
        if time.monotonic() < self.next_check:
            return
        paused = False
        good = 0
        while not stopped():
            try:
                output = subprocess.check_output(
                    [self.smi, '-i', self.args.smi_gpu,
                     '--query-gpu=temperature.gpu,memory.free',
                     '--format=csv,noheader,nounits'],
                    text=True, timeout=5, stderr=subprocess.DEVNULL,
                )
                temp, free = map(int, output.strip().split(','))
                if paused:
                    ready = temp <= self.args.resume_temp and free >= self.args.resume_free
                else:
                    ready = temp < self.args.pause_temp and free >= self.args.pause_free
                status = f'{temp} C | {free} MiB free'
            except (OSError, ValueError, subprocess.SubprocessError):
                ready = False
                status = 'GPU telemetry unavailable'
            good = good + 1 if ready else 0
            if ready and (not paused or good >= 3):
                if paused:
                    print(f'GPU RESUME | {status}', flush=True)
                self.next_check = time.monotonic() + self.args.gpu_poll
                return
            if not paused:
                paused = True
                if before_resume is not None:
                    before_resume()
                print(f'GPU PAUSE | {status}', flush=True)
            print(f'GPU WAIT | {status} | stable {good}/3', flush=True)
            time.sleep(self.args.gpu_poll)


from training_progress import BestTrial, Progress, StopFlag, StopTraining, atomic_save


@torch.no_grad()
def assess(model, bank, count, device, seed, stop):
    model.eval()
    rng = torch.Generator(device=device).manual_seed(seed)
    details = {}
    for task in range(count):
        stop.check()
        for extreme in (False, True):
            inputs, target = sample(384, device, rng, bank, task, extreme)
            residual = model(inputs) - target
            error = residual.square().sum(-1)
            magnitude = target.square().sum(-1)
            active = magnitude > 1e-10
            scale = magnitude[active].mean().clamp_min(1e-12)
            moving = (error[active].mean() / scale).sqrt()
            static = (error[~active].sum() / (~active).sum().clamp_min(1) / scale).sqrt()
            key = f'{task}/' + ('extreme' if extreme else 'normal')
            details[key] = float(residual.abs().amax() / scale.sqrt())
    return details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('checkpoints/fullbody_extreme_v04.pt'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--steps', type=int, default=0)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--patience', type=int, default=6)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--mastery', type=float, default=1e-9)
    parser.add_argument('--extreme-mastery', type=float, default=1e-9)
    parser.add_argument('--precision', choices=('float32', 'float64'), default='float64')
    parser.add_argument('--round-factor', type=float, default=.8)
    parser.add_argument('--passes', type=int, default=3)
    parser.add_argument('--pause-temp', type=int, default=80)
    parser.add_argument('--resume-temp', type=int, default=72)
    parser.add_argument('--pause-free', type=int, default=384)
    parser.add_argument('--resume-free', type=int, default=768)
    parser.add_argument('--gpu-poll', type=float, default=3)
    parser.add_argument('--smi-gpu', default='0')
    args = parser.parse_args()
    if (min(args.batch, args.eval_every, args.patience, args.passes) < 1 or args.steps < 0
            or not 0 < args.mastery < 1 or not 0 < args.extreme_mastery < 1
            or not 0 < args.round_factor < 1 or not 0 < args.resume_temp < args.pause_temp
            or not 0 < args.pause_free < args.resume_free or not 0 < args.gpu_poll <= 60):
        parser.error('Invalid settings')
    if args.output.is_file() != args.resume:
        parser.error('Existing checkpoint requires --resume')
    stop = StopFlag()
    device = torch.device(args.device)
    torch.set_default_dtype(getattr(torch, args.precision))
    torch.manual_seed(args.seed)
    if device.type == 'cpu':
        torch.set_num_threads(4)
    guard = GpuGuard(args) if device.type == 'cuda' else None
    if guard:
        guard.wait(stopped=lambda: stop.requested)
    if stop.requested:
        return
    rng = torch.Generator(device=device).manual_seed(args.seed)
    val_rng = torch.Generator(device=device).manual_seed(args.seed + 1)
    train_bank = .72 + .56 * torch.rand(4096, 3, generator=rng, device=device)
    val_bank = .72 + .56 * torch.rand(384, 3, generator=val_rng, device=device)
    width = 192
    model = nn.Sequential(nn.Linear(16 + len(PARTS), width), nn.SiLU(),
                          nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width),
                          nn.SiLU(), nn.Linear(width, 2)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4)
    trial = BestTrial(4e-4, args.patience)
    progress = Progress(args.output, 'extreme')
    step = stage = streak = 0
    round_number = 1
    limits = [args.mastery, args.extreme_mastery]
    batch = args.batch
    review = 0
    if args.resume:
        state = torch.load(args.output, map_location='cpu', weights_only=True)
        if state.get('schema') not in (4, 5):
            parser.error('Expected a V0.4 or V0.5 checkpoint')
        if 'metric_version' not in state:
            backup = args.output.with_name(args.output.stem + '.before_adaptive.pt')
            if not backup.exists():
                shutil.copy2(args.output, backup)
                print(f'Previous checkpoint backed up: {backup}', flush=True)
        model.load_state_dict(state['model_state'])
        opt.load_state_dict(state['optimizer'])
        train_bank = state['train_bank'].to(device=device, dtype=torch.get_default_dtype())
        val_bank = state['val_bank'].to(device=device, dtype=torch.get_default_dtype())
        step, stage, round_number = state['step'], state['stage'], state['round_number']
        limits = list(state['mastery_limits'])
        args.round_factor = state['round_factor']
        if state['device'] == args.device:
            rng.set_state(state['rng'])
            val_rng.set_state(state['validation_rng'])
        batch = min(batch, state['batch'])
        if state.get('metric_version') == 'max_coordinate_v1' and state.get('precision') == args.precision:
            trial.load(state.get('adaptive'))
        else:
            stage, round_number = 0, 1
            limits = [args.mastery, args.extreme_mastery]
            print('Preserved weights; requalifying from task 1 with the new metric and precision.', flush=True)
        review = 500
        del state
        print(f'RESUMED step {step} | round {round_number} | task {stage+1}', flush=True)
    # Reproducible finite practice sets for the first curriculum round.
    # The validation bank and exam seeds are independent of these examples.
    fixed_rng = torch.Generator(device=device).manual_seed(args.seed + 9000)
    fixed_practice = [sample(2048, device, fixed_rng, train_bank, task)
                      for task in range(len(TASKS))]
    best_path = args.output.with_name(args.output.stem + '.best.pt')
    last_metrics = None

    def save():
        state = dict(schema=5, metric_version='max_coordinate_v1', precision=args.precision, device=args.device, model_state=model.state_dict(),
                     optimizer=opt.state_dict(), rng=rng.get_state(), validation_rng=val_rng.get_state(),
                     train_bank=train_bank, val_bank=val_bank, step=step, stage=stage, streak=streak,
                     batch=batch, width=width, tasks=TASKS, round_number=round_number,
                     mastery_limits=limits, round_factor=args.round_factor,
                     learning_phase='fixed' if round_number % 2 else 'random',
                     adaptive=trial.state(), latest_metrics=last_metrics)
        atomic_save(state, args.output)

    def exam():
        return assess(model, val_bank, stage + 1, device, args.seed + 300, stop)

    def report(scores, action, elapsed):
        normal = max(v for k, v in scores.items() if k.endswith('/normal'))
        extreme = max(v for k, v in scores.items() if k.endswith('/extreme'))
        best = trial.best['scores'] if trial.best else scores
        progress.write(step=step, round=round_number, current=stage, total=len(TASKS),
                       task=TASKS[stage][0], value=normal, second=extreme,
                       best=max(best.values()), details=scores, limits=limits,
                       passed=streak, passes=args.passes, batch=batch,
                       lr=opt.param_groups[0]['lr'], action=action,
                       attempts=trial.attempt, eval_seconds=elapsed,
                       phase='fixed' if round_number % 2 else 'random')
        return progress.last

    end = step + args.steps if args.steps else None
    try:
        if trial.best is None:
            start = time.monotonic()
            scores = exam()
            action = trial.observe(model, opt, scores)
            last_metrics = report(scores, action, time.monotonic() - start)
            atomic_save(dict(model_state=trial.best['model'], scores=scores, width=width,
                             schema=5, stage=stage, round_number=round_number), best_path)
            save()
        while end is None or step < end:
            stop.check()
            if guard:
                guard.wait(torch.cuda.empty_cache, stopped=lambda: stop.requested)
            stop.check()
            task = stage
            if review:
                task = step % (stage + 1)
            elif stage and step % 4 == 3:
                weights = torch.tensor([
                    max(trial.best['scores'].get(f'{i}/normal', 0.),
                        trial.best['scores'].get(f'{i}/extreme', 0.)) + .01
                    for i in range(stage)], device=device)
                task = int(torch.multinomial(weights, 1, generator=rng))
            model.train()
            try:
                if round_number % 2:
                    fixed_inputs, fixed_target = fixed_practice[task]
                    chosen = torch.randint(len(fixed_inputs), (batch,), generator=rng, device=device)
                    inputs, target = fixed_inputs[chosen], fixed_target[chosen]
                else:
                    # New continuous proportions and poses for generalization practice.
                    new_shapes = .72 + .56 * torch.rand(max(batch, 384), 3, generator=rng, device=device)
                    inputs, target = sample(batch, device, rng, new_shapes, task)
                pred = model(inputs)
                active = target.square().sum(-1) > 1e-10
                # Normalize each exercise so small movements still produce useful gradients.
                scale = target.square().sum(-1)[active].mean().detach().clamp_min(1e-8)
                error = (pred - target).square().sum(-1) / scale
                loss = torch.where(active, error * 3, error).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss')
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
            except torch.cuda.OutOfMemoryError:
                if device.type != 'cuda' or batch == 1:
                    raise
                opt.zero_grad(set_to_none=True)
                inputs = target = pred = error = loss = active = scale = None
                torch.cuda.empty_cache()
                batch = max(1, batch // 2)
                continue
            opt.step()
            step += 1
            review = max(0, review - 1)
            inputs = target = pred = error = loss = active = scale = None
            if step % args.eval_every and step != end:
                continue
            start = time.monotonic()
            scores = exam()
            action = trial.observe(model, opt, scores)
            passed = action != 'rollback' and all(
                v <= limits[int(k.endswith('/extreme'))] for k, v in scores.items())
            if passed and not review:
                # Independent samples gate promotion; never used to choose the best trial.
                fresh = assess(model, val_bank, stage + 1, device, args.seed + step + 1000, stop)
                passed = all(v <= limits[int(k.endswith('/extreme'))] for k, v in fresh.items())
            streak = streak + 1 if passed and not review else 0
            last_metrics = report(scores, action, time.monotonic() - start)
            if action == 'best':
                atomic_save(dict(model_state=trial.best['model'], scores=scores, width=width,
                                 schema=5, stage=stage, round_number=round_number), best_path)
            if streak >= args.passes:
                stage += 1
                streak = 0
                if stage == len(TASKS):
                    stage = 0
                    round_number += 1
                trial = BestTrial(4e-4, args.patience)
                start = time.monotonic()
                scores = exam()
                trial.observe(model, opt, scores)
                last_metrics = report(scores, 'new_exercise', time.monotonic() - start)
                atomic_save(dict(model_state=trial.best['model'], scores=scores, width=width,
                                 schema=5, stage=stage, round_number=round_number), best_path)
            save()
    except StopTraining:
        print('Stop requested. Saving completed updates.', flush=True)
    finally:
        save()
        print(f'Saved step {step}: {args.output}', flush=True)


if __name__ == '__main__':
    main()
