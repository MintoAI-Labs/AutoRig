"""Synthetic full-body vertex deformation with a mastery-based curriculum."""
import argparse
from pathlib import Path
import shutil
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


def sample(n, device, rng, bank, stage):
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

    def wait(self, before_resume=None):
        if time.monotonic() < self.next_check:
            return
        paused = False
        good = 0
        while True:
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=60000)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--mastery', type=float, default=.20, help='Maximum relative error per validation task')
    parser.add_argument('--passes', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--output', type=Path, default=Path('checkpoints/fullbody_extreme.pt'))
    parser.add_argument('--pause-temp', type=int, default=80)
    parser.add_argument('--resume-temp', type=int, default=72)
    parser.add_argument('--pause-free', type=int, default=384, help='Free MiB')
    parser.add_argument('--resume-free', type=int, default=768, help='Free MiB')
    parser.add_argument('--gpu-poll', type=float, default=3)
    parser.add_argument('--smi-gpu', default='0', help='nvidia-smi GPU index or UUID')
    args = parser.parse_args()
    if not (0 < args.resume_temp < args.pause_temp and
            0 < args.pause_free < args.resume_free and 0 < args.gpu_poll <= 60):
        parser.error('Invalid GPU pause/resume thresholds')
    if min(args.steps, args.batch, args.eval_every, args.passes) < 1 or not 0 < args.mastery < 1:
        parser.error('Invalid steps, batch, evaluation interval, passes or mastery')
    if args.output.exists():
        parser.error('Choose a fresh output path')
    device = torch.device(args.device)
    if device.type == 'cpu':
        torch.set_num_threads(min(4, torch.get_num_threads()))
    guard = GpuGuard(args) if device.type == 'cuda' else None
    if guard:
        guard.wait()
    torch.manual_seed(args.seed)
    rng = torch.Generator(device=device).manual_seed(args.seed)
    val_rng = torch.Generator(device=device).manual_seed(args.seed + 1)
    train_bank = .72 + .56 * torch.rand(4096, 3, generator=rng, device=device)
    val_bank = .72 + .56 * torch.rand(384, 3, generator=val_rng, device=device)
    validation = [sample(384, device, val_rng, val_bank, stage) for stage in range(len(TASKS))]
    width = 192
    model = nn.Sequential(nn.Linear(16 + len(PARTS), width), nn.SiLU(),
                          nn.Linear(width, width), nn.SiLU(),
                          nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 2)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4)
    stage = streak = 0
    best = float('inf')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    batch_size = args.batch
    for step in range(1, args.steps + 1):
        task = stage
        if stage and step % 4 == 0:
            task = (step // 4 - 1) % stage
        if guard:
            guard.wait(torch.cuda.empty_cache)
        while True:
            try:
                inputs, target = sample(batch_size, device, rng, train_bank, task)
                prediction = model(inputs)
                active = target.norm(dim=-1) > 1e-5
                error = (prediction - target).square().sum(-1)
                loss = torch.where(active, 3 * error, error).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
            except torch.cuda.OutOfMemoryError:
                if not guard or batch_size == 1:
                    raise
            else:
                break
            opt.zero_grad(set_to_none=True)
            inputs = target = prediction = active = error = loss = None
            torch.cuda.empty_cache()
            guard.next_check = 0.0
            guard.wait()
            batch_size = max(1, batch_size // 2)
            print(f'CUDA OOM | retry step {step} with batch {batch_size}', flush=True)
        opt.step()
        inputs = target = prediction = active = error = loss = None
        if step % args.eval_every and step != args.steps:
            continue
        scores = []
        with torch.no_grad():
            for inputs, target in validation[:stage + 1]:
                error = (model(inputs) - target).square().sum(-1)
                magnitude = target.square().sum(-1)
                active = magnitude > 1e-10
                relative = (error[active].mean() / magnitude[active].mean()).sqrt()
                drift = (error[~active].sum() / (~active).sum().clamp_min(1) / magnitude[active].mean()).sqrt()
                scores.append(max(float(relative), float(drift)))
        worst = max(scores)
        print(f'Step {step} | {TASKS[stage][0]} | task scores {scores} | worst {worst:.3f}', flush=True)
        if worst < best:
            best = worst
            tmp = args.output.with_suffix('.tmp')
            torch.save(dict(model_state=model.state_dict(), step=step, stage=stage,
                            tasks=TASKS, scores=scores, width=width), tmp)
            tmp.replace(args.output)
        streak = streak + 1 if worst <= args.mastery else 0
        if streak == args.passes:
            if stage == len(TASKS) - 1:
                print('All curriculum stages passed.', flush=True)
                break
            stage += 1
            streak = 0
            best = float('inf')
            print(f'Next exercise: {TASKS[stage][0]}', flush=True)


if __name__ == '__main__':
    main()