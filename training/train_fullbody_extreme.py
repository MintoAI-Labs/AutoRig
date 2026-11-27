"""Synthetic full-body vertex deformation with a mastery-based curriculum."""
import argparse
from pathlib import Path

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
    args = parser.parse_args()
    if min(args.steps, args.batch, args.eval_every, args.passes) < 1 or not 0 < args.mastery < 1:
        parser.error('Invalid steps, batch, evaluation interval, passes or mastery')
    if args.output.exists():
        parser.error('Choose a fresh output path')
    device = torch.device(args.device)
    if device.type == 'cpu':
        torch.set_num_threads(min(4, torch.get_num_threads()))
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
    for step in range(1, args.steps + 1):
        task = stage
        if stage and step % 4 == 0:
            task = (step // 4 - 1) % stage
        inputs, target = sample(args.batch, device, rng, train_bank, task)
        prediction = model(inputs)
        active = target.norm(dim=-1) > 1e-5
        error = (prediction - target).square().sum(-1)
        loss = torch.where(active, 3 * error, error).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
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
