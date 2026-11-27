"""Shared checkpoint, telemetry and bounded rollback for MintoAI trainers."""
import copy
import json
import math
from pathlib import Path
import signal
import time

import torch


class StopTraining(Exception):
    pass


class StopFlag:
    def __init__(self):
        self.requested = False
        signal.signal(signal.SIGINT, self.request)
        signal.signal(signal.SIGTERM, self.request)

    def request(self, *_):
        self.requested = True

    def check(self):
        if self.requested:
            raise StopTraining()


def atomic_save(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    torch.save(state, temp)
    # Short retries tolerate antivirus or a concurrent dashboard reader on Windows.
    for attempt in range(10):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.1)


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return copy.deepcopy(value)


class BestTrial:
    """Compare fixed exams, protect prior skills, and retry from the accepted state."""
    def __init__(self, base_lr, patience=6):
        self.base_lr = base_lr
        self.patience = patience
        self.best = None
        self.wait = 0
        self.attempt = 0

    def load(self, state):
        if state:
            self.best = state['best']
            self.wait = state['wait']
            self.attempt = state['attempt']
            self.base_lr = state['base_lr']

    def state(self):
        return dict(best=self.best, wait=self.wait, attempt=self.attempt, base_lr=self.base_lr)

    def observe(self, model, optimizer, scores, extras=None):
        finite = bool(scores) and all(math.isfinite(v) for v in scores.values())
        keys = set(scores)
        if self.best and keys != set(self.best['scores']):
            raise ValueError('Best checkpoint belongs to a different exercise')
        if self.best:
            old = self.best['scores']
            # No skill may degrade more than 2%; aggregate must improve by 0.1%.
            protected = all(scores[k] <= old[k] * 1.02 + 1e-15 for k in keys)
            improvement = sum(scores.values()) < sum(old.values()) * .999
        else:
            protected = improvement = True
        if finite and protected and improvement:
            self.best = dict(model=cpu_copy(model.state_dict()),
                             optimizer=cpu_copy(optimizer.state_dict()),
                             scores=dict(scores), extras=cpu_copy(extras or {}))
            self.wait = 0
            return 'best'
        self.wait += 1
        if self.best is None or self.wait < self.patience:
            return 'trial'
        model.load_state_dict(self.best['model'])
        optimizer.load_state_dict(self.best['optimizer'])
        self.attempt += 1
        # Try three learning rates around the accepted state, with a slower decay.
        factor = (.5, .2, 1.)[(self.attempt - 1) % 3]
        rate = max(1e-12, self.base_lr * factor * .8 ** ((self.attempt - 1) // 3))
        for group in optimizer.param_groups:
            group['lr'] = rate
        self.wait = 0
        return 'rollback'


class Progress:
    def __init__(self, checkpoint, kind):
        self.path = Path(checkpoint).with_suffix('.metrics.jsonl')
        self.kind = kind
        self.last = None

    def write(self, **fields):
        row = dict(kind=self.kind, time=time.time(), **fields)
        # Metrics with undefined scores are stored as null; JSON remains portable.
        def clean(value):
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, dict):
                return {str(k): clean(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [clean(v) for v in value]
            return value
        row = clean(row)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size > 16 * 1024 * 1024:
            self.path.replace(self.path.with_name(self.path.name + '.1'))
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        self.last = row
        print(f"Step {row['step']} | round {row['round']} | {row['task']} | "
              f"error {row.get('value')} | {row['action']} | lr {row['lr']:.3g}", flush=True)
