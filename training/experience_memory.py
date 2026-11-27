"""Measured experiment memory. State in checkpoints drives reproducible decisions."""
import json
import math
import sqlite3
import time

POLICIES = {
    'balanced': (1., .5, .1, 1.),
    'rule_rotation': (1., .5, .1, 1.),
    'rule_tps': (1., .5, .1, 1.),
    'rule_hybrid': (1., .5, .1, 1.),
    'hard_vertices': (.5, 2., .1, .5),
    'static_regions': (.5, 1., .1, 3.),
    'edge_coherence': (1., 1., 1., 1.),
    'psd_hint': (1., .5, .1, 1.),
}


class Experience:
    def __init__(self, path, saved=None):
        self.db = sqlite3.connect(path, timeout=10)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS events '
                        '(id INTEGER PRIMARY KEY, time REAL, step INTEGER, kind TEXT, data TEXT)')
        if saved and 'balanced' in saved:
            saved={'moving_vertices':saved}
        self.arms = saved or {}
        for context in ('moving_vertices','static_drift'):
            self.arms.setdefault(context,{})
            for name in POLICIES:
                self.arms[context].setdefault(name,dict(n=0,reward=0.,blocked=0))

    def log(self, step, kind, **data):
        self.db.execute('INSERT INTO events(time,step,kind,data) VALUES(?,?,?,?)',
                        (time.time(), step, kind, json.dumps(data, allow_nan=False)))
        self.db.commit()

    def select(self, trial_number, context='moving_vertices', allow_psd=True):
        arms=self.arms[context]
        available = [k for k, v in arms.items() if v['blocked'] <= trial_number and (allow_psd or k!='psd_hint')]
        if not available:
            available = [k for k in POLICIES if allow_psd or k!='psd_hint']
        total = sum(v['n'] for v in arms.values()) + 1
        def score(key):
            arm = arms[key]
            if not arm['n']:
                return math.inf
            return arm['reward'] / arm['n'] + .3 * math.sqrt(math.log(total + 1) / arm['n'])
        return max(available, key=score)

    def feedback(self, policy, reward, trial_number, context='moving_vertices'):
        arm = self.arms[context][policy]
        arm['n'] += 1
        arm['reward'] += max(-1., min(1., reward))
        if reward < -.02:
            arm['blocked'] = trial_number + 3

    def close(self):
        self.db.close()
