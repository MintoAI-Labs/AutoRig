"""Synthetic eye geometry for testing MintoAI's dataset format only."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'dataset' / 'synthetic_eye_v01'
OUT.mkdir(parents=True, exist_ok=True)
# A strip of triangles connecting two eyelid curves; original synthetic data.
x = np.linspace(-1, 1, 17, dtype=np.float32)
arch = np.sqrt(np.maximum(0, 1 - x*x))
# Keep corners distinct to avoid degenerate triangles at endpoints.
arch = np.maximum(arch, 0.04)
base = np.concatenate([np.column_stack([x, .42*arch]),
                       np.column_stack([x, -.28*arch])]).astype(np.float32)
n = len(x)
triangles = np.array([(i, i+1, i+n) for i in range(n-1)] +
                     [(i+1, i+n+1, i+n) for i in range(n-1)], dtype=np.int32)
values = np.linspace(0, 1, 21, dtype=np.float32)
poses = np.repeat(base[None, :, :], len(values), axis=0)
# Illustrative eyelid closure, not learned and not artist rig data.
for i, p in enumerate(values):
    poses[i, :n, 1] = (.42*p - .10*(1-p))*arch
    poses[i, n:, 1] = (-.28*p - .10*(1-p))*arch
# At the closed endpoint, zero-area triangles are intentional.
delta = poses - base[None, :, :]
np.savez_compressed(OUT / 'mesh.npz', vertices_rest=base,
                    triangles=triangles, parameter_values=values[:, None],
                    vertices_posed=poses, vertex_delta=delta)
meta = {'schema_version': '0.1', 'source': 'synthetic', 'training_eligible': False,
        'purpose': 'pipeline smoke test only', 'parameter_ids': ['ParamEyeLOpen'],
        'parameter_default': [1.0], 'coordinate_system': 'normalized y-up',
        'vertex_groups': {'upper_lid': list(range(n)), 'lower_lid': list(range(n, 2*n))}}
(OUT / 'metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
with np.load(OUT / 'mesh.npz', allow_pickle=False) as data:
    assert data['vertices_posed'].shape == (21, 34, 2)
    assert data['triangles'].shape == (32, 3)
    assert np.isfinite(data['vertices_posed']).all()
    assert np.allclose(data['vertices_rest'] + data['vertex_delta'], data['vertices_posed'])
    assert np.allclose(data['vertices_posed'][-1], data['vertices_rest'])
print('MintoAI synthetic dataset V0.1')
print('Parameter: ParamEyeLOpen')
print('Vertices:', len(base))
print('Triangles:', len(triangles))
print('Poses:', len(values))
print('PASS: dataset written and validated')
print('Output:', OUT)
