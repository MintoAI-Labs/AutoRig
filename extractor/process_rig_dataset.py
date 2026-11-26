"""MintoAI raw-0.1 validator and optional authorized-data converter."""
import argparse
import json
from pathlib import Path
import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def process(path, output, permission):
    d = json.loads(path.read_text(encoding='utf-8-sig'))
    require(d['schema_version'] == 'raw-0.1', 'Unsupported schema')
    meshes, samples = d['meshes'], d['samples']
    m, s = len(meshes), len(samples)
    require(m > 0 and s > 1, 'Need meshes and at least two poses')
    ids = [x['id'] for x in meshes]
    require(len(set(ids)) == m, 'Duplicate mesh IDs')
    counts = np.array([x['vertex_count'] for x in meshes], dtype=np.int64)
    require(np.all(counts > 0), 'Empty mesh')
    offsets = np.r_[0, np.cumsum(counts)]
    rest, uv, faces, face_offsets = [], [], [], [0]
    mask_indices, mask_offsets = [], [0]
    for i, mesh in enumerate(meshes):
        n = counts[i]
        for key, dest in [('vertices_rest', rest), ('uv', uv)]:
            arr = np.asarray(mesh[key], dtype=np.float32)
            require(arr.size == 2*n and np.isfinite(arr).all(), f'{ids[i]} invalid {key}')
            dest.append(arr.reshape(-1, 2))
        tri = np.asarray(mesh['triangles'], dtype=np.int64)
        require(tri.size % 3 == 0 and np.all((tri >= 0) & (tri < n)), f'{ids[i]} invalid triangles')
        faces.append(tri.reshape(-1, 3) + offsets[i])
        face_offsets.append(face_offsets[-1] + tri.size//3)
        masks = np.asarray(mesh['mask_indices'], dtype=np.int64)
        require(np.all((masks >= 0) & (masks < m)), f'{ids[i]} invalid mask')
        mask_indices.extend(masks.tolist())
        mask_offsets.append(len(mask_indices))
    rest = np.concatenate(rest)
    uv = np.concatenate(uv)
    faces = np.concatenate(faces)
    poses = []
    for j, sample in enumerate(samples):
        require(len(sample['vertices_posed']) == m, f'Pose {j} wrong mesh count')
        arrays = []
        for i, raw in enumerate(sample['vertices_posed']):
            a = np.asarray(raw, dtype=np.float32)
            require(a.size == 2*counts[i] and np.isfinite(a).all(), f'Pose {j}, {ids[i]} invalid coordinates')
            arrays.append(a.reshape(-1, 2))
        poses.append(np.concatenate(arrays))
    poses = np.stack(poses)
    values = np.array([x['value'] for x in samples], dtype=np.float32)
    require(np.isfinite(values).all() and np.all(np.diff(values) > 0), 'Invalid parameter sweep')
    p = d['parameter']
    require(np.isfinite([p['min'],p['max'],p['default']]).all() and p['min'] < p['max'], 'Invalid parameter range')
    require(values.min() >= p['min']-1e-5 and values.max() <= p['max']+1e-5, 'Values outside parameter range')
    opacity = np.asarray([x['opacities'] for x in samples], dtype=np.float32)
    order = np.asarray([x['draw_orders'] for x in samples], dtype=np.int64)
    flags = np.asarray([x['dynamic_flags'] for x in samples], dtype=np.int64)
    rest_opacity = np.asarray(d['rest_opacities'], dtype=np.float32)
    rest_order = np.asarray(d['rest_draw_orders'], dtype=np.int64)
    require(opacity.shape == order.shape == flags.shape == (s,m), 'Invalid pose attributes')
    require(rest_opacity.shape == rest_order.shape == (m,), 'Invalid rest attributes')
    require(np.isfinite(opacity).all() and np.isfinite(rest_opacity).all(), 'Invalid opacity')
    delta = poses - rest[None,:,:]
    magnitude = np.linalg.norm(delta, axis=2)
    moved = np.array([magnitude[:,offsets[i]:offsets[i+1]].max() > 1e-6 for i in range(m)])
    opacity_changed = np.any(np.abs(opacity-rest_opacity) > 1e-6, axis=0)
    order_changed = np.any(order != rest_order, axis=0)
    lo, hi = rest.min(axis=0), rest.max(axis=0)
    center = (lo+hi)/2
    scale = float((hi-lo).max())
    require(scale > 0 and np.isfinite(scale), 'Invalid normalization scale')
    rest_norm = (rest-center)/scale
    delta_norm = delta/scale
    print('MintoAI Dataset Processor V0.1')
    print('Source:', d['source'])
    print('Parameter:', p['id'])
    print('Meshes:', m)
    print('Vertices:', len(rest))
    print('Triangles:', len(faces))
    print('Poses:', s)
    print('Moving meshes:', int(moved.sum()))
    print('Opacity-changing meshes:', int(opacity_changed.sum()))
    print('Draw-order-changing meshes:', int(order_changed.sum()))
    print('Normalized rest shape:', rest_norm.shape)
    print('Normalized displacement shape:', delta_norm.shape)
    print('PASS: raw data validated and normalized in memory')
    if permission:
        output.mkdir(parents=True, exist_ok=True)
        target = output/(path.stem+'.npz')
        np.savez_compressed(target, rest=rest_norm, delta=delta_norm, uv=uv, triangles=faces,
            mesh_ids=np.asarray(ids), mesh_offsets=offsets, triangle_offsets=np.asarray(face_offsets),
            mask_indices=np.asarray(mask_indices,dtype=np.int64), mask_offsets=np.asarray(mask_offsets),
            texture_indices=np.asarray([x['texture_index'] for x in meshes]),
            parent_part_indices=np.asarray([x['parent_part_index'] for x in meshes]),
            constant_flags=np.asarray([x['constant_flags'] for x in meshes]),
            moving_meshes=moved, opacities=opacity, rest_opacities=rest_opacity,
            draw_orders=order, rest_draw_orders=rest_order, dynamic_flags=flags,
            parameter_values=values, parameter_id=np.asarray(p['id']),
            parameter_range=np.asarray([p['min'],p['max'],p['default']],dtype=np.float32),
            all_parameter_ids=np.asarray(d['all_parameter_ids']), all_parameter_defaults=np.asarray(d['all_parameter_defaults'],dtype=np.float32),
            center=center, scale=np.float32(scale), source=np.asarray(d['source']), permission_note=np.asarray(permission))
        with np.load(target, allow_pickle=False) as check:
            require(check['delta'].shape == delta_norm.shape, 'NPZ verification failed')
        print('Saved:', target)
    else:
        print('Audit only: no training file saved (permission not supplied).')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input', type=Path)
    ap.add_argument('--output', type=Path, default=Path('dataset/processed'))
    ap.add_argument('--permission', help='Reference to verified permission for SDK/data use in AI training; omit for audit only')
    args = ap.parse_args()
    process(args.input, args.output, args.permission)
