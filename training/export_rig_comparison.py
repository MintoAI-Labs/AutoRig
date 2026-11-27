"""Export a private, offline texture comparison for the locked Real Rig evaluations."""
import argparse
import base64
import gzip
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
import torch

from codex_blink_lesson import DTYPE, geometry
from train_real_rig_curriculum import split_poses
from train_real_rig_v03 import mesh_patches, resolve_textures
from train_real_rig_v04 import PARAMS, ROOT, ParameterRig
from train_real_rig_v01 import load_record
from unified_sources import family, file_hash, to_device


def pack(array, dtype):
    return base64.b64encode(gzip.compress(np.asarray(array, dtype=dtype).tobytes(), compresslevel=3)).decode()


def texture(path):
    with Image.open(path) as source:
        image = source.convert('RGBA')
    image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    stream = io.BytesIO()
    image.save(stream, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()


def attach_render_orders(records, models, sources):
    """Capture Core renderOrders at the same poses without modifying NPZ data."""
    edge = Path(r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe')
    core = ROOT / 'SDK/CubismSdkForWeb-5-r.5/Core/live2dcubismcore.js'
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=edge)
        page = browser.new_page()
        page.add_script_tag(path=core)
        for record in records:
            moc, aliases = sources[record['source']]
            orders = page.evaluate("""({encoded, param, values, count}) => {
              const bytes = Uint8Array.from(atob(encoded), char => char.charCodeAt(0));
              const moc = Live2DCubismCore.Moc.fromArrayBuffer(bytes.buffer);
              const model = Live2DCubismCore.Model.fromMoc(moc);
              try {
                const p = model.parameters, id = p.ids.indexOf(param);
                if (id < 0 || model.drawables.count !== count) throw Error('Drawable topology mismatch');
                p.values.set(p.defaultValues); model.update();
                return values.map(value => {
                  p.values.set(p.defaultValues); p.values[id] = value; model.update();
                  const orders = model.getRenderOrders();
                  if (orders.length !== count) throw Error('Unexpected number of render orders');
                  return Array.from(orders);
                });
              } finally {
                model.release(); moc._release();
              }
            }""", dict(encoded=base64.b64encode(moc.read_bytes()).decode(),
                       param=aliases[record['param']], values=record['values'],
                       count=len(models[record['model']]['meshes'])))
            record['renderOrder'] = pack(orders, '<i4')
        browser.close()

def stats(error, mask, exam):
    if not mask.any():
        return None
    filtered = error[:, mask, :]
    squares = np.square(filtered, dtype=np.float64)
    maxima = np.abs(filtered).max(axis=(1, 2))
    worst = exam[int(np.argmax(maxima[exam]))]
    return dict(rms=float(squares[exam].mean()**.5), max=float(maxima[worst]), pose=worst)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output file')
    torch.set_num_threads(4)
    device = torch.device('cpu')
    folder = ROOT / 'checkpoints'
    files = [folder / 'real_rig_curriculum.best.pt',
             folder / 'codex_gradient_seed42_step50/projected.step050.candidate.pt']
    hashes = [file_hash(path) for path in files]
    checkpoints = [torch.load(path, map_location='cpu', weights_only=True) for path in files]
    best, candidate = checkpoints
    settings = best['curriculum_settings']
    for key in ('scales', 'thresholds', 'curriculum_settings'):
        if best[key] != candidate[key]:
            raise ValueError(f'Incompatible checkpoint setting: {key}')
    if (candidate['step'] != 50 or candidate['config']['seed'] != 42
            or candidate['config']['source_hashes']['best'] != hashes[0]):
        raise ValueError('Expected the verified seed 42 step 50 candidate')
    networks = []
    for checkpoint in checkpoints:
        net = ParameterRig(checkpoint['width']).to(device=device, dtype=DTYPE)
        net.load_state_dict(checkpoint['model_state'])
        networks.append(net.eval())

    root, model_root = Path(settings['data']), Path(settings['models'])
    old_group = settings['groups'][0]
    validation_family, test_family = family(settings['validation']), family(settings['holdout'])
    records, models, sources = [], [], {}
    bounds = np.array([np.inf, np.inf, -np.inf, -np.inf])
    for manifest in sorted(root.glob('*/source.json'), key=lambda p: (p.parent.name != old_group, p.parent.name)):
        meta = json.loads(manifest.read_text(encoding='utf-8-sig'))
        source_family = family(meta['model_path'])
        if source_family == test_family:
            continue
        is_old = manifest.parent.name == old_group
        if not is_old and source_family != validation_family:
            continue
        moc = model_root / meta['model_path']
        sources[moc.stem] = (moc, meta['original_parameter_ids'])
        paths = sorted(manifest.parent.glob('*.npz'))
        if is_old:
            paths = [path for path in paths if path.name.startswith('ParamEyeROpen')]
        paths.sort(key=lambda path: ('EyeROpen' not in path.name, 'EyeLOpen' not in path.name, path.name))
        first = paths[0]
        patches, _ = mesh_patches(first, model_root)
        rgba = torch.from_numpy(patches).to(torch.float32).permute(0, 3, 1, 2) / 255
        appearance = torch.cat((rgba[:, :3] * rgba[:, 3:4], rgba[:, 3:4]), 1).contiguous()
        model_index = len(models)
        with np.load(first, allow_pickle=False) as snapshot:
            rest = snapshot['rest']
            uv = snapshot['uv']
            triangles = snapshot['triangles']
            offsets = snapshot['triangle_offsets']
            mesh_count = len(offsets) - 1
            mask_offsets = snapshot['mask_offsets']
            masks = snapshot['mask_indices']
            models.append(dict(
                rest=pack(rest, '<f4'), uv=pack(uv, '<f4'), triangles=pack(triangles, '<u4'),
                atlases=[texture(path) for path in resolve_textures(manifest.parent, model_root)],
                meshes=[dict(first=int(offsets[i])*3, count=int(offsets[i+1]-offsets[i])*3,
                             atlas=int(snapshot['texture_indices'][i]), flags=int(snapshot['constant_flags'][i]),
                             masks=masks[mask_offsets[i]:mask_offsets[i+1]].tolist())
                        for i in range(mesh_count)],
                vertices=len(rest)))
        for path in paths:
            row = load_record(path, root, device)
            if family(row['source']) != source_family or row['param'] not in PARAMS:
                raise ValueError(f'Snapshot identity differs: {path}')
            with np.load(path, allow_pickle=False) as snapshot:
                if (not np.array_equal(rest, snapshot['rest']) or not np.array_equal(uv, snapshot['uv'])
                        or not np.array_equal(triangles, snapshot['triangles'])):
                    raise ValueError(f'Topology varies across control sweeps: {path}')
                values = snapshot['parameter_values'].tolist()
                opacity = pack(snapshot['opacities'], '<f4')
                order = pack(snapshot['draw_orders'], '<i4')
            param = row['param']
            labels = row['delta']
            moving = np.any(labels != 0, axis=(0, 2))
            exam = split_poses(row)[1] if is_old else list(range(0, len(values), 4))
            model_row = dict(row, visual=appearance, mesh_active=torch.zeros(row['nm']),
                             semantic=torch.full((row['nm'],), 14))
            model_row = to_device(model_row, device, DTYPE)
            scale = best['scales'][param]
            deltas, errors = [labels], []
            for net in networks:
                features, logits = net.encode(model_row)
                threshold = best['thresholds'][param]
                if is_old and threshold != candidate['measurement']['primary']['old_gate']:
                    raise ValueError('Stored gate differs from curriculum evaluator')
                gate = (logits.sigmoid() >= threshold)[row['mesh'], None]
                normalized = torch.cat([geometry(net.heads[param], features, samples) * gate
                                        for samples in model_row['t'].split(8)]).numpy()
                residual = normalized - labels.astype(np.float64) / scale
                errors.append(dict(moving=stats(residual, moving, exam),
                                   static=stats(residual, ~moving, exam),
                                   all=stats(residual, np.ones_like(moving), exam)))
                deltas.append(normalized * scale)
            for delta in deltas:
                posed = rest[None] + delta
                bounds[:2] = np.minimum(bounds[:2], posed.min(axis=(0, 1)))
                bounds[2:] = np.maximum(bounds[2:], posed.max(axis=(0, 1)))
            records.append(dict(
                model=model_index, scope='Bài mắt cũ' if is_old else 'Hiyori',
                source=Path(row['source']).stem, param=param, scale=scale, values=values, exam=exam,
                moving=pack(moving, 'u1'), opacity=opacity, order=order,
                deltas=pack(np.stack(deltas), '<f4'), errors=errors,
            ))
            print(records[-1]['source'], param, len(exam), flush=True)
    if len(records) != 1 + candidate['config']['validation_records']:
        raise ValueError('Unexpected number of old and Hiyori records')
    attach_render_orders(records, models, sources)
    if [file_hash(path) for path in files] != hashes:
        raise RuntimeError('Input checkpoint changed during export')
    payload = dict(records=records, models=models, bounds=bounds.tolist(),
                   seed=42, step=50, hashes=hashes, score=candidate['measurement']['primary'])
    template = Path(__file__).with_name('rig_comparison_template.html').read_text(encoding='utf-8-sig')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(template.replace('__DATA__', json.dumps(
            payload, separators=(',', ':'), ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')))
    print('Saved local texture viewer', args.output, flush=True)


if __name__ == '__main__':
    main()
