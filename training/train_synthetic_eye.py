"""MintoAI V0.1: synthetic-only mesh deformation smoke test.
Run from project/training; no artist data is read.
"""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'dataset' / 'synthetic_eye_v01'
OUT = ROOT / 'checkpoints'
OUT.mkdir(parents=True, exist_ok=True)

def main():
    meta = json.loads((DATA / 'metadata.json').read_text(encoding='utf-8'))
    if meta.get('source') != 'synthetic':
        raise ValueError('This smoke test accepts only the synthetic dataset.')
    with np.load(DATA / 'mesh.npz', allow_pickle=False) as f:
        rest = f['vertices_rest'].astype(np.float32)
        p = f['parameter_values'].astype(np.float32).reshape(-1)
        delta = f['vertex_delta'].astype(np.float32)
    assert rest.shape == (34, 2) and delta.shape == (21, 34, 2)
    assert np.isfinite(rest).all() and np.isfinite(delta).all() and np.isfinite(p).all()
    lid = np.zeros((len(rest), 1), dtype=np.float32)
    lid[meta['vertex_groups']['upper_lid']] = 1.0
    features = np.concatenate([np.broadcast_to(rest, (len(p), *rest.shape)),
                               np.broadcast_to(lid, (len(p), len(rest), 1)),
                               np.broadcast_to(p[:,None,None], (len(p), len(rest), 1))], axis=-1).copy()
    # Split whole poses, not random vertices from the same pose.
    val_ids = np.array([2, 5, 8, 11, 14, 17])
    train_ids = np.setdiff1d(np.arange(len(p)), val_ids)
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        torch.set_num_threads(1)
    print('MintoAI Neural Rigger V0.1 - SYNTHETIC ONLY')
    print('Device:', torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU')
    print('Train poses:', len(train_ids), '| Validation poses:', len(val_ids))
    x = torch.from_numpy(features).to(device)
    y = torch.from_numpy(delta).to(device)
    net = nn.Sequential(nn.Linear(4,64), nn.SiLU(), nn.Linear(64,64), nn.SiLU(), nn.Linear(64,2)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=0.003)
    mse = nn.MSELoss()
    def predict(features):
        # Guarantee identity at the open-eye default (p=1).
        return net(features) * (1.0 - features[..., 3:4])
    for epoch in range(1001):
        net.train()
        opt.zero_grad(set_to_none=True)
        loss = mse(predict(x[train_ids]), y[train_ids])
        loss.backward()
        opt.step()
        if epoch % 200 == 0:
            net.eval()
            with torch.no_grad():
                val = mse(predict(x[val_ids]), y[val_ids]).item()
            print(f'Epoch {epoch:4d} | Train MSE: {loss.item():.8f} | Val MSE: {val:.8f}')
    net.eval()
    with torch.no_grad():
        pred = predict(x).cpu().numpy()
    err = np.linalg.norm(pred[val_ids] - delta[val_ids], axis=-1)
    mean, maximum = float(err.mean()), float(err.max())
    print(f'Validation mean vertex error: {mean:.6f}')
    print(f'Validation max vertex error: {maximum:.6f}')
    path = OUT / 'synthetic_eye_v01.pt'
    torch.save({'state_dict': {k: v.detach().cpu() for k,v in net.state_dict().items()},
                'architecture': [4,64,64,2], 'activation': 'SiLU',
                'features': ['rest_x','rest_y','is_upper_lid','eye_open'],
                'output': 'xy displacement multiplied by (1-eye_open)',
                'source': 'synthetic', 'production_rig': False}, path)
    np.savez_compressed(OUT / 'synthetic_eye_v01_predictions.npz',
                        parameter_values=p, vertices_rest=rest,
                        vertices_predicted=rest[None]+pred, vertices_target=rest[None]+delta,
                        validation_indices=val_ids)
    print('Checkpoint:', path)
    if device.type == 'cuda':
        print(f'Peak allocated VRAM: {torch.cuda.max_memory_allocated()/1024**2:.2f} MiB')
    if np.isfinite(err).all() and mean < 0.015:
        print('PASS: synthetic interpolation smoke test')
    else:
        print('CHECK: synthetic interpolation error is above target')
    print('Not a production rig. No real artist model used for training.')

if __name__ == '__main__':
    main()
