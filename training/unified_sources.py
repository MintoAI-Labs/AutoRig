"""Convert existing NPZs and the Extreme teacher to one mesh/control representation."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch
from torch import nn

import train_real_rig_v03 as loader
from train_real_rig_v01 import PARAMS
from unified_rig import CONTROLS

PARTS = ((0,.73,.18,.21),(0,.71,.12,.06),(0,.82,.14,.04),(0,.30,.24,.37),
         (-.37,.21,.15,.30),(.37,.21,.15,.30),(-.57,-.04,.16,.18),(.57,-.04,.16,.18),
         (-.14,-.35,.115,.31),(.14,-.35,.115,.31),(-.14,-.77,.10,.24),(.14,-.77,.10,.24),
         (-.17,-.98,.13,.065),(.17,-.98,.13,.065))
EXT_CONTROLS = ['ParamAngleZ','ParamArmL','ParamLegL','ParamMouthOpenY',
                'ParamMouthForm','ParamBrowLY','ParamBodyAngleZ','ParamArmR','ParamLegR']


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tree_hash(path):
    h = hashlib.sha256()
    for item in sorted(Path(path).glob('*/*')):
        if item.suffix == '.npz' or item.name == 'source.json':
            h.update(item.relative_to(path).as_posix().encode())
            h.update(bytes.fromhex(file_hash(item)))
    return h.hexdigest()


def family(name):
    stem = Path(name).stem.casefold()
    return re.split(r'_(?:free|pro|sample|basic|vowels)(?:_|$)', stem)[0]


def load_records(data, models, audit, domain, stop):
    audit.mkdir(parents=True, exist_ok=True)
    previous = loader.CHECK_STOP
    loader.CHECK_STOP = stop.check
    try:
        result = loader.load_visual(data, models, torch.device('cpu'), audit)
    finally:
        loader.CHECK_STOP = previous
    for r in result:
        r['domain'] = domain
        r['id'] = f'{domain}/{r["group"]}/{r["param"]}'
        r['semantic'] = torch.full((r['nm'],), 14, dtype=torch.long)
        n = len(r['t'])
        if n < 5:
            raise ValueError(f'Need at least five poses for train/validation: {r["id"]}')
        r['practice'] = [i for i in range(n) if i % 4 != 0]
        r['exam'] = list(range(0, n, 4))
        # The old 1e-5 motion cutoff conflicts with strict sub-nanounit exams.
        moving=np.max(np.abs(r['delta']),axis=(0,2))>0
        r['moving']=torch.tensor(moving)
        r['mesh_active']=torch.tensor((np.bincount(r['mesh'].numpy(),
            weights=moving.astype(float),minlength=r['nm'])>0).astype(np.float32))
        r['teacher_kind'] = 'real' if domain == 'real' else None
    return result


def make_ext_teacher(checkpoint):
    width = int(checkpoint['width'])
    model = nn.Sequential(nn.Linear(30,width), nn.SiLU(), nn.Linear(width,width), nn.SiLU(),
                          nn.Linear(width,width), nn.SiLU(), nn.Linear(width,2))
    model=model.to(dtype=next(iter(checkpoint['model_state'].values())).dtype)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model


def synthetic_records(seed, count, prefix='train'):
    rng = np.random.default_rng(seed)
    axis = np.linspace(-.9,.9,4)
    local = np.stack(np.meshgrid(axis,axis),-1).reshape(-1,2)
    local=local/np.maximum(1.,np.linalg.norm(local,axis=1,keepdims=True)/.95)
    points = np.array(PARTS)
    parts, vertices = len(points), len(local)
    triangles = []
    for m in range(parts):
        for y in range(3):
            for x in range(3):
                i = m*vertices+y*4+x
                triangles += [[i,i+1,i+4],[i+1,i+5,i+4]]
    tri = np.array(triangles)
    edges = np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]])
    edges = np.unique(np.concatenate([edges,edges[:,::-1]]),axis=0).T
    mesh = np.repeat(np.arange(parts), vertices)
    values = np.linspace(-1,1,17)
    records = []
    for index in range(count):
        shape = rng.uniform(.72,1.28,3)
        original = (points[:,None,:2] + local[None]*points[:,None,2:]) * shape[:2]
        rest = original.reshape(-1,2)*.25
        feat = np.empty((len(rest),8))
        for m in range(parts):
            sub = rest[mesh==m]
            center, extent = sub.mean(0), np.ptp(sub,axis=0)
            feat[mesh==m] = np.concatenate([sub,np.broadcast_to(center,sub.shape),
                np.broadcast_to(extent,sub.shape),(sub-center)/max(extent.max(),1e-8)],1)
        for control in EXT_CONTROLS:
            delta = np.zeros((len(values),parts,vertices,2))
            if control in ['ParamAngleZ','ParamArmL','ParamArmR','ParamLegL','ParamLegR','ParamBodyAngleZ']:
                mapping = {
                    'ParamAngleZ':([0,1,2],(0,.53),.42),
                    'ParamArmL':([4,6],(-.22,.49),1.2),'ParamArmR':([5,7],(.22,.49),-1.2),
                    'ParamLegL':([8,10,12],(-.14,-.11),.65),'ParamLegR':([9,11,13],(.14,-.11),-.65),
                    'ParamBodyAngleZ':(list(range(8)),(0,-.12),.3)}
                ids,pivot,strength = mapping[control]
                xy = original[ids]-np.array(pivot)*shape[:2]
                angle = values[:,None,None]*strength*shape[2]
                xx, yy = xy[:,:,0][None], xy[:,:,1][None]
                delta[:,ids] = np.stack([(np.cos(angle)-1)*xx-np.sin(angle)*yy,
                                         np.sin(angle)*xx+(np.cos(angle)-1)*yy],-1)
            elif control == 'ParamMouthOpenY':
                delta[:,1,:,1] = -.08*np.maximum(values[:,None],0)*(1-local[:,1])
            elif control == 'ParamMouthForm':
                delta[:,1,:,1] = .055*values[:,None]*np.abs(local[:,0])
            else:
                delta[:,2,:,1] = .045*values[:,None]*np.abs(local[:,0])
            r = dict(source=f'procedural_{prefix}_{index:03d}',group=f'{prefix}_{index:03d}',
                     param=control, domain='extreme', id=f'extreme/{prefix}_{index}/{control}',
                     feat=torch.tensor(feat),edge=torch.tensor(edges,dtype=torch.long),
                     mesh=torch.tensor(mesh,dtype=torch.long),nm=parts,
                     t=torch.tensor(values),delta=delta.reshape(len(values),-1,2)*.25,
                     visual=torch.zeros(parts,4,48,48),semantic=torch.arange(parts),
                     practice=[i for i in range(len(values)) if i%4],
                     exam=list(range(0,len(values),4)),teacher_kind='extreme',
                     ext_original=original,ext_local=local,ext_shape=shape)
            peak = np.max(np.abs(r['delta']),axis=(0,2))>0
            r['moving'] = torch.tensor(peak)
            r['mesh_active'] = torch.tensor([float(peak[mesh==m].any()) for m in range(parts)])
            records.append(r)
    return records


def to_device(record, device, dtype):
    result = dict(record)
    for key in ('feat','t','visual','mesh_active'):
        result[key] = record[key].to(device=device,dtype=dtype)
    for key in ('mesh','edge','semantic','moving'):
        result[key] = record[key].to(device)
    result['onehot'] = torch.eye(len(PARAMS),device=device,dtype=dtype)[
        PARAMS.index(record['param'])][None] if record['param'] in PARAMS else None
    return result


@torch.no_grad()
def extreme_prediction(teacher, r, indices, device, dtype):
    original, local, shape = r['ext_original'],r['ext_local'],r['ext_shape']
    values = r['t'][indices].to(device=device,dtype=dtype)
    p,k,m = len(values),len(local),len(PARTS)
    base = np.concatenate([original.transpose(1,0,2),
             np.broadcast_to(local[:,None],(k,m,2)),np.broadcast_to(np.eye(m),(k,m,m)),
             np.broadcast_to(shape,(k,m,3)),np.zeros((k,m,9))],-1)
    inputs = torch.tensor(base,device=device,dtype=dtype)[None].expand(p,-1,-1,-1).clone()
    inputs[...,21+EXT_CONTROLS.index(r['param'])] = values[:,None,None]
    prediction = teacher(inputs).transpose(1,2).reshape(p,-1,2)*.25
    return prediction


class DeviceCache:
    """Reuse immutable graph inputs; discard cached graphs when VRAM gets tight."""
    def __init__(self, device, dtype, count=4):
        from collections import OrderedDict
        self.rows=OrderedDict()
        self.device,self.dtype,self.count=device,dtype,count

    def clear(self):
        self.rows.clear()
        if self.device.type=='cuda':
            torch.cuda.empty_cache()

    def get(self, record):
        key=id(record)
        if key in self.rows:
            result=self.rows.pop(key)
            self.rows[key]=result
            return result
        if self.device.type=='cuda':
            needed=sum(record[k].numel()*torch.tensor([],dtype=self.dtype).element_size()
                       for k in ('feat','t','visual','mesh_active'))
            free,_=torch.cuda.mem_get_info()
            if free<needed+768*1024**2:
                self.clear()
        result=to_device(record,self.device,self.dtype)
        self.rows[key]=result
        while len(self.rows)>self.count:
            self.rows.popitem(last=False)
        return result
