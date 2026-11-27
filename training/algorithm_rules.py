"""Differentiable rotation and thin-plate-spline rules, fitted against human rigs."""
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import torch
from torch import nn

SOURCE_ROOT='algorithms/binding/src/main/kotlin/io/github/psd2live/algorithms/binding'
FILES=['ThinPlateSpline.kt','FaceTurn.kt','BodyMotionTuning.kt','EyeLegacyMotion.kt']


def inspect_algorithms(checkout):
    """Read a known source layout and retain source-specific numeric priors and provenance."""
    checkout=Path(checkout).resolve()
    commit=subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
    documents={}
    for name in FILES:
        path=checkout/SOURCE_ROOT/name
        text=path.read_text(encoding='utf-8')
        documents[name]=dict(path=str(path.relative_to(checkout)),
                            sha256=hashlib.sha256(text.encode()).hexdigest(),
                            functions=sorted(set(re.findall(r'fun\s+(\w+)\s*\(',text))))
    tuning=(checkout/SOURCE_ROOT/'BodyMotionTuning.kt').read_text(encoding='utf-8')
    numbers={name:float(value) for name,value in re.findall(
        r'val\s+(\w+)\s*:\s*Float\s*=\s*([-+\d.]+)f',tuning)}
    face=(checkout/SOURCE_ROOT/'FaceTurn.kt').read_text(encoding='utf-8')
    match=re.search(r'ThinPlateSpline\(from,\s*to,\s*([\deE.+-]+)\)',face)
    regularization=float(match[1]) if match else 1e-3
    return dict(schema=1,psd2live_commit=commit,source_files=documents,
                scope='selected mathematical rules; not a general Kotlin interpreter',
                rules=[
                    dict(name='rotation',learn=['angle','pivot_x','pivot_y'],
                         test='compare posed mesh coordinates to human rig'),
                    dict(name='thin_plate_spline',learn=['anchor_displacements','regularization'],
                         test='interpolation constraints and human rig pose error')],
                priors=dict(tps_regularization=regularization,
                            rotation_degrees={'ParamBodyAngleZ':numbers.get('bodyZDegrees',3.),
                                              'ParamArmL':numbers.get('armSwingDegrees',4.),
                                              'ParamArmR':-numbers.get('armSwingDegrees',4.)}))


def kernel(a,b):
    squared=(a[:,None]-b[None]).square().sum(-1)
    return torch.where(squared<1e-12,torch.zeros_like(squared),
                       .5*squared*torch.log(squared.clamp_min(1e-30)))


def tps_weights(points,anchors,regularization):
    n=len(anchors)
    poly=torch.cat([anchors.new_ones(n,1),anchors],-1)
    top=torch.cat([kernel(anchors,anchors)+torch.eye(n,device=anchors.device,dtype=anchors.dtype)*regularization,poly],-1)
    bottom=torch.cat([poly.T,anchors.new_zeros(3,3)],-1)
    matrix=torch.cat([top,bottom],0)
    targets=torch.cat([torch.eye(n,device=anchors.device,dtype=anchors.dtype),
                       anchors.new_zeros(3,n)],0)
    solution=torch.linalg.solve(matrix,targets)
    evaluation=torch.cat([kernel(points,anchors),points.new_ones(len(points),1),points],-1)
    return evaluation@solution


class RuleFields(nn.Module):
    def __init__(self,width,controls):
        super().__init__()
        axis=torch.linspace(-.5,.5,3)
        self.register_buffer('anchors',torch.stack(torch.meshgrid(axis,axis,indexing='xy'),-1).reshape(-1,2))
        self.rotation=nn.ModuleDict()
        self.warp=nn.ModuleDict()
        self.log_regularization=nn.ParameterDict()
        for param in controls:
            for collection,size in ((self.rotation,3),(self.warp,18)):
                head=nn.Sequential(nn.Linear(width+3,width),nn.SiLU(),nn.Linear(width,size))
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
                collection[param]=head
            self.log_regularization[param]=nn.Parameter(torch.tensor(math.log(1e-3)))

    def initialize_priors(self,prior):
        with torch.no_grad():
            for param,degrees in prior.get('rotation_degrees',{}).items():
                if param in self.rotation:
                    self.rotation[param][-1].bias[0]=math.radians(degrees)
            reg=max(1e-12,min(.1,float(prior.get('tps_regularization',1e-3))))
            for value in self.log_regularization.values():
                value.fill_(math.log(reg))

    def forward(self,r,mesh_features,values,mode):
        features=torch.stack([values,values.abs(),values.square()],-1)
        inputs=torch.cat([mesh_features[None].expand(len(values),-1,-1),
                          features[:,None].expand(-1,len(mesh_features),-1)],-1)
        out=r['feat'].new_zeros(len(values),len(r['feat']),2)
        mesh=r['mesh']
        if mode in ('rotation','hybrid'):
            parameters=self.rotation[r['param']](inputs)
            theta=parameters[...,0][:,mesh]*values[:,None]
            xy=r['feat'][:,:2]
            center=r['feat'][:,2:4]
            extent=r['feat'][:,4:6]
            pivot=center[None]+parameters[:,mesh,1:].tanh()*extent[None]
            local=xy[None]-pivot
            dx=(theta.cos()-1)*local[...,0]-theta.sin()*local[...,1]
            dy=theta.sin()*local[...,0]+(theta.cos()-1)*local[...,1]
            out=out+torch.stack([dx,dy],-1)/r['motion_scale']
        if mode in ('tps','hybrid'):
            reg=self.log_regularization[r['param']].exp().clamp(1e-12,.1)
            weights=tps_weights(r['feat'][:,6:8],self.anchors,reg)
            offsets=self.warp[r['param']](inputs).reshape(len(values),r['nm'],9,2)
            selected=offsets[:,mesh]
            out=out+(weights[None,:,:,None]*selected).sum(2)*values[:,None,None]
        return out
