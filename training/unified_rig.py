"""One student for real rigs, procedural full-body motion and PSD2Live labels."""
import copy
import torch
from torch import nn
from train_real_rig_v01 import PARAMS, mean_by
from train_real_rig_v04 import ParameterRig
from algorithm_rules import RuleFields

CONTROLS = [*PARAMS, 'ParamArmL', 'ParamArmR', 'ParamLegL', 'ParamLegR',
            'ParamBodyAngleZ', 'ParamBrowLY']


class UnifiedRig(ParameterRig):
    def __init__(self, width=64):
        super().__init__(width)
        self.embed[0] = nn.Linear(8 + len(CONTROLS) + 32, width)
        for param in CONTROLS[len(PARAMS):]:
            self.gates[param] = copy.deepcopy(self.gates['ParamAngleZ'])
            self.heads[param] = copy.deepcopy(self.heads['ParamAngleZ'])
        self.layer_attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.layer_position = nn.Linear(4, width)
        self.semantic = nn.Embedding(15, width)
        self.attention_mix = nn.Parameter(torch.zeros(()))
        self.semantic_mix = nn.Parameter(torch.zeros(()))
        self.rule_mode='affine'
        self.rule_fields=RuleFields(width,CONTROLS)
        self.affine = nn.ModuleDict()
        for param in CONTROLS:
            head = nn.Sequential(nn.Linear(width + 3, width), nn.SiLU(), nn.Linear(width, 6))
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            self.affine[param] = head

    def initialize_real(self, state):
        own = self.state_dict()
        for key, value in state.items():
            if key == 'embed.0.weight':
                own[key].zero_()
                own[key][:, :8].copy_(value[:, :8])
                own[key][:, 8:8 + len(PARAMS)].copy_(value[:, 8:15])
                own[key][:, 8 + len(CONTROLS):].copy_(value[:, 15:])
            elif key in own and own[key].shape == value.shape:
                own[key].copy_(value)
        self.load_state_dict(own)

    def encode(self, r):
        n = len(r['feat'])
        mesh, nm = r['mesh'], r['nm']
        src, dst = r['edge']
        visual = self.visual(r['visual'])
        identity = r['feat'].new_zeros(1, len(CONTROLS))
        identity[0, CONTROLS.index(r['param'])] = 1
        h = self.embed(torch.cat([r['feat'], identity.expand(n, -1), visual[mesh]], 1))
        h = h + self.semantic_mix * self.semantic(r['semantic'])[mesh]
        for block in self.blocks:
            mh = mean_by(h, mesh, nm)
            glob = mh.mean(0, keepdim=True).expand(n, -1)
            h = h + block(torch.cat([h, mean_by(h[src], dst, n), mh[mesh], glob], 1))
        mh = mean_by(h, mesh, nm)
        geometry = mean_by(r['feat'][:, [2, 3, 4, 5]], mesh, nm)
        tokens = (mh + self.layer_position(geometry))[None]
        attended, _ = self.layer_attention(tokens, tokens, tokens, need_weights=False)
        h = h + self.attention_mix * attended[0, mesh]
        mh = mean_by(h, mesh, nm)
        logits = self.gates[r['param']](
            torch.cat([mh, mh.mean(0, keepdim=True).expand(nm, -1)], 1)).squeeze(-1)
        return h, logits

    def pose_batch(self, r, values, encoded=None, hard_gate=False):
        h, logits = self.encode(r) if encoded is None else encoded
        p, n = len(values), len(h)
        tfeat = torch.stack([values, values.abs(), values.square()], -1)
        conditioning = tfeat[:, None].expand(-1, n, -1)
        raw = self.heads[r['param']](
            torch.cat([h[None].expand(p, -1, -1), conditioning], -1)) * values[:, None, None]
        mh = mean_by(h, r['mesh'], r['nm'])
        affine = self.affine[r['param']](
            torch.cat([mh[None].expand(p, -1, -1),
                       tfeat[:, None].expand(-1, r['nm'], -1)], -1))
        affine = affine[:, r['mesh']] * values[:, None, None]
        local = r['feat'][:, :2] - r['feat'][:, 2:4]
        dx = affine[..., 0] * local[:, 0] + affine[..., 1] * local[:, 1] + affine[..., 2]
        dy = affine[..., 3] * local[:, 0] + affine[..., 4] * local[:, 1] + affine[..., 5]
        if self.rule_mode in ('affine','hybrid'):
            raw=raw+torch.stack([dx,dy],-1)
        if self.rule_mode in ('rotation','tps','hybrid'):
            raw=raw+self.rule_fields(r,mh,values,self.rule_mode)
        gates = logits.sigmoid()
        if hard_gate:
            gates = (gates >= .5).to(raw.dtype)
        prediction = raw * gates[r['mesh']][None, :, None]
        return prediction, raw, logits
