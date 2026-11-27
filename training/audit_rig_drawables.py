"""Read-only Hiyori geometry, gate and drawable audit."""
import argparse
import base64
import gzip
import json
from pathlib import Path
import re
import sys

import numpy as np
from playwright.sync_api import sync_playwright
import torch

from codex_blink_lesson import DTYPE, geometry
from train_real_rig_v01 import load_record
from train_real_rig_v03 import mesh_patches
from train_real_rig_v04 import ParameterRig, ROOT
from unified_rig import UnifiedRig
from unified_sources import family, file_hash, to_device

sys.path.insert(0, str(ROOT / 'extractor'))
from autolearn_v04 import EXTRACT, LOAD

EDGE = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
CORE = ROOT / 'SDK/CubismSdkForWeb-5-r.5/Core/live2dcubismcore.js'


def unpack(value, dtype):
    return np.frombuffer(gzip.decompress(base64.b64decode(value)), dtype=dtype)


def peak(x):
    return float(np.abs(x).max())


def model_meta(browser, moc, params):
    page = browser.new_page()
    page.add_script_tag(path=CORE)
    encoded = base64.b64encode(moc.read_bytes()).decode()
    page.evaluate(LOAD, encoded)
    metadata = page.evaluate("""() => ({
      parts:Array.from(model.parts.ids),
      ids:Array.from(model.drawables.ids),
      parents:Array.from(model.drawables.parentPartIndices)
    })""")
    if params:
        raw = json.loads(page.evaluate(EXTRACT, params))
        page.evaluate("""() => {
          model.parameters.values.set(model.parameters.defaultValues);model.update();
        }""")
        metadata['render_order'] = page.evaluate("""() =>
          Array.from(model.getRenderOrders?.() ?? model.drawables.renderOrders ?? [])""")
        metadata['raw'] = raw
    page.close()
    return metadata


def part_group(id_, name):
    value = id_.lower()
    if 'eye' in value or '目' in name:
        return 'Mắt'
    if 'hair' in value or '髪' in name or 'サイドアップ' in name:
        return 'Tóc'
    if any(term in value for term in ('face', 'cheek', 'brow', 'nose', 'mouth', 'ear')):
        return 'Mặt'
    if any(term in value for term in ('body', 'arm', 'neck', 'leg', 'torso')):
        return 'Thân'
    return 'Khác'


def source_check(z, live):
    raw = live['raw']
    meshes = raw['meshes']
    native_rest = np.concatenate([np.asarray(m['vertices_rest'], np.float32).reshape(-1, 2) for m in meshes])
    native_poses = np.asarray([np.concatenate([np.asarray(m,np.float32).reshape(-1,2)
                                               for m in sample['vertices_posed']])
                               for sample in raw['samples']])
    lo, hi = native_rest.min(0), native_rest.max(0)
    center, scale = (lo + hi) / 2, (hi - lo).max()
    native_uv = np.concatenate([m['uv'] for m in meshes]).astype('f4').reshape(-1, 2)
    native_tri = np.concatenate([np.asarray(m['triangles']).reshape(-1,3)+z['mesh_offsets'][i]
                                 for i,m in enumerate(meshes)])
    if native_uv.shape != z['uv'].shape:
        native_uv = native_uv.reshape(z['uv'].shape)
    return dict(
        ids_equal=live['ids'] == z['mesh_ids'].tolist(),
        counts_equal=np.diff(z['mesh_offsets']).tolist() == [m['vertex_count'] for m in meshes],
        triangles_equal=np.array_equal(native_tri,z['triangles']),
        uv_equal=np.array_equal(native_uv,z['uv']),
        order_equal=np.array_equal([s['draw_orders'] for s in raw['samples']],z['draw_orders']),
        core_render_order_exists=bool(live['render_order']),
        default_render_order_equals_draw_order=live['render_order']==raw['rest_draw_orders'],
        opacity_max=peak(np.asarray([s['opacities'] for s in raw['samples']])-z['opacities']),
        rest_max=peak((native_rest-center)/scale-z['rest']),
        delta_max=peak((native_poses-native_rest)/scale-z['delta']),
        values_equal=np.array_equal(np.asarray([s['value'] for s in raw['samples']],dtype='f4'),z['parameter_values']),
        range_equal=np.array_equal(np.asarray(list(raw['parameter'][k] for k in ('min','max','default')),
                                              dtype='f4'),z['parameter_range']),
        scale=float(scale), scale_equal=bool(scale==z['scale']),
    )


def drawable_scores(ids, offsets, truth, prediction, raw, probs, threshold, vertices, values):
    result = []
    for k, id_ in enumerate(ids):
        start, end = offsets[k:k+2]
        errors = prediction[:,start:end]-truth[:,start:end]
        pose, local, axis = np.unravel_index(np.argmax(np.abs(errors)), errors.shape)
        vertex = start+local
        result.append(dict(
            id=id_, category=vertices[k]['category'], part=vertices[k]['part'], drawable_index=k,
            pose_index=int(pose), pose_value=float(values[pose]), vertex=int(vertex), vertex_in_drawable=int(local),
            axis='xy'[axis], gate_probability=float(probs[k]), gate_threshold=threshold,
            gate_on=bool(probs[k]>=threshold), truth=truth[pose,vertex].tolist(),
            predicted=prediction[pose,vertex].tolist(), predicted_before_gate=raw[pose,vertex].tolist(),
            error_linf=peak(errors), error_percent_of_model=100*peak(errors),
            truth_peak=peak(truth[:,start:end]), static=not bool(np.any(truth[:,start:end])),
        ))
    return sorted(result,key=lambda item:item['error_linf'],reverse=True)


def identity_check(browser, html):
    """Run the original browser draw path with copied reference deltas."""
    page = browser.new_page()
    try:
        page.goto(html.as_uri(), timeout=90_000)
        page.wait_for_function('current !== undefined && current !== null', timeout=90_000)
        records = page.evaluate("""async () => {
          const verified = [];
          for (const record of data.records) {
            const model = await loadModel(record.model);
            const delta = await unpack(record.deltas, Float32Array);
            const n = record.values.length * model.rest.length;
            delta.copyWithin(n, 0, n); delta.copyWithin(n*2, 0, n);
            if (!delta.subarray(0, n).every((value, i) => value === delta[n+i] && value === delta[2*n+i])) {
              throw Error('Identity delta mismatch');
            }
            verified.push({source:record.source, control:record.param, poses:record.values.length});
            if (record.source !== 'Hiyori' || record.param !== 'ParamEyeROpen') continue;
            current = {record, model, delta,
              moving:await unpack(record.moving, Uint8Array),
              opacities:await unpack(record.opacity, Float32Array),
              orders:await unpack(record.renderOrder, Int32Array),
              points:Array.from({length:3}, () => new Float32Array(model.rest.length))
            };
            renderers.forEach(renderer => renderer.upload(model));
            for (const pose of [0, record.errors[1].all.pose, record.values.length-1, 64.5]) {
              slider.value = pose; draw();
              if (!current.points[0].every((value,i) => value === current.points[1][i] &&
                  value === current.points[2][i])) throw Error('Identity overlay mismatch');
              if (Object.values(fields).some(columns =>
                  columns.slice(0,2).some(cell => cell.textContent !== '0 / 0'))) {
                throw Error('Nonzero identity error');
              }
            }
          }
          if (!verified.some(r => r.source === 'Hiyori' && r.control === 'ParamEyeROpen')) {
            throw Error('Missing Hiyori right-eye record');
          }
          return verified;
        }""")
        return dict(html_sha256=file_hash(html),identity_max=0,records=records,renderer_overlay_equal=True)
    finally:
        page.close()

@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path,help='New local audit directory')
    parser.add_argument('--html',type=Path,default=ROOT/'dataset/rig_comparison_seed42_step50.html')
    args=parser.parse_args()
    out=args.output
    out.mkdir(parents=True,exist_ok=False)
    html=args.html.resolve()
    payload=json.loads(re.search(r'<script id="payload" type="application/json">(.*?)</script>',
                                 html.read_text(encoding='utf-8'),re.S)[1])
    paths=[ROOT/'checkpoints/real_rig_curriculum.best.pt',
           ROOT/'checkpoints/codex_gradient_seed42_step50/projected.step050.candidate.pt']
    before=[file_hash(p) for p in paths]
    assert before==payload['hashes']
    ck, candidate=[torch.load(p,map_location='cpu',weights_only=True) for p in paths]
    settings=ck['curriculum_settings']
    root, models=Path(settings['data']),Path(settings['models'])
    torch.set_num_threads(4)
    torch.manual_seed(42)
    device=torch.device('cpu')
    real=ParameterRig(ck['width']).to(dtype=DTYPE).eval()
    real.load_state_dict(ck['model_state'])
    trial=ParameterRig(candidate['width']).to(dtype=DTYPE).eval()
    trial.load_state_dict(candidate['model_state'])
    unified=UnifiedRig(ck['width']).to(dtype=DTYPE).eval()
    unified.initialize_real(ck['model_state'])
    summary=[]
    atlas_meta={}
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=EDGE,args=['--disable-webgl'])
        identity=identity_check(browser,html)
        (out/'identity.json').write_text(json.dumps(identity,ensure_ascii=False,indent=2),encoding='utf-8')
        for group in sorted(root.glob('*/source.json')):
            meta=json.loads(group.read_text(encoding='utf-8-sig'))
            moc=models/meta['model_path']
            name=moc.stem
            records=[r for r in payload['records'] if r['source']==name]
            if not records:
                continue
            eye=next(group.parent.glob('*ParamEyeROpen*.npz'))
            key=records[0]['model']
            with np.load(eye,allow_pickle=False) as z:
                params=dict(id='ParamEyeROpen',actual=meta['original_parameter_ids']['ParamEyeROpen'],
                            count=len(z['parameter_values']),source=name,permission='local audit')
                live=model_meta(browser,moc,params)
                check=source_check(z,live)
                if not check['ids_equal'] or not check['triangles_equal']:
                    raise ValueError(f'Core topology differs for {name}')
                cdi=moc.with_suffix('.cdi3.json')
                labels={p['Id']:p['Name'] for p in json.loads(cdi.read_text(encoding='utf-8-sig')).get('Parts',[])} if cdi.exists() else {}
                ids=live['ids']
                offsets=z['mesh_offsets']
                details=[]
                for index,(did,parent) in enumerate(zip(ids,live['parents'])):
                    part=live['parts'][parent] if parent>=0 else ''
                    label=labels.get(part,'')
                    details.append(dict(id=did,part=part,label=label,category=part_group(part,label),
                                        start=int(offsets[index]),end=int(offsets[index+1])))
                atlas_meta[key]=dict(source=name,core_check=check,drawables=details)
                assert np.array_equal(unpack(payload['models'][key]['rest'],'<f4').reshape(-1,2),z['rest'])
                assert np.array_equal(unpack(payload['models'][key]['triangles'],'<u4').reshape(-1,3),z['triangles'])
            patches,_=mesh_patches(eye,models)
            rgba=torch.from_numpy(patches).float().permute(0,3,1,2)/255
            visual=torch.cat((rgba[:,:3]*rgba[:,3:4],rgba[:,3:4]),1).contiguous()
            for rec in records:
                path,=group.parent.glob(f"*{rec['param']}*.npz")
                row=load_record(path,root,device)
                row=to_device(dict(row,visual=visual,mesh_active=torch.zeros(row['nm']),
                                   semantic=torch.full((row['nm'],),14)),device,DTYPE)
                param=rec['param']
                scale=ck['scales'][param]
                threshold=ck['thresholds'][param]
                with np.load(path,allow_pickle=False) as z:
                    lo,hi,default=z['parameter_range']
                    values=z['parameter_values']
                    expected=(values-default)/np.where(values>=default,max(float(hi-default),1e-6),max(float(default-lo),1e-6))
                    assert np.array_equal(expected.astype('f4'),row['t'].numpy().astype('f4'))
                    assert np.array_equal(z['mesh_ids'],ids) and np.array_equal(z['mesh_offsets'],offsets)
                    assert np.array_equal(values,np.asarray(rec['values'],'f4'))
                    assert np.array_equal(unpack(rec['order'],'<i4').reshape(z['draw_orders'].shape),z['draw_orders'])
                    assert np.array_equal(unpack(rec['opacity'],'<f4').reshape(z['opacities'].shape),z['opacities'])
                    rendered=unpack(rec['deltas'],'<f4').reshape(3,*z['delta'].shape)
                    assert np.array_equal(rendered[0],z['delta'])
                mesh=row['mesh']
                hr, logits=real.encode(row)
                hg, gl=trial.encode(row)
                hu, ul=unified.encode(row)
                prob=logits.sigmoid()
                gate=(prob>=threshold)[mesh,None]
                gate_u=(ul.sigmoid()>=threshold)[mesh,None]
                raw=torch.stack([real.deform(hr,t,param) for t in row['t']])
                cand=torch.cat([geometry(trial.heads[param],hg,block) for block in row['t'].split(8)])
                unified_runs=[unified.pose_batch(row,t,(hu,ul),hard_gate=True) for t in row['t'].split(8)]
                hard_half=torch.cat([r[0] for r in unified_runs])
                unified_raw=torch.cat([r[1] for r in unified_runs])
                real_delta=(raw*gate*scale).numpy()
                candidate_delta=(cand*(gl.sigmoid()>=threshold)[mesh,None]*scale).numpy()
                checks=[peak(rendered[1]-real_delta),peak(rendered[2]-candidate_delta)]
                if max(checks)>1e-7:
                    raise ValueError(f'Preview inference differs for {name} {param}: {checks}')
                static=~np.any(rendered[0]!=0,axis=(0,2))
                result=dict(
                    source=name,param=param,poses=len(values),scale=scale,parameter_default=float(default),
                    nearest_sample=float(values[np.abs(values-default).argmin()]),
                    exact_default_in_sweep=bool(np.any(values==default)),vertex_count=len(mesh),
                    static_vertices=int(static.sum()),static_gate_on_vertices=int(gate.numpy()[static].sum()),
                    original_gate_threshold=threshold, logit_difference=peak((logits-ul).numpy()),
                    raw_difference_model_span=peak(((raw-unified_raw)*scale).numpy()),
                    same_threshold_difference_model_span=peak(real_delta-(unified_raw*gate_u*scale).numpy()),
                    default_half_difference_model_span=peak(real_delta-(hard_half*scale).numpy()),
                    half_gate_changed_drawables=int(((prob>=threshold)!=(ul.sigmoid()>=.5)).sum()),
                    real_static_rms_model_span=float(np.sqrt(np.mean(real_delta[:,static]**2))) if static.any() else None,
                    serialized_prediction_delta_max_difference=checks,
                )
                if param=='ParamEyeROpen':
                    result['real_drawables']=drawable_scores(ids,offsets,rendered[0],real_delta,
                                                              (raw*scale).numpy(),prob.numpy(),threshold,details,values)
                    result['candidate_drawables']=drawable_scores(ids,offsets,rendered[0],candidate_delta,
                                                                  (cand*scale).numpy(),gl.sigmoid().numpy(),threshold,details,values)
                print(name,param,'gates .5 changed',result['half_gate_changed_drawables'],
                      'raw max',result['raw_difference_model_span'],flush=True)
                summary.append(result)
        browser.close()
    assert len(summary)==22 and len(atlas_meta)==4
    assert [file_hash(p) for p in paths]==before
    for key,meta in atlas_meta.items():
        payload['models'][key]['drawables']=meta['drawables']
    result=dict(identity=identity,models=list(atlas_meta.values()),records=summary,checkpoint_hashes=before)
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')
    template=Path(__file__).with_name('rig_comparison_template.html').read_text(encoding='utf-8-sig')
    template = template.replace(
        'gl.drawElements(gl.LINES,scene.lines.length,gl.UNSIGNED_INT,0);',
        '''for(const i of layerOrder){
          const mesh=scene.meshes[i];
          gl.drawElements(gl.LINES,mesh.count*2,gl.UNSIGNED_INT,mesh.first*8);
        }''')
    template = template.replace('draw(points){','draw(points,layerOrder){')
    template = template.replace(
        'for(let i=0;i<scene.lines.length;i+=2){',
        '''for (const id of layerOrder) {
        const mesh=scene.meshes[id], end=(mesh.first+mesh.count)*2;
        for(let i=mesh.first*2;i<end;i+=2){''')
    template = template.replace('      ctx.stroke();','      }\n      ctx.stroke();')
    extension=Path(__file__).with_name('rig_drawable_preview.js').read_text(encoding='utf-8-sig')
    template=template.replace('</html>','<script>'+extension+'</script>\n</html>')
    with (out/'drawable_preview.html').open('x',encoding='utf-8') as stream:
        stream.write(template.replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':'),
                                                            allow_nan=False).replace('<','\\u003c')))
    print('Saved drawable audit and isolation preview.',flush=True)


if __name__=='__main__':
    main()
