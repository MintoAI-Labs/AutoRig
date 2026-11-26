"""Generate an offline browser viewer for MintoAI synthetic-eye predictions."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / 'checkpoints' / 'synthetic_eye_v01_predictions.npz'
MESH = ROOT / 'dataset' / 'synthetic_eye_v01' / 'mesh.npz'
OUT = ROOT / 'checkpoints' / 'synthetic_eye_viewer.html'

def main():
    with np.load(PRED, allow_pickle=False) as f:
        p = f['parameter_values'].reshape(-1)
        rest = f['vertices_rest']
        target = f['vertices_target']
        pred = f['vertices_predicted']
        val = f['validation_indices'].reshape(-1)
    with np.load(MESH, allow_pickle=False) as f:
        tri = f['triangles']
        if not np.allclose(f['vertices_rest'], rest) or not np.allclose(f['parameter_values'].reshape(-1), p) or not np.allclose(f['vertices_posed'], target):
            raise ValueError('Predictions do not match the synthetic dataset.')
    if rest.shape != (34,2) or target.shape != (21,34,2) or pred.shape != target.shape or p.shape != (21,) or tri.shape != (32,3):
        raise ValueError('Unexpected synthetic dataset dimensions.')
    if not all(np.isfinite(x).all() for x in (p,rest,target,pred,tri,val)) or np.any(tri < 0) or np.any(tri >= len(rest)) or np.any(val < 0) or np.any(val >= len(p)):
        raise ValueError('Invalid data.')
    data = dict(p=p.tolist(), target=target.tolist(), pred=pred.tolist(), tri=tri.tolist(), val=val.tolist())
    html = '''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MintoAI | Eye viewer</title>
<style>body{font:16px system-ui,sans-serif;background:#101a20;color:#ecf6f3;margin:0;padding:24px}main{max-width:1000px;margin:auto}h1{font-size:28px;margin:0 0 8px;color:#bce9db}p{line-height:1.5;color:#b6c8c5}.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}.panel,.controls{background:#192830;border:1px solid #38544f;border-radius:12px;padding:16px}h2{font-size:17px;margin:0 0 8px}svg{width:100%;display:block}input[type=range]{width:100%;accent-color:#bce9db}button{background:#bce9db;color:#142a26;border:0;border-radius:7px;padding:10px 16px;font:inherit;cursor:pointer}.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.controls{margin-top:14px}.tag{color:#f6d5e2}.small{font-size:13px}label{display:block;margin:12px 0 6px}</style>
<main><h1>MintoAI · Neural Eye Viewer V0.1</h1><p>So sánh dữ liệu giả lập gốc và kết quả dự đoán đã lưu từ lần training trên máy bạn.</p><div class="panels"><section class="panel"><h2>Dữ liệu chuẩn (Target)</h2><svg id="target" viewBox="0 0 340 180" role="img" aria-label="Target mesh"></svg></section><section class="panel"><h2>AI dự đoán (Prediction)</h2><svg id="pred" viewBox="0 0 340 180" role="img" aria-label="Predicted mesh"></svg></section></div><section class="controls"><div class="row"><button id="play">Phát chớp mắt</button><button id="next">Mốc validation tiếp theo</button><label style="margin:0"><input type="checkbox" id="overlay"> Chồng dữ liệu chuẩn lên AI</label></div><label for="pose">ParamEyeLOpen: <strong id="value"></strong> · <span id="split" class="tag"></span></label><input id="pose" type="range" min="0" max="20" step="1" value="20"><div class="row small"><span>0 = Nhắm mắt</span><span>1 = Mở mắt</span></div><p id="errors" aria-live="polite"></p></section><p class="small">Chỉ dùng 21 trạng thái đã lưu (15 train, 6 validation). Không phải inference trực tiếp từ checkpoint, không dùng model nghệ sĩ. Sai số theo đơn vị tọa độ giả lập, không phải pixel.</p></main>
<script>const data = __DATA__;
const ns='http://www.w3.org/2000/svg'; const $=id=>document.getElementById(id);
const xy=v=>[170+v[0]*140,88-v[1]*140];
function el(name,attrs){const n=document.createElementNS(ns,name); for(const [k,v] of Object.entries(attrs))n.setAttribute(k,v);return n;}
function draw(svg,pts,color,overlay){svg.replaceChildren();for(const t of data.tri){svg.append(el('polygon',{points:t.map(i=>xy(pts[i]).join(',')).join(' '),fill:color,'fill-opacity':'.045',stroke:color,'stroke-opacity':'.4','stroke-width':'.7'}));}for(const [a,b] of [[0,17],[17,34]]){svg.append(el('polyline',{points:pts.slice(a,b).map(v=>xy(v).join(',')).join(' '),fill:'none',stroke:color,'stroke-width':'2.5'}));}for(const v of pts){const [cx,cy]=xy(v);svg.append(el('circle',{cx,cy,r:2.1,fill:color}));}if(overlay){for(const [a,b] of [[0,17],[17,34]])svg.append(el('polyline',{points:overlay.slice(a,b).map(v=>xy(v).join(',')).join(' '),fill:'none',stroke:'#f4c26b','stroke-dasharray':'4 3','stroke-width':'1.5'}));}}
function render(){const i=Number($('pose').value), t=data.target[i], p=data.pred[i];draw($('target'),t,'#bce9db');draw($('pred'),p,'#f6d5e2',$('overlay').checked?t:null);$('value').textContent=data.p[i].toFixed(2);$('split').textContent=data.val.includes(i)?'VALIDATION · chưa học trực tiếp':'TRAIN';const e=t.map((v,j)=>Math.hypot(v[0]-p[j][0],v[1]-p[j][1]));$('errors').textContent='Sai số trung bình: '+(e.reduce((a,b)=>a+b,0)/e.length).toFixed(6)+' | Sai số lớn nhất: '+Math.max(...e).toFixed(6);}
let timer=null,dir=-1;function stop(){if(timer!==null){clearInterval(timer);timer=null;}$('play').textContent='Phát chớp mắt';} $('play').onclick=()=>{if(timer!==null){stop();return;}$('play').textContent='Dừng';timer=setInterval(()=>{let i=Number($('pose').value);if(i===0)dir=1;if(i===20)dir=-1;$('pose').value=i+dir;render();},65);};$('pose').oninput=()=>{stop();render();};$('overlay').onchange=render;$('next').onclick=()=>{stop();const i=Number($('pose').value);$('pose').value=data.val.find(v=>v>i)??data.val[0];render();};render();
</script></html>'''
    OUT.write_text(html.replace('__DATA__',json.dumps(data,allow_nan=False,separators=(',',':'))),encoding='utf-8')
    print('PASS: viewer generated from saved predictions')
    print('Poses:',len(p),'| Validation:',len(val))
    print('Output:',OUT)
if __name__ == '__main__':
    main()
