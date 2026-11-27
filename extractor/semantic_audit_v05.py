"""MintoAI V0.5 semantic metadata audit. SPDX-License-Identifier: AGPL-3.0-only.
Local-only, no training. Reads literal aliases from a separately installed PSD2Live
checkout; does not copy or redistribute its source. Part tags are weak context,
not individual ArtMesh labels. Requires batch_extract_v02.py beside this script.
"""
import argparse, csv, hashlib, json, re, subprocess, unicodedata
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import numpy as np
from batch_extract_v02 import ROOT, CORE, inventory

PAGE = '''<!doctype html><meta charset="utf-8"><title>MintoAI Semantic Audit</title>
<style>body{font:16px system-ui;background:#14252b;color:#e9f5ef;max-width:850px;margin:40px auto;padding:20px}button{padding:10px}pre{white-space:pre-wrap}</style>
<h1>MintoAI V0.5 · Part metadata audit</h1><p>Local only. No training, no pose extraction. Output is private.</p>
<button id="go">Run audit</button><pre id="log"></pre><script src="/core.js"></script><script>
const go=document.getElementById('go'), log=document.getElementById('log');
function say(s){log.textContent+=s+'\\n';}
async function ready(){for(let i=0;i<150;i++){try{if(Live2DCubismCore.Version.csmGetVersion())return;}catch(e){}await new Promise(r=>setTimeout(r,100));}throw Error('Core not ready');}
async function get(url,opts){const res=await fetch(url,opts);const j=await res.json();if(!res.ok)throw Error(j.error||res.status);return j;}
go.onclick=async()=>{go.disabled=true;log.textContent='';let ok=0,fail=0;try{await ready();const models=await get('/manifest');for(const x of models){let moc=null,m=null;try{const res=await fetch('/moc?id='+x.id);if(!res.ok)throw Error('MOC read failed');moc=Live2DCubismCore.Moc.fromArrayBuffer(await res.arrayBuffer());if(!moc)throw Error('Invalid MOC');m=Live2DCubismCore.Model.fromMoc(moc);if(!m)throw Error('Invalid model');const d=m.drawables,p=m.parts;const body={id:x.id,parts:Array.from(p.ids),meshes:Array.from(d.ids),parents:Array.from(d.parentPartIndices)};const result=await get('/audit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});say(x.name+' | '+result.message);ok++;}catch(e){say('FAIL '+x.name+': '+e.message);fail++;}finally{if(m)m.release();if(moc)moc._release();}}say('DONE | PASS '+ok+' | FAIL '+fail);}catch(e){say('FAIL '+e.message);}finally{go.disabled=false;}};
</script>'''

FIELDS=['mesh_index','mesh_id','parent_index','part_id','part_name','part_tag_candidate','side_hint','label_source']

def norm(s):
    return unicodedata.normalize('NFKC',s).strip().casefold()

def read_aliases(path):
    # Deliberately only literal names(...) calls, not a Kotlin interpreter.
    # Ignore dynamically computed PartRoles and other rules. Conflicts abstain.
    source=path.read_text(encoding='utf-8-sig'); found={}
    for m in re.finditer(r'names\(SemanticTag\.([A-Z_]+),\s*((?:"[^"\\]*"\s*,\s*)*"[^"\\]*")\s*\)',source):
        for name in re.findall(r'"([^"\\]*)"',m.group(2)):
            found.setdefault(norm(name),set()).add(m.group(1))
    return {n:next(iter(tags)) for n,tags in found.items() if len(tags)==1}

def hint(name,aliases):
    s=norm(name); side='UNKNOWN'
    # Character side follows explicit textual markers only; no spatial guess.
    m=re.match(r'^(left|right|l|r)[ _.-]+|^([左右])',s)
    if m:
        side='LEFT' if (m.group(1) or m.group(2)) in ('left','l','左') else 'RIGHT'; s=s[m.end():].strip()
    else:
        m=re.search(r'[ _.-]+(left|right|l|r)$|([左右])$',s)
        if m:
            side='LEFT' if (m.group(1) or m.group(2)) in ('left','l','左') else 'RIGHT'; s=s[:m.start()].strip()
    # A broad eye folder cannot establish iris or other pixel-level identity.
    if s in {'eye','eyes','目','眼','眼睛','눈'}: return 'EYE_REGION',side
    tag=aliases.get(s,'UNKNOWN')
    return tag,side

def cdi_names(moc):
    model=moc.with_suffix('.model3.json'); cdi=moc.with_suffix('.cdi3.json')
    if model.is_file():
        refs=json.loads(model.read_text(encoding='utf-8-sig')).get('FileReferences',{})
        if refs.get('DisplayInfo'):
            cdi=(model.parent/refs['DisplayInfo']).resolve()
    if not cdi.is_relative_to((ROOT/'models').resolve()): raise ValueError('DisplayInfo outside models')
    if not cdi.is_file(): return {}
    parts=json.loads(cdi.read_text(encoding='utf-8-sig')).get('Parts',[])
    out={}
    for p in parts:
        if p['Id'] in out: raise ValueError('Duplicate CDI Part ID')
        out[p['Id']]=p.get('Name','')
    return out

def make_rows(parts,meshes,parents,names,aliases):
    if len(meshes)!=len(parents) or len(set(meshes))!=len(meshes) or len(set(parts))!=len(parts): raise ValueError('Invalid runtime IDs')
    rows=[]
    for i,(mid,pi) in enumerate(zip(meshes,parents)):
        if not isinstance(mid,str) or type(pi)!=int or pi < -1 or pi >= len(parts): raise ValueError('Invalid mesh parent')
        pid=parts[pi] if pi>=0 else ''; name=names.get(pid,''); tag,side=hint(name,aliases)
        rows.append(dict(zip(FIELDS,[i,mid,pi,pid,name,tag,side,'CDI_PART_NAME' if name else 'UNKNOWN'])))
    return rows

def validate_npz(paths,meshes,parents):
    for p in paths:
        with np.load(p,allow_pickle=False) as d:
            if d['mesh_ids'].tolist()!=meshes or d['parent_part_indices'].tolist()!=parents:
                raise ValueError(f'Runtime IDs/parent indices differ from {p.name}; no output')

class Handler(BaseHTTPRequestHandler):
    def send(self,data,mime='application/json',status=200):
        self.send_response(status); self.send_header('Content-Type',mime); self.send_header('Content-Length',str(len(data))); self.send_header('Cache-Control','no-store'); self.end_headers(); self.wfile.write(data)
    def js(self,obj,status=200): self.send(json.dumps(obj,ensure_ascii=False).encode(),status=status)
    def do_GET(self):
        u=urlparse(self.path)
        try:
            if u.path=='/': self.send(PAGE.encode(),'text/html; charset=utf-8')
            elif u.path=='/core.js': self.send(CORE.read_bytes(),'application/javascript')
            elif u.path=='/manifest': self.js([{'id':k,'name':x['name']} for k,x in MODELS.items()])
            elif u.path=='/moc': self.send(MODELS[parse_qs(u.query)['id'][0]]['file'].read_bytes(),'application/octet-stream')
            else: self.js({'error':'Not found'},404)
        except Exception as e: self.js({'error':str(e)},400)
    def do_POST(self):
        if self.path!='/audit' or self.headers.get('Origin')!='http://127.0.0.1:8002': return self.js({'error':'Invalid request'},403)
        try:
            n=int(self.headers.get('Content-Length','0'))
            if not 0<n<=2*1024*1024: raise ValueError('Invalid request size')
            b=json.loads(self.rfile.read(n)); x=MODELS[b['id']]; group=ROOT/'dataset/processed_v02'/x['id']; paths=sorted(group.glob('*.npz'))
            if not paths: raise ValueError('No NPZ for this model')
            validate_npz(paths,b['meshes'],b['parents']); names=cdi_names(x['file']); rows=make_rows(b['parts'],b['meshes'],b['parents'],names,ALIASES)
            dest=OUT/x['id']; dest.mkdir(parents=True,exist_ok=True)
            payload={'source':x['name'],'model_sha256':x['id'],'psd2live_commit':COMMIT,'alias_source_sha256':ALIAS_SHA,'verified_npz_count':len(paths),'rows':rows,'note':'Part-level weak hints, not ArtMesh ground truth'}
            (dest/'semantic_metadata.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
            with (dest/'semantic_metadata.csv').open('w',encoding='utf-8-sig',newline='') as f:
                w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
            named=sum(bool(r['part_name']) for r in rows); tagged=sum(r['part_tag_candidate']!='UNKNOWN' for r in rows)
            msg=f'{len(paths)} NPZ verified | {named}/{len(rows)} named parents | {tagged}/{len(rows)} weak tags'; print(x['name']+' | '+msg,flush=True); self.js({'message':msg})
        except Exception as e: self.js({'error':str(e)},400)

def main():
    global MODELS,ALIASES,COMMIT,ALIAS_SHA,OUT
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--psd2live',type=Path,required=True); p.add_argument('--confirm-authorized',action='store_true'); a=p.parse_args()
    if not a.confirm_authorized: p.error('Confirm permission for local model metadata and upstream source use.')
    classifier=a.psd2live/'src/main/kotlin/io/github/psd2live/core/LayerClassifier.kt'; ALIASES=read_aliases(classifier); ALIAS_SHA=hashlib.sha256(classifier.read_bytes()).hexdigest()
    if not ALIASES: raise SystemExit('No literal aliases found; check upstream version')
    COMMIT=subprocess.check_output(['git','-C',str(a.psd2live),'rev-parse','HEAD'],text=True).strip()
    if not CORE.is_file(): raise SystemExit('Cubism Core missing')
    MODELS={x['id']:x for x in inventory() if any((ROOT/'dataset/processed_v02'/x['id']).glob('*.npz'))}
    if not MODELS: raise SystemExit('No models with matching NPZ')
    OUT=ROOT/'dataset/semantic_audit_v05'; OUT.mkdir(parents=True,exist_ok=True)
    print(f'PSD2Live commit: {COMMIT} | {len(ALIASES)} literal aliases | {len(MODELS)} models with NPZ',flush=True)
    print('Private output:',OUT,flush=True); print('Open http://127.0.0.1:8002 and click Run audit. Ctrl+C to stop.',flush=True)
    with ThreadingHTTPServer(('127.0.0.1',8002),Handler) as server:
        try: server.serve_forever()
        except KeyboardInterrupt: pass

if __name__=='__main__': main()
