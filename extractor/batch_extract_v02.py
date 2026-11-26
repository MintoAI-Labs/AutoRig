"""Local MintoAI batch extraction server. Requires existing process_rig_dataset.py."""
import argparse, hashlib, json, tempfile, threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from process_rig_dataset import process

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / 'SDK/CubismSdkForWeb-5-r.5/Core/live2dcubismcore.js'
PAGE = Path(__file__).with_name('batch_extract_v02.html')
OUT = ROOT / 'dataset/processed_v02'
PARAMS = {'ParamEyeLOpen','ParamEyeROpen','ParamMouthOpenY','ParamMouthForm','ParamAngleX','ParamAngleY','ParamAngleZ'}
MAX_BODY = 128*1024*1024
LOCK = threading.Lock()

def inventory():
    seen = set(); result = []
    for p in sorted((ROOT/'models').rglob('*.moc3')):
        if not p.resolve().is_relative_to((ROOT/'models').resolve()): continue
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest in seen: continue
        seen.add(digest)
        result.append({'id': digest, 'name': p.name, 'path': p.relative_to(ROOT/'models').as_posix(), 'file':p})
    return result

class Handler(BaseHTTPRequestHandler):
    def send_data(self, body, mime='application/json', status=200):
        self.send_response(status); self.send_header('Content-Type',mime); self.send_header('Content-Length',str(len(body))); self.send_header('Cache-Control','no-store'); self.end_headers(); self.wfile.write(body)
    def json(self, data, status=200): self.send_data(json.dumps(data,ensure_ascii=False).encode(),status=status)
    def do_GET(self):
        u=urlparse(self.path)
        try:
            if u.path == '/': self.send_data(PAGE.read_bytes(),'text/html; charset=utf-8')
            elif u.path == '/core.js': self.send_data(CORE.read_bytes(),'application/javascript')
            elif u.path == '/manifest': self.json({'models':[{k:v for k,v in x.items() if k!='file'} for x in MODELS], 'parameters':sorted(PARAMS)})
            elif u.path == '/moc':
                x=BY_ID.get(parse_qs(u.query).get('id',[''])[0]); self.send_data(x['file'].read_bytes(),'application/octet-stream') if x else self.json({'error':'Unknown model'},404)
            else: self.json({'error':'Not found'},404)
        except Exception as e: self.json({'error':str(e)},500)
    def do_POST(self):
        if self.path != '/save': return self.json({'error':'Not found'},404)
        if self.headers.get('Origin') != 'http://127.0.0.1:8001': return self.json({'error':'Invalid origin'},403)
        try:
            n=int(self.headers.get('Content-Length','0'))
            if not 0<n<=MAX_BODY: raise ValueError('Invalid request size (128 MiB limit)')
            body=json.loads(self.rfile.read(n)); raw=body['raw']; x=BY_ID[body['id']]; param=raw['parameter']['id']
            if param not in PARAMS or raw['source']!=x['name'] or raw['schema_version']!='raw-0.1': raise ValueError('Invalid source/parameter/schema')
            if len(raw['samples']) not in (21,51,101): raise ValueError('Invalid sample count')
            permission=body.get('permission','').strip()
            if body.get('save') is True and (not permission or raw.get('permission_user_confirmed') is not True): raise ValueError('Permission reference and confirmation required')
            target_dir=OUT/x['id']; stem=f'{param}_{len(raw["samples"])}'; target=target_dir/(stem+'.npz')
            with LOCK:
                if body.get('save') is True and target.exists() and not body.get('overwrite'): return self.json({'ok':True,'message':'SKIP existing '+str(target.relative_to(ROOT))})
                with tempfile.TemporaryDirectory() as td:
                    path=Path(td)/(stem+'.json'); path.write_text(json.dumps(raw),encoding='utf-8')
                    if body.get('save') is True:
                        # Convert to a temporary directory on the same drive, then move validated output atomically.
                        target_dir.mkdir(parents=True,exist_ok=True)
                        with tempfile.TemporaryDirectory(dir=target_dir) as staging:
                            process(path,Path(staging),permission); (Path(staging)/(stem+'.npz')).replace(target)
                        meta={'model_path':x['path'],'sha256':x['id'],'permission_reference':permission,'schema':'raw-0.1'}
                        (target_dir/'source.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
                    else: process(path,Path(td),None)
            self.json({'ok':True,'message':('SAVED '+str(target.relative_to(ROOT))) if body.get('save') is True else 'AUDIT PASS (no training tensor)'})
        except Exception as e: self.json({'error':str(e)},400)

if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    if not CORE.is_file() or not PAGE.is_file(): raise SystemExit('Missing SDK Core or batch_extract_v02.html')
    MODELS=inventory(); BY_ID={x['id']:x for x in MODELS}
    print(f'MintoAI Batch Extractor V0.2 | {len(MODELS)} unique MOC3 files | http://127.0.0.1:8001',flush=True)
    print('Local only. Ctrl+C to stop.',flush=True)
    server=ThreadingHTTPServer(('127.0.0.1',8001),Handler)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
