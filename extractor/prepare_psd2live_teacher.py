"""Run a local PSD2Live checkout and extract its generated rig into training NPZs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--psd2live',type=Path,required=True)
    parser.add_argument('--psd-dir',type=Path,default=ROOT/'models')
    parser.add_argument('--output',type=Path,default=ROOT/'dataset/psd2live_teacher')
    parser.add_argument('--samples',type=int,default=51)
    parser.add_argument('--confirm-authorized',action='store_true')
    args=parser.parse_args()
    if not args.confirm_authorized:
        parser.error('Confirm permission for local artwork processing and training')
    if args.samples<5 or args.samples%2==0:
        parser.error('--samples must be odd and at least five')
    checkout=args.psd2live.resolve()
    psds=args.psd_dir.resolve()
    output=args.output.resolve()
    commit=subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
    tracked_changes=subprocess.check_output(['git','-C',str(checkout),'diff','HEAD','--'],text=True)
    code_hash=hashlib.sha256((commit+'\n'+tracked_changes).encode()).hexdigest()
    if not psds.is_dir():
        parser.error('PSD directory not found')
    unique={}
    for path in sorted(psds.rglob('*')):
        if path.suffix.lower()=='.psd' and path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(psds):
            unique.setdefault(digest(path),path.resolve())
    if not unique:
        parser.error('No local PSD found. Point --psd-dir at your layered PSD collection')
    import autolearn_v07 as extractors
    core=ROOT/'SDK/CubismSdkForWeb-5-r.5/Core/live2dcubismcore.js'
    if not core.is_file():
        parser.error(f'Cubism Core not found at {core}')
    from playwright.sync_api import sync_playwright
    output.mkdir(parents=True,exist_ok=True)
    prepared=False
    accepted=[]
    failed=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,channel='msedge')
        try:
            for source_hash,psd in unique.items():
                identity=hashlib.sha256(f'{source_hash}:{code_hash}:{args.samples}:fresh-moc-v1'.encode()).hexdigest()
                case=output/'cases'/identity
                receipt=case/'case.json'
                if receipt.is_file():
                    old=json.loads(receipt.read_text(encoding='utf-8'))
                    if old.get('complete'):
                        accepted.append(old)
                        print('CACHE',psd.name,flush=True)
                        continue
                try:
                    if not prepared:
                        wrapper=checkout/('gradlew.bat' if __import__('os').name=='nt' else 'gradlew')
                        subprocess.run([str(wrapper),'--no-daemon','writeRunArgs'],cwd=checkout,check=True)
                        prepared=True
                    java=Path((checkout/'build/run/java').read_text(encoding='utf-8-sig').strip())
                    models=case/'models'
                    models.mkdir(parents=True,exist_ok=True)
                    subprocess.run([str(java),'@build/run/jvm.args','--input',str(psd),
                                    '--output',str(models),'--no-physics','--no-motions','--no-cmo3'],
                                   cwd=checkout,check=True)
                    rows,warnings=extractors.scan(models)
                    if not rows:
                        raise ValueError('PSD2Live produced no extractable model3/MOC3 pair')
                    data=case/'data'
                    # Re-create model state for every sweep so rest metadata is consistent.
                    for row in rows:
                        dest=data/row['id']
                        dest.mkdir(parents=True,exist_ok=True)
                        for old in dest.glob('*.npz'):
                            old.unlink()
                        mapping={}
                        for param in extractors.PARAMS:
                            page=browser.new_page()
                            try:
                                page.set_content('<html></html>')
                                page.add_script_tag(content=core.read_text(encoding='utf-8-sig'))
                                page.wait_for_function(
                                    '()=>{try{return !!Live2DCubismCore.Version.csmGetVersion()}catch(e){return false}}',
                                    timeout=30000)
                                import base64
                                ids=page.evaluate(extractors.LOAD,base64.b64encode(row['file'].read_bytes()).decode())
                                actual=param if param in ids else extractors.ALIASES[param]
                                if actual not in ids:
                                    continue
                                raw=page.evaluate(extractors.EXTRACT,dict(id=param,actual=actual,
                                    count=args.samples,source=row['name'],
                                    permission='User authorized local PSD2Live teacher generation'))
                                if len(raw.encode())>extractors.MAX_RAW:
                                    raise ValueError('Extracted parameter exceeds size limit')
                                with tempfile.TemporaryDirectory(dir=dest) as tmp:
                                    source=Path(tmp)/(param+'.json')
                                    source.write_text(raw,encoding='utf-8')
                                    extractors.process(source,dest,'User authorized local PSD2Live teacher generation')
                                mapping[param]=actual
                            finally:
                                page.close()
                        if not mapping:
                            raise ValueError('No supported head/eye/mouth controls in generated model')
                        extractors.write_json(dest/'source.json',dict(
                            model_path=row['path'],sha256=row['id'],schema='raw-0.1',
                            fingerprint=row['fingerprint'],original_parameter_ids=mapping,
                            teacher='PSD2Live',psd_sha256=source_hash,psd2live_commit=commit))
                    extractors.validate_group(data,models)
                    meta=dict(id=identity,complete=True,source_psd=str(psd),psd_sha256=source_hash,
                              psd2live_commit=commit,psd2live_code_hash=code_hash,
                              models=str(models),data=str(data),family=psd.stem,
                              label_type='procedural_pseudo_label',warnings=warnings)
                    write(receipt,meta)
                    accepted.append(meta)
                    print('READY',psd.name,flush=True)
                except Exception as exc:
                    failed.append(dict(psd=str(psd),error=str(exc)))
                    print('FAILED',psd.name,str(exc),flush=True)
                write(output/'index.json',dict(schema=1,cases=accepted,failures=failed))
        finally:
            browser.close()
    write(output/'index.json',dict(schema=1,cases=accepted,failures=failed))
    print(f'PSD2Live teacher cases ready {len(accepted)}; failed {len(failed)}',flush=True)
    if not accepted:
        raise SystemExit(1)


if __name__=='__main__':
    main()
