"""MintoAI AutoLearn V0.7: local PSD, CMO3 and MOC3 intake.
SPDX-License-Identifier: AGPL-3.0-only
Run from extractor/ beside process_rig_dataset.py. No model or license data is uploaded.
Install psd-tools for PSD intake. Optional <stem>.autolearn.json maps psd,
cmo3 and deformers to paths relative to the MOC3. Optional <stem>.deformers.json
contains cmo3_sha256 and a deformers array of externally supplied annotations.
The V0.4 geometry trainer consumes the original rig NPZ records.
"""
import argparse, hashlib, json, os, shutil, stat, subprocess, sys, tempfile, time, zipfile
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from process_rig_dataset import process
ROOT=Path(__file__).resolve().parent.parent
PARAMS=['ParamAngleX','ParamAngleY','ParamAngleZ','ParamEyeLOpen','ParamEyeROpen','ParamMouthForm','ParamMouthOpenY']
ALIASES=dict(zip(PARAMS,['PARAM_ANGLE_X','PARAM_ANGLE_Y','PARAM_ANGLE_Z','PARAM_EYE_L_OPEN','PARAM_EYE_R_OPEN','PARAM_MOUTH_FORM','PARAM_MOUTH_OPEN_Y']))
MAX_ZIP=2*1024**3
MAX_RAW=128*1024**2

def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def write_json(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp'); tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8'); tmp.replace(p)
def read_json(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def under(base,ref,root):
    p=(base/ref).resolve()
    if not p.is_relative_to(root.resolve()) or not p.is_file(): raise ValueError(f'Missing/outside file: {ref}')
    return p

def unpack(zpath, cache, archives, contents, encoding=None):
    digest = sha(zpath)
    legacy = cache / digest
    if (legacy / '.complete').is_file():
        return legacy
    if digest in archives:
        return archives[digest]
    cache.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache) as td:
        staging=Path(td)/'unpacked'; staging.mkdir()
        with zipfile.ZipFile(zpath, metadata_encoding=encoding) as z:
            entries=z.infolist()
            if len(entries)>20000 or sum(i.file_size for i in entries)>MAX_ZIP: raise ValueError('ZIP exceeds limits')
            seen=set()
            for i in entries:
                raw=i.filename.replace('\\','/'); parts=raw.rstrip('/').split('/')
                if any(x in ('__MACOSX','.DS_Store') or x.startswith('._') for x in parts): continue
                if any(not x or x in ('.','..') or x.endswith((' ','.')) or any(c in x for c in ':*?"<>|\x00') or x.split('.')[0].upper() in {'CON','PRN','AUX','NUL',*[f'COM{i}' for i in range(1,10)],*[f'LPT{i}' for i in range(1,10)]} for x in parts): raise ValueError(f'Unsafe ZIP entry: {raw}')
                if stat.S_ISLNK(i.external_attr>>16): raise ValueError('ZIP symlink rejected')
                key='/'.join(parts).casefold()
                if key in seen: raise ValueError(f'Duplicate ZIP path: {raw}')
                seen.add(key); p=staging.joinpath(*parts)
                if i.is_dir(): p.mkdir(parents=True,exist_ok=True); continue
                p.parent.mkdir(parents=True,exist_ok=True)
                with z.open(i) as src,p.open('xb') as dst: shutil.copyfileobj(src,dst)
        files = [(p.relative_to(staging).as_posix(), sha(p))
                 for p in sorted(staging.rglob('*')) if p.is_file()]
        content = hashlib.sha256(json.dumps(files, ensure_ascii=False).encode()).hexdigest()
        if content in contents:
            dest = contents[content]
            meta = read_json(dest / '.complete')
            write_json(dest / '.complete', dict(content=content,
                                                archives=[*meta['archives'], digest]))
            archives[digest] = dest
            print('MERGE identical files', zpath.name, '->', dest.name, flush=True)
            return dest

        # Only write inside the generated cache. Keep immutable paths for provenance.
        stem = ''.join(c if c.isalnum() or c in '-_' else '_' for c in zpath.stem)
        stem = ('model_' + stem[:40]).rstrip('_')
        dest = cache / stem
        existing = {p.name.casefold() for p in cache.iterdir()}
        if dest.name.casefold() in existing:
            dest = cache / f'{stem}_{content[:12]}'
        number = 2
        while dest.name.casefold() in existing:
            dest = cache / f'{stem}_{content[:12]}_{number}'
            number += 1
        write_json(staging / '.complete', dict(content=content, archives=[digest]))
        staging.rename(dest)
    archives[digest] = contents[content] = dest
    print('UNPACK', zpath.name, '->', dest.name, flush=True)
    return dest



def normalize_export(row, models, cache):
    """Build the adjacent manifest layout required by the existing trainer."""
    moc, textures = row['file'], row['textures']
    root = cache / 'normalized'
    root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256((row['fingerprint'] + '\n' + moc.name).encode()).hexdigest()
    target = root / key
    if not (target / '.complete').is_file():
        with tempfile.TemporaryDirectory(dir=root) as temp:
            stage = Path(temp) / 'export'
            stage.mkdir()
            shutil.copy2(moc, stage / moc.name)
            names = []
            for index, texture in enumerate(textures):
                name = f'texture_{index:03d}.png'
                shutil.copy2(texture, stage / name)
                names.append(name)
            write_json(stage / f'{moc.stem}.model3.json',
                       {'FileReferences': {'Moc': moc.name, 'Textures': names}})
            (stage / '.complete').touch()
            stage.rename(target)
    row['path'] = (target / moc.name).relative_to(models).as_posix()
    print('FIX manifest layout', row['name'], '->', row['path'], flush=True)


def scan(models):
    cache=models/'_autolearn_cache'; warnings=[]; active_cache=set()
    archives = {}
    contents = {}
    for marker in sorted(cache.glob('model_*/.complete')):
        if marker.is_symlink() or marker.parent.is_symlink():
            continue
        try:
            meta = read_json(marker)
            contents[meta['content']] = marker.parent
            for digest in meta['archives']:
                archives[digest] = marker.parent
        except (OSError, ValueError, TypeError, KeyError) as exc:
            warnings.append(f'CACHE {marker.parent.name}: {exc}')
    # Each archive path contributes only its current content to this scan.
    def active(p): return cache not in p.parents or any(d in p.parents for d in active_cache)
    for p in sorted(models.rglob('*')):
        if not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(models.resolve()) or cache in p.parents: continue
        if p.suffix.lower()=='.zip':
            try:
                try:
                    active_cache.add(unpack(p, cache, archives, contents))
                except ValueError as exc:
                    if not str(exc).startswith('Unsafe ZIP entry:'):
                        raise
                    active_cache.add(unpack(p, cache, archives, contents, encoding='cp932'))
                    print('FIX ZIP encoding CP932', p.name, flush=True)
            except Exception as e:
                warnings.append(f'ZIP {p.name}: {e}')
        elif p.suffix.lower() in ('.rar','.7z'): warnings.append(f'UNSUPPORTED archive: {p.name} (extract manually)')
    by={}; referenced=set()
    for p in sorted(models.rglob('*')):
        if not active(p) or not p.is_file() or not p.name.lower().endswith('.model3.json') or p.is_symlink() or not p.resolve().is_relative_to(models.resolve()): continue
        if any(x=='__MACOSX' or x.startswith('._') for x in p.parts): continue
        try:
            refs=read_json(p)['FileReferences']; moc=under(p.parent,refs['Moc'],models)
            if moc.suffix.lower()!='.moc3' or moc.stat().st_size>128*1024**2: raise ValueError('Invalid MOC type/size')
            with moc.open('rb') as f:
                if f.read(4)!=b'MOC3': raise ValueError('Invalid MOC3 header')
            textures=[under(p.parent,x,models) for x in refs['Textures']]
            if not textures: raise ValueError('No textures')
            h=sha(moc); fp=hashlib.sha256(('\n'.join([h]+[sha(t) for t in textures])).encode()).hexdigest()
            referenced.add(moc); row=dict(id=h,fingerprint=fp,path=moc.relative_to(models).as_posix(),name=moc.name,manifest=p.relative_to(models).as_posix(),file=moc,textures=textures)
            by.setdefault(h,[]).append(row)
        except Exception as e: warnings.append(f'INVALID {p.name}: {e}')
    result=[]
    for h, rows in by.items():
        # The trainer groups records by MOC3 hash. Pick one texture set per mesh.
        rows.sort(key=lambda row: ('_autolearn_cache/' in row['path'],
                                   row['path'].casefold(), row['manifest'].casefold(),
                                   row['manifest']))
        row = rows[0]
        fingerprints = {item['fingerprint'] for item in rows}
        if len(fingerprints) > 1:
            warnings.append(f'SELECT {h[:12]}: {row["manifest"]}; '
                            f'{len(fingerprints) - 1} alternate texture set(s) skipped')
        manifest = models / row['manifest']
        matching = [item for item in rows if item['file'] == row['file']
                    and (models / item['manifest']).parent == row['file'].parent
                    and item['manifest'].endswith('.model3.json')]
        if (manifest.parent != row['file'].parent
                or not manifest.name.endswith('.model3.json') or len(matching) != 1):
            try:
                normalize_export(row, models, cache)
            except (OSError, ValueError) as exc:
                warnings.append(f'NORMALIZE {row["name"]}: {exc}')
                continue
        result.append(row)
    for p in sorted(models.rglob('*')):
        if not active(p) or not p.is_file() or any(x=='__MACOSX' or x.startswith('._') for x in p.parts): continue
        if p.suffix.lower()=='.moc3' and p.resolve() not in referenced: warnings.append(f'NO VALID MANIFEST {p.relative_to(models)}')
    return sorted(result,key=lambda r:r['path']),warnings

LOAD=r'''async (b64)=>{const bytes=Uint8Array.from(atob(b64),c=>c.charCodeAt(0));window.moc=Live2DCubismCore.Moc.fromArrayBuffer(bytes.buffer);if(!window.moc)throw Error('MOC invalid');window.model=Live2DCubismCore.Model.fromMoc(window.moc);if(!window.model)throw Error('Model invalid');return Array.from(model.parameters.ids)}'''
EXTRACT=r'''({id,actual,count,source,permission})=>{
const p=model.parameters,d=model.drawables,k=p.ids.indexOf(actual),arr=x=>Array.from(x);
const lo=p.minimumValues[k],hi=p.maximumValues[k],def=p.defaultValues[k];if(!(Number.isFinite(lo)&&Number.isFinite(hi)&&hi>lo))throw Error('Invalid range');
function snap(){return {positions:d.vertexPositions.map(arr),opacities:arr(d.opacities),draw_orders:arr(d.drawOrders),dynamic_flags:arr(d.dynamicFlags)}}
p.values.set(p.defaultValues);model.update();const rest=snap();
const meshes=arr(d.ids).map((mid,i)=>({id:mid,vertex_count:d.vertexCounts[i],texture_index:d.textureIndices[i],parent_part_index:d.parentPartIndices[i],constant_flags:d.constantFlags[i],uv:arr(d.vertexUvs[i]),triangles:arr(d.indices[i]),mask_indices:arr(d.masks[i]),vertices_rest:rest.positions[i]}));
const total=meshes.reduce((s,m)=>s+m.vertex_count,0);if(total*count>3500000)throw Error('Sweep exceeds vertex-pose safety limit');
const samples=[];for(let i=0;i<count;i++){p.values.set(p.defaultValues);p.values[k]=lo+(hi-lo)*i/(count-1);model.update();const s=snap();samples.push({value:p.values[k],vertices_posed:s.positions,opacities:s.opacities,draw_orders:s.draw_orders,dynamic_flags:s.dynamic_flags})}
return JSON.stringify({schema_version:'raw-0.1',source,permission_note:permission,permission_user_confirmed:true,core_version:Live2DCubismCore.Version.csmGetVersion(),coordinate_system:'Cubism Core native coordinates, xy interleaved',parameter:{id,min:lo,max:hi,default:def},original_parameter_id:actual,all_parameter_ids:arr(p.ids),all_parameter_defaults:arr(p.defaultValues),meshes,rest_opacities:rest.opacities,rest_draw_orders:rest.draw_orders,samples});}'''

def reuse_old(old,dest,row,permission,count):
    if not (old/'source.json').is_file(): return False
    meta=read_json(old/'source.json')
    if meta.get('sha256')!=row['id'] or meta.get('model_path')!=row['path']: return False
    files=list(old.glob('*.npz')); found=set()
    for p in files:
        with np.load(p,allow_pickle=False) as z:
            param=str(z['parameter_id'].item())
            if param not in PARAMS or param in found or len(z['parameter_values'])!=count or str(z['source'].item())!=row['name'] or not str(z['permission_note'].item()).strip(): return False
            found.add(param)
    if not found: return False
    # Old cache may omit legacy parameters or be partial: only reuse if all seven exist.
    if found!=set(PARAMS): return False
    for p in files: shutil.copy2(p,dest/p.name)
    return True

def extract(page,core,row,dest,count,permission):
    import base64
    page.set_content('<html></html>'); page.add_script_tag(content=core); page.wait_for_function('()=>{try{return !!Live2DCubismCore.Version.csmGetVersion()}catch(e){return false}}',timeout=30000)
    ids=page.evaluate(LOAD,base64.b64encode(row['file'].read_bytes()).decode('ascii'))
    mapping={p:p if p in ids else ALIASES[p] for p in PARAMS if p in ids or ALIASES[p] in ids}
    if not mapping: raise ValueError('No supported facial parameters')
    for p,actual in mapping.items():
        raw=page.evaluate(EXTRACT,dict(id=p,actual=actual,count=count,source=row['name'],permission=permission))
        if len(raw.encode('utf-8'))>MAX_RAW: raise ValueError('Raw sweep too large')
        with tempfile.TemporaryDirectory(dir=dest) as td:
            path=Path(td)/(p+f'_{count}.json'); path.write_text(raw,encoding='utf-8'); process(path,dest,permission)
    print('EXTRACTED',row['name'],mapping,flush=True)
    return mapping

def validate_group(folder,models):
    # Run the exact trainer's texture and topology validation before admitting the group.
    import torch
    sys.path.insert(0,str(ROOT/'training'))
    from train_real_rig_v03 import load_visual
    # load_visual expects a directory containing hash subdirectories.
    audit = folder.parent / 'visual_checks'
    audit.mkdir(parents=True, exist_ok=True)
    records = load_visual(folder, models, torch.device('cpu'), audit)
    return {r['param'] for r in records}


def source_assets(row, models):
    """Pair exact-stem sources in the export directory and its parent."""
    moc = row['file']
    roots = [moc.parent]
    if moc.parent != models and moc.parent.parent.is_relative_to(models):
        roots.append(moc.parent.parent)
    pairing = moc.with_suffix('.autolearn.json')
    names = read_json(pairing) if pairing.is_file() else {}
    if not isinstance(names, dict) or set(names) - {'psd', 'cmo3', 'deformers'}:
        raise ValueError('Invalid .autolearn.json keys')
    assets = {}
    for key, suffix in (('psd', '.psd'), ('cmo3', '.cmo3'), ('deformers', '.deformers.json')):
        if key in names:
            candidate = under(moc.parent, names[key], models)
            if not candidate.name.lower().endswith(suffix):
                raise ValueError(f'Expected {suffix}: {candidate.name}')
            assets[key] = candidate
            continue
        matches = sorted({
            p.resolve() for root in roots for p in root.iterdir()
            if p.name.casefold() == (moc.stem + suffix).casefold()
            and p.is_file() and not p.is_symlink()
        })
        if len(matches) > 1:
            raise ValueError(f'Ambiguous {key}; set it in {pairing.name}')
        if matches:
            assets[key] = matches[0]
    if 'deformers' in assets and 'cmo3' not in assets:
        raise ValueError('Deformer metadata requires the matching CMO3')
    return assets


def export_source_features(row, dest):
    """Write spatial PSD features and optional editor annotations beside rig records."""
    assets = row['assets']
    if not assets:
        return
    layers = []
    if 'psd' in assets:
        from psd_tools import PSDImage
        psd = PSDImage.open(assets['psd'])
        paths = {}
        pixel_data = {}
        for index, layer in enumerate(psd.descendants()):
            label = str(layer.name)
            path = [*paths.get(id(layer.parent), []), label]
            paths[id(layer)] = path
            box = list(layer.bbox)
            item = dict(index=index, name=label, group_path=path[:-1],
                        kind=layer.kind, bbox=box, visible=layer.is_visible())
            if not layer.is_group() and box[2] > box[0] and box[3] > box[1]:
                image = layer.composite()
                if image is not None:
                    image = image.convert('RGBA')
                    image.thumbnail((128, 128))
                    key = f'layer_{index}'
                    pixel_data[key] = np.asarray(image, dtype=np.uint8)
                    item['rgba_key'] = key
            layers.append(item)
        np.savez_compressed(dest/'artwork_layers.npz', **pixel_data)
        canvas = [psd.width, psd.height]
    else:
        canvas = None
    deformers = None
    if 'deformers' in assets:
        deformers = read_json(assets['deformers'])
        if (not isinstance(deformers, dict)
                or deformers.get('cmo3_sha256') != row['asset_hashes']['cmo3']
                or not isinstance(deformers.get('deformers'), list)
                or not all(isinstance(item, dict) for item in deformers['deformers'])):
            raise ValueError('Expected cmo3_sha256 and a deformers array tied to this CMO3')
    write_json(dest/'multimodal.json', dict(
        schema='autolearn-multimodal-0.4', moc3_sha256=row['id'],
        assets={key: dict(name=path.name, sha256=row['asset_hashes'][key])
                for key, path in assets.items()},
        canvas=canvas, layers=layers, deformers=deformers,
        rig_records=sorted(p.name for p in dest.glob('*.npz')
                           if p.name != 'artwork_layers.npz'),
    ))


def watch(args):
    """Foreground local watcher; never approves new files automatically."""
    root=ROOT/'models'; root.mkdir(exist_ok=True)
    def signature():
        items=[]
        for p in root.rglob('*'):
            if '_autolearn_cache' in p.parts or not p.is_file() or p.suffix.lower() in ('.tmp','.part','.crdownload','.download'): continue
            try:
                st=p.stat(); items.append((str(p.relative_to(root)),st.st_size,st.st_mtime_ns))
            except OSError: pass
        return tuple(sorted(items))
    last=None; candidate=None; unchanged_since=time.monotonic()
    child_args=[x for x in sys.argv[1:] if x!='--watch']
    print('WATCH models/ | Ctrl+C to stop | waits for files to stabilize',flush=True)
    try:
        while True:
            sig=signature()
            if sig!=candidate: candidate=sig; unchanged_since=time.monotonic()
            if sig!=last and time.monotonic()-unchanged_since>=10:
                last=sig
                result=subprocess.run([sys.executable,str(Path(__file__).resolve()),*child_args])
                print('Pipeline exit:',result.returncode,'| watching for changes',flush=True)
            time.sleep(3)
    except KeyboardInterrupt: print('Watcher stopped.',flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--watch',action='store_true'); ap.add_argument('--force-train',action='store_true'); ap.add_argument('--scan-only',action='store_true'); ap.add_argument('--review',action='store_true'); ap.add_argument('--extract-only',action='store_true'); ap.add_argument('--steps',type=int,default=4000); ap.add_argument('--samples',type=int,choices=(21,51,101),default=101); ap.add_argument('--device',choices=('cpu','cuda'),default='cuda'); ap.add_argument('--confirm-sdk-authorized',action='store_true'); ap.add_argument('--browser',default='msedge'); ap.add_argument('--browser-executable',type=Path); ap.add_argument('--validation',default='Hiyori'); ap.add_argument('--holdout',default='Ren'); a=ap.parse_args()
    if a.steps<1: ap.error('steps must be positive')
    if a.watch:
        if not a.scan_only and not a.confirm_sdk_authorized: ap.error('Confirm SDK permission before watching')
        return watch(a)
    models=ROOT/'models'; models.mkdir(exist_ok=True); home=ROOT/'dataset/auto_ingest_v04'; home.mkdir(parents=True,exist_ok=True)
    rows,warnings=scan(models); path=home/'permissions.json'; permissions=read_json(path) if path.exists() else {}
    candidates = []
    for r in rows:
        try:
            r['assets'] = source_assets(r, models)
            r['asset_hashes'] = {key: sha(path) for key, path in r['assets'].items()}
            if r['assets']:
                r['fingerprint'] = hashlib.sha256(json.dumps(
                    [r['fingerprint'], r['asset_hashes']], sort_keys=True
                ).encode()).hexdigest()
            print('SOURCES', r['name'], ', '.join(sorted(r['assets'])) or 'MOC3 only', flush=True)
            if 'cmo3' in r['assets'] and 'deformers' not in r['assets']:
                warnings.append(f'CMO3 paired, deformer annotations absent: {r["name"]}')
            candidates.append(r)
        except (OSError, ValueError, TypeError) as exc:
            warnings.append(f'SOURCE MATCH {r["name"]}: {exc}')
    rows = candidates
    for r in rows:
        suggestion='Model C family' if r['file'].stem.casefold() in ('c6','c_v2') else r['file'].stem
        entry=permissions.setdefault(r['fingerprint'],dict(approved=False,reference='',family=suggestion,path=r['path']))
        entry['path']=r['path']
        print(('APPROVED' if entry.get('approved') is True and entry.get('reference') and entry.get('family') else 'PENDING'),r['name'],r['fingerprint'][:12],flush=True)
        if a.review and not (entry.get('approved') is True and entry.get('reference') and entry.get('family')):
            print('Approve ONLY if all rights holders allow processing all listed source assets and AI training.'); ref=input('Permission reference (Enter to skip): ').strip()
            if ref:
                fam=input(f'Character family [{entry["family"]}]: ').strip() or entry['family']; entry.update(approved=True,reference=ref,family=fam)
                write_json(path,permissions)
    write_json(path,permissions)
    for w in warnings: print(w,flush=True)
    print('Permission register:',path,flush=True)
    if a.scan_only: return
    if not a.confirm_sdk_authorized: ap.error('Confirm SDK permission with --confirm-sdk-authorized')
    approved=[r for r in rows if permissions[r['fingerprint']].get('approved') is True and all(isinstance(permissions[r['fingerprint']].get(k),str) and permissions[r['fingerprint']][k].strip() for k in ('reference','family'))]
    if not approved: raise SystemExit('No approved models. Use --review --scan-only first.')
    core=ROOT/'SDK/CubismSdkForWeb-5-r.5/Core/live2dcubismcore.js'
    if not core.is_file(): raise SystemExit('Missing Cubism Core')
    from playwright.sync_api import sync_playwright
    multimodal=home/'multimodal'; multimodal.mkdir(exist_ok=True)
    cache=home/'records'; cache.mkdir(exist_ok=True); ok=[]; failures=[]
    core_js=core.read_text(encoding='utf-8-sig')
    with sync_playwright() as pw:
        opts={'executable_path':str(a.browser_executable)} if a.browser_executable else ({} if a.browser=='chromium' else {'channel':a.browser})
        browser=pw.chromium.launch(headless=True,**opts)
        try:
            for r in approved:
                perm=permissions[r['fingerprint']]['reference']; key=r['fingerprint']+'_'+str(a.samples); base=cache/key; dest=base/r['id']; meta=dest/'source.json'; page=None
                try:
                    if not (base/'.complete').is_file():
                        if base.exists(): shutil.rmtree(base)
                        base.mkdir(); dest.mkdir()
                        if not reuse_old(ROOT/'dataset/processed_v02'/r['id'],dest,r,perm,a.samples):
                            page=browser.new_page(); mapping=extract(page,core_js,r,dest,a.samples,perm)
                        else: mapping={p:p for p in PARAMS}; print('REUSE V0.2',r['name'],flush=True)
                        write_json(meta,dict(model_path=r['path'],sha256=r['id'],permission_reference=perm,schema='raw-0.1',fingerprint=r['fingerprint'],original_parameter_ids=mapping))
                        params=validate_group(base,models); (base/'.complete').touch()
                    else:
                        old=read_json(meta); old.update(model_path=r['path'],permission_reference=perm); write_json(meta,old)
                        params=validate_group(base,models); print('CACHE',r['name'],flush=True)
                    if r['assets']:
                        output = multimodal/key
                        if not (output/'multimodal.json').is_file():
                            with tempfile.TemporaryDirectory(dir=multimodal) as tmp:
                                staging = Path(tmp)/r['id']
                                staging.mkdir()
                                for source in dest.glob('*.npz'):
                                    try:
                                        os.link(source, staging/source.name)
                                    except OSError:
                                        shutil.copy2(source, staging/source.name)
                                export_source_features(r, staging)
                                staging.rename(output)
                    ok.append((r,dest,params))
                except Exception as e:
                    failures.append(f'{r["name"]}: {e}'); print('SKIP MODEL',failures[-1],flush=True)
                    if base.exists(): shutil.rmtree(base)
                finally:
                    if page: page.close()
        finally: browser.close()
    print('Eligible extracted models:',len(ok),'Failed:',len(failures),flush=True)
    write_json(home/'last_report.json',dict(eligible=[dict(name=r['name'],id=r['id'],parameters=sorted(p)) for r,d,p in ok],failures=failures,warnings=warnings))
    if a.extract_only: return
    # Keep evaluation families isolated, including differently named versions.
    def family(r): return permissions[r['fingerprint']]['family'].strip().casefold()
    vg=[r for r,d,p in ok if Path(r['name']).stem.casefold()==a.validation.casefold()]
    hg=[r for r,d,p in ok if Path(r['name']).stem.casefold()==a.holdout.casefold()]
    if len(vg)!=1 or len(hg)!=1 or family(vg[0])==family(hg[0]): raise SystemExit('Need distinct approved validation and holdout models')
    vf,hf=family(vg[0]),family(hg[0]); selected=[]
    for r,d,p in ok:
        if family(r) in (vf,hf) and r['id'] not in (vg[0]['id'],hg[0]['id']): print('EXCLUDE evaluation-family variant:',r['name']); continue
        selected.append((r,d,p))
    train=[p for r,d,p in selected if family(r) not in (vf,hf)]
    val=next(p for r,d,p in selected if r['id']==vg[0]['id'])
    if not train or set.union(*train)!=set(PARAMS) or val!=set(PARAMS): raise SystemExit('Training and validation need all 7 supported parameters')
    # Avoid training again on exactly the same approved dataset/configuration.
    signature=dict(records=sorted((r['fingerprint'],family(r),sorted(p)) for r,d,p in selected),samples=a.samples,steps=a.steps,device=a.device,validation=a.validation,holdout=a.holdout,trainer_sha256=sha(ROOT/'training/train_real_rig_v04.py'))
    stamp=home/'last_successful_train.json'
    if stamp.exists() and not a.force_train and read_json(stamp)==json.loads(json.dumps(signature)):
        print('NO NEW TRAINING DATA: previous run succeeded; use --force-train to repeat.',flush=True); return
    # A fresh snapshot excludes revoked/unapproved/stale records. Hardlinks avoid duplicate tensors.
    snapshot=home/'snapshots'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'); snapshot.mkdir(parents=True)
    for r,d,p in selected:
        out=snapshot/r['id']; out.mkdir()
        for f in d.glob('*.npz'):
            try: os.link(f,out/f.name)
            except OSError: shutil.copy2(f,out/f.name)
        shutil.copy2(d/'source.json',out/'source.json')
    print('Train from fresh initialization on approved old + new data:',snapshot,flush=True)
    subprocess.run([sys.executable,str(ROOT/'training/train_real_rig_v04.py'),'--data',str(snapshot),'--models',str(models),'--steps',str(a.steps),'--device',a.device,'--validation',a.validation,'--holdout',a.holdout,'--confirm-authorized'],check=True)
    write_json(stamp,signature)
if __name__=='__main__': main()