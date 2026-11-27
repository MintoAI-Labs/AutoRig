"""Read-only LAN dashboard for MintoAI curriculum checkpoints (Python + PyTorch)."""
import ast, math, argparse, ctypes, io, json, os, re, secrets, socket, threading, time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs

PAGE = '<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>MintoAI Training</title>\n<style>\n:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;padding:20px 14px;background:#101916;color:#edf5ef;font:15px system-ui;max-width:1150px;margin:auto}\nheader{display:flex;justify-content:space-between;gap:15px;align-items:center;flex-wrap:wrap;margin-bottom:20px}h1{font-size:29px;margin:4px 0}h2{font-size:23px;margin:6px 0 14px}h3{font-size:16px}select{font:inherit;padding:9px;background:#22372e;color:inherit;border:1px solid #57675f;border-radius:8px}\n.panels{display:grid;grid-template-columns:1fr 1fr;gap:16px}.panel{background:#18251f;border:1px solid #354d40;border-radius:18px;padding:18px;min-width:0;--accent:#bce9db}.unified{grid-column:1/-1}.extreme{background:#281f26;--accent:#f6d5e2}\n.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:15px 0}.tile{border:1px solid #ffffff14;background:#ffffff06;padding:12px;border-radius:12px;min-width:0}\n.label,.note{color:#adbbb4;line-height:1.5;font-size:13px}.value{font-size:25px;color:var(--accent);font-weight:650;overflow-wrap:anywhere;font-variant-numeric:tabular-nums}.smallvalue{font-size:15px}.task{color:var(--accent);margin:12px 0}\ncanvas{width:100%;height:180px;display:block}.hardware{margin-top:16px}.hardware .grid{grid-template-columns:repeat(3,1fr)}hr{border:0;border-top:1px solid #ffffff18;margin:18px 0}pre{font:12px ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;text-align:right;border-bottom:1px solid #ffffff16;white-space:nowrap}th:first-child,td:first-child{text-align:left}\n@media(max-width:700px){.panels{grid-template-columns:1fr}.hardware .grid{grid-template-columns:1fr 1fr}}\n</style>\n<header><div><h1 data-t="title"></h1><div class="note" id="connection"></div></div><label><span data-t="language"></span> <select id="language"><option value="vi">Tiếng Việt</option><option value="en">English</option><option value="ja">日本語</option></select></label></header>\n<main id="panels" class="panels"></main>\n<section class="panel hardware"><h2 data-t="hardware"></h2><div id="hwtiles" class="grid"></div><div id="sources" class="note"></div><h3 data-t="temperature"></h3><canvas id="temperature"></canvas><h3 data-t="usage"></h3><canvas id="usage"></canvas><h3 data-t="memory"></h3><canvas id="memory"></canvas><div class="table"><table id="hwstats"></table></div><p class="note" data-t="hw_note"></p></section><p class="note" data-t="note"></p>\n<script>\nconst translations={"vi": {"title": "MintoAI Training", "language": "Ngôn ngữ", "connecting": "Đang kết nối…", "connected": "Đã kết nối", "disconnected": "Mất kết nối", "real": "Model Live2D thật", "extreme": "Bài tập toàn thân", "step": "Bước đã lưu", "round": "Vòng", "position": "Bài hiện tại", "passes": "PASS liên tiếp", "limit": "Ngưỡng qua bài", "current": "Sai số hiện tại", "best": "Mốc tốt nhất", "secondary": "Extreme / ôn bài cũ", "action": "Lần kiểm tra gần nhất", "lr": "Learning rate", "rate": "Bước/giây trong cửa sổ", "delta": "Sai số thay đổi trong 15 phút", "age": "Giây từ lần ghi log", "nodata": "Chưa có log đo sai số", "hardware": "Phần cứng · 15 phút gần nhất", "cpu_temp": "Nhiệt CPU", "gpu_temp": "Nhiệt GPU", "cpu_load": "Tải CPU", "gpu_load": "Tải GPU", "ram_load": "RAM", "vram_load": "VRAM", "temperature": "Nhiệt độ CPU / GPU", "usage": "Tải CPU / GPU", "memory": "Bộ nhớ", "fixed": "Bài cố định", "random": "Sinh bài ngẫu nhiên", "phase": "Giai đoạn học", "best_action": "Đã nhận mốc tốt hơn", "trial": "Đang thử tiếp", "rollback": "Quay về mốc tốt nhất, đổi learning rate", "new_exercise": "Đã chuyển bài", "details": "Sai số từng bài / parameter", "read_error": "Chưa đọc được checkpoint", "cpu_missing": "Chưa đọc được nhiệt CPU", "gpu_missing": "Chưa đọc được GPU", "sensor_wait": "Đang chờ cảm biến", "now": "Nay", "min": "Min", "mean": "Trung bình", "max": "Max", "head tilt": "Nghiêng đầu", "one arm": "Một tay", "one leg": "Một chân", "facial expressions": "Biểu cảm", "both arms and torso": "Hai tay và thân", "steps and balance": "Bước và thăng bằng", "combined poses": "Pose phối hợp", "note": "Chỉ đọc. Số bước lấy từ checkpoint hoặc log đã ghi. Tuổi log cho biết lần cập nhật, không xác nhận tiến trình còn chạy. Biểu đồ chỉ so cùng vòng và bài trong 15 phút gần nhất.", "hw_note": "Dữ liệu phần cứng của toàn máy. Cảm biến thiếu để trống. Lịch sử phần cứng bắt đầu khi mở dashboard.", "saved": "Checkpoint đã đọc", "eval_seconds": "Thời gian kiểm tra (giây)", "unified": "Mạng chung · rig thật làm chuẩn", "joint_training": "Luyện phối hợp các nguồn", "policy": "Cách tính loss đang thử", "pose_batch": "Pose mỗi lượt", "attempts": "Số đợt thử", "diagnosis": "Chẩn đoán", "algorithm": "Quy tắc đã thử", "rule_rotation": "Thử quy tắc xoay", "rule_tps": "Thử TPS", "rule_hybrid": "Phối hợp quy tắc", "psd_hint": "Thử gợi ý PSD2Live", "static_drift": "Vùng đứng yên bị trôi", "moving_vertices": "Vùng chuyển động"}, "en": {"title": "MintoAI Training", "language": "Language", "connecting": "Connecting…", "connected": "Connected", "disconnected": "Disconnected", "real": "Real Live2D models", "extreme": "Full-body exercises", "step": "Saved step", "round": "Round", "position": "Current exercise", "passes": "Consecutive passes", "limit": "Promotion limits", "current": "Current error", "best": "Best accepted error", "secondary": "Extreme / previous skills", "action": "Latest evaluation", "lr": "Learning rate", "rate": "Steps/s in window", "delta": "Error change over 15 minutes", "age": "Seconds since metrics update", "nodata": "No error metrics yet", "hardware": "Hardware · last 15 minutes", "cpu_temp": "CPU temperature", "gpu_temp": "GPU temperature", "cpu_load": "CPU usage", "gpu_load": "GPU usage", "ram_load": "RAM", "vram_load": "VRAM", "temperature": "CPU / GPU temperature", "usage": "CPU / GPU usage", "memory": "Memory usage", "fixed": "Fixed exercises", "random": "Random exercises", "phase": "Learning phase", "best_action": "Accepted improvement", "trial": "Continuing trial", "rollback": "Restored best and changed learning rate", "new_exercise": "Moved to next exercise", "details": "Per-exercise / parameter errors", "read_error": "Checkpoint unavailable", "cpu_missing": "CPU temperature unavailable", "gpu_missing": "GPU telemetry unavailable", "sensor_wait": "Waiting for sensors", "now": "Now", "min": "Min", "mean": "Mean", "max": "Max", "head tilt": "Head tilt", "one arm": "One arm", "one leg": "One leg", "facial expressions": "Facial expressions", "both arms and torso": "Both arms and torso", "steps and balance": "Steps and balance", "combined poses": "Combined poses", "note": "Read only. Steps come from saved checkpoints or metrics. Log age shows freshness and does not prove a process is running. Charts compare the same round and exercise within the last 15 minutes.", "hw_note": "Hardware readings cover the whole computer. Missing sensors are left blank. Hardware history starts with the dashboard.", "saved": "Checkpoint loaded", "eval_seconds": "Evaluation time (seconds)", "unified": "Unified network · human-rig reference", "joint_training": "Joint training across sources", "policy": "Loss policy under trial", "pose_batch": "Poses per batch", "attempts": "Experiment count", "diagnosis": "Diagnosis", "algorithm": "Rule under trial", "rule_rotation": "Rotation rule", "rule_tps": "TPS rule", "rule_hybrid": "Combined rules", "psd_hint": "PSD2Live proposal", "static_drift": "Static-region drift", "moving_vertices": "Moving regions"}, "ja": {"title": "MintoAI トレーニング", "language": "言語", "connecting": "接続中…", "connected": "接続済み", "disconnected": "接続が切れました", "real": "実際のLive2Dモデル", "extreme": "全身の練習", "step": "保存済みステップ", "round": "ラウンド", "position": "現在の課題", "passes": "連続合格", "limit": "合格基準", "current": "現在の誤差", "best": "採用済みの最小誤差", "secondary": "極端な姿勢・復習", "action": "最新の評価", "lr": "学習率", "rate": "ウィンドウ内のステップ/秒", "delta": "15分間の誤差変化", "age": "メトリクス更新からの秒数", "nodata": "誤差ログはまだありません", "hardware": "ハードウェア・直近15分", "cpu_temp": "CPU温度", "gpu_temp": "GPU温度", "cpu_load": "CPU使用率", "gpu_load": "GPU使用率", "ram_load": "RAM", "vram_load": "VRAM", "temperature": "CPU・GPU温度", "usage": "CPU・GPU使用率", "memory": "メモリ使用率", "fixed": "固定課題", "random": "ランダム課題", "phase": "学習段階", "best_action": "改善を採用", "trial": "試行を継続中", "rollback": "最良状態に戻し学習率を変更", "new_exercise": "次の課題へ進みました", "details": "課題・パラメータ別の誤差", "read_error": "チェックポイントを読み取れません", "cpu_missing": "CPU温度を取得できません", "gpu_missing": "GPU情報を取得できません", "sensor_wait": "センサーを待機中", "now": "現在", "min": "最小", "mean": "平均", "max": "最大", "head tilt": "頭の傾き", "one arm": "片腕", "one leg": "片脚", "facial expressions": "表情", "both arms and torso": "両腕と胴体", "steps and balance": "足運びとバランス", "combined poses": "複合ポーズ", "note": "読み取り専用です。保存済みのステップとログを表示します。ログの時刻だけではプロセスの稼働を確認できません。直近15分の同じラウンド・課題を比較します。", "hw_note": "ハードウェア情報はマシン全体の値です。取得できないセンサーは空欄です。履歴はダッシュボード起動時から記録します。", "saved": "チェックポイントを読み込みました", "eval_seconds": "評価時間（秒）", "unified": "統合ネットワーク・手作業リグ基準", "joint_training": "複数ソースの共同学習", "policy": "試行中の損失関数", "pose_batch": "バッチ内のポーズ数", "attempts": "実験回数", "diagnosis": "診断", "algorithm": "試行した規則", "rule_rotation": "回転規則", "rule_tps": "TPS規則", "rule_hybrid": "規則の組み合わせ", "psd_hint": "PSD2Liveの候補", "static_drift": "静止部分のずれ", "moving_vertices": "動く部分"}};\nconst $=id=>document.getElementById(id);\nlet language=localStorage.getItem(\'minto-language\')||\'vi\';\nif(!translations[language])language=\'vi\';\nconst t=k=>translations[language][k]||k;\nconst locale=()=>({vi:\'vi-VN\',en:\'en-US\',ja:\'ja-JP\'}[language]);\nconst numeric=v=>typeof v===\'number\'&&Number.isFinite(v);\nconst fmt=v=>numeric(v)?v.toLocaleString(locale(),{maximumSignificantDigits:6}):(v??\'—\');\nconst token=new URLSearchParams(location.search).get(\'token\');\nlet latest=null;\nconst fields=[\'step\',\'round\',\'position\',\'passes\',\'phase\',\'limit\',\'current\',\'secondary\',\'best\',\'action\',\'lr\',\'rate\',\'delta\',\'age\',\'eval_seconds\',\'policy\',\'pose_batch\',\'attempts\',\'diagnosis\',\'algorithm\'];\nfor(const kind of [\'real\',\'extreme\',\'unified\']){\n const panel=document.createElement(\'section\');panel.className=\'panel \'+kind;panel.id=kind;\n panel.innerHTML=\'<h2 data-t="\'+kind+\'"></h2><div class="note" data-k="status"></div><div class="task" data-k="task"></div><div class="grid">\'+fields.filter(k=>kind===\'unified\'||![\'policy\',\'pose_batch\',\'attempts\',\'diagnosis\',\'algorithm\'].includes(k)).map(k=>\'<div class="tile"><div class="label" data-t="\'+k+\'"></div><div class="value \'+([\'action\',\'phase\'].includes(k)?\'smallvalue\':\'\')+\'" data-k="\'+k+\'">—</div></div>\').join(\'\')+\'</div><canvas data-k="chart"></canvas><details><summary data-t="details"></summary><pre data-k="details"></pre></details><pre class="note" data-k="file"></pre>\';\n $(\'panels\').append(panel);\n}\nconst metrics=[\'cpu_temp\',\'gpu_temp\',\'cpu_load\',\'gpu_load\',\'ram_load\',\'vram_load\'];\nfor(const k of metrics){const div=document.createElement(\'div\');div.className=\'tile\';div.innerHTML=\'<div class="label" data-t="\'+k+\'"></div><div class="value" id="\'+k+\'">—</div><div class="note" id="\'+k+\'_max"></div>\';$(\'hwtiles\').append(div)}\nfunction chart(canvas,points,keys,maxY=null){\n const width=canvas.clientWidth,height=180,ratio=devicePixelRatio||1;\n canvas.width=width*ratio;canvas.height=height*ratio;\n const ctx=canvas.getContext(\'2d\');ctx.scale(ratio,ratio);ctx.font=\'11px system-ui\';ctx.fillStyle=\'#adbbb4\';\n const values=points.flatMap(p=>keys.map(k=>p[k])).filter(numeric);\n if(!values.length){ctx.fillText(t(\'nodata\'),8,25);return}\n let lo=maxY?0:Math.min(...values),hi=maxY||Math.max(...values);\n if(lo===hi){lo=Math.max(0,lo-.01);hi+=.01}\n const left=65,right=Math.max(66,width-8),top=18,bottom=155;\n for(let i=0;i<4;i++){const y=top+(bottom-top)*i/3;ctx.strokeStyle=\'#ffffff15\';ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();ctx.fillText(fmt(hi-(hi-lo)*i/3),2,y+3)}\n const now=Date.now()/1000;\n const x=p=>left+(right-left)*Math.max(0,Math.min(1,(p.time-now+900)/900));\n const y=v=>bottom-(bottom-top)*(v-lo)/(hi-lo);\n keys.forEach((k,i)=>{ctx.strokeStyle=ctx.fillStyle=[\'#bce9db\',\'#f6d5e2\'][i];ctx.lineWidth=2;let previous=null;for(const p of points){if(!numeric(p[k])){previous=null;continue}if(previous&&p.time-previous.time<=120){ctx.beginPath();ctx.moveTo(x(previous),y(previous[k]));ctx.lineTo(x(p),y(p[k]));ctx.stroke()}else{ctx.beginPath();ctx.arc(x(p),y(p[k]),2,0,2*Math.PI);ctx.fill()}previous=p}});\n ctx.fillStyle=\'#adbbb4\';ctx.fillText(\'-15m\',left,177);ctx.fillText(t(\'now\'),right-30,177);\n}\nfunction renderTrainer(kind,data){\n const root=$(kind),put=(k,v)=>{const element=root.querySelector(\'[data-k="\'+k+\'"]\');if(element)element.textContent=fmt(v)};\n const rows=data.points||[],last=rows.at(-1);\n put(\'status\',data.warning?t(\'read_error\'):t(\'saved\'));put(\'step\',last?.step??data.step);put(\'round\',data.round);\n put(\'position\',data.total?(data.current+1)+\'/\'+data.total:null);\n put(\'passes\',last?last.passed+\'/\'+last.passes:(data.passes?data.streak+\'/\'+data.passes:null));\n put(\'phase\',t(last?.phase||data.phase||\'—\'));put(\'task\',t(last?.task||data.task||\'—\'));\n put(\'limit\',(last?.limits||data.limits||[]).map(fmt).join(\' / \'));\n put(\'current\',last?.value);put(\'secondary\',last?.second);put(\'best\',last?.best);\n put(\'action\',t(last?.action===\'best\'?\'best_action\':last?.action||\'—\'));put(\'lr\',last?.lr);\n let first=rows[0],seconds=last&&first?last.time-first.time:0;\n put(\'rate\',seconds>0?(last.step-first.step)/seconds:null);\n put(\'delta\',rows.length>1&&numeric(last.value)&&numeric(first.value)?last.value-first.value:null);\n put(\'policy\',t(last?.policy||\'—\'));put(\'algorithm\',last?.algorithm);put(\'pose_batch\',last?.pose_batch);put(\'attempts\',last?.attempts);put(\'diagnosis\',t(last?.diagnosis||\'—\'));\n put(\'age\',last?Math.round(Date.now()/1000-last.time):null);put(\'eval_seconds\',last?.eval_seconds);\n put(\'details\',last?.details?JSON.stringify(last.details,null,2):t(\'nodata\'));\n put(\'file\',data.file||\'\');chart(root.querySelector(\'canvas\'),rows,[\'value\',\'second\']);\n}\nfunction renderHardware(hw){\n const rows=hw?.history||[],p=hw?.latest||{};\n for(const k of metrics){const unit=k.endsWith(\'temp\')?\'°C\':\'%\';$(k).textContent=numeric(p[k])?fmt(p[k])+unit:\'—\';const values=rows.map(r=>r[k]).filter(numeric);$(k+\'_max\').textContent=values.length?t(\'max\')+\' \'+fmt(Math.max(...values))+unit:\'\'}\n $(\'sources\').textContent=[p.cpu_temp==null?t(\'cpu_missing\'):p.cpu_source,p.gpu_name,p.gpu_temp==null?t(\'gpu_missing\'):p.gpu_source].filter(Boolean).join(\' · \');\n const plot=rows.map(r=>({...r,time:r.t}));chart($(\'temperature\'),plot,[\'cpu_temp\',\'gpu_temp\']);chart($(\'usage\'),plot,[\'cpu_load\',\'gpu_load\'],100);chart($(\'memory\'),plot,[\'ram_load\',\'vram_load\'],100);\n const table=$(\'hwstats\');table.replaceChildren();\n const add=(cells,header=false)=>{const tr=document.createElement(\'tr\');for(const value of cells){const td=document.createElement(header?\'th\':\'td\');td.textContent=value;tr.append(td)}table.append(tr)};\n add([\'\',t(\'min\'),t(\'mean\'),t(\'max\'),\'Δ\'],true);\n for(const k of metrics){const values=rows.map(r=>r[k]).filter(numeric);add([t(k),...(values.length?[Math.min(...values),values.reduce((a,b)=>a+b,0)/values.length,Math.max(...values),values.length>1?values.at(-1)-values[0]:null]:[null,null,null,null]).map(fmt)])}\n}\nfunction translate(){document.documentElement.lang=language;document.querySelectorAll(\'[data-t]\').forEach(el=>el.textContent=t(el.dataset.t));$(\'language\').value=language;if(latest)render()}\nfunction render(){for(const kind of [\'real\',\'extreme\',\'unified\'])renderTrainer(kind,latest[kind]);renderHardware(latest.hardware)}\n$(\'language\').onchange=()=>{language=$(\'language\').value;localStorage.setItem(\'minto-language\',language);translate()};\nwindow.onresize=()=>{if(latest)render()};\nasync function poll(){try{const response=await fetch(\'/api?token=\'+encodeURIComponent(token),{cache:\'no-store\'});if(!response.ok)throw Error(\'HTTP \'+response.status);latest=await response.json();render();$(\'connection\').textContent=t(\'connected\')+\' · \'+new Date().toLocaleTimeString(locale())}catch(error){$(\'connection\').textContent=t(\'disconnected\')+\' · \'+error.message}finally{setTimeout(poll,5000)}}\ntranslate();poll();\n</script></html>'

def read_shared(path):
    # Windows: allow atomic replacement by the trainer while this reader is open.
    if os.name != 'nt':
        return path.read_bytes()
    import msvcrt
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = create(str(path), 0x80000000, 7, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except Exception:
        kernel.CloseHandle(handle)
        raise
    with os.fdopen(fd, 'rb') as stream:
        return stream.read()


NUM=r'[\d.eE+\-]+'
REAL=re.compile(rf'Step (\d+)\s*\| round (\d+)\s*\| model (\d+)/\d+\s+(.*?)\s*\| exam ({NUM}) <= {NUM}\s*\| pass (\d+/\d+)\s*\| validation ({NUM})')
EXTREME=re.compile(r'Step (\d+)\s*\| round (\d+)\s*\| (.*?)\s*\| limits .*?\| normal/extreme: (\[.*?\])\s*\| pass (\d+/\d+)')

def log_points(path,kind):
    if not path: return []
    raw=read_shared(path)[-262144:]
    enc='utf-16-le' if raw.startswith(b'\xff\xfe') or raw.count(b'\x00')>len(raw)//5 else 'utf-8'
    if enc=='utf-16-le' and len(raw)%2: raw=raw[1:]
    result=[]
    for m in (REAL if kind=='real' else EXTREME).finditer(raw.decode(enc,errors='replace')):
        try:
            if kind=='real':
                step,rnd,idx,name,val,passed,second=m.groups(); group=(int(rnd),int(idx)); val,second=float(val),float(second)
            else:
                step,rnd,name,scores,passed=m.groups(); group=(int(rnd),name); pairs=ast.literal_eval(scores); val=max(float(v[0]) for v in pairs); second=max(float(v[1]) for v in pairs)
            if not all(map(math.isfinite,(val,second))):continue
            result.append(dict(step=int(step),group=group,value=val,second=second,**{'pass':passed}))
        except (ValueError,TypeError,IndexError,SyntaxError):continue
    if not result:return []
    # Reset on a new run with a lower step, and chart only latest round/exercise.
    start=0
    for i in range(1,len(result)):
        if result[i]['step']<result[i-1]['step']:start=i
    return [v for v in result[start:] if v['group']==result[-1]['group']][-120:]

def summarize(s,kind):
    if kind=='unified':
        cfg=s['config']
        return dict(step=int(s['step']),round=1,current=0,total=1,streak=int(s.get('streak',0)),
                    passes=3,limits=[float(cfg['mastery'])],eval_every=int(cfg['eval_every']),
                    phase=s.get('phase'),task='joint_training')
    if kind=='real':
        cfg=s['curriculum_settings']; return dict(step=int(s['steps']),round=int(s['round_number']),current=int(s['current']),total=len(cfg['groups']),streak=int(s['streak']),passes=int(cfg['passes']),limits=[float(s['limit'])],eval_every=int(cfg['eval_every']))
    if s.get('schema') not in (4,5):raise ValueError('Fullbody requires schema 4')
    idx=int(s['stage']); tasks=s['tasks']; return dict(step=int(s['step']),round=int(s['round_number']),current=idx,total=len(tasks),task=str(tasks[idx][0]),batch=int(s['batch']),device=str(s['device']),limits=list(map(float,s['mastery_limits'])))

def structured_points(path, current):
    if not path.is_file():
        return []
    raw = read_shared(path)[-4_000_000:]
    result = []
    for line in raw.splitlines():
        try:
            row = json.loads(line)
            if (row.get('time', 0) >= time.time() - 900
                    and row.get('round') == current.get('round')
                    and row.get('current') == current.get('current')):
                result.append(row)
        except (ValueError, TypeError):
            continue
    if not result:
        return []
    start = 0
    for i in range(1, len(result)):
        if result[i]['step'] < result[i-1]['step']:
            start = i
    return result[start:]


def make_reader(path,log,kind,torch):
    cache={}; last_stamp=None; lock=threading.Lock()
    def status():
        nonlocal cache,last_stamp
        with lock:
            try:
                st=path.stat(); stamp=(st.st_mtime_ns,st.st_size)
                if stamp!=last_stamp:
                    s=torch.load(io.BytesIO(read_shared(path)),map_location='cpu',weights_only=True)
                    cache=dict(summarize(s,kind),file=str(path),saved=st.st_mtime,phase=s.get('learning_phase',s.get('phase')));last_stamp=stamp
                data=dict(cache)
            except Exception as e:
                data=dict(cache);data['warning']='Chưa đọc được checkpoint: '+type(e).__name__
            data.setdefault('file',str(path))
            if 'saved' in data:data['age']=max(0,time.time()-data['saved'])
            try:
                data['points'] = structured_points(path.with_suffix('.metrics.jsonl'), data)
                if not data['points'] and log:
                    old = log_points(log, kind)
                    data['points'] = [dict(v, time=time.time(), passed=v['pass'].split('/')[0],
                                           passes=v['pass'].split('/')[1], action='trial') for v in old]

            except Exception as e:data['points']=[];data['log_warning']='Không đọc được log: '+type(e).__name__
            return data
    return status

# Hardware telemetry is sampled independently of browser requests.
from collections import deque
import csv, shutil, subprocess

SAMPLE_SECONDS = 5
WINDOW_SECONDS = 900

def number(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None

def command(argv, timeout=4):
    return subprocess.check_output(argv, timeout=timeout, stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)).decode('utf-8-sig', errors='replace').strip()

def cpu_temp(psutil):
    if os.name == 'nt':
        try:
            from urllib.request import build_opener, ProxyHandler
            opener = build_opener(ProxyHandler({}))
            with opener.open('http://127.0.0.1:8085/data.json', timeout=2) as response:
                if response.geturl() != 'http://127.0.0.1:8085/data.json':
                    raise ValueError('Unexpected sensor redirect')
                document = json.loads(response.read(4_000_000))
            stack = [(document, False)]
            temperatures = []
            while stack:
                node, under_cpu = stack.pop()
                if not isinstance(node, dict):
                    continue
                under_cpu = under_cpu or str(node.get('ImageURL', '')).endswith('/cpu.png')
                ident = str(node.get('SensorId', '')).lower()
                name = str(node.get('Text', ''))
                if (under_cpu or re.search(r'/(intelcpu|amdcpu)/', ident)) and re.search(
                        r'CPU Package|Core Max|CPU Core #|Tctl|Tdie', name, re.I):
                    match = re.fullmatch(r'\s*([+-]?\d+(?:[.,]\d+)?)\s*°\s*C\s*',
                                         str(node.get('Value', '')), re.I)
                    if match:
                        value = float(match[1].replace(',', '.'))
                        if 0 < value < 130:
                            temperatures.append(value)
                stack.extend((child, under_cpu) for child in node.get('Children', []))
            if temperatures:
                return max(temperatures), 'Libre Hardware Monitor · localhost'
        except (OSError, ValueError, TypeError):
            pass

    if os.name == 'nt':
        # Read actual Intel CPU package sensor, not motherboard ACPI temperature.
        script = r"""$ErrorActionPreference='Stop'; Get-WmiObject -Namespace 'root\LibreHardwareMonitor' -Class Sensor | Where-Object { $_.SensorType -eq 'Temperature' -and $_.Identifier -match '^/(intelcpu|amdcpu|cpu)/' -and $_.Name -in @('CPU Package','Core (Tctl/Tdie)','Core (Tdie)') } | Select-Object Name,Value | ConvertTo-Json -Compress"""
        try:
            raw = command(['powershell.exe','-NoProfile','-NonInteractive','-Command',script], timeout=4)
            items = json.loads(raw) if raw else []
            if isinstance(items, dict): items = [items]
            for name in ('CPU Package','Core (Tctl/Tdie)','Core (Tdie)'):
                vals = [number(x.get('Value')) for x in items if x.get('Name') == name]
                vals = [v for v in vals if v is not None and 0 < v < 150]
                if vals: return max(vals), 'LibreHardwareMonitor · '+name
        except (OSError, subprocess.SubprocessError, ValueError, TypeError): pass
        return None, 'Chưa có cảm biến CPU từ LibreHardwareMonitor (WMI)'
    try:
        sensors = psutil.sensors_temperatures() if psutil else {}
        for key in ('coretemp','k10temp','zenpower'):
            entries = sensors.get(key, [])
            for item in entries:
                if any(x in item.label.lower() for x in ('package','tctl','tdie')):
                    return number(item.current), key+' · '+item.label
    except (OSError, AttributeError): pass
    return None, 'Không tìm thấy cảm biến CPU package'

def gpu_sample(index):
    exe = shutil.which('nvidia-smi')
    if not exe and os.name == 'nt':
        for loc in (Path(os.environ.get('SystemRoot',r'C:\Windows'))/'System32/nvidia-smi.exe', Path(os.environ.get('ProgramFiles',r'C:\Program Files'))/'NVIDIA Corporation/NVSMI/nvidia-smi.exe'):
            if loc.is_file(): exe=str(loc); break
    if not exe: return {}, 'Không tìm thấy nvidia-smi'
    try:
        raw = command([exe,'-i',str(index),'--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'])
        row = next(csv.reader(raw.splitlines(), skipinitialspace=True))
        if len(row)!=5: raise ValueError('Unexpected GPU response')
        name,temp,load,used,total = row
        used,total=number(used),number(total)
        return dict(gpu_name=name,gpu_temp=number(temp),gpu_load=number(load),vram_used=used,vram_total=total,vram_load=100*used/total if used is not None and total else None), 'NVIDIA · nvidia-smi'
    except (OSError, subprocess.SubprocessError, ValueError, StopIteration): return {}, 'Không đọc được GPU NVIDIA'

METRICS=('cpu_temp','gpu_temp','cpu_load','gpu_load','ram_load','vram_load')

def make_hardware_sampler(index):
    try: import psutil
    except ImportError: psutil=None
    history=deque(maxlen=240); lock=threading.Lock(); stop=threading.Event()
    def loop():
        if psutil: psutil.cpu_percent(interval=None)
        while not stop.is_set():
            start=time.monotonic()
            point={k:None for k in METRICS}
            try:
                if psutil:
                    point['cpu_load']=number(psutil.cpu_percent(interval=None)) if history else None
                    mem=psutil.virtual_memory(); point.update(ram_load=number(mem.percent),ram_used=(mem.total-mem.available)/2**30,ram_total=mem.total/2**30)
                point['cpu_temp'],point['cpu_source']=cpu_temp(psutil)
                gpu,source=gpu_sample(index);point.update(gpu);point['gpu_source']=source
                if not psutil:point['usage_warning']='Thiếu psutil: chưa đọc được tải CPU và RAM.'
            except Exception as e:point['usage_warning']='Lỗi đọc cảm biến: '+type(e).__name__
            point['t']=time.time()
            with lock:
                history.append(point)
                while history and history[0]['t']<point['t']-WINDOW_SECONDS:history.popleft()
            stop.wait(max(0,SAMPLE_SECONDS-(time.monotonic()-start)))
    def status():
        now=time.time()
        with lock: rows=[dict(p) for p in history if p['t']>=now-WINDOW_SECONDS]
        return dict(now=now,window=WINDOW_SECONDS,interval=SAMPLE_SECONDS,latest=rows[-1] if rows else {},history=rows)
    threading.Thread(target=loop,daemon=True,name='hardware-sampler').start()
    return status,stop

def main():
    p=argparse.ArgumentParser(description='MintoAI dual read-only LAN dashboard')
    p.add_argument('--unified-checkpoint',type=Path,default=Path('checkpoints/unified_rig_v06.pt'))
    p.add_argument('--checkpoint','--real-checkpoint',dest='real_checkpoint',type=Path,default=Path('checkpoints/real_rig_curriculum.pt'))
    p.add_argument('--extreme-checkpoint',type=Path,default=Path('checkpoints/fullbody_extreme_v04.pt'))
    p.add_argument('--log','--real-log',dest='real_log',type=Path)
    p.add_argument('--extreme-log',type=Path)
    p.add_argument('--port',type=int,default=8765)
    p.add_argument('--host',default='0.0.0.0')
    p.add_argument('--gpu-index',default='0',help='NVIDIA GPU index or UUID')
    args=p.parse_args()
    import torch
    token=secrets.token_urlsafe(24)
    real=make_reader(args.real_checkpoint,args.real_log,'real',torch)
    extreme=make_reader(args.extreme_checkpoint,args.extreme_log,'extreme',torch)
    unified=make_reader(args.unified_checkpoint,None,'unified',torch)
    hardware,stop_hardware=make_hardware_sampler(args.gpu_index)
    def status():return dict(real=real(),extreme=extreme(),unified=unified(),hardware=hardware())
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*_): pass
        def do_GET(self):
            u=urlsplit(self.path)
            if not secrets.compare_digest(parse_qs(u.query).get('token',[''])[0],token):
                self.send_error(403);return
            if u.path=='/': body=PAGE.encode(); kind='text/html; charset=utf-8'
            elif u.path=='/api': body=json.dumps(status(),ensure_ascii=False).encode();kind='application/json; charset=utf-8'
            else: self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.send_header('Referrer-Policy','no-referrer');self.end_headers();self.wfile.write(body)
    server=ThreadingHTTPServer((args.host,args.port),Handler)
    ips={'127.0.0.1'}
    if args.host=='0.0.0.0':
        try: ips.update(socket.gethostbyname_ex(socket.gethostname())[2])
        except OSError: pass
        try:
            with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as s:
                s.connect(('192.0.2.1',9));ips.add(s.getsockname()[0])
        except OSError: pass
    else: ips={args.host}
    print('MintoAI read-only dashboard. Private network only. Ctrl+C closes dashboard, not trainer.',flush=True)
    for ip in sorted(ips):print(f'http://{ip}:{args.port}/?token={token}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        stop_hardware.set()
        server.server_close()

if __name__=='__main__':main()