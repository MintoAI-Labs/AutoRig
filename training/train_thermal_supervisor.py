"""Windows launcher. Keep Libre Hardware Monitor running with WMI sensors available."""
import argparse
import base64
import json
import math
import re
from urllib.request import ProxyHandler, build_opener
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

CPU_COMMAND = """
$ErrorActionPreference = 'Stop'
$s = Get-CimInstance -Namespace root/LibreHardwareMonitor -ClassName Sensor
$s | Where-Object {
  $_.SensorType -eq 'Temperature' -and
  $_.Identifier -match '/(intelcpu|amdcpu)/' -and
  $_.Name -match 'CPU Package|Core Max|CPU Core #|Tctl|Tdie'
} | ForEach-Object { $_.Value.ToString([cultureinfo]::InvariantCulture) }
"""


class WorkerControl:
    def __init__(self):
        ref = os.environ.get('MINTO_TRAIN_CONTROL')
        if not ref:
            raise SystemExit('Start this trainer via train_thermal_supervisor.py')
        self.root = Path(ref)

    def state(self):
        if (self.root / 'stop').exists():
            return 'stop'
        try:
            state = json.loads((self.root / 'state.json').read_text())
            return state['status'] if time.time() - state['time'] < 20 else 'stop'
        except (OSError, ValueError, KeyError):
            return 'stop'

    def stopped(self):
        return self.state() == 'stop'

    def ready(self, stopped, release_cache=None):
        state = self.state()
        if state == 'pause' and release_cache:
            release_cache()
        while state == 'pause' and not stopped():
            time.sleep(.25)
            state = self.state()
        return state == 'run' and not stopped()


def web_cpu(document):
    """Read only CPU temperature sensors in the Libre Hardware Monitor tree."""
    values = []
    pending = [(document, False)]
    while pending:
        node, under_cpu = pending.pop()
        if not isinstance(node, dict):
            continue
        name = str(node.get('Text', ''))
        ident = str(node.get('SensorId', '')).lower()
        under_cpu = under_cpu or str(node.get('ImageURL', '')).endswith('/cpu.png')
        is_cpu = bool(re.search(r'/(intelcpu|amdcpu)/', ident)) or under_cpu
        if is_cpu and re.search(r'CPU Package|Core Max|CPU Core #|Tctl|Tdie', name, re.I):
            raw = node.get('Value', '')
            match = re.fullmatch(r'\s*([+-]?\d+(?:[.,]\d+)?)\s*°\s*C\s*', str(raw), re.I)
            if match:
                values.append(float(match[1].replace(',', '.')))
        pending.extend((child, under_cpu) for child in node.get('Children', []))
    return valid_cpu(values)


def valid_cpu(values):
    if not values or not all(math.isfinite(v) and 0 < v < 130 for v in values):
        raise ValueError('No valid CPU temperature in Celsius')
    return max(values)


def read_cpu(powershell, port, options):
    try:
        result = subprocess.run(
            [powershell, '-NoProfile', '-NonInteractive', '-EncodedCommand',
             base64.b64encode(CPU_COMMAND.encode('utf-16-le')).decode('ascii')],
            capture_output=True, **options)
        if result.returncode:
            raise ValueError(result.stderr.strip() or f'WMI exit {result.returncode}')
        return valid_cpu(list(map(float, result.stdout.split())))
    except (ValueError, OSError, subprocess.SubprocessError) as wmi_error:
        try:
            opener = build_opener(ProxyHandler({}))
            with opener.open(f'http://127.0.0.1:{port}/data.json', timeout=3) as response:
                if response.geturl() != f'http://127.0.0.1:{port}/data.json':
                    raise ValueError('Unexpected telemetry redirect')
                return web_cpu(json.loads(response.read(4_000_000)))
        except (OSError, ValueError) as web_error:
            raise ValueError(f'CPU WMI: {wmi_error} | Libre Hardware Monitor web: {web_error}') from web_error


def read_sensors(powershell, smi, gpu, port):
    options = dict(text=True, timeout=4, creationflags=subprocess.CREATE_NO_WINDOW)
    package = read_cpu(powershell, port, options)
    card = subprocess.check_output(
        [smi, '-i', gpu, '--query-gpu=temperature.gpu,memory.free',
         '--format=csv,noheader,nounits'], **options)
    temp, free = map(int, card.strip().split(','))
    if not (0 < temp < 130 and free >= 0):
        raise ValueError('Invalid GPU sensor readings')
    return package, temp, free


def decision(cpu, gpu, free, paused, stable, gpu_job, now, hot_since, pause_temps):
    hot = {'cpu': cpu >= 85, 'gpu': gpu >= 80}
    duration = {}
    for sensor, is_hot in hot.items():
        if is_hot:
            duration[sensor] = now - hot_since.setdefault(sensor, now)
        else:
            hot_since.pop(sensor, None)
    if paused:
        if pause_temps is not None:
            before_cpu, before_gpu = pause_temps
            if ((hot['cpu'] and cpu >= before_cpu + 2)
                    or (hot['gpu'] and gpu >= before_gpu + 2)):
                return 'stop', 0, pause_temps, 'temperature rose >=2 C after pause'
        cool = cpu <= 75 and gpu <= 72 and (not gpu_job or free >= 768)
        stable = stable + 1 if cool else 0
        if stable >= 3:
            return 'run', stable, None, ''
        return 'pause', stable, pause_temps, ''
    critical = False
    for sensor, is_critical in (('critical_cpu', cpu >= 95), ('critical_gpu', gpu >= 87)):
        if is_critical:
            critical |= now - hot_since.setdefault(sensor, now) >= 5
        else:
            hot_since.pop(sensor, None)
    sustained = max(duration.values(), default=0)
    if critical or sustained >= 30 or (gpu_job and free < 384):
        reason = 'critical temperature' if critical else ('sustained heat' if sustained >= 30 else 'low VRAM')
        return 'pause', 0, (cpu, gpu), reason
    detail = f'heat timer {sustained:.1f}/30 s' if duration else ''
    return 'run', 0, None, detail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lhm-port', type=int, default=8085)
    parser.add_argument('--extreme-checkpoint', type=Path)
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--job', choices=['both', 'real', 'extreme'], default='both')
    parser.add_argument('--real-device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--gpu', default='0', help='Physical GPU index for nvidia-smi')
    parser.add_argument('--confirm-authorized', action='store_true')
    args = parser.parse_args()
    if os.name != 'nt':
        parser.error('This launcher requires Windows')
    root = Path(__file__).resolve().parent.parent
    checkpoints = root / 'checkpoints'
    jobs = []
    if args.job in ('both', 'real'):
        if not args.confirm_authorized or not args.snapshot or not args.snapshot.is_dir():
            parser.error('Real trainer requires --snapshot and --confirm-authorized')
        jobs.append(('train_real_rig_guarded.py', args.real_device,
                     ['--data', str(args.snapshot.resolve()), '--confirm-authorized'],
                     checkpoints / f'real_rig_guarded_{args.real_device}.pt'))
    if args.job in ('both', 'extreme'):
        jobs.append(('train_fullbody_guarded.py', 'cuda', ['--batch', '128'],
                     (args.extreme_checkpoint or checkpoints / 'fullbody_extreme_v04_guarded.pt').resolve()))
    powershell = shutil.which('powershell.exe')
    smi = shutil.which('nvidia-smi')
    if not powershell or not smi:
        parser.error('PowerShell and nvidia-smi must be available in PATH')
    checkpoints.mkdir(exist_ok=True)
    gpu_job = any(device == 'cuda' for _, device, _, _ in jobs)
    children = []
    stop = False

    def request_stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    with tempfile.TemporaryDirectory(prefix='minto-train-') as directory:
        control = Path(directory)

        def publish(status):
            temp = control / 'state.tmp'
            temp.write_text(json.dumps(dict(status=status, time=time.time())))
            temp.replace(control / 'state.json')

        status, stable = 'pause', 0
        hot_since = {}
        pause_temps = None
        try:
            # Start workers only after both sensors have been read and cleared.
            while not stop:
                cpu, gpu, free = read_sensors(powershell, smi, args.gpu, args.lhm_port)
                status, stable, pause_temps, detail = decision(
                    cpu, gpu, free, status == 'pause', stable, gpu_job,
                    time.monotonic(), hot_since, pause_temps)
                suffix = f' | {detail}' if detail else ''
                print(f'{status.upper()} | CPU {cpu:g} C | GPU {gpu} C | VRAM {free} MiB free{suffix}', flush=True)
                publish(status)
                if status == 'stop':
                    break
                if status == 'run' and not children and not stop:
                    for filename, device, extra, output in jobs:
                        command = [sys.executable, str(Path(__file__).with_name(filename)),
                                   '--device', device, '--output', str(output), *extra]
                        if output.is_file():
                            command.append('--resume')
                        env = dict(os.environ, MINTO_TRAIN_CONTROL=directory,
                                   CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=args.gpu)
                        children.append(subprocess.Popen(command, cwd=root, env=env,
                                                         creationflags=subprocess.CREATE_NEW_PROCESS_GROUP))
                if any(child.poll() is not None for child in children):
                    print('A trainer exited. Saving and stopping the remaining trainer.', flush=True)
                    break
                time.sleep(2)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            print(f'Telemetry or launcher failure: {exc}', flush=True)
        finally:
            (control / 'stop').touch()
            if children:
                print('STOP | waiting up to 30 seconds for checkpoint saves', flush=True)
                deadline = time.monotonic() + 30
                for child in children:
                    try:
                        child.wait(timeout=max(0, deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        print('Trainer did not stop in time; retaining last completed checkpoint.', flush=True)
                        child.kill()
                        child.wait()


if __name__ == '__main__':
    main()