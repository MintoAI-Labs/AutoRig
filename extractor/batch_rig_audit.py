"""MintoAI: batch-audit raw-0.1 JSON without saving training tensors."""
import argparse
import contextlib
import io
import json
from pathlib import Path
from process_rig_dataset import process

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'dataset')
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    if not dataset.is_dir():
        parser.error(f'Dataset folder does not exist: {dataset}')
    report_path = dataset / 'audit_report.json'
    results = []
    skipped = 0
    for path in sorted(dataset.rglob('*.json')):
        if path == report_path:
            continue
        relative = str(path.relative_to(dataset))
        record = {'file': relative}
        try:
            data = json.loads(path.read_text(encoding='utf-8-sig'))
            if not isinstance(data, dict) or data.get('schema_version') != 'raw-0.1':
                # A malformed raw export must not be silently skipped.
                if isinstance(data, dict) and ('meshes' in data or 'samples' in data):
                    raise ValueError('Unknown or missing raw dataset schema')
                skipped += 1
                continue
            record['source'] = data.get('source')
            record['parameter'] = data.get('parameter', {}).get('id')
            del data
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                process(path, dataset / 'processed', None)
            record['status'] = 'PASS'
            record['details'] = captured.getvalue().splitlines()
        except Exception as exc:
            record['status'] = 'FAIL'
            record['error'] = str(exc)
        results.append(record)
        print(f"[{record['status']}] {relative}")
        if record['status'] == 'FAIL':
            print('  ', record['error'])
    passed = sum(r['status'] == 'PASS' for r in results)
    failed = len(results) - passed
    report = {'mode': 'audit_only', 'files_checked': len(results), 'passed': passed,
              'failed': failed, 'skipped_non_raw': skipped, 'results': results}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nMintoAI Batch Audit V0.1')
    print('Files checked:', len(results))
    print('PASS:', passed)
    print('FAIL:', failed)
    print('Skipped non-raw JSON:', skipped)
    print('Report:', report_path)
    print('Audit only: no training tensors saved.')
    if not results:
        print('No raw datasets found.')
    return 1 if failed or not results else 0


if __name__ == '__main__':
    raise SystemExit(main())
