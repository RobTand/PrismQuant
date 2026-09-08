"""PB-only paired allocator profiles against an exact Git baseline/input receipt."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pstats
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.parse
import urllib.request


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit_outputs(output, inv, root):
    """Compare artifact values, retaining but excluding measured solver times."""
    results = [json.loads((output / arm / 'result.json').read_text())
               for arm in ('before', 'after')]
    assert all(r['returncode'] == 0 and not r['netdata_errors'] for r in results)
    original = output / 'comparison.json'
    if original.exists():
        for rel, digest in json.loads(original.read_text())['artifacts'].items():
            assert sha(output / rel) == digest, rel
    before = output / 'before' / 'frontier'
    after = output / 'after' / 'frontier'
    files = sorted(path.relative_to(before) for path in before.rglob('*') if path.is_file())
    assert files == sorted(path.relative_to(after) for path in after.rglob('*') if path.is_file())
    differences = []
    timing_keys = []
    for rel in files:
        a = (before / rel).read_text().replace(str(before), '<FRONTIER>')
        b = (after / rel).read_text().replace(str(after), '<FRONTIER>')
        if str(rel) == 'terminal-bf16-control.json':
            a, b = json.loads(a), json.loads(b)
            left = a['__prismaquant__']['solve_diagnostics']
            right = b['__prismaquant__']['solve_diagnostics']
            assert left.keys() == right.keys()
            for target in left:
                for side in (left, right):
                    seconds = side[target].pop('solver_seconds')
                    assert type(seconds) is float and math.isfinite(seconds) and seconds >= 0
                timing_keys.append(target)
        if a != b:
            differences.append(str(rel))
    assert sha(root / 'joint-allocation-cost.pkl') == inv['input_sha256']
    report = {'arms': results, 'output_files': len(files), 'differences': differences,
              'excluded_solver_seconds_targets': sorted(timing_keys),
              'artifacts': {str(p.relative_to(output)): sha(p) for p in output.rglob('*')
                            if p.is_file() and p.name != 'comparison-audited.json'}}
    (output / 'comparison-audited.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'output_files': len(files), 'differences': differences,
                      'excluded_solver_seconds': len(timing_keys),
                      'audit_sha256': sha(output / 'comparison-audited.json')}), flush=True)
    assert not differences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--invocation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit-existing', action='store_true')
    args = parser.parse_args()
    inv = json.loads(args.invocation.read_text())
    root = args.invocation.parent
    assert sha(root / 'joint-allocation-cost.pkl') == inv['input_sha256']
    if args.audit_existing:
        audit_outputs(args.output, inv, root)
        return
    args.output.mkdir(exist_ok=False, parents=True)
    current = Path.cwd()
    results = []
    with tempfile.TemporaryDirectory(prefix='joint-validation-pair-') as tmp:
        tmp = Path(tmp)
        baseline = tmp / 'baseline'
        baseline.mkdir()
        archive = tmp / 'baseline.tar'
        with archive.open('wb') as stream:
            subprocess.run(['git', 'archive', inv['source_commit']], stdout=stream, check=True)
        subprocess.run(['tar', '-xf', str(archive), '-C', str(baseline)], check=True)
        for arm, cwd in [('before', baseline), ('after', current)]:
            out = args.output / arm
            out.mkdir()
            profile = tmp / (arm + '.pstats')
            argv = [s.replace(str(root / 'frontier-01'), str(out / 'frontier')) for s in inv['command']]
            argv = argv[:1] + ['-m', 'cProfile', '-o', str(profile)] + argv[1:]
            (out / 'invocation.json').write_text(json.dumps({'argv': argv, 'env': inv['env'], 'cwd': str(cwd)}, indent=2))
            stopped = threading.Event()
            errors = []
            query = urllib.parse.urlencode({'format': 'json', 'filter': 'system.cpu system.ram system.io system.load nvidia_smi.*_power_draw'})
            def sample(host, address):
                with (out / (host + '-netdata.jsonl')).open('w') as stream:
                    while not stopped.is_set():
                        record = {'host': host, 'time': time.time()}
                        try:
                            with urllib.request.urlopen('http://' + address + ':19999/api/v1/allmetrics?' + query, timeout=3) as response:
                                record['metrics'] = json.load(response)
                            assert 'system.cpu' in record['metrics']
                        except Exception as exc:
                            record['error'] = repr(exc)
                            errors.append(record)
                        stream.write(json.dumps(record) + '\n')
                        stream.flush()
                        stopped.wait(5)
            threads = [threading.Thread(target=sample, args=host) for host in [
                ('dl380g10', '127.0.0.1'), ('sparky', '192.168.1.180'), ('sparklina', '192.168.1.110')]]
            start = time.time()
            for thread in threads:
                thread.start()
            try:
                with (out / 'allocator.log').open('w') as stream:
                    completed = subprocess.run(argv, cwd=cwd, env={**os.environ, **inv['env']}, stdout=stream, stderr=subprocess.STDOUT)
                elapsed = time.time() - start
            finally:
                stopped.set()
                for thread in threads:
                    thread.join()
            shutil.copy2(profile, out / 'allocator.pstats')
            with (out / 'profile-top.txt').open('w') as stream:
                pstats.Stats(str(profile), stream=stream).strip_dirs().sort_stats('cumulative').print_stats(45)
                pstats.Stats(str(profile), stream=stream).strip_dirs().sort_stats('tottime').print_stats(30)
            result = {'arm': arm, 'elapsed_s': elapsed, 'returncode': completed.returncode, 'started_unix': start, 'netdata_errors': errors}
            results.append(result)
            (out / 'result.json').write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
            assert completed.returncode == 0
        audit_outputs(args.output, inv, root)



if __name__ == '__main__':
    main()
