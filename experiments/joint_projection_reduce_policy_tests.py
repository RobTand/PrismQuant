"""Run CPU contract/schema regressions inside an admitted container."""
import argparse
import json
from pathlib import Path
import py_compile
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tooling = '/tmp/pq-fused-policy-tools'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--disable-pip-version-check',
                    '--target', tooling, 'pytest==8.4.2', 'pytest-xdist==3.8.0'], check=True)
    sys.path.insert(0, tooling)
    import os
    os.environ['PYTHONPATH'] = tooling + ':' + os.environ['PYTHONPATH']
    import pytest
    files = ['tests/test_joint_projection_backend.py', 'tests/test_joint_aura_projection.py',
             'tests/test_joint_aura_streamed.py', 'tests/test_joint_aura_packed.py',
             'tests/test_joint_aura_microbatch.py', 'tests/test_joint_aura_lease_lifetime.py',
             'tests/test_joint_aura_one_squaring.py', 'tests/test_joint_aura_assignment_diagnostics.py',
             'tests/test_joint_aura_allocator_currency.py', 'tests/test_tessera_joint_aura.py',
             'tests/test_native_operator_panel.py', 'tests/test_native_moe_panel.py',
             'tests/test_allocator_measured_runtime_cli.py', 'tests/test_architecture_doc.py',
             'tests/test_docs_staleness.py']
    for name in ['prismaquant/joint_projection_backend.py', 'prismaquant/joint_aura.py',
                 'prismaquant/aura_cost.py', 'prismaquant/tessera_joint_aura.py',
                 'experiments/joint_projection_reduce_production.py']:
        py_compile.compile(name, cfile=str(args.output / (Path(name).name + 'c')), doraise=True)
    code = pytest.main(['-q', '-n', '4', '-p', 'no:cacheprovider', *files,
                        '--junitxml', str(args.output / 'pytest.xml')])
    (args.output / 'receipt.json').write_text(json.dumps({'test_exit_code': code, 'files': files,
                                                       'mode': 'CPU-only; four pytest workers, native threads one'}, indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
