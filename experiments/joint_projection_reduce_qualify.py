"""Run the CUDA numerical/stream/layout gate inside one admitted action."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from prismaquant.kernels.joint_projection_reduce import build_identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tooling = '/tmp/pq-fused-test-tools'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--disable-pip-version-check',
                    '--target', tooling, 'pytest==8.4.2'], check=True)
    sys.path.insert(0, tooling)
    import pytest
    identity = build_identity()
    binary = Path(identity['binary_path'])
    shutil.copy2(binary, args.output / binary.name)
    shutil.copy2(binary.parent / 'build.ninja', args.output / 'build.ninja')
    code = pytest.main(['-q', '-p', 'no:cacheprovider', 'tests/test_joint_projection_reduce.py',
                        '--junitxml', str(args.output / 'pytest.xml')])
    (args.output / 'receipt.json').write_text(json.dumps({'candidate': identity, 'test_exit_code': code}, indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
