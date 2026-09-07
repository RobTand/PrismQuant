"""PB CPU action: compile candidate against the pinned container and seal inputs."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import torch

from prismaquant.kernels.joint_projection_reduce import build_identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    identity = build_identity()
    identity['nvcc_version'] = subprocess.check_output(['/usr/local/cuda/bin/nvcc', '--version'], text=True)
    identity['cxx_version'] = subprocess.check_output(['c++', '--version'], text=True)
    binary = Path(identity['binary_path'])
    shutil.copy2(binary, args.output / binary.name)
    shutil.copy2(binary.parent / 'build.ninja', args.output / 'build.ninja')
    include = Path(torch.__file__).parent / 'include'
    for relative in identity['headers']:
        path = args.output / 'headers' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(include / relative, path)
    (args.output / 'identity.json').write_text(json.dumps(identity, indent=2))
    print(json.dumps(identity), flush=True)


if __name__ == '__main__':
    main()
