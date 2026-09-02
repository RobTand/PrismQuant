#!/usr/bin/env python3
"""Assert the built artifacts carry prismaquant's runtime data files.

prismaquant is pure Python, so the packaging risk is not compiled sources —
it is the JSON and tensor tables that the code *reads at runtime*: model,
serving and lane specs; exact IQ/CB lookup tables; and the
integration contract. Those are declared in
`[tool.setuptools.package-data]`, which is easy to break silently: the package
still imports, and the failure only appears when something asks for a profile
— i.e. on a user's first real run, not in any import smoke test.

`run-pipeline.sh` is checked for the same reason:
they are the shipped orchestration surface.

Usage: check_dist.py <dist-dir> <source-root>
"""
from __future__ import annotations

import glob
import os
import sys
import tarfile
import zipfile


def _expected(source_root: str) -> set[str]:
    """Every runtime data file present in the source tree, as package paths."""
    out: set[str] = set()
    for pattern in (
        "prismaquant/model_profiles/specs/*.json",
        "prismaquant/serving_profile_specs/*.json",
        "prismaquant/lane_specs/*.json",
        "prismaquant/data/*.pt",
        "prismaquant/run-pipeline.sh",
    ):
        for path in glob.glob(os.path.join(source_root, pattern)):
            out.add(os.path.relpath(path, source_root))
    if not out:
        sys.exit(f"no runtime data files found under {source_root!r} — the "
                 "check itself is misconfigured, refusing to pass vacuously")
    return out


def main() -> None:
    dist_dir, source_root = sys.argv[1], sys.argv[2]
    expected = _expected(source_root)
    print(f"expecting {len(expected)} runtime data files in each artifact")

    wheels = glob.glob(os.path.join(dist_dir, "*.whl"))
    sdists = glob.glob(os.path.join(dist_dir, "*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        sys.exit(f"expected exactly one wheel and one sdist, got "
                 f"{wheels} / {sdists}")

    with zipfile.ZipFile(wheels[0]) as zf:
        names = set(zf.namelist())
    missing = sorted(n for n in expected if n not in names)
    if missing:
        sys.exit(f"wheel {os.path.basename(wheels[0])} is missing "
                 f"{len(missing)} runtime data file(s): {missing}. Fix "
                 "[tool.setuptools.package-data] in pyproject.toml — an "
                 "installed prismaquant cannot resolve a profile without "
                 "these, and it fails at first use, not at import.")
    print(f"wheel OK: all {len(expected)} runtime data files present")

    # The sdist prefixes everything with `<name>-<version>/`.
    with tarfile.open(sdists[0]) as tf:
        members = tf.getnames()
    prefix = os.path.commonprefix(members).split("/")[0]
    sd = {m[len(prefix) + 1:] for m in members if m.startswith(prefix + "/")}
    missing = sorted(n for n in expected if n not in sd)
    if missing:
        sys.exit(f"sdist {os.path.basename(sdists[0])} is missing "
                 f"{len(missing)} runtime data file(s): {missing}")
    print(f"sdist OK: all {len(expected)} runtime data files present")


if __name__ == "__main__":
    main()
