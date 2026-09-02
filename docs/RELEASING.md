# Releasing prismaquant

The pipeline is `.github/workflows/release.yml`. It fires on `v*` tag pushes
only, so merging it — or any later commit to `main` — can never trigger an
upload. A release is:

```bash
# 1. bump the version in pyproject.toml and CHANGELOG.md, merge it to main
# 2. tag and push the tag
git tag -a v0.7.0 -m "prismaquant 0.7.0"
git push origin v0.7.0
```

The workflow then builds, gates, publishes to PyPI via OIDC, and creates the
GitHub Release with the sdist + wheel attached. A pre-release version
(`v0.2.0rc1`) routes to TestPyPI instead.

## One-time PyPI setup — read the trap first

**The trap, which cost a release on the sibling `gridbook` project: a PyPI
*pending* publisher only becomes a real publisher when the project is created
*through that publisher*.** On gridbook, a pending publisher was configured and
then an API-token upload was used as a dry run. That token upload created the
project, orphaning the pending publisher, and the tagged pipeline run died with
`invalid-publisher: valid token, but no corresponding publisher` — despite
perfectly correct OIDC claims. Build and verify had both passed; only the
irreversible step failed.

So: **if trusted publishing is intended, the first upload must come from the
workflow. Never token-upload a rehearsal first.**

As of 2026-08-02, the `prismaquant` project exists and releases 0.2.0 through
0.6.0 were published successfully by this workflow through trusted OIDC. The
active configuration is therefore **case 2** below. Case 1 is retained only for
maintainers creating a new sibling project.

1. **If the `prismaquant` project does NOT yet exist on PyPI** — use the
   *account-level* "pending publisher" form
   (<https://pypi.org/manage/account/publishing/>) and let the first tagged
   workflow run create the project:

   | field | value |
   |---|---|
   | PyPI project name | `prismaquant` |
   | Owner | `RobTand` |
   | Repository name | `prismaquant` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

2. **For PrismaQuant today: the project already exists.** If the publisher must
   be recreated,
   the account-level pending form will NOT work. Use the *project-scoped*
   publisher form at `https://pypi.org/manage/project/prismaquant/settings/publishing/`
   with the same four values. That is the form that works once a project exists;
   on gridbook the first re-check failed because the account-level pending form
   had been used again.

3. **Create the GitHub environments** `pypi` and `testpypi`
   (Settings → Environments). Adding a required reviewer to `pypi` puts a human
   approval gate in front of the only irreversible step.

4. **Optional, for rehearsals:** repeat step 1 on TestPyPI
   (<https://test.pypi.org/manage/account/publishing/>) with environment
   `testpypi`.

No API token and no repository secret is ever needed. If a token was used at any
point, treat it as compromised if it was ever pasted into a chat, and revoke it —
nothing in this pipeline needs one.

## What the pipeline gates before it publishes

- `twine check --strict` on both artifacts.
- **Runtime data files are packaged** (`.github/scripts/check_dist.py`).
  prismaquant is pure Python, so the packaging risk is not compiled sources: it
  is the model/serving/lane JSON, canonical IQ and CB tensor tables, the
  immutable Tessera serving pin and its resolver, and `run-pipeline.sh`. (The
  Gridbook pin and resolver were in this list until 2026-09-02, when that lane
  was retired — `archive/gridbook_lane_2026-09-02/`.) A package-data regression
  still imports cleanly and only fails on a user's first real run, so every
  asset is checked against both wheel and sdist.
- **The tag matches the built version.** A mismatch means the wrong artifact is
  about to be published under the wrong name.
- **The wheel works non-editably** (`.github/scripts/check_installed.py`): run
  from a temp directory with no `PYTHONPATH`, so `import prismaquant` cannot
  fall back to the checkout. The install must resolve from site-packages, every
  serving profile must load out of the installed JSON, the model-structure spec
  directory must be present inside the package, IQ and CB tables must load,
  `run-pipeline.sh` must be there, and the allocator and shipcard CLIs must
  run. The packaged-Gridbook-helper check was removed 2026-09-02 with that lane
  (`archive/gridbook_lane_2026-09-02/`).
- **The tag commit is on `main`.** The workflow fetches full history and refuses
  a tag cut from an unmerged release branch.

The release job deliberately does **not** re-run the whole test suite against
the install. A substantial part of the suite asserts *repository* invariants
rather than package behaviour — the pinned recipe defaults in
`run-pipeline.sh`, doc staleness, paper claims, calibration-anchor source docs
— and several tests read prismaquant's own `.py` files as source text from a
repo-relative path. Those cannot pass from outside a checkout by construction,
and making them pass would mean weakening them. CI runs the full suite from the
checkout on every push and pull request, which is the right home for repo
invariants; the release job's job is to prove the *artifact* is complete and
usable. (Three tests that drive repo-root `tools/` scripts now skip cleanly
outside a checkout instead of failing collection.)

## After a publish

The index lags for roughly 30 seconds after an upload, so a plain
`pip install prismaquant` immediately afterwards can still resolve the previous
version. Re-check with `--no-cache-dir` before concluding anything about
resolution.

## Scope note

The GPU work — kernel numerics, real-checkpoint streaming, anything touching a
served artifact — is a manual gate on the reference box and is not covered by
either workflow. CI and this pipeline cover the allocator / solver / footprint /
profile logic, which is where a silent regression is hardest to notice.
