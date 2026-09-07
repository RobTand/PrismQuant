"""Closed, explicit source transition for the interrupted 2026-09-07 joint run.

This is not a caller-selected source hash. Admission reconstructs the complete
old package with only the reviewed empty-lease change and exact API glue undone.
The independent receipt binds the actual new package (including this verifier),
its Git commit, the original inputs and every preserved unit. No old byte moves.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import weakref

VERSION = "empty_joint_lease_v1"
SCHEMA = "prismaquant.joint_aura.source_transition.v1"
_CONTRACT = {
    "source_sha256": "735a5712fc43153709c1f9e463fe1fff649cb50ac8c28c5c305ad6369c3043ba",
    "git_commit": "c81ddaa55b9d6970087285797f93a1e6a5143d05",
    "manifest_sha256": "c6a5df125c564984be020888d265ac1233a54dacc4756049538fdd6ff74f8f96",
    "identity_sha256": "35097589e2bd3ec5385b5977058edf1ce60dd855dc1e346883d6d9af75757660",
    "inspection_sha256": "61ae11fcfdfad75f846c1c11d770c3ab645dcc380c9da79d0033a3ebdc15f49c",
    "plan_sha256": "ee8148f7e34cbecd82804bbcc3231fbe4e32a95912c7d2ecb2b9af64d274372c",
    "prepared_sha256": "50be6355a9c103f4558e8817f0ca550a979d3815b1c6db2d5b4d251d02cefe87",
    "production_cache_sha256": "c8463ba61d7fdf0e6feb842136f1276ec106cb8de9b8316431558e3ca3b01525",
    "preserved_units": 1260,
    "total_units": 2142,
}
# Exact old/new snippets, not patterns or caller-provided exemptions. Reverse
# these, omit this new module only, and the entire old package must match.
_SOURCE_REWRITES = {'aura_cost.py': [('            joint_lease = None\n            if joint_activation:\n',
                   '            joint_lease = None\n'
                   '            # Completed layers still propagate cotangents to pending earlier\n'
                   '            # layers, but have no target device or projections to lease.\n'
                   '            if joint_activation and pending:\n'),
                  ('    joint_projection_backend=None,\n    profile=None,\n) -> dict:',
                   '    joint_projection_backend=None,\n'
                   '    source_transition=None,\n'
                   '    profile=None,\n'
                   ') -> dict:'),
                  ('    if type(probe_microbatch) is not int or probe_microbatch < 0:\n',
                   '    if source_transition is not None:\n'
                   '        from prismaquant.joint_aura_source_transition import '
                   'require_verified_transition\n'
                   '        source_transition = require_verified_transition(\n'
                   '            source_transition, checkpoint_dir=checkpoint_dir,\n'
                   '            resume=resume, joint_activation=joint_activation,\n'
                   '        )\n'
                   '    if type(probe_microbatch) is not int or probe_microbatch < 0:\n'),
                  ('            "producer_source_sha256": _aura_source_sha256(),\n'
                   '            "source_execution": source_execution_identity(runner.model),',
                   '            "producer_source_sha256": (_aura_source_sha256() if source_transition '
                   'is None\n'
                   '                                       else '
                   'source_transition.measurement_source_sha256),\n'
                   '            "source_execution": source_execution_identity(runner.model),'),
                  ('        checkpoint_root, checkpoint_identity_sha256, completed_states = (\n'
                   '            _prepare_aura_checkpoints(',
                   '        if source_transition is not None:\n'
                   '            identity = source_transition.measurement_identity(identity)\n'
                   '        checkpoint_root, checkpoint_identity_sha256, completed_states = (\n'
                   '            _prepare_aura_checkpoints('),
                  ('        if execution_partition is not None:\n'
                   '            payload["provenance"]["streamed_microbatch"] = execution_partition',
                   '        if source_transition is not None:\n'
                   '            payload["provenance"]["source_transition"] = '
                   'source_transition.final_provenance()\n'
                   '        if execution_partition is not None:\n'
                   '            payload["provenance"]["streamed_microbatch"] = execution_partition'),
                  ('                        ), **({"joint_aura_rows": joint_rows[name]} if '
                   'joint_activation else {})},',
                   '                        ), **({"joint_aura_rows": joint_rows[name]} if '
                   'joint_activation else {}),\n'
                   '                        **({"execution_provenance": '
                   'source_transition.execution_provenance}\n'
                   '                           if source_transition is not None else {})},')],
 'tessera_joint_aura.py': [('def execute(command, config, *, plan_sha256, prepared=None, '
                            'resume=False):\n'
                            '    """Execute one admitted preparation or one dependent cost action."""\n',
                            'def execute(command, config, *, plan_sha256, prepared=None, resume=False, '
                            'source_transition=None):\n'
                            '    """Execute one admitted preparation or one dependent cost action."""\n'
                            '    if source_transition is not None:\n'
                            '        from .joint_aura_source_transition import load_transition\n'
                            '        _require(command == "run" and resume, "source transition requires '
                            'run --resume")\n'
                            '        source_transition = load_transition(\n'
                            '            source_transition, config=config, plan_sha256=plan_sha256,\n'
                            '            prepared=prepared, checkpoint_dir=Path(config["output_root"]) '
                            '/ "checkpoints",\n'
                            '        )\n'),
                           ('        implementation = _aura_source_sha256()\n'
                            '        if command == "prepare":',
                            '        implementation = (_aura_source_sha256() if source_transition is '
                            'None\n'
                            '                          else '
                            'source_transition.measurement_source_sha256)\n'
                            '        if source_transition is not None:\n'
                            '            result["source_transition"] = '
                            'source_transition.execution_provenance\n'
                            '        if command == "prepare":'),
                           ('                joint_projection_backend=projection_backend,\n'
                            '                include_routed_experts=True',
                            '                joint_projection_backend=projection_backend,\n'
                            '                **({"source_transition": source_transition} if '
                            'source_transition is not None else {}),\n'
                            '                include_routed_experts=True'),
                           ('    parser.add_argument("--resume", action="store_true")\n'
                            '    args = parser.parse_args(argv)',
                            '    parser.add_argument("--resume", action="store_true")\n'
                            '    parser.add_argument("--source-transition", type=Path)\n'
                            '    parser.add_argument("--source-transition-sha256")\n'
                            '    args = parser.parse_args(argv)\n'
                            '    if bool(args.source_transition) != '
                            'bool(args.source_transition_sha256):\n'
                            '        parser.error("--source-transition and --source-transition-sha256 '
                            'are required together")'),
                           ('        resume=args.resume)\n    print(json.dumps',
                            '        resume=args.resume,\n'
                            '        **({"source_transition": {"path": str(args.source_transition),\n'
                            '                                  "sha256": '
                            'args.source_transition_sha256}}\n'
                            '           if args.source_transition is not None else {}))\n'
                            '    print(json.dumps')]}


def _require(ok, message):
    if not ok:
        raise ValueError(f"joint source transition: {message}")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _bound(record, label):
    _require(isinstance(record, dict) and set(record) == {"path", "sha256"},
             f"{label} requires independently bound path/SHA256")
    path = Path(record["path"])
    _require(path.is_file() and _sha(path) == record["sha256"], f"{label} bytes changed")
    return path


def source_proof(package_root=None):
    """Reconstruct old bytes; any additional source change fails closed."""
    root = Path(package_root) if package_root is not None else Path(__file__).resolve().parent
    current, original = hashlib.sha256(), hashlib.sha256()
    seen = set()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.relative_to(root).parts or path.suffix in {".pyc", ".pyo"}:
            continue
        name = path.relative_to(root).as_posix()
        payload = path.read_bytes()
        def update(digest, data):
            encoded = name.encode()
            digest.update(len(encoded).to_bytes(4, "big"))
            digest.update(encoded)
            digest.update(len(data).to_bytes(8, "big"))
            digest.update(data)
        update(current, payload)
        if name == "joint_aura_source_transition.py":
            seen.add(name)
            continue
        for old, new in reversed(_SOURCE_REWRITES.get(name, ())):
            _require(payload.count(new.encode()) == 1, f"unapproved or missing source hunk in {name}")
            payload = payload.replace(new.encode(), old.encode(), 1)
            seen.add(name)
        update(original, payload)
    _require(seen == set(_SOURCE_REWRITES) | {"joint_aura_source_transition.py"}, "incomplete source proof")
    _require(original.hexdigest() == _CONTRACT["source_sha256"], "unapproved producer package change")
    return {"producer_source_sha256": current.hexdigest(),
            "reconstructed_source_sha256": original.hexdigest(),
            "transition_module_sha256": _sha(root / "joint_aura_source_transition.py")}


def _actual_execution():
    _require(not os.environ.get("PRISMAQUANT_IDENTITY_GIT_COMMIT"),
             "legacy Git identity override is forbidden for an explicit transition")
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                            capture_output=True, text=True, timeout=10).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", "prismaquant"],
                            cwd=root, check=True, capture_output=True, text=True, timeout=10).stdout
    _require(not status.strip(), "producer package must be committed and clean")
    return {"git_commit": commit, **source_proof()}


def _load_inputs(bindings, checkpoint_dir):
    _require(set(bindings) == {"plan", "prepared", "inspection"}, "unexpected input bindings")
    loaded = {}
    for label in ("plan", "prepared", "inspection"):
        _require(bindings[label]["sha256"] == _CONTRACT[f"{label}_sha256"], f"unapproved {label}")
        loaded[label] = json.loads(_bound(bindings[label], label).read_bytes())
    plan, prepared, inspection = (loaded[k] for k in ("plan", "prepared", "inspection"))
    _require(prepared["implementation_sha256"] == _CONTRACT["source_sha256"], "prepared source mismatch")
    _require(prepared["plan_sha256"] == bindings["plan"]["sha256"], "prepared plan mismatch")
    _require(prepared["production_cache"]["sha256"] == _CONTRACT["production_cache_sha256"], "PWC binding mismatch")
    _bound(prepared["production_cache"], "prepared production cache")
    manifest_path = Path(checkpoint_dir) / "manifest.json"
    _require(_sha(manifest_path) == _CONTRACT["manifest_sha256"], "original manifest changed")
    manifest = json.loads(manifest_path.read_bytes())
    _require(manifest["identity_sha256"] == _CONTRACT["identity_sha256"] ==
             hashlib.sha256(_canonical(manifest["identity"])).hexdigest(), "original identity seal mismatch")
    _require(manifest["identity"]["git_commit"] == _CONTRACT["git_commit"] and
             manifest["identity"]["producer_source_sha256"] == _CONTRACT["source_sha256"], "original producer mismatch")
    _require(inspection["identity_sha256"] == manifest["identity_sha256"] and
             inspection["manifest_sha256"] == _CONTRACT["manifest_sha256"] and
             inspection["original_source_commit"] == _CONTRACT["git_commit"], "inspection identity mismatch")
    _require(inspection["completed_units"] == len(inspection["units"]) == _CONTRACT["preserved_units"] and
             inspection["total_units"] == len(manifest["units"]) == _CONTRACT["total_units"], "original roster count mismatch")
    return loaded, manifest


def _unit_roster(checkpoint_dir, manifest, preserved, execution_provenance=None, adopted=()):
    # Reuse the existing envelope/payload validator. Original envelopes are
    # additionally byte-bound; new envelopes must carry this exact execution.
    from .aura_cost import _aura_unit_checkpoint_path, _load_aura_unit_checkpoint
    root = Path(checkpoint_dir)
    expected = {}
    for row in manifest["units"]:
        name = row["qname"]
        path = _aura_unit_checkpoint_path(root, name)
        _require(row["file"] == path.relative_to(root).as_posix(), "noncanonical unit filename")
        _require(name not in expected, "duplicate manifest unit")
        expected[name] = path
    old = {row["qname"]: row for row in preserved}
    _require(len(old) == len(preserved) and set(old) <= set(expected), "invalid preserved roster")
    _require(set((root / "units").glob("*.pkl")) <= set(expected.values()), "unexpected unit file")
    prior = {row["qname"]: row for row in adopted}
    _require(len(prior) == len(adopted) and set(prior) <= set(expected) - set(old), "invalid adopted roster")
    new = []
    for name, path in expected.items():
        if name in old:
            row = old[name]
            _require(path.is_file() and path.stat().st_size == row["bytes"] and
                     _sha(path) == row["sha256"] and row["file"] == path.relative_to(root).as_posix(),
                     f"preserved unit bytes changed: {name}")
        elif name in prior:
            row = prior[name]
            _require(path.is_file() and path.stat().st_size == row["bytes"] and
                     _sha(path) == row["sha256"] and row["file"] == path.relative_to(root).as_posix(),
                     f"predecessor unit bytes changed: {name}")
        elif not path.exists():
            continue
        state = _load_aura_unit_checkpoint(path, qname=name, identity_sha256=manifest["identity_sha256"])
        with path.open("rb") as handle:
            envelope = pickle.load(handle)
        if name in old:
            _require(envelope["payload_sha256"] == old[name]["payload_sha256"], f"preserved payload changed: {name}")
        else:
            expected_execution = prior[name]["execution_provenance"] if name in prior else execution_provenance
            _require(expected_execution is not None and
                     state.get("execution_provenance") == expected_execution,
                     f"new unit lacks bound execution provenance: {name}")
            if name in prior:
                _require(envelope["payload_sha256"] == prior[name]["payload_sha256"],
                         f"predecessor payload changed: {name}")
            new.append({"qname": name, "file": path.relative_to(root).as_posix(),
                        "sha256": _sha(path), "payload_sha256": envelope["payload_sha256"],
                        "bytes": path.stat().st_size, "execution_provenance": expected_execution})
    return new


def _execution_provenance(receipt, digest):
    return {"schema": SCHEMA, "version": VERSION,
            "receipt": {"sha256": digest}, "execution": receipt["execution"],
            "measurement_identity_sha256": receipt["original"]["identity_sha256"]}


def _read_chain(bound, *, execution, bindings=None, seen=frozenset()):
    _require(isinstance(bound, dict) and set(bound) == {"path", "sha256"}, "invalid predecessor binding")
    _require(bound["sha256"] not in seen and len(seen) < 128, "cyclic or excessive predecessor chain")
    path = _bound(bound, "transition receipt")
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == bound["sha256"], "receipt changed during read")
    receipt = json.loads(raw)
    _require(set(receipt) == {"schema", "version", "execution", "original", "inputs", "preserved_units",
                             "predecessor", "adopted_units"}, "unexpected receipt fields")
    _require(receipt["schema"] == SCHEMA and receipt["version"] == VERSION and receipt["original"] == _CONTRACT,
             "unapproved transition contract")
    actual = receipt["execution"]
    _require(set(actual) == set(execution) and
             re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", actual.get("git_commit", "")) is not None and
             {k: v for k, v in actual.items() if k != "git_commit"} ==
             {k: v for k, v in execution.items() if k != "git_commit"},
             "predecessor execution source differs from current package")
    if bindings is not None:
        _require(receipt["inputs"] == bindings, "predecessor input bindings changed")
    adopted = receipt["adopted_units"]
    _require(isinstance(adopted, list), "invalid adopted unit records")
    by_name = {}
    for row in adopted:
        _require(isinstance(row, dict) and set(row) == {"qname", "file", "sha256", "payload_sha256", "bytes",
                                                       "execution_provenance"}, "invalid adopted unit record")
        _require(row["qname"] not in by_name, "duplicate adopted unit")
        by_name[row["qname"]] = row
    predecessor = receipt["predecessor"]
    if predecessor is None:
        _require(not adopted, "adopted units require a bound predecessor receipt")
    else:
        prior, _ = _read_chain(predecessor, execution=execution, bindings=receipt["inputs"],
                              seen=seen | {bound["sha256"]})
        _require(prior["preserved_units"] == receipt["preserved_units"], "predecessor original roster changed")
        inherited = {row["qname"]: row for row in prior["adopted_units"]}
        _require(set(inherited) <= set(by_name), "predecessor adopted units disappeared")
        provenance = _execution_provenance(prior, predecessor["sha256"])
        for name, row in by_name.items():
            if name in inherited:
                _require(row == inherited[name], "inherited unit binding changed")
            else:
                _require(row["execution_provenance"] == provenance, "fabricated predecessor unit provenance")
    return receipt, raw


def create_transition(*, bindings, checkpoint_dir, output, predecessor=None):
    """Create once; never rewrite a checkpoint, prepared cache or old receipt."""
    execution = _actual_execution()
    loaded, manifest = _load_inputs(bindings, checkpoint_dir)
    preserved = loaded["inspection"]["units"]
    adopted = []
    if predecessor is None:
        _unit_roster(checkpoint_dir, manifest, preserved)
    else:
        prior, _ = _read_chain(predecessor, execution=execution, bindings=bindings)
        _require(prior["preserved_units"] == preserved, "predecessor original roster changed")
        adopted = _unit_roster(checkpoint_dir, manifest, preserved,
                               _execution_provenance(prior, predecessor["sha256"]), prior["adopted_units"])
    receipt = {"schema": SCHEMA, "version": VERSION, "execution": execution,
               "original": dict(_CONTRACT), "inputs": bindings, "preserved_units": preserved,
               "predecessor": predecessor, "adopted_units": adopted}
    raw = _canonical(receipt) + b"\n"
    with Path(output).open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return {"path": str(output), "sha256": hashlib.sha256(raw).hexdigest()}


_ISSUED = weakref.WeakSet()


@dataclass(frozen=True, eq=False)
class VerifiedTransition:
    """Factory-issued immutable capability; arbitrary mappings are rejected."""
    _receipt_bytes: bytes
    _receipt_path: str
    _receipt_sha256: str
    _checkpoint_dir: str
    _manifest_bytes: bytes

    @property
    def measurement_source_sha256(self):
        return _CONTRACT["source_sha256"]

    @property
    def execution_provenance(self):
        receipt = json.loads(self._receipt_bytes)
        return _execution_provenance(receipt, self._receipt_sha256)

    def measurement_identity(self, actual):
        receipt = json.loads(self._receipt_bytes)
        _require(actual["git_commit"] == receipt["execution"]["git_commit"] and
                 actual["producer_source_sha256"] == receipt["execution"]["producer_source_sha256"],
                 "actual checkpoint source does not match admitted execution")
        identity = dict(actual)
        identity["git_commit"] = receipt["original"]["git_commit"]
        identity["producer_source_sha256"] = receipt["original"]["source_sha256"]
        expected = json.loads(self._manifest_bytes)["identity"]
        _require(identity == expected, "non-source checkpoint identity changed")
        return identity

    def final_provenance(self):
        receipt = json.loads(self._receipt_bytes)
        new = _unit_roster(self._checkpoint_dir, json.loads(self._manifest_bytes),
                           receipt["preserved_units"], self.execution_provenance, receipt["adopted_units"])
        _require(len(new) + len(receipt["preserved_units"]) == receipt["original"]["total_units"],
                 "incomplete final checkpoint coverage")
        return {**self.execution_provenance, "preserved_units": receipt["preserved_units"],
                "new_units": new, "preserved_count": len(receipt["preserved_units"]), "new_count": len(new)}


def load_transition(bound_receipt, *, config, plan_sha256, prepared, checkpoint_dir):
    execution = _actual_execution()
    receipt, raw = _read_chain(bound_receipt, execution=execution)
    path = Path(bound_receipt["path"])
    _require(receipt["execution"] == execution, "actual execution source differs from receipt")
    loaded, manifest = _load_inputs(receipt["inputs"], checkpoint_dir)
    _require(config == loaded["plan"] and plan_sha256 == receipt["inputs"]["plan"]["sha256"], "runtime plan changed")
    _require(prepared == receipt["inputs"]["prepared"], "runtime prepared binding changed")
    _require(receipt["preserved_units"] == loaded["inspection"]["units"], "preserved roster changed")
    verified = VerifiedTransition(raw, str(path), bound_receipt["sha256"],
                                  str(Path(checkpoint_dir).resolve()), _canonical(manifest))
    _unit_roster(checkpoint_dir, manifest, receipt["preserved_units"], verified.execution_provenance,
                 receipt["adopted_units"])
    _ISSUED.add(verified)
    return verified


def require_verified_transition(value, *, checkpoint_dir, resume, joint_activation):
    _require(type(value) is VerifiedTransition and value in _ISSUED,
             "transition must be issued by the verified receipt loader")
    _require(resume is True and joint_activation is True and checkpoint_dir is not None and
             Path(checkpoint_dir).resolve() == Path(value._checkpoint_dir), "transition is restricted to bound joint resume")
    receipt = json.loads(value._receipt_bytes)
    _require(_sha(value._receipt_path) == value._receipt_sha256, "admitted receipt bytes changed")
    execution = _actual_execution()
    _require(receipt["execution"] == execution, "producer source changed after admission")
    _read_chain({"path": value._receipt_path, "sha256": value._receipt_sha256}, execution=execution)
    _require(_sha(Path(checkpoint_dir) / "manifest.json") == receipt["original"]["manifest_sha256"],
             "manifest changed after admission")
    _unit_roster(checkpoint_dir, json.loads(value._manifest_bytes),
                 receipt["preserved_units"], value.execution_provenance, receipt["adopted_units"])
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "prepared", "inspection"):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predecessor", type=Path)
    parser.add_argument("--predecessor-sha256")
    args = parser.parse_args(argv)
    if bool(args.predecessor) != bool(args.predecessor_sha256):
        parser.error("--predecessor and --predecessor-sha256 are required together")
    bindings = {name: {"path": str(getattr(args, name)), "sha256": getattr(args, name + "_sha256")}
                for name in ("plan", "prepared", "inspection")}
    print(json.dumps(create_transition(bindings=bindings, checkpoint_dir=args.checkpoint_dir, output=args.output,
        predecessor=None if args.predecessor is None else {"path": str(args.predecessor),
                                                          "sha256": args.predecessor_sha256})))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
