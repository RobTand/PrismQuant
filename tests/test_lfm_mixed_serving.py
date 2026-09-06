"""Portable actual filesystem/command refusal tests; no model or CUDA imports."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import types
import sys
import unittest
from unittest.mock import patch

from experiments import lfm_mixed_serving as m


class ServingTests(unittest.TestCase):
    def test_assembly_binds_actual_result_and_payload_population(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "model"
            artifact.mkdir()
            (artifact / "weights").write_bytes(b"encoded bytes")
            record = {"schema": "prismabuild.tessera-model.v1", "index": None, "files": {"weights": m.sha(artifact / "weights")}}
            m.write(artifact / "pb-result.json", record)
            receipt = root / "result.json"
            receipt.write_text("assembly log\nPB_TESSERA_RESULT=" + json.dumps(record) + "\n")
            digest = m.sha(receipt)
            self.assertEqual(m.verify_assembly(artifact, receipt, digest), record)
            (artifact / "weights").write_bytes(b"changed bytes")
            with self.assertRaisesRegex(ValueError, "bytes/population"):
                m.verify_assembly(artifact, receipt, digest)
            (artifact / "weights").write_bytes(b"encoded bytes")
            (artifact / "extra").write_text("extra")
            with self.assertRaisesRegex(ValueError, "bytes/population"):
                m.verify_assembly(artifact, receipt, digest)

    def test_partition_cannot_masquerade_as_assembly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = {"schema": "prismabuild.tessera-model.v1", "index": 0, "files": {"weights": "x"}}
            m.write(root / "pb-result.json", record)
            blob = root / "cas.stdout"
            blob.write_text("PB_TESSERA_RESULT=" + json.dumps(record) + "\n")
            with self.assertRaisesRegex(ValueError, "assembled"):
                m.verify_assembly(root, blob, m.sha(blob))

    def test_encoder_full_closure_refuses_added_or_changed_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "encoder"
            source.mkdir()
            (source / "script.py").write_text("original")
            manifest = root / "manifest.json"
            m.write(manifest, {"commit": m.ENCODER, "files": {"script.py": m.sha(source / "script.py")}})
            digest = m.sha(manifest)
            m.verify_encoder(source, manifest, digest)
            (source / "script.py").write_text("changed")
            with self.assertRaisesRegex(ValueError, "closure"):
                m.verify_encoder(source, manifest, digest)
            (source / "script.py").write_text("original")
            (source / "extra.py").write_text("additional")
            with self.assertRaisesRegex(ValueError, "closure"):
                m.verify_encoder(source, manifest, digest)

    def handoff(self):
        plan = {f"{grid}-{i}": {"grid": grid, "q256": rung}
                for grid, rung, count in (("E4M3", 1024, 22), ("E2M1x2", 896, 6), ("BF16", 1792, 60))
                for i in range(count)}
        source = {"files": {"weights": "bf16source"}}
        identity = {"source": source, "runtime_image": m.IMAGE, "encoder_fixture_id": "numerics",
                    "code_sha256": "code", "options": {"plan": plan, "hessian_sha256": None,
                    "input_scales_sha256": m.SCALES}}
        manifest = {"export_identity": identity, "merged_from": [{}], "totals": {"modules": 74, "units": 2178}}
        calibration = {"mode": "calibrate", "weights_only_export": True, "hessian": None}
        return plan, manifest, source, calibration

    def test_handoff_requires_real_source_image_scales_and_full_population(self):
        self.assertTrue(m.validate_handoff(*self.handoff()))
        for field, value in (("runtime_image", "other-image"), ("source", {}), ("encoder_fixture_id", "")):
            args = self.handoff()
            args[1]["export_identity"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                m.validate_handoff(*args)
        args = self.handoff()
        args[1]["export_identity"]["options"]["input_scales_sha256"] = "other-scales"
        with self.assertRaisesRegex(ValueError, "calibration binding"):
            m.validate_handoff(*args)
        args = self.handoff()
        args[1]["totals"]["units"] = 96
        with self.assertRaisesRegex(ValueError, "full-body"):
            m.validate_handoff(*args)

    def test_passthrough_cannot_count_as_trellis16(self):
        args = self.handoff()
        args[0]["BF16-0"] = {"grid": "PASSTHROUGH", "q256": 1792}
        with self.assertRaisesRegex(ValueError, "family population"):
            m.validate_handoff(*args)

    def test_strict_refusal_does_not_become_attestation_or_mask_raw_failure(self):
        observed = {"verdict": "passed", "require_attested": False,
                    "expected_owners": list(range(74)), "expected_projection_units": 2178,
                    "cell_launch_agreement": {"structures": {
                        structure: {"phases": {phase: {"covered_by_cell": covered, "unattested": missing}
                                   for phase in ("prefill", "decode")}}
                        for structure, covered, missing in (("dense", 0, 52), ("routed_moe", 22, 0))}}}
        strict = {"verdict": "REFUSED", "problems": ["current contract does not attest every planned dense owner in both driven phases: details"]}
        self.assertEqual(m.classify_strict(observed, strict, 1), "dense_cells_unattested")
        with self.assertRaisesRegex(ValueError, "unexpected"):
            m.classify_strict(observed, {"verdict": "REFUSED", "problems": ["wrong source"]}, 1)
        altered = copy.deepcopy(observed)
        altered["cell_launch_agreement"]["structures"]["routed_moe"]["phases"]["decode"]["unattested"] = 1
        with self.assertRaisesRegex(ValueError, "another coverage defect"):
            m.classify_strict(altered, strict, 1)
        with self.assertRaisesRegex(ValueError, "raw route"):
            m.classify_strict({**observed, "verdict": "REFUSED"}, strict, 1)

    def inspected(self):
        return {"Id": "a" * 64, "Config": {"Labels": {m.LABEL: "ours"},
                "Env": [f"{k}={v}" for k, v in m.LIMITS.items()]},
                "HostConfig": {"Memory": 64 * 2**30, "MemorySwap": 64 * 2**30, "CpusetCpus": "5-8"}}

    def test_actual_container_resource_and_owner_inspection(self):
        record = self.inspected()
        m.verify_owned(record, "a" * 64, "ours", [5, 6, 7, 8])
        for change in ("owner", "cpus", "threads"):
            altered = copy.deepcopy(record)
            if change == "owner": altered["Config"]["Labels"][m.LABEL] = "someone-else"
            if change == "cpus": altered["HostConfig"]["CpusetCpus"] = "0-15"
            if change == "threads": altered["Config"]["Env"] = ["OMP_NUM_THREADS=32"]
            with self.subTest(change=change), self.assertRaises(ValueError):
                m.verify_owned(altered, "a" * 64, "ours", [5, 6, 7, 8])

    def test_public_container_drops_arbitrary_env_and_daemon_fields(self):
        record = self.inspected()
        record["Config"]["Env"] += ["PRISMABUILD_RESOURCE_SCOPE_TOKEN=private-token"]
        record["secret_metadata"] = "private-token"
        public = m.public_container(record)
        self.assertNotIn("private-token", json.dumps(public))
        self.assertEqual(public["native_environment"], m.LIMITS)

    def test_preflight_runs_handoff_and_returns_before_docker(self):
        # Drive the real CLI/control flow and real seal files; substitute only
        # the external producer/header APIs and the already measured inputs.
        plan, manifest, teacher, calibration = self.handoff()
        calibration["files"] = {}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            encoder = root / "encoder"
            (encoder / "experiments").mkdir(parents=True)
            (encoder / "experiments/moe_greedy_smoke_prompts.json").write_text("prompts")
            calibrator = root / "calibration"
            calibrator.mkdir()
            m.write(calibrator / "preparation-seal.json", calibration)
            m.write(calibrator / "plan.json", plan)
            planpath = root / "plan.json"
            m.write(planpath, plan)
            artifact = Path("/mnt/shared/fixture-student")
            source = Path("/mnt/shared/fixture-teacher")
            modules = {}
            for name in ("tessera", "tessera.serving", "tessera.serving_parts",
                         "tessera.serving.build_identity", "tessera.serving.contract"):
                modules[name] = types.ModuleType(name)
            modules["tessera.serving_parts"].source_identity = lambda p: teacher if p == source else {"student": "bytes"}
            modules["tessera.serving.build_identity"].is_complete = lambda p: True
            modules["tessera.serving.build_identity"].incomplete_reason = lambda p: ""
            modules["tessera.serving.contract"].derive_smoke_status = lambda p: "recorded"
            gate = types.ModuleType("fixture_gate")
            gate._population = unittest.mock.Mock()
            loader = unittest.mock.Mock()
            spec = types.SimpleNamespace(loader=loader)
            original_read = m.read
            def read(path):
                if path == artifact / "tessera_serving_manifest.json": return manifest
                if path == artifact / "config.json": return {}
                return original_read(path)
            argv = ["wrapper", "--out", str(root / "out"), "--encoder", str(encoder),
                    "--encoder-manifest", str(root / "manifest.json"), "--encoder-manifest-sha256", "bound",
                    "--artifact", str(artifact), "--source", str(source), "--plan", str(planpath),
                    "--calibration", str(calibrator), "--assembly-result", str(root / "assembly.json"),
                    "--assembly-result-sha256", "bound", "--preflight-only"]
            with patch.dict(sys.modules, modules), patch.object(sys, "argv", argv), \
                 patch.dict(m.os.environ, {"PRISMABUILD_CONTAINER_OWNER": "test-owner"}), \
                 patch.object(m.os, "sched_getaffinity", return_value={1}), \
                 patch.object(m, "verify_encoder", return_value={"commit": m.ENCODER}), \
                 patch.object(m, "verify_assembly", return_value={"files": {"fixture": "digest"}}), patch.object(m, "read", side_effect=read), \
                 patch.object(m, "CALIBRATION_SEAL", m.sha(calibrator / "preparation-seal.json")), \
                 patch.object(m, "PROMPTS", m.sha(encoder / "experiments/moe_greedy_smoke_prompts.json")), \
                 patch.object(m.importlib.util, "spec_from_file_location", return_value=spec), \
                 patch.object(m.importlib.util, "module_from_spec", return_value=gate), \
                 patch.object(m.subprocess, "check_output", side_effect=AssertionError("Docker/model must not start")):
                m.main()
            gate._population.assert_called_once_with(plan, {}, manifest)
            receipt = original_read(root / "out/preflight.json")
            self.assertEqual(receipt["status"], "verified")
            self.assertEqual(receipt["artifact_seal_sha256"], m.sha(root / "out/artifact-seal.json"))
            self.assertFalse(receipt["gpu_or_container_launched"])

    def test_archive_preserves_output_bytes_excludes_build_caches_and_refuses_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / "out"
            out.mkdir()
            (out / "ext").mkdir()
            (out / "ext/kernel.so").write_bytes(b"compiled")
            (out / "receipt.json").write_text('{"observed": true}')
            archive = root / "archive"
            record = m.publish_output(out, archive)
            self.assertEqual(record["files"]["receipt.json"]["sha256"], m.sha(out / "receipt.json"))
            self.assertFalse((archive / "ext").exists())
            with self.assertRaisesRegex(ValueError, "fresh shared"):
                m.publish_output(out, archive)

    def test_cleanup_never_stops_a_differently_owned_container(self):
        with tempfile.TemporaryDirectory() as directory:
            cidfile = Path(directory) / "cid"
            cidfile.write_text("a" * 64)
            record = self.inspected()
            record["Config"]["Labels"][m.LABEL] = "other"
            result = subprocess.CompletedProcess([], 0, json.dumps([record]), "")
            with patch.object(m.subprocess, "run", return_value=result), patch.object(m, "cleanup_container") as stop:
                with self.assertRaisesRegex(ValueError, "differently owned"):
                    m.cleanup_owned("name", cidfile, "ours")
                stop.assert_not_called()

    def test_cleanup_uses_captured_id_not_name(self):
        with tempfile.TemporaryDirectory() as directory:
            cidfile = Path(directory) / "cid"
            cidfile.write_text("a" * 64)
            result = subprocess.CompletedProcess([], 0, json.dumps([self.inspected()]), "")
            with patch.object(m.subprocess, "run", return_value=result), patch.object(m, "require_container_name_available"), \
                 patch.object(m, "cleanup_container", return_value={"safe": True}) as stop:
                self.assertTrue(m.cleanup_owned("reusable-name", cidfile, "ours")["safe"])
                stop.assert_called_once_with("a" * 64)

    def test_missing_cid_requires_name_absence(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(m, "require_container_name_available", side_effect=ValueError("name exists")), \
                 patch.object(m, "cleanup_container") as stop:
                with self.assertRaisesRegex(ValueError, "name exists"):
                    m.cleanup_owned("name", Path(directory) / "absent", "ours")
                stop.assert_not_called()


if __name__ == "__main__":
    unittest.main()
