"""Portable stdlib regression fixtures; run through PrismaBuild, never a GPU."""
import hashlib
import json
from pathlib import Path
import pickle
import runpy
import tempfile
import types
import unittest
from unittest.mock import patch


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "prior"
        self.root.mkdir()
        self.out = self.base / "new"
        self.out.mkdir()
        self.ns = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                                    "experiments/pq183_lfm_bound.py"))
        self.g = self.ns["continue_export"].__globals__
        self.image_check = patch.dict(self.g, verify_producer_image=lambda *_: None)
        self.image_check.start()
        self.addCleanup(self.image_check.stop)
        wire = self.root / "cache/wire"
        wire.mkdir(parents=True)
        self.hessian = self.root / "cache/hessian_capture.pt"
        self.hessian.write_bytes(b"fixture H; numerical validity belongs to the shared export gate")
        self.hessian.with_suffix(".pt.provenance.json").write_text("{}")
        self.fmt = "TESSERA_E4M3_K1_R1024"
        receipts, costs, cost_wires = {}, {}, {}
        for index, name in enumerate(sorted(self.ns["EXPECTED_UNITS"])):
            blob = f"wire fixture {index}".encode()
            filename = f"{index}.tessera"
            (wire / filename).write_bytes(blob)
            rec = {"file": filename, "blob_sha256": hashlib.sha256(blob).hexdigest(),
                   "blob_bytes": len(blob), "identity": {"unit": name}}
            receipts[name] = rec
            cost_wires[name] = {self.fmt: rec}
            costs[name] = {self.fmt: {"output_mse_measured": True, "wire_bytes": len(blob),
                                     "hessian_identity": {"applied": True}}}
        target = {"platform": "sm_121", "runtime_image": self.ns["IMAGE"],
                  "execution_mode": "eager", "residency": "resident"}
        scope = {"target": target, "by_unit": {}}
        data = {"costs": costs, "tessera_expert_wires": cost_wires,
                "provenance": {"wire_dir": str(wire), "hessian": {"capture_path": str(self.hessian)},
                               "tessera_serving_scope": scope}}
        (self.root / "cost.pkl").write_bytes(pickle.dumps(data))
        cost_sha = self.ns["sha"](self.root / "cost.pkl")
        meta = {"tessera_expert_wires": receipts, "tessera_expert_wire_dir": str(wire),
                "tessera_expert_stack_formats": {self.ns["STACK"]: self.fmt},
                "measurement_recipe": {"cost_sha256": cost_sha}, "tessera_serving_scope": scope}
        self.assignment = {name: self.fmt for name in receipts}
        self.assignment.update({"other.body.unit": "BF16", "__prismaquant__": meta})
        self.dump("layer_config.json", self.assignment)
        self.dump("recipe.json", {"selected_units": 96, "cost_sha256": cost_sha})
        flags = {"--menu-mode": "attested", "--anchors": "1", "--max-rounds": "1", "--anchor-budget": "1",
                 "--nsamples": "32", "--seqlen": "512", "--seed": "0", "--max-act-rows": "512",
                 "--layer-stride": "13", "--hessian": "require", "--tp-degree": "1"}
        self.dump("campaign.command.json", {"exit_code": 0, "argv": [x for pair in flags.items() for x in pair]})
        self.dump("host-status.json", {"source_snapshot": "a" * 40, "producer_image_id": self.ns["PRODUCER_IMAGE"],
                  "serving_image": self.ns["IMAGE"], "tessera_source": {"commit": self.ns["TESSERA_COMMIT"]},
                  "cleanup": {"prior-container": {"safe": True}}})
        self.dump("producer-image.json", {})  # Image qualification is a separate gate, mocked above.
        self.dump("serving-image.json", {"RepoDigests": [self.ns["IMAGE"]]})
        self.dump("producer-dependencies.json", {"passed": True})
        self.seal = self.base / "inputs.json"
        manifest = self.ns["campaign_input_description"](self.root)
        self.seal.write_text(json.dumps(manifest))
        self.args = types.SimpleNamespace(campaign_input=self.root, out=self.out,
                    campaign_input_manifest=self.seal, campaign_input_manifest_sha256=self.ns["sha"](self.seal),
                    model=Path("/model"))
        self.host_source = "b" * 40
        (self.out / "host-status.json").write_text(json.dumps({
            "schema": "prismaquant.pq183-host-observation.v1", "source_snapshot": self.host_source,
            "campaign_source_snapshot": "a" * 40,
            "phases": {"continue-export": {
                "campaign_input_manifest_sha256": self.args.campaign_input_manifest_sha256}}}))

    def dump(self, name, data):
        (self.root / name).write_text(json.dumps(data))

    def test_exact_roster_and_unchanged_bytes_are_required(self):
        manifest = self.ns["verify_campaign_inputs"](self.args)
        self.assertEqual(manifest["measured_units"], 96)
        self.assertEqual(manifest["priced_wires"], 96)
        self.assertEqual(len(manifest["files"]), 106)
        # Unconsumed static scales are not an input to this E4M3 export.
        (self.root / "cache/input_scales.safetensors").write_bytes(b"unused")
        self.ns["verify_campaign_inputs"](self.args)
        (self.root / "cache/wire/0.tessera").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "input changed"):
            self.ns["verify_campaign_inputs"](self.args)

    def test_failed_or_incomplete_campaign_cannot_be_sealed(self):
        self.dump("campaign.command.json", {"exit_code": 1})
        with self.assertRaisesRegex(ValueError, "did not exit"):
            self.ns["campaign_input_description"](self.root)

    def test_changed_cost_or_hessian_refuses_before_deserialization(self):
        for relative in ("cost.pkl", "cache/hessian_capture.pt"):
            with self.subTest(relative=relative):
                path = self.root / relative
                original = path.read_bytes()
                path.write_bytes(original + b"changed")
                with patch.object(pickle, "load", side_effect=AssertionError("must not unpickle")):
                    with self.assertRaisesRegex(ValueError, "input changed"):
                        self.ns["verify_campaign_inputs"](self.args)
                path.write_bytes(original)

    def test_incomplete_cost_population_is_refused(self):
        path = self.root / "cost.pkl"
        data = pickle.loads(path.read_bytes())
        del data["costs"][next(iter(data["costs"]))]
        path.write_bytes(pickle.dumps(data))
        with self.assertRaisesRegex(ValueError, "complete measured 96-unit"):
            self.ns["campaign_input_description"](self.root)

    def test_unsafe_prior_cleanup_is_refused(self):
        path = self.root / "host-status.json"
        previous = json.loads(path.read_text())
        previous["cleanup"]["prior-container"]["safe"] = False
        path.write_text(json.dumps(previous))
        with self.assertRaisesRegex(ValueError, "cleanup is unverified"):
            self.ns["campaign_input_description"](self.root)

    def test_output_collision_preserves_existing_bytes_and_input(self):
        (self.out / "cost.pkl").write_bytes(b"existing output")
        before = self.ns["verify_campaign_inputs"](self.args)
        with patch.dict(self.g, producer_preflight=lambda _: None):
            with self.assertRaisesRegex(ValueError, "output collision"):
                self.ns["continue_export"](self.args)
        self.assertEqual((self.out / "cost.pkl").read_bytes(), b"existing output")
        self.assertEqual(before, self.ns["verify_campaign_inputs"](self.args))

    def test_output_inside_original_is_refused(self):
        self.args.out = self.root / "new"
        with self.assertRaisesRegex(ValueError, "outside its input"):
            self.ns["verify_campaign_inputs"](self.args)

    def test_missing_host_receipt_fails_before_producer_work(self):
        (self.out / "host-status.json").unlink()
        with patch.dict(self.g, producer_preflight=lambda _: self.fail("must refuse before producer work")):
            with self.assertRaises(FileNotFoundError):
                self.ns["continue_export"](self.args)
        self.assertFalse((self.out / "cost.pkl").exists())

    def test_malformed_or_misbound_host_identity_fails_before_producer_work(self):
        path = self.out / "host-status.json"
        original = path.read_text()
        for field, value in (("source_snapshot", ""), ("source_snapshot", "b" * 39),
                             ("source_snapshot", None), ("schema", "wrong"),
                             ("campaign_source_snapshot", "c" * 40), ("phases", {})):
            with self.subTest(field=field, value=value):
                receipt = json.loads(original)
                receipt[field] = value
                path.write_text(json.dumps(receipt))
                with patch.dict(self.g, producer_preflight=lambda _: self.fail("must refuse before producer work")):
                    with self.assertRaisesRegex(ValueError, "host source receipt"):
                        self.ns["continue_export"](self.args)
                self.assertFalse((self.out / "cost.pkl").exists())

    def test_missing_cost_in_seal_refuses_before_unpickle(self):
        manifest = json.loads(self.seal.read_text())
        del manifest["files"]["cost.pkl"]
        self.seal.write_text(json.dumps(manifest))
        self.args.campaign_input_manifest_sha256 = self.ns["sha"](self.seal)
        with patch.object(pickle, "load", side_effect=AssertionError("must not unpickle")):
            with self.assertRaisesRegex(ValueError, "omits the authoritative"):
                self.ns["verify_campaign_inputs"](self.args)

    def test_continuation_preserves_assignment_and_never_requantizes(self):
        before = self.ns["verify_campaign_inputs"](self.args)
        receipts = self.assignment["__prismaquant__"]["tessera_expert_wires"]
        events = []
        def preflight(model, **kw):
            events.append("shared-preflight")
            self.assertFalse(kw["cached_expert_units"])
            self.assertEqual(kw["hessian_path"], self.hessian)
            self.assertEqual(Path(kw["assignment_path"]).read_bytes(), (self.root / "layer_config.json").read_bytes())
            self.assertFalse((self.out / "cached-expert-units").exists())
            return {"build": {}, "selected_serving_scope": {"expert_projection": {
                "source": {}, "units": receipts, "wire_dir": str(self.root / "cache/wire")}}}
        def bundle(projection):
            events.append("bundle")
            path = Path(projection["wire_dir"]) / "cached-units.json"
            path.write_text(json.dumps(projection))
            return path
        def exported(args, hessian, digest):
            events.append("export")
            self.assertEqual(digest, self.ns["sha"](self.out / "build.json"))
            for name in ("cost.pkl", "layer_config.json", "recipe.json"):
                self.assertEqual((self.out / name).read_bytes(), (self.root / name).read_bytes())
            self.assertEqual(len(list((self.out / "cached-expert-units").glob("*.tessera"))), 96)
        fake_lane = types.SimpleNamespace(preflight=preflight, write_cached_expert_units=bundle)
        fake_scope = types.SimpleNamespace(ServingTarget=lambda **kw: kw)
        def forbidden(*_):
            raise AssertionError("must not call a campaign or allocator")
        with patch.dict("sys.modules", {"prismaquant.tessera_export_lane": fake_lane,
                                        "prismaquant.tessera_serving_scope": fake_scope}), patch.dict(
                self.g, producer_preflight=lambda _: None, campaign=forbidden, allocate=forbidden,
                export_from_build=exported), patch.object(
                    self.g["subprocess"], "check_output", side_effect=FileNotFoundError("git absent in producer")):
            self.ns["continue_export"](self.args)
        self.assertEqual(events, ["shared-preflight", "bundle", "export"])
        self.assertEqual(before, self.ns["verify_campaign_inputs"](self.args))
        self.assertEqual(json.loads((self.out / "continuation.json").read_text())[
            "continuation_source_snapshot"], self.host_source)


if __name__ == "__main__":
    unittest.main()
