"""Actual PB exporter stdout envelope, not a fabricated bare-JSON CAS result."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments import lfm_mixed_serving as m


class AssemblyStdoutTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        artifact = root / "model"
        artifact.mkdir()
        (artifact / "weights").write_bytes(b"actual encoded bytes")
        record = {"schema": "prismabuild.tessera-model.v1", "index": None, "count": 24,
                  "contract": "frozen export contract", "files": {"weights": m.sha(artifact / "weights")}}
        m.write(artifact / "pb-result.json", record)
        return root, artifact, record

    def test_real_log_and_unique_completion_record_bind_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root, artifact, record = self.fixture(directory)
            blob = root / "cas.stdout"
            blob.write_text("producer loading source headers\nPB_TESSERA_RESULT=" + json.dumps(record) + "\n")
            self.assertEqual(m.verify_assembly(artifact, blob, m.sha(blob)), record)

    def test_missing_and_duplicate_completion_record_refuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root, artifact, record = self.fixture(directory)
            line = "PB_TESSERA_RESULT=" + json.dumps(record) + "\n"
            blob = root / "cas.stdout"
            for text in ("logs only\n", line + line):
                blob.write_text(text)
                with self.subTest(text=text), self.assertRaisesRegex(ValueError, "exactly one.*completion"):
                    m.verify_assembly(artifact, blob, m.sha(blob))

    def test_altered_completion_record_and_changed_log_digest_refuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root, artifact, record = self.fixture(directory)
            blob = root / "cas.stdout"
            blob.write_text("PB_TESSERA_RESULT=" + json.dumps({**record, "contract": "other export"}) + "\n")
            with self.assertRaisesRegex(ValueError, "actual PB result"):
                m.verify_assembly(artifact, blob, m.sha(blob))
            digest = m.sha(blob)
            blob.write_text("changed log\n" + blob.read_text())
            with self.assertRaisesRegex(ValueError, "assembly result changed"):
                m.verify_assembly(artifact, blob, digest)


if __name__ == "__main__":
    unittest.main()
