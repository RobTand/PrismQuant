"""Code archive bytes and full extracted closure remain one immutable input."""
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

from experiments import lfm_mixed_source_run as stage
from experiments import lfm_mixed_serving as serving


class SourceStageTests(unittest.TestCase):
    def fixture(self, root, extra=False):
        source = root / "source.py"
        source.write_bytes(b"frozen source\n")
        archive = root / "source.tar"
        with tarfile.open(archive, "w") as stream:
            for name in (["source.py", "extra.py"] if extra else ["source.py"]):
                data = source.read_bytes()
                member = tarfile.TarInfo(name)
                member.size = len(data)
                stream.addfile(member, io.BytesIO(data))
        manifest = root / "manifest.json"
        serving.write(manifest, {"commit": serving.ENCODER, "archive_sha256": serving.sha(archive),
            "archive_bytes": archive.stat().st_size, "files": {"source.py": serving.sha(source)}})
        return archive, manifest

    def test_extracts_actual_archive_and_verifies_local_source_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, manifest = self.fixture(root)
            binding = stage.extract_source(archive, manifest, serving.sha(manifest), root / "local")
            self.assertEqual(binding["files"], 1)
            self.assertEqual((root / "local/source.py").read_bytes(), (root / "source.py").read_bytes())
            archive.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "archive bytes"):
                stage.extract_source(archive, manifest, serving.sha(manifest), root / "other")
            self.assertFalse((root / "other").exists())

    def test_additional_archive_file_cannot_evade_full_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive, manifest = self.fixture(root, extra=True)
            with self.assertRaisesRegex(ValueError, "closure"):
                stage.extract_source(archive, manifest, serving.sha(manifest), root / "local")


if __name__ == "__main__":
    unittest.main()
