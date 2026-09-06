"""Saved Tessera strings survive the shared reader in a fresh process.

Isolate the documented torch-free modules from the package initializer, which
imports the numerical runtime. The real export CLI is qualified separately.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


_REPO = Path(__file__).resolve().parents[1]
_BOOTSTRAP = """import sys, types
from pathlib import Path
package = types.ModuleType('prismaquant')
package.__path__ = [str(Path(sys.argv[1]) / 'prismaquant')]
sys.modules['prismaquant'] = package
from prismaquant.layer_config import canonicalize_format, load_assignment
"""


class TesseraStringAssignmentTests(unittest.TestCase):
    def run_reader(self, code, *args):
        result = subprocess.run([sys.executable, '-c', _BOOTSTRAP + code,
                                 str(_REPO), *map(str, args)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_string_matches_existing_dictionary_entry(self):
        self.run_reader("""name = 'TESSERA_E4M3_K1_R1024'
expected = canonicalize_format({'data_type': 'tessera', 'tessera_format': name})
assert canonicalize_format(name) == expected
assert canonicalize_format(name.lower()) == expected
assert 'torch' not in sys.modules and 'tessera' not in sys.modules
""")

    def test_fresh_reader_loads_campaign_assignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'assignment.json'
            path.write_text(json.dumps({
                'model.layers.13.feed_forward.experts.0.w1': 'TESSERA_E4M3_K1_R1024',
                'model.layers.0.feed_forward.w1': 'BF16'}))
            self.run_reader("""assignment = load_assignment(sys.argv[2])
assert assignment['model.layers.13.feed_forward.experts.0.w1'] == 'TESSERA_E4M3_K1_R1024'
assert assignment['model.layers.0.feed_forward.w1'] == 'BF16'
assert 'torch' not in sys.modules and 'tessera' not in sys.modules
""", path)

    def test_malformed_or_group_names_are_not_rungs(self):
        names = ['TESSERA_E4M3_K1_G1024', 'TESSERA_E4M3_K1_R1024_extra',
                 'prefix_TESSERA_E4M3_K1_R1024', 'TESSERA_E4M3_K1_R1024\n',
                 'TESSERA_E4M3_Kx_R1024']
        self.run_reader("""import json
for name in json.loads(sys.argv[2]):
    try:
        canonicalize_format(name)
    except ValueError:
        continue
    raise AssertionError('malformed rung accepted: ' + repr(name))
""", json.dumps(names))

    def test_malformed_dictionary_names_are_rejected(self):
        names = ['TESSERA_E4M3_K1_R1024\n', 'TESSERA_E4M3_K1_R1024_extra',
                 'TESSERA_E4M3_K1_G1024', 'tessera_e4m3_k1_r1024']
        self.run_reader("""import json
for name in json.loads(sys.argv[2]):
    try:
        canonicalize_format({'data_type': 'tessera', 'tessera_format': name})
    except ValueError:
        continue
    raise AssertionError('malformed dictionary rung accepted: ' + repr(name))
""", json.dumps(names))


if __name__ == '__main__':
    unittest.main()
