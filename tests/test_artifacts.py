import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.artifacts import json_fingerprint


class ArtifactTests(unittest.TestCase):
    def test_fingerprint_ignores_checkout_formatting_but_detects_data_changes(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "rainfall.json"
            self.assertIsNone(json_fingerprint(path))
            path.write_bytes(b'{\r\n  "rain": 3000,\r\n  "lat": 22.5\r\n}\r\n')
            original = json_fingerprint(path)
            path.write_bytes(b'{"lat":22.5,"rain":3000}\n')
            self.assertEqual(json_fingerprint(path), original)
            path.write_text(json.dumps({"rain": 3100, "lat": 22.5}))
            self.assertNotEqual(json_fingerprint(path), original)
