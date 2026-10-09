"""Exercise artifact integrity against tampering and unsafe manifest paths."""
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("validate", Path(__file__).parents[1] / "scripts/validate.py")
validate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validate)


class IntegrityTests(unittest.TestCase):
    def test_detects_corrupted_artifact(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            file = root / "weights"
            file.write_bytes(b"original")
            hashes = {"weights": hashlib.sha256(b"original").hexdigest()}
            validate.check_hashes(root, hashes)
            file.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                validate.check_hashes(root, hashes)

    def test_rejects_manifest_escape(self):
        with self.assertRaisesRegex(ValueError, "Unsafe filename"):
            validate.check_hashes(Path("."), {"../outside": "a"})

    def test_rejects_empty_manifest(self):
        with self.assertRaisesRegex(ValueError, "Missing checksums"):
            validate.check_hashes(Path("."), {})
