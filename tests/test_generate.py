import json
import tempfile
import unittest
from pathlib import Path

from capstone.core import ROOT
from capstone.generate import generate, load_run
from capstone.transformer import execute


class InferenceArtifactTests(unittest.TestCase):
    def test_reload_generation_and_context_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            result = execute(ROOT / "configs/tiny-transformer.json", output_root=Path(directory))
            path = Path(result["artifact_dir"])
            model, config, manifest = load_run(path)
            first = generate(model, "a", 8, config["architecture"]["context"])
            restored, _, _ = load_run(path)
            self.assertEqual(first, generate(restored, "a", 8, 32))
            self.assertEqual(first["completion_bytes"], 8)
            self.assertEqual(len(bytes.fromhex(first["bytes_hex"])), 8)
            self.assertEqual(manifest["step"], 40)
            with self.assertRaisesRegex(ValueError, "exceeds context"):
                generate(model, "☃" * 8, 8, 32)

    def test_checkpoint_and_manifest_contract_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            result = execute(ROOT / "configs/tiny-transformer.json", output_root=Path(directory))
            path = Path(result["artifact_dir"])
            checkpoint = path / "checkpoint.pt"
            original = checkpoint.read_bytes()
            checkpoint.write_bytes(original + b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_run(path)
            checkpoint.write_bytes(original)
            manifest = json.loads((path / "manifest.json").read_text())
            manifest["implementation_sha256"] = "0" * 64
            (path / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "implementation_sha256"):
                load_run(path)


if __name__ == "__main__":
    unittest.main()
