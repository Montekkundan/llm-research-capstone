import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

import torch

from capstone.core import ROOT
from capstone.transformer import AtlasByteModel, compare_configs, execute, load_config


@unittest.skipUnless(importlib.util.find_spec("atlas"), "install the optional llm-model-atlas package")
class AtlasBackendTests(unittest.TestCase):
    def test_all_five_presets_accept_byte_inputs_and_update(self):
        for preset in ("olmo2", "gemma3", "mistral_small31", "qwen3_dense", "deepseek_v3_style"):
            with self.subTest(preset=preset):
                model = AtlasByteModel(preset)
                output = model(torch.tensor([[256, 1, 2]]))
                self.assertEqual(output.shape, (1, 3, 256))
                output.mean().backward()
                self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_dense_and_moe_use_same_runner_and_resume_exactly(self):
        for name in ("atlas-qwen3.json", "atlas-deepseek-v3.json"):
            with self.subTest(config=name), tempfile.TemporaryDirectory() as direct_root, tempfile.TemporaryDirectory() as resume_root:
                path = ROOT / "configs" / name
                direct = execute(path, output_root=Path(direct_root))
                partial = execute(path, output_root=Path(resume_root), stop_after=3)
                self.assertFalse(partial["report"]["complete"])
                resumed = execute(path, output_root=Path(resume_root), resume=True)
                self.assertEqual(direct["report"], resumed["report"])
                self.assertEqual(direct["manifest"]["architecture_sha256"], resumed["manifest"]["architecture_sha256"])
                self.assertEqual(direct["manifest"]["tokens_seen"], resumed["manifest"]["tokens_seen"])
                self.assertGreater(direct["manifest"]["parameter_count"], 0)
                self.assertEqual(direct["report"]["held_out"]["bytes"], 270)
                self.assertTrue(math.isfinite(direct["report"]["held_out"]["bits_per_byte"]))
                a = torch.load(Path(direct["artifact_dir"]) / "checkpoint.pt", weights_only=True)
                b = torch.load(Path(resumed["artifact_dir"]) / "checkpoint.pt", weights_only=True)
                for key, tensor in a["model"].items():
                    torch.testing.assert_close(tensor, b["model"][key], atol=0, rtol=0)
                checkpoint_path = Path(resumed["artifact_dir"]) / "checkpoint.pt"
                b["implementation_sha256"] = "0" * 64
                torch.save(b, checkpoint_path)
                with self.assertRaisesRegex(ValueError, "implementation differs"):
                    execute(path, output_root=Path(resume_root), resume=True)

    def test_preset_is_a_single_declared_comparison_variable(self):
        dense = load_config(ROOT / "configs/atlas-qwen3.json")
        moe = load_config(ROOT / "configs/atlas-deepseek-v3.json")
        compare_configs(dense, moe, "architecture.preset")


if __name__ == "__main__":
    unittest.main()
