import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from capstone.core import ROOT, read_and_split
from capstone.transformer import CausalTransformer, compare_configs, evaluate, examples, execute, load_config


class TransformerExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_path = ROOT / "configs/tiny-transformer.json"
        cls.swap_path = ROOT / "configs/tiny-transformer-depth-swap.json"

    def test_causal_mask_blocks_future_tokens(self):
        torch.manual_seed(3)
        model = CausalTransformer(16, 2, 1, 8).eval()
        first = torch.tensor([[256, 10, 11, 12]])
        second = torch.tensor([[256, 10, 99, 98]])
        with torch.no_grad():
            original, changed = model(first), model(second)
        torch.testing.assert_close(original[:, :2], changed[:, :2], rtol=0, atol=0)
        self.assertFalse(torch.equal(original[:, 2:], changed[:, 2:]))

    def test_each_byte_is_evaluated_once_and_documents_do_not_mix(self):
        config = load_config(self.config_path)
        train, heldout, provenance = read_and_split(config)
        self.assertFalse(set(provenance["train_document_indices"]) & set(provenance["eval_document_indices"]))
        items = examples(heldout, config["architecture"]["context"])
        self.assertTrue(all(source[0] == 256 and len(source) == len(target) for source, target in items))
        self.assertEqual(sum(len(target) for _, target in items), sum(map(len, heldout)))
        self.assertNotEqual(train, heldout)

    def test_resume_matches_uninterrupted_weights_and_metrics(self):
        with tempfile.TemporaryDirectory() as whole_root, tempfile.TemporaryDirectory() as resumed_root:
            whole = execute(self.config_path, output_root=Path(whole_root))
            partial = execute(self.config_path, output_root=Path(resumed_root), stop_after=13)
            self.assertFalse(partial["report"]["complete"])
            resumed = execute(self.config_path, output_root=Path(resumed_root), resume=True)
            self.assertEqual(whole["report"], resumed["report"])
            a = torch.load(Path(whole["artifact_dir"]) / "checkpoint.pt", weights_only=True)
            b = torch.load(Path(resumed["artifact_dir"]) / "checkpoint.pt", weights_only=True)
            self.assertEqual(a["step"], b["step"])
            self.assertEqual(a["tokens_seen"], b["tokens_seen"])
            for name in a["model"]:
                torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)
            self.assertEqual(a["config_sha256"], whole["manifest"]["config_sha256"])
            self.assertEqual(whole["manifest"]["tokens_seen"], resumed["manifest"]["tokens_seen"])
            self.assertTrue(math.isfinite(resumed["report"]["held_out"]["bits_per_byte"]))
            self.assertEqual(resumed["report"]["held_out"]["bytes"], 270)

    def test_training_improves_this_toy_heldout_split(self):
        with tempfile.TemporaryDirectory() as initial_root, tempfile.TemporaryDirectory() as trained_root:
            initial = execute(self.config_path, output_root=Path(initial_root), stop_after=0)
            trained = execute(self.config_path, output_root=Path(trained_root))
            self.assertLess(trained["report"]["held_out"]["bits_per_byte"], initial["report"]["held_out"]["bits_per_byte"])
            self.assertGreater(trained["manifest"]["tokens_seen"], 0)

    def test_checkpoint_rejects_changed_config_and_data(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            execute(self.config_path, output_root=root_path, stop_after=2)
            with self.assertRaises(FileExistsError):
                execute(self.config_path, output_root=root_path)
            changed = load_config(self.config_path)
            changed["training"]["learning_rate"] = 0.01
            config_path = root_path / "changed.json"
            config_path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "config, dataset, architecture or implementation differs"):
                execute(config_path, output_root=root_path, resume=True)
            checkpoint_path = root_path / "tiny-transformer/checkpoint.pt"
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            checkpoint["data_sha256"] = "0" * 64
            torch.save(checkpoint, checkpoint_path)
            with self.assertRaisesRegex(ValueError, "config, dataset, architecture or implementation differs"):
                execute(self.config_path, output_root=root_path, resume=True)

    def test_one_variable_architecture_comparison_guard(self):
        base, swap = load_config(self.config_path), load_config(self.swap_path)
        compare_configs(base, swap, "architecture.layers")
        bad = copy.deepcopy(swap)
        bad["training"]["steps"] += 1
        with self.assertRaisesRegex(ValueError, "only architecture.layers"):
            compare_configs(base, bad, "architecture.layers")

    def test_evaluation_is_finite_and_does_not_mutate_weights(self):
        model = CausalTransformer(16, 2, 1, 8)
        state = {key: value.clone() for key, value in model.state_dict().items()}
        result = evaluate(model, examples([b"abc", b"defghi"], 8))
        self.assertEqual(result["bytes"], 9)
        self.assertGreater(result["bits_per_byte"], 0)
        for key, value in state.items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
