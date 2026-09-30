import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from capstone.compare import assert_one_variable
from capstone.core import ROOT, ByteNgram, evaluate, execute, load_config, read_and_split


class CapstoneContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_path = ROOT / "configs/baseline.json"
        cls.swap_path = ROOT / "configs/bigram-swap.json"
        cls.base = load_config(cls.base_path)
        cls.swap = load_config(cls.swap_path)

    def test_document_split_is_reproducible_and_disjoint(self):
        train_a, eval_a, prov_a = read_and_split(self.base)
        train_b, eval_b, prov_b = read_and_split(self.base)
        self.assertEqual((train_a, eval_a, prov_a), (train_b, eval_b, prov_b))
        self.assertFalse(set(prov_a["train_document_indices"]) & set(prov_a["eval_document_indices"]))
        self.assertEqual(len(train_a) + len(eval_a), prov_a["document_count"])
        train_hashes = {prov_a["document_sha256"][i] for i in prov_a["train_document_indices"]}
        eval_hashes = {prov_a["document_sha256"][i] for i in prov_a["eval_document_indices"]}
        self.assertFalse(train_hashes & eval_hashes)

    def test_duplicate_content_is_rejected_before_split(self):
        config = json.loads(json.dumps(self.base))
        config["dataset"]["path"] = "documents.txt"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "documents.txt").write_bytes(b"same text\nother text\nthird text\nsame text\n")
            with patch("capstone.core.ROOT", root), self.assertRaisesRegex(ValueError, "duplicate document bytes"):
                read_and_split(config)

    def test_repeat_run_is_byte_identical_and_metric_finite(self):
        with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
            first = execute(self.base_path, output_root=Path(one))
            second = execute(self.base_path, output_root=Path(two))
            for filename in ("checkpoint.json", "manifest.json", "report.json"):
                self.assertEqual(
                    (Path(first["artifact_dir"]) / filename).read_bytes(),
                    (Path(second["artifact_dir"]) / filename).read_bytes(),
                )
            metric = first["report"]["evaluation"]["bits_per_byte"]
            self.assertTrue(math.isfinite(metric) and metric > 0)

    def test_swap_changes_only_one_scientific_field(self):
        assert_one_variable(self.base, self.swap, "architecture.variant")
        bad = json.loads(json.dumps(self.swap))
        bad["seed"] = 8
        with self.assertRaisesRegex(ValueError, "changed"):
            assert_one_variable(self.base, bad, "architecture.variant")

    def test_unknown_family_is_rejected(self):
        altered = json.loads(json.dumps(self.base))
        altered["architecture"]["family"] = "transformer"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "unsupported.json"
            path.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "only the byte-ngram"):
                load_config(path)

    def test_heldout_eval_uses_no_training_documents(self):
        _, heldout, provenance = read_and_split(self.base)
        source = (ROOT / provenance["path"]).read_bytes().splitlines()
        self.assertEqual(heldout, [source[i] for i in provenance["eval_document_indices"]])

    def test_evaluation_does_not_change_checkpoint_state(self):
        model = ByteNgram("bigram", 0.5)
        model.fit([b"aa"])
        before = model.state()
        evaluate(model, [b"zx"])
        self.assertEqual(model.state(), before)


if __name__ == "__main__":
    unittest.main()
