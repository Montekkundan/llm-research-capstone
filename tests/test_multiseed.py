import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from capstone import multiseed
from capstone.multiseed import (T_CRITICAL_95, format_report, mean_and_sd, paired_summary,
                                run_study, t_critical_95)


class PairedStatisticsTests(unittest.TestCase):
    def test_mean_and_sample_sd(self):
        # 1, 2, 3, 4: mean 2.5, squared deviations 2.25 + 0.25 + 0.25 + 2.25 = 5, sd = sqrt(5 / 3).
        mean, sd = mean_and_sd([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(mean, 2.5)
        self.assertAlmostEqual(sd, math.sqrt(5 / 3))
        with self.assertRaises(ValueError):
            mean_and_sd([1.0])

    def test_t_table_spot_values_and_range(self):
        for df, expected in ((1, 12.706), (2, 4.303), (3, 3.182), (7, 2.365), (30, 2.042)):
            self.assertEqual(t_critical_95(df), expected)
        values = [T_CRITICAL_95[df] for df in range(1, 31)]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertGreater(min(values), 1.96)
        for df in (0, 31):
            with self.assertRaises(ValueError):
                t_critical_95(df)

    def test_interval_that_includes_zero(self):
        # Differences -0.1, -0.3, -0.2: mean -0.2, sd 0.1, SE 0.1 / sqrt(3) = 0.0577350,
        # t(df = 2) = 4.303, half-width 0.2484338, interval [-0.4484338, +0.0484338].
        result = paired_summary([4.0, 5.0, 6.0], [3.9, 4.7, 5.8])
        self.assertEqual((result["n"], result["degrees_of_freedom"], result["t_critical_95"]), (3, 2, 4.303))
        self.assertAlmostEqual(result["mean_difference"], -0.2)
        self.assertAlmostEqual(result["sd_difference"], 0.1)
        self.assertAlmostEqual(result["standard_error"], 0.0577350, places=6)
        self.assertAlmostEqual(result["ci95"][0], -0.4484338, places=6)
        self.assertAlmostEqual(result["ci95"][1], 0.0484338, places=6)
        self.assertFalse(result["excludes_zero"])

    def test_interval_that_excludes_zero(self):
        # Differences -0.30, -0.28, -0.32, -0.30: mean -0.30, squared deviations sum 0.0008,
        # sd = sqrt(0.0008 / 3) = 0.0163299, SE = sd / 2 = 0.0081650, t(df = 3) = 3.182,
        # half-width 0.0259810, interval [-0.3259810, -0.2740190].
        result = paired_summary([5.0] * 4, [4.70, 4.72, 4.68, 4.70])
        self.assertAlmostEqual(result["sd_difference"], 0.0163299, places=7)
        self.assertAlmostEqual(result["standard_error"], 0.0081650, places=7)
        self.assertAlmostEqual(result["ci95"][0], -0.3259810, places=6)
        self.assertAlmostEqual(result["ci95"][1], -0.2740190, places=6)
        self.assertTrue(result["excludes_zero"])

    def test_sign_convention_and_degenerate_spread(self):
        # candidate minus baseline: a worse candidate gives a positive mean.
        self.assertGreater(paired_summary([1.0, 1.0], [2.0, 3.0])["mean_difference"], 0)
        constant = paired_summary([1.0, 2.0], [0.0, 1.0])  # both differences -1: zero spread
        self.assertEqual((constant["sd_difference"], constant["ci95"]), (0.0, [-1.0, -1.0]))
        self.assertTrue(constant["excludes_zero"])
        self.assertFalse(paired_summary([1.0, 2.0], [1.0, 2.0])["excludes_zero"])

    def test_documented_eight_seed_study(self):
        # Per-seed held-out bits/byte of `python3 -m capstone.multiseed --seeds 8`, rounded to
        # four places. The README quotes: 1-layer 4.110 (sd 0.018), 2-layer 4.088 (sd 0.025),
        # paired difference -0.022 with SE 0.0066.
        one = [4.1375, 4.0962, 4.0957, 4.1150, 4.0935, 4.1333, 4.0924, 4.1141]
        two = [4.0835, 4.0671, 4.0803, 4.0998, 4.0845, 4.1420, 4.0578, 4.0879]
        result = paired_summary(one, two)
        self.assertAlmostEqual(mean_and_sd(one)[0], 4.110, places=3)
        self.assertAlmostEqual(mean_and_sd(one)[1], 0.018, places=3)
        self.assertAlmostEqual(mean_and_sd(two)[0], 4.088, places=3)
        self.assertAlmostEqual(mean_and_sd(two)[1], 0.025, places=3)
        self.assertAlmostEqual(result["mean_difference"], -0.02185, places=5)
        self.assertAlmostEqual(result["standard_error"], 0.0066, places=4)
        # t(df = 7) = 2.365: -0.02185 -/+ 2.365 * 0.006617 = [-0.03750, -0.00620].
        self.assertAlmostEqual(result["ci95"][0], -0.0375, places=4)
        self.assertAlmostEqual(result["ci95"][1], -0.0062, places=4)
        self.assertTrue(result["excludes_zero"])
        # The single run in the README (seed 7) is one point inside the per-seed spread.
        self.assertAlmostEqual(result["differences"][7], -0.0262, places=4)
        self.assertLess(abs(result["differences"][7] - result["mean_difference"]), result["sd_difference"])

    def test_invalid_inputs(self):
        for baseline, candidate in (([1.0], [2.0]), ([1.0, 2.0], [1.0]), ([], [])):
            with self.assertRaises(ValueError):
                paired_summary(baseline, candidate)


class MultiseedSmokeTests(unittest.TestCase):
    def test_tiny_study_is_paired_and_split_is_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = run_study(seeds=[0, 1], split_seed=7, steps=2, output_root=root)
            self.assertEqual((result["baseline_layers"], result["candidate_layers"], result["steps"]), (1, 2, 2))
            self.assertEqual(len(result["baseline_bits_per_byte"]), 2)
            self.assertEqual(len(result["candidate_bits_per_byte"]), 2)
            self.assertTrue(all(math.isfinite(value) for value in
                                result["baseline_bits_per_byte"] + result["candidate_bits_per_byte"]))
            expected = [after - before for before, after in
                        zip(result["baseline_bits_per_byte"], result["candidate_bits_per_byte"])]
            self.assertEqual(result["differences"], expected)
            self.assertAlmostEqual(result["mean_difference"], sum(expected) / 2)
            self.assertIsInstance(result["excludes_zero"], bool)
            self.assertEqual(json.loads((root / "multiseed.json").read_text())["seeds"], [0, 1])
            text = format_report(result)
            self.assertIn("interval excludes zero: " + ("yes" if result["excludes_zero"] else "no"), text)
            # Same frozen split and run seed in both arms; only depth differs.
            manifests = {name: json.loads((root / name / "manifest.json").read_text())
                         for name in ("multiseed-layers1-seed0", "multiseed-layers2-seed0",
                                      "multiseed-layers1-seed1", "multiseed-layers2-seed1")}
            splits = {json.dumps(m["data"], sort_keys=True) for m in manifests.values()}
            self.assertEqual(len(splits), 1)
            self.assertEqual({m["data"]["split_seed"] for m in manifests.values()}, {7})
            self.assertEqual([m["architecture"]["layers"] for m in manifests.values()], [1, 2, 1, 2])
            configs = [json.loads((root / name / "config.json").read_text()) for name in manifests]
            self.assertEqual([c["seed"] for c in configs], [0, 0, 1, 1])
            self.assertEqual({c["training"]["steps"] for c in configs}, {2})
            # Re-running into the same directory says how to proceed instead of overwriting.
            with self.assertRaisesRegex(FileExistsError, "empty --output-root"):
                run_study(seeds=[0, 1], split_seed=7, steps=2, output_root=root)

    def test_depth_is_the_only_field_that_may_differ(self):
        candidate = json.loads(multiseed.CANDIDATE_CONFIG.read_text())
        candidate["training"]["learning_rate"] = 0.01
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.json"
            path.write_text(json.dumps(candidate))
            with mock.patch.object(multiseed, "CANDIDATE_CONFIG", path):
                with self.assertRaisesRegex(ValueError, "only architecture.layers"):
                    run_study(seeds=[0, 1], steps=1)

    def test_rejects_too_few_or_repeated_seeds(self):
        for seeds in ([0], [3, 3], [-1, 0]):
            with self.assertRaises(ValueError):
                run_study(seeds=seeds, steps=1)


if __name__ == "__main__":
    unittest.main()
