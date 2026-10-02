"""Paired multi-seed study of the 1-layer versus 2-layer Transformer comparison.

    python3 -m capstone.multiseed            # 3 seeds, a quick look
    python3 -m capstone.multiseed --seeds 8  # 8 seeds: the study quoted in the README

For each seed the two configs from the README (``configs/tiny-transformer.json`` and
``configs/tiny-transformer-depth-swap.json``) are trained with the same seed, the same
frozen document split and the same minibatch sequence, and differ only in
``architecture.layers`` (the one-field guard of ``compare_configs`` is enforced for
every seed). The seed controls initialization and minibatch sampling. The split is held
fixed at ``dataset.split_seed`` (default 7, the split behind the single-run numbers), so
the spread below is seed uncertainty on one five-document held-out set. It says nothing
about other documents, other splits, or compute-matched depth changes.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import tempfile
from pathlib import Path
from typing import Any

from .core import ROOT
from .transformer import compare_configs, execute, load_config

BASE_CONFIG = ROOT / "configs/tiny-transformer.json"
CANDIDATE_CONFIG = ROOT / "configs/tiny-transformer-depth-swap.json"

# Two-sided 95% Student t quantiles, t(0.975, df), for df = 1..30. Hard-coded so the
# study needs no scipy. Larger studies should extend the table or use a statistics package.
T_CRITICAL_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
    23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
    30: 2.042,
}


def t_critical_95(degrees_of_freedom: int) -> float:
    if degrees_of_freedom not in T_CRITICAL_95:
        raise ValueError("the built-in t table covers 1 to 30 degrees of freedom (2 to 31 seeds)")
    return T_CRITICAL_95[degrees_of_freedom]


def mean_and_sd(values: list[float]) -> tuple[float, float]:
    """Mean and sample standard deviation sqrt(sum((x - mean)^2) / (n - 1))."""
    if len(values) < 2:
        raise ValueError("a standard deviation needs at least two values")
    mean = sum(values) / len(values)
    return mean, math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def paired_summary(baseline: list[float], candidate: list[float]) -> dict[str, Any]:
    """Summarize d_i = candidate_i - baseline_i over n paired runs.

        mean = sum(d) / n
        sd   = sqrt(sum((d - mean)^2) / (n - 1))
        SE   = sd / sqrt(n)
        95% interval = mean +/- t(0.975, n - 1) * SE

    A negative difference means the candidate has the lower loss. The interval excludes
    zero when it lies strictly on one side of it.
    """
    if len(baseline) != len(candidate):
        raise ValueError("paired runs need equal-length score lists")
    count = len(baseline)
    if count < 2:
        raise ValueError("an interval needs at least two paired seeds")
    differences = [after - before for before, after in zip(baseline, candidate)]
    mean, sd = mean_and_sd(differences)
    standard_error = sd / math.sqrt(count)
    t_value = t_critical_95(count - 1)
    low, high = mean - t_value * standard_error, mean + t_value * standard_error
    return {"n": count, "differences": differences, "mean_difference": mean,
            "sd_difference": sd, "standard_error": standard_error,
            "degrees_of_freedom": count - 1, "t_critical_95": t_value,
            "ci95": [low, high], "excludes_zero": bool(low > 0 or high < 0)}


def _seeded(config: dict[str, Any], *, seed: int, split_seed: int, steps: int | None) -> dict[str, Any]:
    config = copy.deepcopy(config)
    config["seed"] = seed
    config["dataset"]["split_seed"] = split_seed
    if steps is not None:
        config["training"]["steps"] = steps
    config["run_id"] = f"multiseed-layers{config['architecture']['layers']}-seed{seed}"
    return config


def run_study(*, seeds: list[int], split_seed: int = 7, steps: int | None = None,
              output_root: Path | None = None) -> dict[str, Any]:
    """Train both arms for every seed and return per-seed scores plus the paired summary."""
    if len(seeds) != len(set(seeds)) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    if len(seeds) < 2:
        raise ValueError("an interval needs at least two paired seeds")
    base, candidate = load_config(BASE_CONFIG), load_config(CANDIDATE_CONFIG)
    arms = [(_seeded(base, seed=seed, split_seed=split_seed, steps=steps),
             _seeded(candidate, seed=seed, split_seed=split_seed, steps=steps)) for seed in seeds]
    for one, two in arms:
        compare_configs(one, two, "architecture.layers")
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch) if output_root is None else Path(output_root)
        taken = [config["run_id"] for pair in arms for config in pair
                 if (root / config["run_id"] / "checkpoint.pt").exists()]
        if taken:
            raise FileExistsError(f"{root} already holds {taken[0]} from an earlier study; "
                                  "pass an empty --output-root, or omit --output-root to use a temporary directory")
        (root / "configs").mkdir(parents=True, exist_ok=True)
        scores: list[list[float]] = [[], []]
        for pair in arms:
            for index, config in enumerate(pair):
                path = root / "configs" / f"{config['run_id']}.json"
                path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
                result = execute(path, output_root=root)
                scores[index].append(result["report"]["held_out"]["bits_per_byte"])
        summary = paired_summary(*scores)
        (baseline_mean, baseline_sd), (candidate_mean, candidate_sd) = map(mean_and_sd, scores)
        result = {
            "baseline_layers": base["architecture"]["layers"],
            "candidate_layers": candidate["architecture"]["layers"],
            "seeds": seeds, "split_seed": split_seed,
            "steps": arms[0][0]["training"]["steps"],
            "baseline_bits_per_byte": scores[0], "candidate_bits_per_byte": scores[1],
            "baseline_mean": baseline_mean, "baseline_sd": baseline_sd,
            "candidate_mean": candidate_mean, "candidate_sd": candidate_sd,
            **summary,
            "caveats": ["Seeds change initialization and minibatch order; the five-document held-out set is fixed.",
                        "The 2-layer model has more parameters, so this is not a compute-matched comparison.",
                        "An interval that excludes zero here is a statement about this split, not about architectures."],
        }
        if output_root is not None:
            (root / "multiseed.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return result


def format_report(result: dict[str, Any]) -> str:
    low, high = result["ci95"]
    lines = [f"{result['baseline_layers']}-layer vs {result['candidate_layers']}-layer, "
             f"{result['steps']} steps, split_seed {result['split_seed']}, held-out bits per byte"]
    for seed, before, after, difference in zip(result["seeds"], result["baseline_bits_per_byte"],
                                               result["candidate_bits_per_byte"], result["differences"]):
        lines.append(f"seed {seed}: {before:.4f} -> {after:.4f}  difference {difference:+.4f}")
    lines += [
        f"{result['baseline_layers']}-layer mean {result['baseline_mean']:.4f} (sd {result['baseline_sd']:.4f}), "
        f"{result['candidate_layers']}-layer mean {result['candidate_mean']:.4f} (sd {result['candidate_sd']:.4f})",
        f"mean paired difference ({result['candidate_layers']}-layer minus {result['baseline_layers']}-layer): "
        f"{result['mean_difference']:+.4f}",
        f"standard error: {result['standard_error']:.4f} (sd of differences {result['sd_difference']:.4f}, n = {result['n']})",
        f"95% t interval (df = {result['degrees_of_freedom']}, t = {result['t_critical_95']}): "
        f"[{low:+.4f}, {high:+.4f}]",
        "interval excludes zero: " + ("yes" if result["excludes_zero"] else "no"),
        *("note: " + caveat for caveat in result["caveats"]),
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=3,
                        help="number of seeds, 0..N-1 (default 3; use 8 for the README study)")
    parser.add_argument("--split-seed", type=int, default=7, help="frozen document split (default 7)")
    parser.add_argument("--steps", type=int, help="override training steps for both arms (default: config value)")
    parser.add_argument("--output-root", type=Path,
                        help="keep configs, checkpoints and multiseed.json here (must not hold an earlier study)")
    parser.add_argument("--json", action="store_true", help="print the result as JSON instead of text")
    args = parser.parse_args()
    result = run_study(seeds=list(range(args.seeds)), split_seed=args.split_seed,
                       steps=args.steps, output_root=args.output_root)
    print(json.dumps(result, sort_keys=True) if args.json else format_report(result))


if __name__ == "__main__":
    main()
