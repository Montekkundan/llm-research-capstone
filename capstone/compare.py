"""Enforce one-variable toy ablations before comparing held-out metrics."""

import argparse
import json
from pathlib import Path
from typing import Any
from .core import execute, load_config


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    result = {}
    for key, child in value.items():
        result.update(flatten(child, f"{prefix}.{key}" if prefix else key))
    return result


def assert_one_variable(base: dict[str, Any], candidate: dict[str, Any], varied_field: str) -> None:
    if varied_field not in {"architecture.variant", "architecture.alpha"}:
        raise ValueError("only architecture.variant or architecture.alpha may be varied in this starter")
    a, b = flatten(base), flatten(candidate)
    differences = {key for key in a.keys() | b.keys() if a.get(key) != b.get(key)} - {"run_id"}
    if differences != {varied_field}:
        raise ValueError(f"expected only {varied_field} to change; changed: {sorted(differences)}")
    if base["run_id"] == candidate["run_id"]:
        raise ValueError("comparison runs need distinct run_id values")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--vary", required=True)
    args = parser.parse_args()
    a, b = load_config(args.base), load_config(args.candidate)
    assert_one_variable(a, b, args.vary)
    base = execute(args.base)
    candidate = execute(args.candidate)
    comparison = {
        "scope": "toy byte-ngram ablation, not an LLM architecture claim",
        "varied_field": args.vary,
        "matched": ["dataset bytes", "document split and seed", "evaluation metric", "smoothing alpha" if args.vary != "architecture.alpha" else "architecture variant"],
        "base_run": base["report"]["run_id"],
        "candidate_run": candidate["report"]["run_id"],
        "base_bits_per_byte": base["report"]["evaluation"]["bits_per_byte"],
        "candidate_bits_per_byte": candidate["report"]["evaluation"]["bits_per_byte"],
        "candidate_minus_base_bits_per_byte": candidate["report"]["evaluation"]["bits_per_byte"] - base["report"]["evaluation"]["bits_per_byte"],
        "uncertainty": "One tiny held-out split; no interval or population-level conclusion.",
    }
    output = Path(candidate["artifact_dir"]) / "comparison.json"
    output.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"comparison": str(output), "delta_bits_per_byte": comparison["candidate_minus_base_bits_per_byte"]}, sort_keys=True))


if __name__ == "__main__":
    main()
