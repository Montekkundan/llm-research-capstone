"""Train or resume the tiny causal Transformer capstone."""

import argparse
import json
from pathlib import Path

from .transformer import compare_configs, execute, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compare-config", type=Path)
    parser.add_argument("--vary")
    args = parser.parse_args()
    if args.compare_config and (args.resume or args.stop_after is not None or not args.vary):
        parser.error("comparison requires --vary and does not accept --resume/--stop-after")
    if args.vary and not args.compare_config:
        parser.error("--vary requires --compare-config")
    if args.compare_config:
        compare_configs(load_config(args.config), load_config(args.compare_config), args.vary)
    base = execute(args.config, output_root=args.output_root, stop_after=args.stop_after, resume=args.resume)
    print(json.dumps({"artifact_dir": base["artifact_dir"], "held_out": base["report"]["held_out"], "step": base["report"]["step"]}, sort_keys=True))
    if args.compare_config:
        candidate = execute(args.compare_config, output_root=args.output_root)
        comparison = {"varied_field": args.vary, "base_manifest": base["manifest"],
                      "candidate_manifest": candidate["manifest"],
                      "delta_bits_per_byte": candidate["report"]["held_out"]["bits_per_byte"] - base["report"]["held_out"]["bits_per_byte"],
                      "limitations": "One seed and tiny corpus; a one-field config change does not equalize parameter count or compute."}
        (Path(candidate["artifact_dir"]) / "comparison.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"comparison": str(Path(candidate["artifact_dir"]) / "comparison.json"), "delta_bits_per_byte": comparison["delta_bits_per_byte"]}, sort_keys=True))


if __name__ == "__main__":
    main()
