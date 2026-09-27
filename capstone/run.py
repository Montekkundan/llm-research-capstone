"""Run one toy baseline or architecture swap: python -m capstone.run --config ..."""

import argparse
import json
from pathlib import Path
from .core import execute


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = execute(args.config)
    print(json.dumps({"artifact_dir": result["artifact_dir"], "evaluation": result["report"]["evaluation"]}, sort_keys=True))


if __name__ == "__main__":
    main()
