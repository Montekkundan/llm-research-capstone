"""Restore a capstone run and inspect a bounded, greedy byte continuation."""

import argparse
import json
from pathlib import Path

import torch

from .core import canonical, read_and_split, sha256
from .transformer import BOS, build_model, implementation_hash, load_config


def load_run(directory: Path):
    config = load_config(directory / "config.json")
    manifest = json.loads((directory / "manifest.json").read_text())
    checkpoint_path = directory / "checkpoint.pt"
    if manifest["checkpoint_sha256"] != sha256(checkpoint_path.read_bytes()):
        raise ValueError("checkpoint checksum differs from manifest")
    model, architecture = build_model(config["architecture"])
    _, _, provenance = read_and_split(config)
    expected = {
        "config_sha256": sha256(canonical(config)),
        "architecture_sha256": sha256(canonical(architecture)),
        "implementation_sha256": implementation_hash(config["architecture"]),
        "data_sha256": provenance["sha256"],
    }
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    for key, value in expected.items():
        recorded = manifest["data"]["sha256"] if key == "data_sha256" else manifest[key]
        if saved[key] != value or recorded != value:
            raise ValueError(f"restore contract differs: {key}")
    model.load_state_dict(saved["model"], strict=True)
    return model.eval(), config, manifest


@torch.inference_mode()
def generate(model, prompt: str, max_new_bytes: int, context: int) -> dict:
    if type(max_new_bytes) is not int or max_new_bytes < 1:
        raise ValueError("max_new_bytes must be a positive integer")
    ids = [BOS, *prompt.encode("utf-8")]
    if len(ids) + max_new_bytes > context:
        raise ValueError("prompt plus generation budget exceeds context")
    output = bytearray()
    for _ in range(max_new_bytes):
        token = int(model(torch.tensor([ids], dtype=torch.long))[0, -1].argmax())
        output.append(token)
        ids.append(token)
    return {"text": output.decode("utf-8", errors="replace"), "bytes_hex": output.hex(),
            "prompt_bytes": len(ids) - len(output) - 1, "completion_bytes": len(output),
            "finish_reason": "length"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-bytes", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, config, manifest = load_run(args.run)
    result = generate(model, args.prompt, args.max_new_bytes, config["architecture"]["context"])
    print(json.dumps({"run_id": config["run_id"], "checkpoint_sha256": manifest["checkpoint_sha256"],
                      **result}, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
