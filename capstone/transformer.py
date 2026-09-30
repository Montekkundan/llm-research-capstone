"""Small, auditable byte-level causal Transformer experiment."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .core import ROOT, canonical, read_and_split, sha256

BOS = 256
PAD = 257
IGNORE = -100


class CausalTransformer(nn.Module):
    def __init__(self, width: int, heads: int, layers: int, context: int):
        super().__init__()
        self.context = context
        self.token = nn.Embedding(258, width)
        self.position = nn.Embedding(context, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=4 * width,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 256, bias=False)

    def forward(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2 or tokens.shape[1] > self.context:
            raise ValueError("tokens must have shape [batch, time <= context]")
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        states = self.token(tokens) + self.position(positions)
        causal = torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device, dtype=torch.bool).triu(1)
        return self.output(self.norm(self.blocks(states, mask=causal)))


class AtlasByteModel(nn.Module):
    def __init__(self, preset: str):
        super().__init__()
        try:
            from atlas import TinyLanguageModel, load_preset
        except ImportError as error:
            raise ImportError("atlas backend requires `pip install -e ../llm-model-atlas`") from error
        self.spec = replace(load_preset(preset), vocab_size=258)
        self.spec.validate()
        self.model = TinyLanguageModel(self.spec)

    def forward(self, tokens: Tensor) -> Tensor:
        # BOS/PAD are input-only; normalizing over them would change bits/byte.
        return self.model(tokens)[..., :256]


def build_model(architecture: dict[str, Any]) -> tuple[nn.Module, dict[str, Any]]:
    if architecture["family"] == "atlas":
        model = AtlasByteModel(architecture["preset"])
        return model, asdict(model.spec)
    model = CausalTransformer(architecture["width"], architecture["heads"],
                              architecture["layers"], architecture["context"])
    return model, architecture


def implementation_hash(architecture: dict[str, Any]) -> str:
    sources = [Path(__file__), Path(__file__).with_name("core.py")]
    if architecture["family"] == "atlas":
        from atlas import model as atlas_model, spec as atlas_spec
        sources.extend((Path(atlas_model.__file__), Path(atlas_spec.__file__)))
    return sha256(canonical([sha256(path.read_bytes()) for path in sources]))


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if set(config) != {"schema_version", "run_id", "seed", "dataset", "architecture", "training", "evaluation"} or config["schema_version"] != 2:
        raise ValueError("expected the exact v2 Transformer run fields")
    if not isinstance(config["run_id"], str) or not config["run_id"].replace("-", "").replace("_", "").isalnum():
        raise ValueError("run_id must use letters, digits, hyphens or underscores")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    dataset = config["dataset"]
    if set(dataset) != {"path", "train_fraction"} or not isinstance(dataset["path"], str) or not isinstance(dataset["train_fraction"], (int, float)) or not 0 < dataset["train_fraction"] < 1:
        raise ValueError("dataset needs path and train_fraction in (0, 1)")
    arch = config["architecture"]
    if arch.get("family") == "atlas":
        if set(arch) != {"family", "preset", "context"} or arch["preset"] not in {
            "olmo2", "gemma3", "mistral_small31", "qwen3_dense", "deepseek_v3_style"
        }:
            raise ValueError("atlas backend needs one of the five named text presets")
    elif arch.get("family") == "causal-byte-transformer":
        if set(arch) != {"family", "width", "heads", "layers", "context"}:
            raise ValueError("causal-byte-transformer needs width, heads, layers and context")
        for key in ("width", "heads", "layers"):
            if type(arch[key]) is not int or arch[key] <= 0:
                raise ValueError(f"architecture.{key} must be a positive integer")
        if arch["width"] % arch["heads"]:
            raise ValueError("width must divide by heads")
    else:
        raise ValueError("architecture.family must be causal-byte-transformer or atlas")
    if type(arch["context"]) is not int or arch["context"] < 2:
        raise ValueError("architecture.context must be at least two")
    training = config["training"]
    if set(training) != {"steps", "batch_size", "learning_rate"}:
        raise ValueError("training needs steps, batch_size and learning_rate")
    if any(type(training[key]) is not int or training[key] <= 0 for key in ("steps", "batch_size")):
        raise ValueError("steps and batch_size must be positive integers")
    if not isinstance(training["learning_rate"], (int, float)) or not math.isfinite(training["learning_rate"]) or training["learning_rate"] <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if config["evaluation"] != {"metric": "bits_per_byte"}:
        raise ValueError("evaluation metric must be bits_per_byte")
    return config


def examples(documents: list[bytes], context: int) -> list[tuple[list[int], list[int]]]:
    # Documents never share a context. Chunk boundaries reset to BOS.
    result = []
    for doc in documents:
        for start in range(0, len(doc), context):
            chunk = doc[start:start + context]
            result.append(([BOS, *chunk[:-1]], list(chunk)))
    return result


def batch(items: list[tuple[list[int], list[int]]]) -> tuple[Tensor, Tensor]:
    length = max(len(inputs) for inputs, _ in items)
    inputs = torch.full((len(items), length), PAD, dtype=torch.long)
    targets = torch.full((len(items), length), IGNORE, dtype=torch.long)
    for row, (source, target) in enumerate(items):
        inputs[row, :len(source)] = torch.tensor(source)
        targets[row, :len(target)] = torch.tensor(target)
    return inputs, targets


@torch.no_grad()
def evaluate(model: nn.Module, items: list[tuple[list[int], list[int]]]) -> dict[str, float | int]:
    model.eval()
    total_nats = 0.0
    count = 0
    for item in items:
        source, target = batch([item])
        logits = model(source)
        total_nats += F.cross_entropy(logits.reshape(-1, 256), target.reshape(-1), reduction="sum").item()
        count += int((target != IGNORE).sum())
    bits = total_nats / count / math.log(2)
    return {"bits_per_byte": bits, "perplexity_per_byte": math.exp(total_nats / count), "bytes": count}


def compare_configs(base: dict[str, Any], candidate: dict[str, Any], varied_field: str) -> None:
    if varied_field not in {"architecture.width", "architecture.heads", "architecture.layers", "architecture.context", "architecture.preset"}:
        raise ValueError("vary one declared architecture dimension")
    from .compare import flatten
    a, b = flatten(base), flatten(candidate)
    changes = {key for key in a.keys() | b.keys() if a.get(key) != b.get(key)} - {"run_id"}
    if changes != {varied_field} or base["run_id"] == candidate["run_id"]:
        raise ValueError(f"expected distinct run_ids and only {varied_field} to change; changed: {sorted(changes)}")


def execute(config_path: Path, *, output_root: Path | None = None, stop_after: int | None = None, resume: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    train_docs, eval_docs, provenance = read_and_split(config)
    context = config["architecture"]["context"]
    train_items, eval_items = examples(train_docs, context), examples(eval_docs, context)
    arch, training = config["architecture"], config["training"]
    steps = training["steps"]
    end = steps if stop_after is None else stop_after
    if type(end) is not int or not 0 <= end <= steps:
        raise ValueError("stop_after must be between zero and configured steps")
    torch.set_num_threads(1)
    torch.manual_seed(config["seed"])
    model, resolved_architecture = build_model(arch)
    if arch["family"] == "atlas" and context > resolved_architecture["max_context"]:
        raise ValueError("configured context exceeds atlas preset max_context")
    architecture_hash = sha256(canonical(resolved_architecture))
    code_hash = implementation_hash(arch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"])
    out = (output_root or ROOT / "artifacts") / config["run_id"]
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "checkpoint.pt"
    config_hash = sha256(canonical(config))
    start = 0
    tokens_seen = 0
    if resume:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (saved["config_sha256"] != config_hash or saved["data_sha256"] != provenance["sha256"]
                or saved["architecture_sha256"] != architecture_hash
                or saved["implementation_sha256"] != code_hash):
            raise ValueError("checkpoint config, dataset, architecture or implementation differs from this run")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start = saved["step"]
        tokens_seen = saved["tokens_seen"]
    elif checkpoint_path.exists():
        raise FileExistsError(f"{checkpoint_path} exists; use --resume or another run_id")
    if end < start:
        raise ValueError("stop_after is earlier than checkpoint step")
    for step in range(start, end):
        generator = torch.Generator().manual_seed(config["seed"] + step)
        indices = torch.randint(len(train_items), (training["batch_size"],), generator=generator).tolist()
        source, target = batch([train_items[i] for i in indices])
        tokens_seen += int((target != IGNORE).sum())
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(source)
        loss = F.cross_entropy(logits.reshape(-1, 256), target.reshape(-1), ignore_index=IGNORE)
        loss.backward()
        optimizer.step()
    save = {"step": end, "tokens_seen": tokens_seen, "config_sha256": config_hash,
            "data_sha256": provenance["sha256"], "architecture_sha256": architecture_hash,
            "implementation_sha256": code_hash,
            "model": model.state_dict(), "optimizer": optimizer.state_dict()}
    pending = out / "checkpoint.tmp"
    torch.save(save, pending)
    pending.replace(checkpoint_path)
    checkpoint_hash = sha256(checkpoint_path.read_bytes())
    manifest = {"schema_version": 2, "run_id": config["run_id"], "step": end, "tokens_seen": tokens_seen,
                "config_sha256": config_hash, "checkpoint_sha256": checkpoint_hash,
                "architecture_sha256": architecture_hash, "implementation_sha256": code_hash,
                "data": provenance, "parameter_count": sum(p.numel() for p in model.parameters()),
                "implementation": "atlas.TinyLanguageModel/v4" if arch["family"] == "atlas" else "capstone.transformer.CausalTransformer/v1",
                "architecture": resolved_architecture, "torch": torch.__version__}
    (out / "config.json").write_bytes(canonical(config))
    (out / "manifest.json").write_bytes(canonical(manifest))
    report = {"run_id": config["run_id"], "step": end, "complete": end == steps,
              "train": evaluate(model, train_items), "held_out": evaluate(model, eval_items),
              "limitations": ["Tiny original corpus: the metrics demonstrate mechanics, not general language quality.",
                              "Byte tokens and independent context chunks; no document packing or full-context evaluation.",
                              "CPU-only teaching decoder; no post-training, GPU profile or distributed path."]}
    (out / "report.json").write_bytes(canonical(report))
    return {"artifact_dir": str(out), "manifest": manifest, "report": report}
