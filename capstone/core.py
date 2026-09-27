"""Deterministic byte-ngram *toy* experiment, not an LLM trainer."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
VOCAB_SIZE = 256
BOS = 256


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    required = {"schema_version", "run_id", "seed", "dataset", "architecture", "evaluation"}
    if set(config) != required or config["schema_version"] != 1:
        raise ValueError("config must have exactly the v1 manifest fields")
    if not isinstance(config["seed"], int) or isinstance(config["seed"], bool):
        raise ValueError("seed must be an integer")
    if not isinstance(config["run_id"], str) or not config["run_id"].replace("-", "").replace("_", "").isalnum():
        raise ValueError("run_id must use letters, numbers, hyphens, or underscores")
    dataset, architecture, evaluation = config["dataset"], config["architecture"], config["evaluation"]
    if set(dataset) != {"path", "train_fraction"} or not 0 < dataset["train_fraction"] < 1:
        raise ValueError("dataset requires path and train_fraction between zero and one")
    if set(architecture) != {"family", "variant", "alpha"} or architecture["family"] != "byte-ngram":
        raise ValueError("only the byte-ngram teaching family is implemented")
    if architecture["variant"] not in {"unigram", "bigram"} or not isinstance(architecture["alpha"], (int, float)) or architecture["alpha"] <= 0:
        raise ValueError("variant must be unigram/bigram and alpha positive")
    if evaluation != {"metric": "bits_per_byte"}:
        raise ValueError("only bits_per_byte evaluation is implemented")
    return config


def read_and_split(config: dict[str, Any]) -> tuple[list[bytes], list[bytes], dict[str, Any]]:
    raw_path = ROOT / config["dataset"]["path"]
    if not raw_path.is_file() or ROOT not in raw_path.resolve().parents:
        raise ValueError("dataset must be an existing file within this project")
    raw = raw_path.read_bytes()
    # Each nonempty line is an independent document. Split documents before any context construction.
    docs = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(docs) < 4 or any(len(doc) == 0 for doc in docs):
        raise ValueError("at least four nonempty documents are required")
    indices = list(range(len(docs)))
    random.Random(config["seed"]).shuffle(indices)
    train_count = max(1, min(len(docs) - 1, int(len(docs) * config["dataset"]["train_fraction"])))
    train_ids, eval_ids = indices[:train_count], indices[train_count:]
    provenance = {
        "path": config["dataset"]["path"],
        "sha256": sha256(raw),
        "document_count": len(docs),
        "train_document_indices": train_ids,
        "eval_document_indices": eval_ids,
        "split_unit": "line-document",
    }
    return [docs[i] for i in train_ids], [docs[i] for i in eval_ids], provenance


class ByteNgram:
    def __init__(self, variant: str, alpha: float):
        self.variant = variant
        self.alpha = alpha
        self.counts: dict[int, Counter[int]] = defaultdict(Counter)

    def context(self, previous: int) -> int:
        return BOS if self.variant == "unigram" else previous

    def fit(self, documents: list[bytes]) -> None:
        for doc in documents:
            previous = BOS
            for byte in doc:
                self.counts[self.context(previous)][byte] += 1
                previous = byte

    def probability(self, previous: int, byte: int) -> float:
        # Evaluation must not create a new row in the checkpoint for an unseen context.
        row = self.counts.get(self.context(previous), Counter())
        return (row[byte] + self.alpha) / (sum(row.values()) + self.alpha * VOCAB_SIZE)

    def state(self) -> dict[str, Any]:
        return {
            "family": "byte-ngram",
            "variant": self.variant,
            "alpha": self.alpha,
            "counts": {str(context): {str(byte): n for byte, n in sorted(row.items())}
                       for context, row in sorted(self.counts.items())},
        }


def evaluate(model: ByteNgram, documents: list[bytes]) -> dict[str, Any]:
    loss = 0.0
    n = 0
    for doc in documents:
        previous = BOS
        for byte in doc:
            loss -= math.log2(model.probability(previous, byte))
            previous = byte
            n += 1
    return {"bits_per_byte": loss / n, "eval_bytes": n, "eval_documents": len(documents)}


def execute(config_path: Path, *, output_root: Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    train, heldout, data = read_and_split(config)
    model = ByteNgram(config["architecture"]["variant"], config["architecture"]["alpha"])
    model.fit(train)
    metrics = evaluate(model, heldout)
    out = (output_root or ROOT / "artifacts") / config["run_id"]
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = canonical(model.state())
    (out / "checkpoint.json").write_bytes(checkpoint)
    manifest = {
        "schema_version": 1,
        "scope": "standard-library byte-ngram mechanics demo; no LLM or GPU training",
        "config_sha256": sha256(canonical(config)),
        "data": data,
        "checkpoint_sha256": sha256(checkpoint),
        "python": platform.python_version(),
        "implementation": "capstone.core.ByteNgram/v1",
    }
    (out / "manifest.json").write_bytes(canonical(manifest))
    report = {
        "run_id": config["run_id"],
        "architecture": config["architecture"],
        "evaluation": metrics,
        "evidence": "held-out line-documents from the bundled original toy corpus",
        "limitations": [
            "The dataset is tiny and authored for a mechanics check.",
            "This model is a smoothed byte n-gram, not a transformer or capable LLM.",
            "No GPU throughput, trained assistant quality, or scaling claim is measured.",
        ],
    }
    (out / "report.json").write_bytes(canonical(report))
    return {"report": report, "manifest": manifest, "artifact_dir": str(out)}
