# Research capstone starter (lectures 90–96)

This is a **CPU-only experiment-contract starter**, not a pretrained model, LLM trainer, GPU benchmark, DeepSeek-V3 reproduction, or assistant. It gives students a small, executable way to practice provenance, held-out evaluation, one-variable ablations, report writing, and architecture-swap discipline before attempting the much larger capstones. The only implemented learners are Laplace-smoothed byte unigram and bigram models on a 20-line original teaching corpus. Do not interpret their scores as language-model research results.

## Run the toy experiment

Python 3.10+ and its standard library are sufficient. From the repository root:

```sh
python3 -m unittest discover -s tests -v
python3 -m capstone.run --config configs/baseline.json
python3 -m capstone.compare --base configs/baseline.json --candidate configs/bigram-swap.json --vary architecture.variant
```

Each run writes `artifacts/<run_id>/checkpoint.json`, `manifest.json`, and `report.json`; the comparison additionally writes `comparison.json` under the candidate run. The manifest records the input SHA-256, document indices, config and checkpoint hashes, implementation label, and Python version. Splitting is by line-document **before** byte contexts are formed. The metric is held-out negative log likelihood in bits per byte; lower is better for this *same* byte corpus and protocol. No confidence interval is estimated.

[`examples/toy-unigram-report.json`](examples/toy-unigram-report.json) and [`examples/toy-ablation-comparison.json`](examples/toy-ablation-comparison.json) are snapshots from the verified local run; regenerate them when changing code or data. Live run artifacts remain in the ignored `artifacts/` directory.

The locally verified example on Python 3.14.7 yields 4.2982 bits/byte for the unigram and 5.2249 for the bigram on 270 held-out bytes (five documents). The bigram's worse score is a useful warning about sparse context counts, not evidence that unigram models are preferable generally. Rerun the commands to obtain evidence for your machine; do not copy these numbers into an LLM comparison.

## What students extend

- [Lesson map](LESSON_MAP.md) links each of 90–96 to a deliverable and a verification gate.
- [Experiment contracts](CONTRACTS.md) define a single-factor ablation and the interface a real architecture adapter must eventually satisfy. These are requirements, **not implemented GPU or transformer modules**.
- `configs/baseline.json` is the v1 run manifest; `configs/bigram-swap.json` changes only `run_id` and `architecture.variant`. The comparator rejects a simultaneous data, split, seed, smoothing, or metric change.
- The bundled corpus is intentionally original and tiny. For a real run, use a licensed dataset, document its provenance and exclusions, split by a leakage-safe unit, and independently evaluate the resulting checkpoint.

The full tokenizer → pretraining → post-training → evaluation → inference path is illustrated in [Andrej Karpathy's nanochat repository](https://github.com/karpathy/nanochat), which is a reference for a *future* student implementation, not a dependency of this starter. [Hoffmann et al., *Training Compute-Optimal Large Language Models* (2022)](https://arxiv.org/abs/2203.15556) motivates reporting model, token, and compute budgets together before a scaling comparison. [Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (2023)](https://arxiv.org/abs/2305.13245) is a primary source for a future attention-family swap. [DeepSeek-AI et al., *DeepSeek-V3 Technical Report* (2024)](https://arxiv.org/abs/2412.19437) is the primary structural source for the optional V3-style capstone. A tiny byte n-gram contains none of GQA, MLA, MoE, SFT, or distributed training.

## Evidence boundary

The tests prove local determinism, document split separation, config rejection, and report generation. They do **not** prove a capable model, equal-compute comparison, GPU correctness, training recovery, SFT quality, or paper fidelity. Each later extension needs its own parity tests, hardware profile, budget ledger, seeds, and disclosed differences from its cited architecture.
