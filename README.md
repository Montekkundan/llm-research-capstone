# Research capstone experiments (lectures 90–96)

This repository pairs a standard-library byte n-gram baseline with **tiny trained causal models**. The CPU trainer can use its built-in dense Transformer or any of five optional [model-atlas](https://github.com/Montekkundan/llm-model-atlas) text-path presets under the same byte data, optimizer, checkpoint/resume, and held-out evaluation loop. These are executable teaching experiments on an original 20-line corpus, not pretrained LLMs, GPU benchmarks, published-model reproductions, or assistants. Code covers parts of lectures 90, 92, 94 and 95 only; 91, 93 and 96 are protocols with no code (see [Which lectures have code](#which-lectures-have-code)).

## Train and compare the Transformer

Use Python 3.10+ and install the PyTorch version in `requirements.txt` in a virtual environment (verified with 2.9.1). From the repository root:

```sh
python3 -m unittest discover -s tests -v
python3 -m capstone.transformer_run --config configs/tiny-transformer.json
python3 -m capstone.transformer_run --config configs/tiny-transformer-depth-swap.json
python3 -m capstone.multiseed --seeds 8
```

To make an interrupted run and resume it with the **same config**:

```sh
python3 -m capstone.transformer_run --config configs/tiny-transformer.json --output-root /tmp/capstone-resume --stop-after 13
python3 -m capstone.transformer_run --config configs/tiny-transformer.json --output-root /tmp/capstone-resume --resume
```

To enforce and run the one-field depth comparison in a fresh output directory:

```sh
python3 -m capstone.transformer_run --config configs/tiny-transformer.json --compare-config configs/tiny-transformer-depth-swap.json --vary architecture.layers --output-root /tmp/capstone-depth-study
```

The runner refuses to overwrite an existing checkpoint without `--resume`; the error says to pass `--resume` or use another `run_id`, and `--output-root` writes to a fresh directory. (The n-gram baseline below rewrites its deterministic artifacts instead of refusing.) Each run writes `config.json`, `checkpoint.pt`, `manifest.json`, and `report.json`; a comparison also writes `comparison.json`. The manifest records the raw corpus hash, per-document hashes, disjoint document indices, config, implementation and checkpoint hashes, step, training tokens seen, parameter count, and PyTorch version. Identical normalized line-documents are rejected before splitting; this is exact deduplication, not a near-duplicate detector. The 32-byte chunks reset attention context to BOS at every document and chunk boundary; each byte is scored once. `held_out.bits_per_byte` is the summed next-byte cross-entropy divided by held-out bytes and by ln(2). `perplexity_per_byte = 2 ** bits_per_byte`.

Inspect a continuation from the saved run rather than an untrained model:

```sh
python3 -m capstone.generate --run artifacts/tiny-transformer --prompt a --max-new-bytes 8
```

The loader checks checkpoint bytes, configuration, corpus, resolved architecture and implementation before strict state-dictionary loading. Keep the training corpus available for this research-run check. Generation recomputes the prefix, emits only byte IDs, and ends at its fixed byte budget: there is no EOS token or inference cache. `bytes_hex` preserves raw output even when the display uses UTF-8 replacement characters. This capstone's 258 input IDs and 256 output classes differ from the PicoLLM API's 259-token release schema; it cannot be handed to that service without a separate adapter and contract. Source-hash changes deliberately invalidate older research checkpoints: start a fresh run after updating source, rather than changing its manifest to bypass the check.

The example 40-step local run on PyTorch 2.9.1 gave 4.1141 held-out bits/byte across 270 bytes for one layer and 4.0879 for two layers. This is one seed, one tiny split, and a parameter-count-changing ablation: its 0.0262-bit difference is **not** a general architecture finding. Training loss and a real evaluation set would need many more independent documents and seeds. The n-gram baseline has a different context protocol; its score is not a matched-budget Transformer comparison. That single pair is seed 7 of the seed study in the next section.

For a same-data seed study, set `dataset.split_seed` explicitly, for example `"dataset": {"path": "data/original_toy_corpus.txt", "train_fraction": 0.75, "split_seed": 7}`. Hold that value fixed while varying top-level `seed` and `run_id`; the run seed controls model initialization and minibatch sampling. Both schemas accept a nonnegative integer split seed, and manifests record its resolved value. Omitting it keeps the original behavior of splitting with the run seed. Tests verify identical document assignments and held-out target counts across two different initializations, then resume both for two updates. This is a reproducibility check, not evidence of a multi-seed architecture benefit.

### Quantify seed noise for the depth comparison

`python3 -m capstone.multiseed` trains the 1-layer and 2-layer configs above for each seed, with the same seed, the same frozen split (`split_seed` 7, the split behind the numbers above) and the same minibatches in both arms, so only `architecture.layers` differs (the one-field guard is enforced per seed). It prints each seed's held-out bits/byte, the mean paired difference (2-layer minus 1-layer), its standard error (sd of the paired differences divided by sqrt(n)), a 95% interval using a built-in Student t table, and whether that interval excludes zero. The default is 3 seeds (about 7 seconds on CPU); `--seeds 8` runs seeds 0 to 7, `--steps` shortens training for a smoke run, and `--output-root` keeps the runs (it must be empty).

An earlier 8-seed study gave 1-layer 4.110 (sd 0.018), 2-layer 4.088 (sd 0.025) and a paired difference of -0.022 with standard error 0.0066, and `--seeds 8` reproduces it: mean difference -0.0219, interval [-0.0375, -0.0062], excludes zero. The README's single-run -0.0262 is seed 7 of those eight. The per-seed differences run from +0.009 to -0.054 (sd 0.019), so one run cannot separate a 0.026 gain from seed noise; the single-run delta is within seed-scale noise. With the default 3 seeds the mean difference is -0.033 and the interval [-0.081, +0.016] includes zero. Even an interval that excludes zero measures seed uncertainty on this one five-document held-out split and a parameter-count-changing change, so it is not a general architecture finding. The statistics have hand-computed tests in `tests/test_multiseed.py`.

### Build corpus autocomplete and detect overfitting

Build a tiny sentence autocompleter, then use held-out documents to check whether readable output generalizes. `configs/autocomplete.json` fixes the original corpus, seed, 64-byte context, and total budget of **400 updates**. Start in a fresh output directory. Pausing at 40 saves a checkpoint; `--resume` restores model and AdamW state and completes the remaining 360 updates under the same config.

```sh
python3 -m capstone.transformer_run --config configs/autocomplete.json --output-root runs/autocomplete-demo --stop-after 40
cat runs/autocomplete-demo/autocomplete/report.json
python3 -m capstone.generate --run runs/autocomplete-demo/autocomplete --prompt "attention " --max-new-bytes 41
python3 -m capstone.transformer_run --config configs/autocomplete.json --output-root runs/autocomplete-demo --resume
cat runs/autocomplete-demo/autocomplete/report.json
python3 -m capstone.generate --run runs/autocomplete-demo/autocomplete --prompt "attention " --max-new-bytes 41
python3 -m capstone.generate --run runs/autocomplete-demo/autocomplete --prompt "the model " --max-new-bytes 37
```

The verified CPU run on PyTorch 2.9.1 scored:

| Updates | Training bits/byte | Held-out bits/byte |
| ---: | ---: | ---: |
| 40 | 3.9392 | 4.0576 |
| 400 | 0.0906 | 10.7554 |

At 400, `attention ` continues with `joins a query with stored keys and values`, exactly reproducing a training sentence. The held-out prefix `the model ` produces `aller sa  hairema t l r an ecel tue a` instead of its reference continuation. Readability improved through memorization while held-out loss worsened. The fixed 41- and 37-byte budgets use known reference lengths; generation has no EOS. This corpus-autocomplete lab demonstrates overfitting and checkpoint recovery on one tiny split, with no assistant or general language-quality claim. Repeatedly inspected held-out documents serve as validation; selecting a budget needs a separate untouched test set for a final generalization estimate.

## Swap model families without changing the trainer

Clone `llm-model-atlas` next to this repository and install it into the same virtual environment:

```sh
pip install -e ../llm-model-atlas
python3 -m unittest discover -s tests -v
python3 -m capstone.transformer_run --config configs/atlas-qwen3.json
python3 -m capstone.transformer_run --config configs/atlas-deepseek-v3.json
```

Set `architecture.preset` to `olmo2`, `gemma3`, `mistral_small31`, `qwen3_dense`, or `deepseek_v3_style` to use another atlas text path. The adapter expands each preset's input vocabulary to 258 IDs (256 bytes, BOS, PAD), excludes BOS/PAD from output normalization, and preserves the preset's architecture choices. `manifest.json` records the resolved preset, its source reference, and hashes of that spec and the implementation source files; resume rejects either changing. The atlas repository's own README and tests document each preset's omissions. The included Qwen3 and DeepSeek-style configs both use 80 steps; each runs in seconds on the verified CPU environment. Their scores are **not comparable evidence of model-family quality**: parameter counts, routing compute, and optimization needs differ substantially.

The same guarded comparison CLI can verify that only the preset name changed while keeping corpus, split, seed, step count, batch size, context, and learning rate fixed:

```sh
python3 -m capstone.transformer_run --config configs/atlas-qwen3.json --compare-config configs/atlas-deepseek-v3.json --vary architecture.preset --output-root /tmp/capstone-atlas-study
```

This is a configuration check, not a matched-compute study. A defensible comparison must additionally match or disclose parameters, active FLOPs, training tokens, optimizer tuning, multiple seeds, and hardware measurements.

## Run the n-gram baseline

Python 3.10+ and its standard library are sufficient for this baseline:

```sh
python3 -m unittest discover -s tests -p test_capstone.py -v
python3 -m capstone.run --config configs/baseline.json
python3 -m capstone.compare --base configs/baseline.json --candidate configs/bigram-swap.json --vary architecture.variant
```

Each run writes `artifacts/<run_id>/checkpoint.json`, `manifest.json`, and `report.json`; the comparison additionally writes `comparison.json` under the candidate run. The manifest records the input SHA-256, document indices, config and checkpoint hashes, implementation label, and Python version. Splitting is by line-document **before** byte contexts are formed. The metric is held-out negative log likelihood in bits per byte; lower is better for this *same* byte corpus and protocol. No confidence interval is estimated.

[`examples/toy-unigram-report.json`](examples/toy-unigram-report.json) and [`examples/toy-ablation-comparison.json`](examples/toy-ablation-comparison.json) are snapshots from the verified local run; regenerate them when changing code or data. Live run artifacts remain in the ignored `artifacts/` directory.

The locally verified example on Python 3.14.7 yields 4.2982 bits/byte for the unigram and 5.2249 for the bigram on 270 held-out bytes (five documents). The bigram's worse score is a useful warning about sparse context counts, not evidence that unigram models are preferable generally. Rerun the commands to obtain evidence for your machine; do not copy these numbers into an LLM comparison.

## Which lectures have code

| Lecture | In this repository | Only a protocol (no code here) |
| --- | --- | --- |
| 90 | Byte-level CPU Transformer training with a hashed manifest, document-level split, checkpoint and exact resume, plus the n-gram baseline. | Tokenizer, packed batches, licensed data, GPU compute plan and profile. Inputs are 256 bytes plus BOS and PAD, and each document is cut into independent 32-byte chunks (64 for the autocomplete config). |
| 91 | Greedy byte continuation in `capstone/generate.py` only. **No code** for SFT, preference data or a base-versus-chat evaluation. | The whole post-training stage. |
| 92 | One depth swap (`architecture.layers` 1 versus 2) with a one-field guard, and the paired multi-seed script for it. | Dense/GQA/MLA/MoE comparison, compute-matched budgets, parameter and active-FLOP ledger, failure analysis. |
| 93 | **No code.** | Size/data/compute sweep, scaling fit and held-out prediction. |
| 94 | The optional `deepseek_v3_style` atlas preset trained through the same runner for 80 CPU steps. | Precision and training recipe of the technical report, inference cache, hardware profile. |
| 95 | The five-preset atlas adapter, the Qwen3 and DeepSeek-style configs, the guarded preset comparison and resume tests. | A matched-budget comparison and any speed, memory or quality measurement. |
| 96 | **No code.** | Source-to-code trace, new operator, preregistered critique. |

## What students extend

- [Lesson map](LESSON_MAP.md) links each of 90–96 to a deliverable and a verification gate.
- [Experiment contracts](CONTRACTS.md) distinguish the implemented CPU family swap from the still-unimplemented GPU, broad-evaluation, and assistant stages.
- `configs/baseline.json` is the v1 run manifest; `configs/bigram-swap.json` changes only `run_id` and `architecture.variant`. The comparator rejects a simultaneous data, split, seed, smoothing, or metric change.
- The bundled corpus is intentionally original and tiny. For a real run, use a licensed dataset, document its provenance and exclusions, split by a leakage-safe unit, and independently evaluate the resulting checkpoint.

The causal attention mechanism traces to [Vaswani et al., *Attention Is All You Need* (2017)](https://arxiv.org/abs/1706.03762); the built-in path uses PyTorch's [TransformerEncoderLayer](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html) with an explicit causal mask as a compact decoder-only implementation, not a reproduction of the paper's encoder-decoder training. The atlas presets draw their structural choices from primary model reports or official configurations recorded in each resolved manifest; [Ainslie et al., *GQA* (2023)](https://arxiv.org/abs/2305.13245) and the [DeepSeek-V3 Technical Report (2024)](https://arxiv.org/abs/2412.19437) provide background for two of the mechanisms. The full tokenizer → pretraining → post-training → evaluation → inference path is illustrated in [Andrej Karpathy's nanochat repository](https://github.com/karpathy/nanochat), a reference for a later larger project, not a dependency here. [Hoffmann et al., *Chinchilla* (2022)](https://arxiv.org/abs/2203.15556) motivates a proper model/data/compute ledger.

## Evidence boundary

The tests prove local causality, exact interruption/resume parity for the built-in, Qwen3-dense, and DeepSeek-style paths on one CPU/PyTorch environment, document split separation, config rejection, and finite held-out metrics. They do **not** prove a capable model, equal-compute comparison, GPU correctness, SFT quality, or full paper fidelity. The multi-seed script measures seed noise only, on one fixed split. The atlas paths omit inference caches, full published-model dimensions, and other details documented there. Each later extension needs parity tests, a hardware profile, budget ledger, seeds, and disclosed differences from its cited architecture.
