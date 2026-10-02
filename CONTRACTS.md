# Experiment and architecture contracts

## Implemented v1 toy run contract

The JSON config names one `run_id`, integer `seed`, corpus path, document-level train fraction, byte-ngram `variant` and smoothing `alpha`, and `bits_per_byte` evaluation. `capstone.core.execute` hashes the raw data and config, records exact split indices, fits only on training documents, evaluates on held-out documents, and emits a hashed JSON checkpoint plus manifest and report. The same config and input should produce byte-identical files on a given Python version. Generated files are ignored by Git; archive them with a real experiment when you need an audit trail.

The CLI comparator allows exactly one scientific field to change: `architecture.variant` or `architecture.alpha`. A different `run_id` is required so artifacts cannot overwrite each other. It is a *guardrail*, not proof of a fair LLM ablation. For 92–93, additionally preregister the question; match data, tokenizer, evaluation, token/compute or wall-clock budget as appropriate; disclose parameter count and active FLOPs; use multiple seeds; report spread and failure modes. Do not change the architecture and budget simultaneously while attributing a difference to architecture alone.

## Implemented v2 causal Transformer contract

`configs/tiny-transformer.json` fixes corpus/split, model width, attention heads, depth, context, optimizer, batch size, learning rate, and total steps. `capstone.transformer.execute` trains the byte-level decoder on CPU. Its upper-triangular attention mask prevents access to future positions; examples never span documents. The checkpoint includes model/AdamW state and step. A deterministic step-indexed batch generator makes interruption/resume weight-exact on the tested CPU/PyTorch environment. Resume rejects changed config, corpus, resolved architecture, or implementation source hashes. Held-out loss is summed over every document byte; each 32-byte chunk begins a new context. The depth-swap config changes only `architecture.layers`, so parameter count and FLOPs change with depth and are **not** matched. The comparison is a mechanics exercise, not a fair scaling claim. `capstone.multiseed` repeats the pair over seeds on a frozen split and reports the mean paired difference, its standard error and a 95% t interval; it measures seed noise only, not data or split uncertainty.

The optional atlas backend accepts five validated tiny text-path presets without changing the data, optimizer, evaluation, checkpoint, or resume functions. It changes the token vocabulary to 258 for byte inputs and scores only the 256 byte outputs. The resolved spec and its primary `source_url`/`source_revision` are embedded in the manifest. The included Qwen3 dense and DeepSeek-V3-style configs are short CPU smoke runs; exact resume parity and finite held-out metrics are tested for both. Their config-only preset comparison is not matched for parameter count, active FLOPs, or convergence.

## Future larger-model adapter contract (specification only)

A larger replacement must preserve these external interfaces. The v2 CPU decoder and optional atlas preset adapter implement part of this contract; no GPU or production-scale adapter ships here.

1. `prepare_data(source_manifest, tokenizer_manifest) -> train/eval iterators`: document split and dedup happen before packing; save hashes and license/provenance. No train text enters evaluation by construction.
2. `build_model(architecture_config) -> model`: declare family, exact dimensions, attention/position/FFN/routing choices, weight tying, parameter count, active parameter count, context length, and expected KV or recurrent state layout. Cite the primary report and list deviations.
3. `train(model, train_iterator, runtime_config, resume_manifest) -> checkpoint`: record optimizer/scheduler, RNG, data cursor, precision, mesh, gradient accumulation, global step, tokens seen, measured memory/throughput, and interruption/restart parity. The v2 CPU decoder records optimizer and step and proves local resume parity, but does not implement GPU mesh, precision variants, or throughput measurement.
4. `evaluate(checkpoint, frozen_eval_manifest) -> report`: use the same tokenizer, held-out records, metrics and generation settings across families; add quality rubrics appropriate to the model, not only loss. Never compare a base model against a chat-tuned model without marking the difference.
5. `export(checkpoint) -> weights + config + tokenizer + manifest`: round-trip reload, hashes, exact model family/version, and state-layout version are required. A failed or incompatible reload must be explicit.

For DeepSeek-V3-style work, use [the technical report](https://arxiv.org/abs/2412.19437) as a source for structural choices and disclose omissions instead of calling a generic MoE “V3.” For GQA, cite [the GQA report](https://arxiv.org/abs/2305.13245) and test both full-prefix and cached decoding. For scaling, cite [Chinchilla](https://arxiv.org/abs/2203.15556) and show the actual fitted range and held-out prediction rather than borrowing reported coefficients as your own result.

## Stop conditions

The tests validate the tiny CPU Transformer, five atlas forward/backward paths, two atlas train/resume paths, and n-gram harness. Do not mark lectures 90–96 implementation-complete from these checks. GPU timing, distributed correctness, post-training, matched-budget family comparisons, and broad evaluation are outstanding until independently measured.
