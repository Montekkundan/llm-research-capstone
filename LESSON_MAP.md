# Lectures 90–96: project map

Each row describes a **future student deliverable**. Only the toy provenance/ablation mechanics in 90, 92, and 96 have an executable starter here. All model-training and distributed gates require later code and hardware.

| Lecture | Capstone task | Minimum evidence before claiming completion |
| --- | --- | --- |
| 90 — From raw documents to a repeatable GPU pretraining run | Replace the toy corpus and n-gram with licensed documents, tokenizer, packed batches, fixed compute plan, and a trained base checkpoint. | Data/license/source hashes, split audit, tokenizer round trip, throughput and memory profile, restartable recipe, held-out metric. |
| 91 — From base model to a measured assistant | Add SFT and an optional preference stage through the same checkpoint format. | Base-vs-chat eval set and rubric, leakage check, regression cases, stage-specific checkpoint manifests. |
| 92 — Compare architectures without changing five things at once | Preregister a dense/GQA/MLA/MoE question and the exact matched budget; run multiple seeds. | Config diff, parameter and active-FLOP ledger, same data and evaluation, uncertainty interval and failure analysis. The current toy `compare` checks only one changed field. |
| 93 — Test scaling claims with your own runs | Sweep model size, data, and compute; fit on training runs and predict a held-out run. | Raw run manifests, fit code, residual plots, held-out error, explicit extrapolation range. |
| 94 — Build a small DeepSeek-V3-style model end to end | Implement a disclosed small structural reproduction of MLA, experts, precision and training choices. | Operator parity, routing tests, checkpoint, actual hardware profile, eval report, and a list of differences from the DeepSeek-V3 technical report. |
| 95 — Switch to another model family without rewriting the trainer | Implement a second architecture adapter under the same data/runtime/checkpoint/eval contracts. | Two-family smoke train, optimizer and resume parity, matched-budget comparison, speed/memory/quality measurements. |
| 96 — Port and defend a new architecture | Trace a primary report to a spec, implement one missing operator, then critique a preregistered claim. | Source-to-code trace, numerical and gradient tests, ablation, reproducible artifact, limitations and negative results. |

The sequence is not a claim that the capstones are implemented. A result is ready to teach as a completed lab only when its evidence column is satisfied. The tiny starter helps students practice the bookkeeping before paying for a GPU run.
