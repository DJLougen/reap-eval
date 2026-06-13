# Producer checklist — shipping a pruned MoE

If you publish a pruned/compressed MoE checkpoint, ship **all** of these — or label it experimental and mean it. Derived from [`../README.md`](../README.md) §10.

- [ ] **Base-vs-pruned Tier-1 benchmarks** with the dense baseline, sample size ≥100, and the harness named (SWE-Bench Verified, LiveCodeBench, HumanEval+). No placeholder/`{{VAR}}` cells.
- [ ] **Prune ratio and "%-removed" in the model name** — let buyers see the tradeoff before pulling 100+ GB.
- [ ] **Public** calibration + observation datasets (no 401/gated), each with schema, size, and license.
- [ ] **Repetition-loop disclosure** — which generation modes are safe (bounded / agentic) vs unsafe (open-ended), and the exact sampling settings you validated.
- [ ] **Prune-vs-quant separation** for every quantized variant (don't blame pruning for a 2-bit quant's damage).
- [ ] **Correct citation** matching the arXiv ID; no leaked local paths, tokens, or hostnames in the card.
- [ ] **Recovery deltas** — if you ran distillation / LoRA TuneComp, post before/after quality. Recovery without a measured delta is not a result.

> Rule of thumb: throughput, context length, and "fits on 1 GPU" are facts pruning preserves almost for free. They are **not** quality evidence. Ship the quality numbers.
