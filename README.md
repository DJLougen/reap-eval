# Evaluating MoE Expert-Pruned Model Releases — A Methodology

**A reusable protocol for judging whether a pruned/compressed Mixture-of-Experts checkpoint is worth running.**

**Slides:** a standalone presentation of this methodology lives at **https://djlougen.github.io/reap-eval/** (source: [`docs/index.html`](docs/index.html)).

Distilled from the 0xSero REAP analysis (April–June 2026). Applies to **any** expert-pruning or one-shot MoE compression release — REAP, expert-skipping, expert-merging, layer-dropping — not just one author or method. The worked-example results that motivated it are summarized in §9 and demonstrated in [`examples/`](examples/).

> **One-line thesis:** A pruned MoE is guilty until proven innocent. The burden of proof is **base-vs-pruned quality benchmarks on agentic/coding tasks**, posted by the producer or run by you. Download counts, throughput numbers, observation dumps, and a clean-looking name are **not** evidence of quality.

---

## What's in this repo

- **[`README.md`](README.md)** — this methodology (the centerpiece).
- **[`docs/index.html`](docs/index.html)** — the [slide deck](https://djlougen.github.io/reap-eval/) (reveal.js, self-contained).
- **[`templates/scorecard.md`](templates/scorecard.md)** — copy-paste; fill one per model evaluated.
- **[`templates/producer-checklist.md`](templates/producer-checklist.md)** — minimum evidence to ship a pruned MoE.
- **[`examples/`](examples/)** — worked scorecards (e.g. [`kimi-k2.6-519b.md`](examples/kimi-k2.6-519b.md)).
- **[`CONTRIBUTING.md`](CONTRIBUTING.md)** — how to add red flags and case studies.

---

## 0. Quick reference (one page)

**The 6-step pass**
1. **Identify** — base model, prune ratio, quant, claimed size reduction.
2. **Read the card** — find the disclosed tradeoff and every warning word.
3. **Verify provenance** — are the cited calibration/benchmark datasets public and real? (Check for 401/404.)
4. **Classify the evidence** — Tier-weight the benchmarks, assign a confidence level, compute *relative* drop.
5. **Separate prune vs quant** — attribute degradation; never blame pruning for a 2-bit quant's damage.
6. **Test it yourself** — repetition-loop probe, bounded-vs-open-ended, sampling sweep, long-context needle.

**Verdict taxonomy**
| Verdict | Bar |
|---|---|
| **Clear Win** | All Tier-1 drops <5% relative, baseline posted, n≥100, no warnings, no repetition pathology |
| **Marginal** | Tier-1 drops 5–15% relative, functional, minor degradation |
| **Failure** | Any Tier-1 drop >15% relative, OR repetition/collapse, OR "do not use / not stable" |
| **Uncertain / High-Risk** | Insufficient data, no baseline, n<50, "experimental"/"alpha", or private evidence |

**Relative-drop thresholds:** `<5%` minimal · `5–15%` marginal · `>15%` failure. (Always relative, not pp.)
**Tier-1 veto:** a model **cannot** be a Clear Win with `>10%` relative drop on **any** Tier-1 (agentic/coding) benchmark, no matter how good Tier-3 looks.
**Default suspicion:** ≥40% prune + open-ended generation = assume repetition-loop risk until you've tested otherwise.

---

## 1. Purpose & scope

**Use this when** someone publishes a pruned/compressed MoE checkpoint and claims it is "near-lossless," "97% retained," "production-ready," or a drop-in for the base model — and you have to decide whether to download 50–300 GB and trust it.

**Core research question (per release):** *Does this checkpoint preserve enough Tier-1 (agentic/coding) capability to justify its size reduction and the compute spent producing it — with evidence, not vibes?*

**What pruning does and does not change** (know this before reading any card):
- **Changes:** number of routed experts, expert tensors, expert-ID mapping, checkpoint size, memory footprint.
- **Preserves (by design):** context length, tokenizer, attention architecture, layer count, hidden size, experts-active-per-token, chat format.
- **Therefore:** "200K context works" and "loads on 1 GPU" and "fast prefill" are **fit/throughput** facts the pruning preserves almost for free. They are **not** quality evidence. Do not let them stand in for quality.

---

## 2. The evaluation protocol (step by step)

### 2.1 Identify the artifact
Record, from the card/config:

| Field | Where to find it | Notes |
|---|---|---|
| Base model | card header / `base_model` tag | If the base is unnamed ("—"), treat as **High-Risk** immediately. |
| Prune ratio | card body / `prune_summary.json` | Experts kept vs original per MoE layer. Compute `% removed`. |
| Quantization | tags / config | None / W4A16 / GGUF-QN / NVFP4 / FP8 / MXFP4. |
| Params removed | card / config | Cross-check against the size-reduction claim. |
| Active params/token | config | Should be ~unchanged after pruning (router still picks top-k). |

**Compute the prune ratio yourself** from `experts_kept / experts_original`. Do not trust the name — see §6.2.

### 2.2 Read the card for the disclosed tradeoff
Scan for, and copy verbatim, every instance of: `experimental`, `alpha`, `not stable`, `do not use`, `private`, `proof-of-concept`, `repetition`, `loop`, `collapse`, `nonsense`, `interrupted`, `timed out`, `evaluate against your own tasks`, `not for claiming model quality`.

These are the producer telling you the answer. A card that says "validate generation before production" is **not** claiming a win — score it accordingly.

### 2.3 Verify provenance & evidence (the step most people skip)
A serious release cites its calibration data and benchmark artifacts. **Open every cited dataset link and confirm it is public and real.**

- `404` → dangling/false citation.
- `401` / "gated" / "private" → **the evidence backing the claim is unverifiable by you.** Treat the claim as unsupported, full stop. (In the June reassessment, the three newest flagships' cited "calibration evidence"/"benchmark traces" datasets all returned 401.)
- One-line dataset card on a high-traffic artifact → undocumented; cannot reproduce.
- Check the **citation itself**: does the bibtex match the arXiv ID? (Wrong-paper citations are a real integrity tell — see §6.7.)

If the evidence is private or absent, you do the benchmarking yourself (§2.6) or you treat the model as **Uncertain/High-Risk**.

### 2.4 Classify the benchmark evidence
For every benchmark the card reports, compute and record:

```
Absolute score:        XX.X%
Dense baseline:        YY.Y%      <- if absent, the number is uninterpretable
Percentage-point drop: XX.X - YY.Y = Z.Z pp
Relative percent drop: (XX.X - YY.Y) / YY.Y * 100 = W.W%
```

Then Tier-weight (§4), assign confidence (§5), and apply the thresholds. **A score with no baseline is not a measurement** — log it as an evidence gap, not a result.

### 2.5 Separate pruning from quantization
Degradation has two sources; don't conflate them.

- Same pruned checkpoint at different quants → isolates the **quant** contribution.
- Dense model at the same quant → isolates the **prune** contribution.
- Rule of thumb observed in practice: heavy quant (`<3 BPW`) does **more** damage than the prune; e.g. each ~1 BPW reduction ≈ 10–12% score drop. Don't credit/blame pruning for a 2-bit GGUF's collapse.
- If you can't separate them, label **"Combined (prune+quant)"** and flag the uncertainty.

### 2.6 Test it yourself (behavioral battery — §7)
The single highest-value step, because most cards lack quality numbers. Run the repetition-loop probe, the bounded-vs-open-ended split, the sampling sweep, and a long-context needle. Details and exact params in §7.

### 2.7 Score and write the verdict
Fill the scorecard (§8). Assign one verdict from the taxonomy. State the confidence and the top evidence gap in one line.

---

## 3. Metric calculation rules

- **Always use *relative* percent drop** for classification, not percentage points. `80% → 73%` is `7 pp` but `8.75%` relative — the relative figure is what crosses thresholds.
- Thresholds: **<5% minimal · 5–15% marginal · >15% failure** (or any repetition/collapse/"do not use", regardless of numbers).
- Report perplexity separately; it correlates poorly with task quality and degrades faster than benchmarks (a 20% prune showed +39% PPL with <4% coding drop). Use it as a smoke signal, not a verdict.

---

## 4. Benchmark weighting

| Tier | Weight | Benchmarks | Why |
|---|---|---|---|
| **Tier 1** | High | SWE-Bench Verified, Terminal-Bench, browser-use, HumanEval/+/MBPP/+, LiveCodeBench | Agentic/coding = what pruned local models are actually used for; most sensitive to expert loss. |
| **Tier 2** | Medium | GSM8K, MATH, GPQA, BBH | Reasoning/math. |
| **Tier 3** | Low | MMLU, ARC, HellaSwag, CommonsenseQA, knowledge QA | Cheap to preserve; least diagnostic of real degradation. |

**Decision rule (Tier-1 veto):** A model **CANNOT** be a "Clear Win" if it shows `>10%` relative drop on **any** Tier-1 benchmark — even with perfect Tier-3.
**Knowledge-collapse watch:** Tier-3 world-knowledge tasks (e.g. World Religions) can crater (90% → 48% at 30% prune) while math/coding hold. A model can be "fine for coding, lobotomized on knowledge." Report both.

---

## 5. Confidence levels

| Level | Requires |
|---|---|
| **High** | Direct dense-baseline comparison posted; n≥100 (coding/agentic), n≥250 (MMLU); lm-eval-harness or multiple independent evals; full card. |
| **Medium** | Proxy/sampled benchmarks (n=50–100); single/self-reported evaluator; partial baseline. |
| **Low** | Qualitative claims only ("works well"); no baseline; n<50; "not benchmarked"/"experimental"/"alpha"; private/inaccessible evidence. |

**Sample-size floors:** do not count toward a verdict any result from a proxy benchmark with <50 samples, or any "97% retained"-style claim with no baseline.

---

## 6. Red-flag catalog (the "tells")

Each flag: **how to detect** → **why it matters**. These are the patterns that most often hide a lossy prune behind a clean release.

**6.1 Templated/placeholder benchmark tables.** *Detect:* literal `{{VAR}}` / `<...>` / `TBD` cells under a "Headline numbers"/"Benchmarks" heading. *Matters:* a benchmark heading with no data is worse than no heading — it manufactures the *appearance* of evidence. (Observed live on a model with 1,000+ downloads.)

**6.2 "Pruned/%/REAP" scrubbed from the model name.** *Detect:* name reads like a first-party release (`Qwen3.6-28B`) while the card body reveals a prune; compare to the base model's real name/size. *Matters:* downloaders can't see the tradeoff before pulling 100+ GB; discoverability of "this is lossy" drops as reach rises.

**6.3 Buried warning words.** *Detect:* `experimental`, `alpha`, `not stable`, `do not use`, `proof-of-concept` anywhere in the card. *Matters:* this is the producer's own verdict; it caps confidence at Low and often = Failure.

**6.4 Benchmark "interrupted / timed out / debug-only."** *Detect:* phrases like "do not treat as a final score," "useful for debugging the benchmark path, not for claiming model quality." *Matters:* an uploaded artifact dataset is not a score; absence of a completed run is absence of evidence.

**6.5 Private/dangling evidence (401/404).** *Detect:* open every cited calibration/trace dataset URL. *Matters:* unverifiable provenance = unsupported claim (§2.3).

**6.6 Unbacked retention percentages.** *Detect:* "97.9% capability retained" with no benchmark, no baseline, no sample size, or on a model that no longer exists. *Matters:* a precise number implies measurement that wasn't shown.

**6.7 Citation drift.** *Detect:* the bibtex title/authors don't match the arXiv ID; the same ID cited differently across the author's repos. *Matters:* signals copy-paste provenance and weakens trust in the rest of the card.

**6.8 Throughput/fit presented as quality.** *Detect:* the only numbers are tok/s, TTFT, context length, "fits on 1 GPU." *Matters:* pruning preserves these almost for free (§1); they say nothing about output quality.

**6.9 Narrow eval suites.** *Detect:* English-only, n=50, single-domain evals; no multilingual/long-form/adversarial coverage. *Matters:* clean English n=50 can coexist with "Korean is broken" — verify the dimensions the suite ignores.

**6.10 Download count as proof.** *Detect:* "Proven" / popularity cited in lieu of validation. *Matters:* downloads measure hype and file size, not quality.

**6.11 Hygiene inconsistency (integrity smoke).** *Detect:* leaked local paths (`/Users/.../...`), tokens, or hostnames in a card that elsewhere claims paths are "intentionally scrubbed." *Matters:* minor on its own, but a reliable signal of low-rigor card generation; raise scrutiny on everything else.

---

## 7. Behavioral test battery (runnable)

Run these against the served checkpoint when the card lacks quality numbers (the common case). All four are cheap and decisive.

**7.1 Repetition-loop probe (the most important test for ≥40% prunes).**
- Prompt with **open-ended, long-form** requests ("Tell me about cats and cat allergens"; "Write a 1,500-word essay on X"). The prompt-dependent attractor is the point.
- Generate to a high `max_tokens` (≥2,048).
- **Fail conditions:** single-token or short-phrase loops (`saliva saliva saliva…`), duplicate-line loops, never-terminating reasoning (with thinking on, an un-closed `<think>` that fills context and returns empty content), Unicode replacement chars, or hitting max length without natural stop.
- **Severity is prune-dependent:** 50% prunes frequently loop on prose while passing all bounded tasks.

**7.2 Bounded vs open-ended split (characterize, don't just pass/fail).**
- **Bounded set** (where pruned MoEs usually survive): structured JSON, tool/function calls, code generation that you execute, short math/Q&A, agentic/terminal tasks with stop conditions.
- **Open-ended set** (where they break): free-form prose, long essays, multi-paragraph explanations.
- Report the model as e.g. "reliable bounded/agentic, loops on open-ended" rather than a single verdict — that's the actionable truth for ≥40% prunes.

**7.3 Sampling sweep (prove loops are/aren't fixable).**
- Try `temperature=0` with `repetition_penalty ∈ {1.0, 1.05, 1.12, 1.15}`, and `temperature∈{0.6,0.7}` with presence/frequency penalties.
- If open-ended loops persist across the sweep, **the pathology is structural, not a sampling bug** — do not let "just raise rep_penalty" off the hook. (Note `rep_penalty=1.05` has been seen to *cause* deterministic loops on short structured output; tune per checkpoint.)

**7.4 Long-context needle + stability sweep.**
- Insert a known fact ("SUM=166") at varied depths; generate at short / 32k / 128k / 200k / near-max context; verify retrieval and natural termination at each band.
- Watch for context-band-specific failures (e.g. loops only in the 96k–160k band, sometimes fixed by raising `max_num_batched_tokens`, not by changing weights).

**7.5 Gate-repair audit (for producers / deep due-diligence).**
- After pruning, every surviving router gate weight and `e_score_correction_bias` row must equal the original source row selected by the keep-plan. Compare element-wise; expect **max abs diff = 0.0** across all MoE layers, and confirm config (`n_routed_experts`, `num_experts_per_tok`, `n_group`, `topk_group`) is repaired to the new expert count. A nonzero diff means the router points at the wrong experts.

---

## 8. Scorecard template (copy-paste)

```
Model: <id>            Base: <base>        Prune: <%>    Quant: <fmt>
Size: <base GB> -> <pruned GB>  (<reduction %>)   Active/tok: <unchanged?>

EVIDENCE
  Baseline posted?        Y/N
  Cited datasets public?  Y/N  (404/401 list: ____)
  Card warnings:          "<verbatim>"
  Confidence:             High / Medium / Low

BENCHMARKS (relative drop)
  Tier-1 | <bench> | base __ | pruned __ | rel __% | tier-1 veto? Y/N
  Tier-2 | ...
  Tier-3 | ...   (note any knowledge-collapse)

PRUNE vs QUANT
  Attribution: prune __% / quant __% / interaction __%  (or "Combined")

BEHAVIORAL (run by me)
  Repetition loop (open-ended):   pass/FAIL  (attractor: ____)
  Bounded/agentic:                pass/FAIL
  Sampling sweep fixes loops?     Y/N
  Long-context needle @ <ctx>:    pass/FAIL

EVIDENCE GAPS
  [ ] no baseline  [ ] n<50  [ ] qualitative only  [ ] private/401 evidence
  [ ] throughput-as-quality  [ ] narrow suite  [ ] prune/quant unseparated

VERDICT: Clear Win / Marginal / Failure / Uncertain-High-Risk
ONE-LINE: ____
```

---

## 9. Stopping rule (for producers & funders)

When is the whole pruning effort *not* economically justified?

**Primary stopping rule.** After **3 checkpoint-level attempts** that each had (a) adequate calibration (≥100k samples), (b) reasonable hardware (≥2× RTX 4090 / A100-class), (c) conservative ratio (≤30%), and (d) a real evaluation attempted — if the cumulative **clear-win rate is <25%** (0–1 of 3), stop.

**Secondary (faster) stopping rule.** After **2 consecutive** attempts with any of: `>15%` relative Tier-1 drop, repetition/collapse, or a "do not use" warning.

**Do not count** attempts whose only evidence is <50-sample proxies, no baseline, or "not benchmarked."

**Economic-justification threshold (when to call the method validated):** ≥3 checkpoint-level **clear wins**, success rate **≥50%**, and **≥1 success at a 40–50% ratio**.

**Worked-example result** (0xSero REAP releases, 22 checkpoint-level attempts): 0 clear wins, ~23% marginal, 50%-prune failure ~60% → **not justified** for solo/community settings; the June reassessment (62 models) did not move this. A filled scorecard for one such model is in [`examples/kimi-k2.6-519b.md`](examples/kimi-k2.6-519b.md).

---

## 10. Producer checklist — minimum evidence to ship a pruned MoE

If you publish one, ship **all** of these or label it experimental and mean it:

1. **Base-vs-pruned Tier-1 benchmarks** with baseline, sample size ≥100, and the harness named (SWE-Bench Verified, LiveCodeBench, HumanEval+). No placeholders.
2. **Prune ratio and "%-removed" in the model name** — let buyers see the tradeoff pre-download.
3. **Public** calibration + observation datasets (no 401), with schema, size, and license.
4. **A repetition-loop disclosure**: which generation modes are safe (bounded/agentic) vs unsafe (open-ended), with the sampling settings you validated.
5. **Prune-vs-quant separation** for every quantized variant.
6. **Correct citation** matching the arXiv ID; no leaked local paths/tokens.
7. If you ran **recovery** (distillation/LoRA TuneComp), post **before/after** quality — recovery without a measured delta is not a result.

---

*Methodology v1.0 — June 13, 2026. Thresholds, tiers, confidence levels, and stopping rules are carried verbatim from the source statistical analysis; the red-flag catalog and behavioral battery are generalized from the April 2026 analysis and June 2026 reassessment of 0xSero's REAP releases. Applies to any expert-pruning / MoE-compression release.*
