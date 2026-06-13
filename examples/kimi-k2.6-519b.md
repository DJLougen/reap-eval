# Evaluation scorecard — `0xSero/Kimi-K2.6-519B-NVFP4`

Worked example demonstrating the methodology. All quotes are verbatim from the live model/dataset cards (June 2026). Filled from [`../templates/scorecard.md`](../templates/scorecard.md); section numbers reference [`../README.md`](../README.md).

```
Model: 0xSero/Kimi-K2.6-519B-NVFP4   Base: moonshotai/Kimi-K2.6 (~1.026T)   Prune: 50% (keep 192 of 384 routed experts)   Quant: NVFP4
Size: ~1.026T params -> 519B          Active/tok: unchanged (router still selects top-k)
```

## Evidence (§2.3, §5)

| Check | Answer |
|---|---|
| Dense baseline posted? | **N** — no base-vs-pruned quality benchmark on the card |
| Cited datasets public? | **Partly.** `kimi-k2.6-reap-observations-v1` is public; the cited `kimi-k2-6-nvfp4-reap-keep192-endpoint-benchmark-traces-v1` returns **HTTP 401** (private) |
| Card warning words (verbatim) | "alpha testing phase, this model is not stable … falls into nonsense and repetition loops on open-ended / long-form generation" |
| Confidence level | **Low** (for quality) — warnings + no baseline; Medium for the *fit/serving* claims |

## Benchmarks — relative drop (§3, §4)

| Tier | Benchmark | Base | Pruned | Rel. drop | Tier-1 veto? |
|---|---|---|---|---|---|
| 1 | SWE-Bench Verified / LiveCodeBench / HumanEval+ | — | — | n/a | n/a — none posted |
| 2 | GSM8K / MATH | — | — | n/a | — |
| 3 | MMLU | — | — | n/a | — |

> No quality benchmark with a baseline exists on the card. Every cell is an evidence gap, not a measurement.

## Prune vs quant (§2.5)

```
Attribution:  Combined — cannot separate. 50% expert prune AND NVFP4 quant applied together; no same-prune/different-quant or dense/same-quant comparison published.
```

## Behavioral tests (§7)

| Test | Result | Notes |
|---|---|---|
| Repetition loop (open-ended, ≥2048 tok) | **FAIL** | Author-confirmed: loops into "nonsense and repetition loops on open-ended / long-form generation"; documented attractors (`saliva saliva saliva…`) |
| Bounded / agentic (JSON, tools, code, short Q&A) | **pass** | Author: "reliable for structured, tool-calling, code, math, agentic (bounded outputs)" |
| Sampling sweep fixes loops? | **N** | Still loops at `temperature=0` with `repetition_penalty` up to 1.15 → pathology is structural, not a sampling bug |
| Long-context needle @ ~260k | **pass** | Context validated to 259,943 tokens; gate-repair audit matched all 60 MoE layers to max abs diff 0.0 |

## Evidence gaps (§6)

- [x] no baseline   - [x] private/401 evidence (benchmark-traces dataset)   - [x] prune/quant unseparated
- [x] throughput-as-quality (decode ~71 tok/s on 4× RTX PRO 6000 cited; no quality numbers)
- [ ] placeholder table   - [x] "pruned/%" scrubbed from name (`Kimi-K2.6-519B` reads as a release; "50% prune" only in body)

## Verdict

**`Failure` for general use  /  `Marginal` for bounded-output use.**

**One line:** A 50% expert prune that reproduces the known repetition-loop pathology on open-ended generation (author-labeled "not stable"); genuinely rigorous fit/serving QA, but reliable only for bounded/agentic outputs — and the headline quality claim has no public baseline.

---

*This example shows the method giving a split, honest verdict: real credit for the serving/QA engineering (260k context validated, gate audit to 0.0), an unflinching `Failure` on the open-ended quality claim the producer's own warning confirms.*
