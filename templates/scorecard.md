# Evaluation scorecard — `<model-id>`

Copy this file to `evaluations/<model-id>.md` (gitignored) or, for a contribution, to `examples/<model-id>.md`.
Fill every field. "Unknown" and "no baseline" are valid, informative answers — leave nothing blank.
See the methodology in [`../README.md`](../README.md); section numbers below point to it.

```
Model: <id>            Base: <base>        Prune: <%>    Quant: <fmt>
Size: <base GB> -> <pruned GB>  (<reduction %>)   Active/tok: <unchanged?>
```

## Evidence (§2.3, §5)

| Check | Answer |
|---|---|
| Dense baseline posted? | Y / N |
| Cited datasets public? (list any 404/401) | |
| Card warning words (verbatim) | "" |
| Confidence level | High / Medium / Low |

## Benchmarks — relative drop (§3, §4)

| Tier | Benchmark | Base | Pruned | Rel. drop | Tier-1 veto? |
|---|---|---|---|---|---|
| 1 | | | | % | Y / N |
| 2 | | | | % | — |
| 3 | | | | % | — |

> Tier-1 veto = `>10%` relative drop on any agentic/coding benchmark ⇒ cannot be a Clear Win.
> Note any knowledge-collapse on Tier-3 (e.g. world-knowledge cratering while coding holds).

## Prune vs quant (§2.5)

```
Attribution:  prune __%  /  quant __%  /  interaction __%      (or "Combined — cannot separate")
```

## Behavioral tests I ran (§7)

| Test | Result | Notes |
|---|---|---|
| Repetition loop (open-ended, ≥2048 tok) | pass / FAIL | attractor: |
| Bounded / agentic (JSON, tools, code, short Q&A) | pass / FAIL | |
| Sampling sweep fixes loops? (temp 0; rep_pen 1.0–1.15) | Y / N | |
| Long-context needle @ `<ctx>` | pass / FAIL | |

## Evidence gaps (§6)

- [ ] no baseline  - [ ] n<50  - [ ] qualitative only  - [ ] private/401 evidence
- [ ] throughput-as-quality  - [ ] narrow suite (English-only / single-domain)  - [ ] prune/quant unseparated
- [ ] placeholder/templated benchmark table  - [ ] "pruned/%" scrubbed from name  - [ ] citation drift

## Verdict

**`Clear Win` / `Marginal` / `Failure` / `Uncertain-High-Risk`**

**One line:** ____
