# Case study — the analysis behind the methodology

These are the two original documents the methodology in [`../README.md`](../README.md) was distilled from: a full statistical analysis of the REAP-pruned community models published by HuggingFace user **0xSero**, plus the analysis protocol it used.

| File | What it is |
|---|---|
| [`REAP_Final_Report.md`](REAP_Final_Report.md) | The full report — a 22-attempt statistical analysis (April 2026) **plus the June 2026 reassessment addendum** covering 62 models, the published datasets, and the funding angle. The worked-example verdicts cited in the methodology come from here. |
| [`REAP_statistical_analysis.md`](REAP_statistical_analysis.md) | The analysis protocol (v2.0) — metric-calculation rules, tier weighting, confidence levels, attempt-counting, evidence-gap checklist, and stopping rules. The methodology's rubric is carried **verbatim** from this document. |

**Headline result:** across 22 checkpoint-level attempts (and unchanged when rescaled to 62 models in the June reassessment) — **0 clear wins**, ~23% marginal, ~60% failure at 50% prune → REAP-in-practice is not economically justified for solo/community settings.

> These two documents name 0xSero specifically and quote their model/dataset cards verbatim. The methodology, templates, and slides in the rest of the repo generalize those findings into a model-agnostic protocol you can apply to any pruned/compressed MoE release.
