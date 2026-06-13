# Contributing

This repo is a methodology, a set of templates, and a growing library of worked examples for evaluating pruned/compressed Mixture-of-Experts model releases. Contributions are welcome in three forms.

## 1. New red flags (`README.md` §6)

Add a "tell" that helps a reader spot a hidden-lossy or unsupported release. Each entry **must** follow the existing format:

> **N.M Short name.** *Detect:* how to find it mechanically. *Matters:* why it changes the verdict.

A red flag without a concrete detection step is not actionable — it will be asked to add one.

## 2. Worked examples (`examples/`)

Apply the methodology to a real release and submit a filled `templates/scorecard.md` as `examples/<model-id>.md`. Requirements:

- **Cite verbatim.** Every warning, claim, and number is quoted from the live model/dataset card or a benchmark run you performed — with the source linked. No paraphrased "the author says it's good."
- **Show the math.** Relative drops computed from a stated baseline; mark "no baseline" explicitly when absent.
- **Separate prune from quant** (§2.5) or label "Combined."
- **Run at least one behavioral test** (§7) if the card lacks quality numbers, and report exactly what you ran.
- **No unsourced retention percentages.** "97% retained" with no baseline is logged as an evidence gap, not a result.

## 3. Methodology changes (`README.md` §§3–5, 9)

Thresholds, tiers, confidence levels, and stopping rules are deliberately fixed so verdicts are comparable across contributors. Proposals to change them must come with a rationale and, ideally, examples where the current rule misclassifies. Open an issue first.

## House style

- Evidence over adjectives. Lead with the fact, then the implication.
- Quotes are verbatim and attributed. Inferences are labeled as inferences.
- Give credit where the engineering is real; hold the line on missing quality evidence. The point is an honest verdict, not a takedown.
