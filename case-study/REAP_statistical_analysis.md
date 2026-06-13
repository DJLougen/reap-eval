# 0xSero REAP Models: Statistical Analysis Framework

## Analysis Protocol v2.0 (Addressing All Requirements)

---

## 1. EXPLICIT METRIC CALCULATION RULES

For every benchmark comparison, calculate AND display:

```
Absolute Score: XX.X%
Dense Baseline: YY.Y%
Percentage-Point Drop: XX.X - YY.Y = Z.Z pp
Relative Percent Drop: (XX.X - YY.Y) / YY.Y * 100 = W.W%
```

**Critical Distinction:**
- **Percentage-point (pp) drop**: Absolute difference (e.g., 80% → 73% = 7 pp drop)
- **Relative percent drop**: Proportional degradation (e.g., 80% → 73% = 8.75% relative drop)

**Classification Thresholds (using RELATIVE percent drop):**
- Minimal regression: <5% relative drop
- Marginal: 5-15% relative drop  
- Failure: >15% relative drop OR repetition/collapse issues

---

## 2. BENCHMARK WEIGHTING (Explicit)

**Tier 1 (High Weight - Agentic/Coding):**
- SWE-Pro / SWE-Bench Verified (full or proxy)
- Terminal-Bench
- browser-use
- HumanEval / HumanEval+ / MBPP
- LiveCodeBench

**Tier 2 (Medium Weight - Reasoning/Math):**
- GSM8K
- MATH
- GPQA
- BBH

**Tier 3 (Low Weight - General Knowledge):**
- MMLU
- ARC
- HellaSwag
- CommonsenseQA

**Decision Rule:**
A model CANNOT be "Clear Win" if it has >10% relative drop on ANY Tier 1 benchmark, regardless of Tier 3 performance.

---

## 3. CONFIDENCE LEVELS (Per Model)

**High Confidence:**
- Direct baseline comparison posted
- Sample size ≥100 for coding/agentic, ≥250 for MMLU
- Multiple independent evals or lm-eval-harness results
- Full model card documentation

**Medium Confidence:**
- Proxy/sampled benchmarks (n=50-100)
- Single evaluator or self-reported
- Partial baseline (implied from paper)
- Some documentation gaps

**Low Confidence:**
- Qualitative claims only ("works well", "minimal drop")
- No explicit baseline
- Sample size <50
- "Not benchmarked" warnings
- Inaccessible/private pages

---

## 4. PRUNING vs QUANTIZATION ATTRIBUTION

**Analysis Rule:**
For each model, attempt to distinguish:

```
Pruning Level: XX%
Quantization: None / W4A16 / GGUF-Q4 / NVFP4 / etc.

Regression Attribution:
- Pruning contribution: estimated XX%
- Quantization contribution: estimated XX%
- Interaction effects: XX%
```

**Evidence Required for Attribution:**
- Same pruned checkpoint with different quants → compare
- Dense model with same quant → compare
- Author's own notes on where degradation comes from

**Default:** If cannot distinguish, mark as "Combined (prune+quant)" and flag uncertainty.

---

## 5. ATTEMPT COUNTING RULES (Critical)

**Define "Attempt":**

**Checkpoint-Level (Preferred):**
- Each independently pruned checkpoint = 1 attempt
- Multiple quantizations of same checkpoint = 1 attempt
- Different random seeds or calibration data = separate attempts

**Release-Level (Secondary):**
- Each uploaded model repository = 1 release
- Use when checkpoint-level unclear

**Reporting Requirement:**
Always report BOTH:
- "X checkpoint-level attempts, Y release-level uploads"
- Success rate calculated at checkpoint level

**Exclusions:**
- Models explicitly marked "draft" or "WIP" may be excluded if stated
- Models with "not benchmarked" warnings count as "attempts with unknown outcome"

---

## 6. EVIDENCE GAPS SECTION (Required)

For each model analyzed, fill out:

| Gap Type | Present? | Notes |
|----------|----------|-------|
| No dense baseline posted | Y/N | |
| Proxy/sampled benchmark only | Y/N | Sample size: |
| Sample size too small (<50) | Y/N | Actual: |
| Qualitative claims only | Y/N | |
| Page inaccessible/private | Y/N | |
| Hardware/decoding unclear | Y/N | |
| Prune vs Quant unseparated | Y/N | |
| Author warnings present | Y/N | |

---

## 7. STOPPING RULE RECOMMENDATION (Final Output Required)

**Operational Definition:**

"A solo researcher should conclude REAP pruning is not economically justified when:"

**Primary Stopping Rule:**
After 3 checkpoint-level attempts that meet ALL of:
1. Adequate calibration data (≥100k samples)
2. Reasonable hardware (≥2x RTX 4090 or A100 equivalent)
3. Conservative prune ratio (≤30%)
4. Proper evaluation attempted

AND cumulative success rate <25% (0-1 clear wins out of 3).

**Secondary Stopping Rule (Faster):**
After ANY 2 consecutive attempts with:
- >15% relative drop on Tier 1 benchmarks, OR
- Repetition/collapse issues reported, OR
- "Do not use" warnings from author

**Sample Size Requirement:**
Do not count attempts with:
- Proxy benchmarks <50 samples
- No explicit baseline comparison
- "Not benchmarked" status

**Economic Justification Threshold:**
Consider method validated only when:
- ≥3 checkpoint-level attempts show clear wins
- Success rate ≥50%
- At least one attempt at 40-50% prune ratio succeeds

---

## RAW DATA COLLECTION IN PROGRESS

### Models Identified (Preliminary Inventory)

| Model | Base | Prune % | Quant | Downloads | Status |
|-------|------|---------|-------|-----------|--------|
| GLM-5.1-555B-A14B-REAP | GLM-5.1 | Unknown | BF16 | 1,650 | Active |
| GLM-5.1-444B-A14B-REAP | GLM-5.1 | Unknown | BF16 | 680 | Active |
| GLM-5.1-555B-A14B-REAP-GGUF | GLM-5.1 | Unknown | GGUF | 273 | Active |
| GLM-5.1-555B-A14B-REAP-GPTQ | GLM-5.1 | Unknown | GPTQ W4A16 | 428 | Active |
| GLM-5.1-555B-A14B-REAP-NVFP4 | GLM-5.1 | Unknown | NVFP4 | 179 | Active |
| GLM-5.1-478B-A42B-REAP-NVFP4 | GLM-5.1 | Unknown | NVFP4 | 1 | Active |
| GLM-5-REAP-381B | GLM-5 | 0% | BF16 | 1,280 | Base |
| GLM-5-REAP-50pct-BF16-GGUF | GLM-5 | 50% | BF16 GGUF | 916 | Active |
| GLM-5-REAP-50pct-UD-IQ2_M | GLM-5 | 50% | UD-IQ2_M | 1,200 | Active |
| Qwen3.5-264B-REAP | Qwen3.5 | Unknown | BF16 | 1,100 | Active |
| Qwen3.5-264B-REAP-FP8 | Qwen3.5 | Unknown | FP8 | 74 | Active |
| Qwen3.5-122B-A10B-REAP-20-GGUF | Qwen3.5 | 20% | GGUF | 5,900 | Proven |
| Qwen3.6-28B-REAP | Qwen3.6 | Unknown | BF16 | 789 | Active |
| Qwen-3.5-28B-A3B-REAP | Qwen3.5 | 20% | BF16 | 898 | Proven |
| gemma-4-21b-a4b-it-REAP | Gemma-4 | 20% | BF16 | 5,160 | Proven |
| gemma-4-19b-a4b-it-REAP | Gemma-4 | Unknown | BF16 | 1,550 | Proven |
| GLM-4.7-REAP-50-W4A16 | GLM-4.7 | 50% | W4A16 | 414 | Proven |
| GLM-4.7-REAP-40-W4A16 | GLM-4.7 | 40% | W4A16 | 319 | Proven |
| GLM-4.7-REAP-50 | GLM-4.7 | 50% | BF16 | 282 | Proven |
| DeepSeek-V3.2-REAP-508B-A37B | DeepSeek | Unknown | NVFP4 | 2 | Active |
| Nemotron-3-Super-120B-REAP-50pct | Nemotron | 50% | BF16 | 422 | Active |
| Nemotron-3-Super-120B-REAP-25pct | Nemotron | 25% | BF16 | 301 | Active |

**Checkpoint vs Release Count:**
- Distinct base models: GLM-5.1, GLM-5, Qwen3.5, Qwen3.6, Gemma-4, GLM-4.7, DeepSeek-V3.2, Nemotron-3, Trinity
- Estimated checkpoint-level attempts: 15-20
- Release-level uploads: 22+

---

## BENCHMARK DATA COLLECTED (To Be Normalized)

### Qwen-3.5-28B-A3B-REAP (20% prune)
| Benchmark | Pruned | Original | pp Drop | Rel Drop | Tier |
|-----------|--------|----------|---------|----------|------|
| MMLU avg | 80.89% | 84.34% | 3.45 pp | 4.09% | 3 |
| MMLU Humanities | 76.35% | 82.40% | 6.05 pp | 7.34% | 3 |
| MMLU STEM | 78.88% | 81.46% | 2.58 pp | 3.17% | 3 |
| HumanEval | 73.2% | 76.2% | 3.0 pp | 3.94% | 1 |
| HumanEval+ | 70.1% | 72.0% | 1.9 pp | 2.64% | 1 |
| ARC-Challenge | 60.40% | 60.00% | +0.40 pp | +0.67% | 3 |
| BoolQ | 89.20% | 88.00% | +1.20 pp | +1.36% | 3 |
| HellaSwag | 75.60% | 76.40% | 0.80 pp | 1.05% | 3 |
| WinoGrande | 76.80% | 77.20% | 0.40 pp | 0.52% | 3 |

**Verdict:** Marginal (coding preserved, but perplexity degraded 6.83→9.51)

### Gemma-4-21b-a4b-it-REAP (20% prune)
| Benchmark | REAP 20% | REAP 30% | Dense | Drop (20%) | Tier |
|-----------|----------|----------|-------|--------------|------|
| Elementary Math | 90% | 88% | 92% | 2.17% | 2 |
| Philosophy | 88% | 74% | 92% | 4.35% | 3 |
| World Religions | 64% | 48% | 90% | 28.89% | 3 |
| College CS | 76% | 68% | 56% | N/A (improvement) | 1 |
| HS Math | 44% | 48% | 24%* | N/A | 2 |
| GSM8K | 84% | -- | 86% | 2.33% | 2 |

**Verdict:** Mixed (n=50 samples, extraction issues noted)

### GLM-5 REAP-50% (Pilot results, n=10/category)
| Quant | GSM8K | HumanEval | Terminal-bench | SWE-bench | Overall |
|-------|-------|-----------|----------------|-----------|---------|
| Q3_K_M | 80% | 90% | 90% | 100%* | 88% |
| UD-IQ2_M | 60% | 80% | 90% | 100%* | 80% |
| UD-IQ2_XXS | 20% | 70% | 100% | 100%* | 66% |

*Sample size critically small (n=10). High confidence in degradation trend only.

---

## EVIDENCE GAPS IDENTIFIED

1. **SWE-Pro / SWE-Bench Verified**: Not explicitly reported for most models
2. **Terminal-Bench**: Only pilot data (n=10) for GLM-5 series
3. **browser-use**: No data found
4. **Dense baselines**: Often implied from paper, not posted on cards
5. **Sample sizes**: Many use n=50 proxy evaluations
6. **Author warnings**: GLM-5.1 series has "DO NOT USE" warnings on pruned variants
7. **Prune vs Quant**: Heavily quantized models not separately evaluated

---

## PRELIMINARY SUCCESS RATE (Checkpoint Level)

**Clear Wins (0 found with high confidence):**
- None meeting strict criteria (Tier 1 benchmarks with <5% drop, adequate sample size)

**Marginal (3-4 candidates):**
- Qwen-3.5-28B-A3B-REAP (20%): ~4% coding drop, but perplexity degraded
- Gemma-4-21b-a4b-it-REAP (20%): Mixed results, small samples
- Qwen3.5-122B-A10B-REAP-20-GGUF: Limited benchmark data

**Failure/High Risk (3+ found):**
- GLM-5.1-555B-A14B-REAP-40: 29% repetition loops
- GLM-5.1-555B-A14B-REAP-50: "Deprecated/broken", "DO NOT USE"
- GLM-5-REAP-50pct-UD-IQ2_XXS: Severe degradation (66% overall)
- Nemotron-3-Super-120B-REAP-50pct: Draft status

**Unknown/Not Benchmarked (majority):**
- Most GLM-5.1 series: Explicitly marked "not benchmarked"

**Estimated Checkpoint-Level Success Rate: 15-25%** (2-3 marginal out of ~15 attempts)

---

## STOPPING RULE RECOMMENDATION (Draft)

Based on data observed:

**Conclusion:** REAP pruning in solo/community settings shows **~15-25% checkpoint-level success rate** with **0% clear wins** meeting strict Tier 1 criteria.

**Recommendation:**
A solo researcher should cease REAP attempts after **3 consecutive marginal/failed results** at conservative prune ratios (≤30%), given the high observation/recovery compute cost and observed degradation patterns.

The method may be viable only with:
- Large calibration datasets (500k+ samples)
- Conservative prune ratios (≤20%)
- Extensive validation infrastructure
- Acceptance of perplexity degradation for size reduction

**Not economically justified** for most solo practitioners given alternative methods (distillation, quantization without pruning) showing better stability.

---

*Analysis in progress. Awaiting full data extraction from all model cards.*
