# Statistical models research (for course framing)

This note is the statistics-course map for Anomaly Guard. It records which interpretable procedures were recommended, which ones are now in the detector, and which ones stayed out on purpose. Current numbers and the operating point are in [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md). This file is not a second results table.

**Checked-in full run, ordinary-only 1% cutoff** (see the technical report for the group split):

| Mode | Malicious recall | Ordinary test FPR | Mixed legitimate FPR |
|------|------------------|-------------------|----------------------|
| `rules_only` | 0.24 | 0 | 0 |
| `entropy_only` | 0.18 | 0.011 | 0.07 |
| `rules_plus_full_stats` | 0.92 | 0 | 0.53 |
| `payl_only` | 0.98 | 0.015 | 0.66 |

The mixed rate is high because difficult text and distribution shift do not set the cutoff and then fail it. Quoting only the 0.92 recall without the 0.53 mixed false-positive rate is the accuracy trap this note warned about.

---

## 1. How to frame the project in a statistics course

| Statistical idea | Where it already appears | What to argue in class |
|------------------|--------------------------|------------------------|
| **Novelty detection** | Train on legitimate only; score new requests | Same split discipline as one-class / outlier lit.; not supervised classification of attacks |
| **Robust location & scale** | `0.6745 * \|x − median\| / MAD` on `log1p(length)`; the constant is in the numerator | Resistant to heavy tails. A reported “z > 3” now matches that formula |
| **Goodness-of-fit** | χ²/n on character categories, plus length-band χ². Cell contributions are in the explanation | The p-value is baseline incompatibility. Characters in one string are not independent trials |
| **Prediction and proportion intervals** | `ci_only`: Student-t interval for a new log length; Wilson intervals for category proportions | Distance is in interval widths |
| **Empirical calibration** | Percentile pools on non-attack validation. The 1% cutoff uses **ordinary** validation only | The threshold is estimated. Difficult and shifted traffic are a stress test, not the budget |
| **Base rate** | `base_rate` rows in `metrics_summary.json` | P(attack \| alert) at the file’s 36% mixture and at 1% and 0.1% priors. Precision on this file is not an operational rate |
| **Binomial inference** | Wilson interval on the legitimate false-positive rate | Uncertainty on that rate. For rules, the observed zero should be quoted with the Wilson upper end |
| **Precision–recall** | One curve and one threshold sweep per mode | The sweep is descriptive. It does not move the deployed cutoff |
| **Multiple testing** | Request-level percentile cutoff | Benjamini–Hochberg is coded and tested, and it is **not** called from a decision. Do not claim it as a correction |
| **Ablations** | Rules, each statistic, mean fusion, second-highest percentile, raw PAYL | Each mode is one comparison |

**Caution (good for class points):** A χ² *p*-value or an “empirical percentile” is **incompatibility with a reference traffic model**, not a posterior probability of attack (ASA statement on *p*-values).

---

## 2. Why length and chi-square stay near the size slice

On the current ordinary-only cutoff, `combined_stats` malicious recall is about **0.12**, with ordinary false-positive rate about **1.1%** and mixed legitimate false-positive rate about **0.20**. Signature-easy recall is 0. Rule-evasive recall is about 0.05. The size slice is about 0.50.

1. **Task definition.** Statistics-only modes do not use signature rules. Textbook strings are not what length and chi-square are for.
2. **Features.** Unordered character bins miss token structure. PAYL and the mean-fused stack are what move rule-evasive recall, and they spend the difficult and shift groups to do it.
3. **Two false-positive rates.** Ordinary test traffic can sit near 1% while the mixed rate is much higher. The stress groups are the difference. Do not average them into one sentence.
4. **Aggregation.** Max of length and chi-square does not recover PAYL. Mean fusion does, and it also alerts on most difficult text once the cutoff is not lifted by the shift group.

**Implication.** Another black-box model is not the next statistics question. The open claim is which false-positive population the 1% budget is allowed to see.

---

## 3. Recommended upgrades, and where they stand

The subsections below are the original recommendations. Status as of the ordinary-cutoff run:

### 3.1 Stratified length bands for composition — implemented

Length-quartile chi-square lives in `anomaly_guard/core/advanced_stats.py` and is part of `full_stats`. It is not a separate ablation mode.

**Problem:** Comparing a 12-character login username to a 200-character comment baseline confounds length and composition.

**Method:** On training legitimate data only, bin each field into length quantiles (e.g. quartiles). Estimate category probabilities **within bin** (*p*ᵢ|bin). At score time, use the bin for observed length; run χ² only if all *E*ᵢ ≥ 5 in that bin, else `insufficient_evidence`.

**Course tie-in:** Stratified goodness-of-fit; randomization / blocking in experimental design.

---

### 3.2 χ² cell contributions in explanations — implemented

The composition detail string lists the top cells. The decision still uses χ²/n plus a percentile, not the teaching p-value.

Replace a single disguised scalar with:

- Cell contribution: *(Oᵢ − Eᵢ)² / Eᵢ*
- Signed share difference: *p̂ᵢ − pᵢ* (RankShift-style, arXiv:2608.28922)
- Optional **adjusted standardized residuals** when df > 1 (Agresti/Haberman; Cornell note on contingency tables)

**Outputs:** “Quotes ↑ vs baseline; digits ↓” per field—ideal for alert inspector text.

**Course tie-in:** Omnibus test + **follow-up** residual analysis (standard practice in categorical data).

---

### 3.3 log1p before robust length scoring — implemented

Scoring uses the MAD of the `log1p` sample. The robust z is `0.6745 * |x − median| / MAD`.

HTTPS/payload lengths are right-skewed. Literature (e.g. robust Z in security analytics; Stratosphere-style pipelines) uses:

*y = log(1 + length_bytes)* then median/MAD with **σ_robust = max(1.4826 × MAD, ε_floor)*.

**Course tie-in:** Transform for variance stabilization; robust z on non-Gaussian data.

---

### 3.4 N-gram and PAYL profiles — implemented

`payl_only` scores a 256-bin byte histogram and thresholds the **raw** distance. `ngram_only` uses median and MAD of the top bigrams. On this file PAYL’s ordinary false-positive rate is about 1.5%, and its mixed rate is about 0.66 because difficult and shifted traffic all clear that raw cutoff.

**Reference:** Krüger et al. / PAYL (payload histograms); IEEE ANTS 2018 n-gram profiling (median/average counts of *k*-grams on **normal** HTTP).

**Procedure (class-friendly):**

1. On legitimate training, for each endpoint+field, record count of each byte *k*-gram (k = 2–4).
2. Store **median count** (or median + MAD) per gram.
3. Score = max or sum of robust z over grams that **appear** in the test string; optional **anomaly-to-normal ratio** (ANTS paper).

**Why for you:** Captures `uni/**/on`, doubled `%` patterns, etc., without new regex rules.

**Course tie-in:** Multivariate counting; robust profiling; multiple testing across grams (top-*M* contributors only in explanation).

---

### 3.5 Empirical null of the composite score — implemented as a validation quantile, not a permutation test

The alert cutoff is the ordinary-validation quantile of the aggregate score at a 1% target. Component scores are still turned into percentiles, then combined by max, mean, or the second-highest type percentile. There is no permutation or leave-one-out redraw of that null. A permutation test that shuffles characters inside a string would still be the wrong null for dependent text, and it was not added.

**Course tie-in:** Estimating a critical value under the working hypothesis that ordinary validation traffic is exchangeable with the traffic you are willing to call normal.

---

### 3.6 Bonferroni or Benjamini–Hochberg — not used in decisions

`benjamini_hochberg` is unit-tested and never called from scoring. The request-level cutoff is the error rate that is actually controlled. Citing the unused function as a multiple-testing correction would overclaim.

Several fields and components are scored on each request. Max-of-percentiles would let one of them alert. Mean fusion and the second-highest percentile are the combinations that were measured instead of a p-value correction.

**Course tie-in:** Multiple comparisons. The unit that has a budget here is the request, not the field.

---

### 3.7 Wilson intervals on proportions — implemented; beta-binomial is not

`proportion_ci` scores a field by how far a category share sits outside the training Wilson interval. There is no beta-binomial posterior predictive.

For large fields (comments), compare observed category shares to baseline using:

- **Wilson / Agresti–Coull** intervals on each proportion, or
- **Beta–Binomial** posterior predictive: P(observed | legit baseline).

**Course tie-in:** Categorical data; Bayesian intro without neural nets.

---

### 3.8 Mahalanobis on the feature vector — implemented; the mode name is historical

`lda_only` is one-class distance to the legitimate mean with a ridge-stabilized covariance. There is no attack class, so it is not linear discriminant analysis. The empirical percentile remains the alarm. An F or χ² critical value on that distance was not adopted, because the features are skewed counts and fractions. Feature loadings are not exported.

**Course tie-in:** Multivariate distance as an outlier score, not as a classifier trained on attacks.

---

### 3.9 What to **de-prioritize** for the statistics narrative

| Approach | Role in your report |
|----------|---------------------|
| **Isolation Forest** (already optional) | Nonparametric benchmark; cite impurity / isolation path as black-box contrast |
| **Deep sequence models** | Out of scope; breaks explainability requirement |
| **Retuning `rules.yaml` to raise statistics-only recall** | Confounds the comparison. Rule-evasive strings exist so signatures can fail |

---

## 4. Experimental matrix that was actually run

Train on legitimate only. Percentile pools use non-attack validation. The 1% cutoff uses ordinary validation. Test metrics are alert-only, with review separate, and with ordinary / difficult / shift false-positive rates side by side.

| Estimator | What it shows on this file |
|-----------|----------------------------|
| Rules only | Signature-easy recall about 0.57. Size slice 0. Ordinary and stress FPR 0. XSS alert recall 0, review rate about 0.69 |
| Length or chi-square | Size-slice recall about 0.50. Signature-easy 0. Ordinary FPR about 1% |
| Entropy | Rule-evasive recall about 0.17, size 0.50, mixed FPR about 0.07 |
| PAYL raw distance | Rule-evasive recall about 0.96. Ordinary FPR about 1.5%. Difficult and shift alert rates are 1 |
| Mean-fused stack, with or without rules | Malicious recall about 0.92. Ordinary FPR 0. Mixed FPR about 0.53 |
| OLS residual | Recall 0. Kept as a negative result |

The class hypothesis that “hybrid dominates rules at a 1% mixed false-positive rate” is **not** what this cutoff shows. The hybrid dominates rules on recall at a 1% **ordinary** rate, and it does not hold a 1% rate once difficult and shifted traffic are included. The descriptive mixed-mask sweep still has a point near 1% mixed FPR with hybrid recall about 0.33. That point is not the deployed cutoff.

---

## 5. Where the code lives

| Idea | Module |
|------|--------|
| Length, chi-square, entropy, prediction interval, Wilson proportions | `anomaly_guard/core/statistics.py` |
| PAYL, bigrams, length-band χ², OLS, Mahalanobis | `anomaly_guard/core/advanced_stats.py` |
| Mean, max, and second-highest fusion; raw PAYL cutoff; settings load | `anomaly_guard/core/decision.py` |
| Ordinary validation split | `validation_ordinary_records` in `anomaly_guard/evaluation/splits.py` |
| Group false-positive rates, base-rate table, contamination refit | `anomaly_guard/evaluation/experiments.py` |

---

## 6. References (starting set)

1. PAYL-style payload histograms — Wang & Stolfo, *Anomalous Payload-Based Network Intrusion Detection* (classic citation).
2. N-gram statistical profiling — IEEE ANTS 2018, “Statistical Profiling of n-grams for Payload Based Anomaly Detection for HTTP Web Traffic” (DOI 10.1109/ANTS.2018.8710080).
3. RankShift / categorical share divergence — arXiv:2608.28922 (explainable shift scores).
4. Robust z / MAD in security analytics — Google SecOps community blog on MAD robust z-scores; Stratosphere HTTPS AD module (log1p + median/MAD).
5. Chi-square interpretation — Agresti, *Categorical Data Analysis*; adjusted residuals (Cornell CSCU note).
6. Wilson intervals — already used via `statsmodels` in `evaluation/metrics.py`.
7. scikit-learn novelty detection docs — novelty vs outlier framing (your `fit`/`score` API).

---

## 7. Short answer for instructors

The statistical upgrades in this note are in the detector: log-length MAD, chi-square with cell contributions and length bands, Wilson proportion intervals, a prediction interval, PAYL, bigrams, entropy, and Mahalanobis distance. The cutoff is a validation quantile on ordinary traffic.

The result to teach is not a single recall. Rules stay quiet on every legitimate group and miss the size slice. Entropy is the quiet statistical mode. PAYL and the mean-fused hybrid catch evasive and size attacks and then alert on traffic that is legitimate but not the training population. At a 0.1% attack prior, the hybrid’s mixed false-positive rate makes P(attack | alert) about 0.2%. Sequential tests, Hawkes processes, and Kalman filters do not fit this generator and were not added.
