# Statistical models research (for course framing)

This note supports the **defense / statistics class** angle for Anomaly Guard. It recommends **interpretable statistical procedures** first—not deep learning—and maps them to what you already measure and to published HTTP payload baselines.

**Current checked-in full run (reference):** `combined_stats` malicious recall ≈ **9%** (alert only); **`rules_plus_stats` ≈ 33%**; **`size_distribution` slice** ≈ **50%** for stats vs **0%** for rules. The low global stats number is largely **slice mix** (evasive attacks) plus **honest ablations**, not only “under-trained.”

---

## 1. How to frame the project in a statistics course

| Statistical idea | Where it already appears | What to argue in class |
|------------------|--------------------------|------------------------|
| **Novelty detection** | Train on legitimate only; score new requests | Same split discipline as one-class / outlier lit.; not supervised classification of attacks |
| **Robust location & scale** | Median + MAD length z | Resistant to heavy tails; MAD×1.4826 links to σ under normality |
| **Goodness-of-fit** | Character category counts vs baseline *p*ᵢ | Pearson χ²; discuss **cell contributions** and **expected count ≥ 5** |
| **Empirical calibration** | Percentiles from validation legitimate traffic | Estimating an **operating threshold** from data (not fixed 3σ myth) |
| **Binomial inference** | Wilson CI on legitimate FPR | Uncertainty on **false-positive rate**, not “P(attack)” |
| **Precision–recall** | Exported curves on test | Imbalanced positives; prevalence **not** internet attack rate |
| **Multiple testing** | Many fields × two components | Risk of family-wise false alarms; Bonferroni or FDR as discussion |
| **Ablations** | `rules_only` vs `combined_stats` vs hybrid | Which **hypothesis** each mode tests |

**Caution (good for class points):** A χ² *p*-value or an “empirical percentile” is **incompatibility with a reference traffic model**, not a posterior probability of attack (ASA statement on *p*-values).

---

## 2. Why pure “combined_stats” is ~9% (substantive, not only ML failure)

1. **Task definition:** Statistics-only modes deliberately **ignore** signature rules. Most test positives are **rule-evasive** strings that match legitimate length/composition tolerances.
2. **Features:** Histogram bins over **unordered** characters miss token structure (e.g. `uni/**/on` looks like letters/slashes).
3. **Calibration:** Validation targets **1%** legitimate FPR; on test, alert FPR was **~6.4%**—threshold transfer + shift groups stress the baseline.
4. **Aggregation:** Max of length/composition **percentiles** is conservative; one weak field does not rescue another (appropriate statistically, harsh for recall).

**Implication for “train a better model”:** For a stats class, prefer **better estimators, stratification, and test design** before opaque models.

---

## 3. Recommended statistical upgrades (priority order)

### 3.1 Stratified length bands for composition (high value, stays interpretable)

**Problem:** Comparing a 12-character login username to a 200-character comment baseline confounds length and composition.

**Method:** On training legitimate data only, bin each field into length quantiles (e.g. quartiles). Estimate category probabilities **within bin** (*p*ᵢ|bin). At score time, use the bin for observed length; run χ² only if all *E*ᵢ ≥ 5 in that bin, else `insufficient_evidence`.

**Course tie-in:** Stratified goodness-of-fit; randomization / blocking in experimental design.

---

### 3.2 χ² with **per-category contributions** in explanations (not just one scalar)

Replace a single disguised scalar with:

- Cell contribution: *(Oᵢ − Eᵢ)² / Eᵢ*
- Signed share difference: *p̂ᵢ − pᵢ* (RankShift-style, arXiv:2608.28922)
- Optional **adjusted standardized residuals** when df > 1 (Agresti/Haberman; Cornell note on contingency tables)

**Outputs:** “Quotes ↑ vs baseline; digits ↓” per field—ideal for alert inspector text.

**Course tie-in:** Omnibus test + **follow-up** residual analysis (standard practice in categorical data).

---

### 3.3 **log1p** transform before robust length scoring (heavy tails)

HTTPS/payload lengths are right-skewed. Literature (e.g. robust Z in security analytics; Stratosphere-style pipelines) uses:

*y = log(1 + length_bytes)* then median/MAD with **σ_robust = max(1.4826 × MAD, ε_floor)*.

**Course tie-in:** Transform for variance stabilization; robust z on non-Gaussian data.

---

### 3.4 N-gram **occurrence profiling** (PAYL family, still statistical)

**Reference:** Krüger et al. / PAYL (payload histograms); IEEE ANTS 2018 n-gram profiling (median/average counts of *k*-grams on **normal** HTTP).

**Procedure (class-friendly):**

1. On legitimate training, for each endpoint+field, record count of each byte *k*-gram (k = 2–4).
2. Store **median count** (or median + MAD) per gram.
3. Score = max or sum of robust z over grams that **appear** in the test string; optional **anomaly-to-normal ratio** (ANTS paper).

**Why for you:** Captures `uni/**/on`, doubled `%` patterns, etc., without new regex rules.

**Course tie-in:** Multivariate counting; robust profiling; multiple testing across grams (top-*M* contributors only in explanation).

---

### 3.5 Empirical null / **permutation calibration** of composite score

Instead of pooling all length scores into one percentile ladder:

1. On validation legitimate, for each request compute vector of component scores **s** = (z_len, χ²_comp, …).
2. Build **null distribution** of *T(s)* = max(percentiles) by **permutation** or **leave-one-out** on validation (no test peeking).
3. Alert if *T* exceeds validation (1 − α) quantile (α = 0.01).

**Course tie-in:** Estimating critical values; exchangeability under H₀ “legitimate.”

---

### 3.6 Multiple fields: **Bonferroni or Benjamini–Hochberg FDR**

You test several fields × (length, composition). Family-wise error inflates legitimate FPR.

**Options:**

- **Bonferroni:** alert if any field *p*ᵢ ≤ α/m (conservative).
- **BH-FDR:** control expected false discovery rate among field alerts at q = 0.05.

**Course tie-in:** Multiple comparisons chapter; connect to your ~6.4% test FPR vs 1% target.

---

### 3.7 **Two-sample** or **empirical Bayes** for proportions (alternative to χ² on one string)

For large fields (comments), compare observed category shares to baseline using:

- **Wilson / Agresti–Coull** intervals on each proportion, or
- **Beta–Binomial** posterior predictive: P(observed | legit baseline).

**Course tie-in:** Categorical data; Bayesian intro without neural nets.

---

### 3.8 **LDA / sparse LDA** on the **same feature vector** (optional “multivariate stats” extension)

Mahalanobis / LDA on engineered counts is still **classical multivariate statistics** if trained on legitimate-only and scored by distance to legitimate mean (GSAD/RePIDS line of work—UTS thesis).

Use as **`comparison` mode only**, same splits as MAD+χ²; report **loadings** on features for interpretability.

**Course tie-in:** Discriminant analysis as outlier distance, not attack classifier.

---

### 3.9 What to **de-prioritize** for the statistics narrative

| Approach | Role in your report |
|----------|---------------------|
| **Isolation Forest** (already optional) | Nonparametric benchmark; cite impurity / isolation path as black-box contrast |
| **Deep sequence models** | Out of scope; breaks explainability requirement |
| **Tuning rules to fix 9%** | Confounds the stats experiment |

---

## 4. Proposed experimental matrix (for the write-up)

Hold **train = legitimate only**, **calibrate on validation legitimate**, report **test** with slices:

| Estimator | Detection target | Primary metric |
|-----------|------------------|----------------|
| Rules only | Signature-easy | Recall on `signature_easy` |
| MAD length (log1p) | Size-only + some evasive length | Recall on `size_distribution` |
| χ² + residuals | Composition shift | Recall on evasive + shift |
| N-gram median profile | Token structure evasion | Recall on `rule_evasive` |
| Hybrid (rules + calibrated max percentile) | Operational | FPR on legit (Wilson), recall by slice |

**Hypothesis for class:** “Composition and length baselines **increment** detection on size/shift slices; on **rule-evasive** slice, **n-gram or residual-guided** composition adds signal beyond MAD; hybrid **dominates** rules on overall recall at stated FPR budget.”

---

## 5. Mapping to your codebase (implementation sketch when approved)

| Idea | Likely module |
|------|----------------|
| Length bins → composition profiles | `core/statistics.py` fit/score |
| χ² contributions in `ComponentScore.detail` | `core/statistics.py` |
| log1p length | `_robust_length_score` input |
| N-gram tables | new `core/ngram_profile.py` + optional mode `ngram_only` |
| Permutation threshold | `evaluation/experiments.py` calibrate |
| FDR across fields | `core/decision.py` after per-field p/percentile |

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

**“Should we train a better model?”**  
For this class, **yes—but statistically**: stratify baselines, add **n-gram median profiling** and **χ² residual explanations**, stabilize length with **log1p + MAD**, and calibrate composite scores with a **validation-derived null**. That targets the **15% rule-evasive** and **50%-capable size** slices without collapsing the experiment into regex tuning. The **9%** headline is the wrong single number; slice-level and hybrid comparisons are the defensible result.
