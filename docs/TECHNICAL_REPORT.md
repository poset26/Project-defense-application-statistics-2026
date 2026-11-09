# Anomaly Guard — Technical Report

This document describes the **methodology implemented in code** as of commit `69708ba` (route-stratified splits and refined evaluation metrics). It is written for readers who may not open the repository. For operational commands, see [USAGE.md](USAGE.md); for module-level data flow, see [ARCHITECTURE.md](ARCHITECTURE.md).

**Scope disclaimer:** Anomaly Guard is an academic MVP. **Alerts indicate suspicious or statistically unusual HTTP requests relative to YAML rules and a legitimate baseline—not successful attacks.** Ground-truth labels record **simulator scenario intent**, not penetration-test outcomes.

---

## 1. Research question (as implemented)

The project asks whether **learned legitimate-traffic profiles** (robust field length and character-composition statistics), combined with **explicit rules**, can flag held-out **malicious-intent** simulated requests while keeping false positives low on legitimate-family traffic—compared with ablations that disable parts of the stack.

The implementation answers that question only to the extent the **dataset, splits, and mode wiring** allow. Section 6 and [UPDATE_PLAN.md](UPDATE_PLAN.md) document why the **checked-in small-demo results must not be treated as a definitive answer**.

---

## 2. Traffic and labels

### 2.1 Simulator

`anomaly_guard/simulation/` generates paired JSONL files:

- **`requests.jsonl`** — fields sufficient to replay HTTP-like requests (method, route, raw query, headers, body).
- **`ground_truth.jsonl`** — `scenario_group`, `scenario_id`, optional `attack_family`, `template_id`, and assigned `split`.

Scenario groups:

| Group | Role |
|-------|------|
| `legitimate` | Ordinary traffic templates |
| `difficult_legitimate` | Unicode, apostrophes, benign `<html>` mentions, etc. |
| `distribution_shift` | Legitimate but off-template (e.g. longer comments) |
| `benign_invalid` | Contract violations without attack payloads |
| `malicious_intent` / `hard_malicious_intent` | Attack payloads aligned with rule categories |

Attack families in the generator include SQLi, XSS, path traversal, and oversized payloads (see scenarios and `rules.yaml`).

### 2.2 Application surface

Three routes are in scope: **GET `/search`**, **POST `/login`**, **POST `/comment`**, defined in `anomaly_guard/config/routes.yaml`.

---

## 3. Detection methodology

### 3.1 Parsing and features

Requests are parsed under configured byte and field limits. Features include per-field lengths, entropy, encoding indicators, and **category fractions** over eight mutually exclusive character bins (`letter`, `digit`, `whitespace`, `quote`, `bracket`, `slash`, `other`).

### 3.2 Rule engine

Rules enforce route **schema** (types, required fields, lengths) and **suspicious patterns** (e.g. SQL tautology fragments, script tags, traversal sequences) with severities low / medium / high. Every `AnomalyGuard.score()` call evaluates rules; explanations always list firings.

### 3.3 Statistical baseline

**Training data:** train-split rows with `scenario_group == legitimate` only.

For each endpoint and configured field with enough non-empty training examples (≥5):

1. **Length:** median and MAD; scoring uses a robust z-style magnitude (`|x − median| / (0.6745 × MAD)`) or discrete-length logic when variation is negligible.
2. **Composition:** training category counts are merged for rare bins; at score time observed counts are compared to expected counts derived from training proportions. **`scipy.stats.chisquare`** yields a statistic; the implementation uses **`sqrt(chi2 / n)`** as the raw score. The chi-square **p-value is discarded** (it is not interpreted as an attack probability).

**Calibration:** On validation-split legitimate-family traffic, the detector collects raw aggregate scores (max over OK components), sorts them, and maps new requests to **empirical percentiles**. CLI `train` and `AnomalyGuard.calibrate()` also set alert/review percentile cutoffs from validation scores toward `max_false_positive_rate_target` (default **1%** on validation).

**Cold start:** Unknown endpoints or insufficient field evidence yield `insufficient_baseline`; decisions fall back toward rules and explanations note missing coverage.

### 3.4 Detector modes (ablations)

| Mode | Statistical use | Decision (high level) |
|------|-----------------|------------------------|
| `rules_only` | Ignored for decision | Rule severities only |
| `length_only` | Length components only; aggregate recomputed | Hybrid decision path |
| `composition_only` | Composition only | Hybrid |
| `combined_stats` | Both; aggregate = max raw component scores | Hybrid |
| `rules_plus_stats` | Full stats | Pattern short-circuit + percentiles + schema review |
| `isolation_forest` | IF on feature vector + rules | Rules first, then IF thresholds |

**Isolation Forest** (100 estimators, `contamination=0.02`, fixed decision thresholds) is included as an optional **multivariate comparison**, not the primary explainable path.

### 3.5 Decisions and explanations

Outputs: **`normal`**, **`review`**, **`alert`**. Hybrid logic (`_decision_hybrid`) can elevate **high-severity suspicious patterns** to alert before statistics; statistical aggregate percentiles above calibrated thresholds elevate to alert/review; schema rule hits can force **review** in several statistical modes.

The Streamlit **Normal vs unusual** page displays baseline **medians, MAD values, training counts, and composition `category_probs`** from saved JSON—it does **not** overlay a live request on those distributions.

---

## 4. Evaluation methodology

### 4.1 Splits (post–69708ba)

- Malicious and benign-invalid: split by **template group** hash to limit leakage.
- Legitimate-family: **per-request, route-aware** hashing so each HTTP route appears in train and validation (fix for earlier template-only hashing that stranded entire routes in one partition).

**Train:** legitimate train only → `fit`.  
**Validation:** legitimate, difficult_legitimate, distribution_shift → `calibrate`.  
**Test:** all groups present in test partition → reported metrics.

### 4.2 Experiment runner

`anomaly-guard evaluate` runs all six modes on the test split, exports predictions and `metrics_summary.json`, plots `detector_comparison.png`, and writes `error_analysis.json` for `rules_plus_stats`.

### 4.3 Metrics (as coded)

**Malicious-intent detection** (`compute_malicious_attack_metrics`):

- **Recall:** flagged / all malicious-intent test rows.
- **Precision:** TP / (TP + FP on **legitimate-family** rows only); benign-invalid FPs excluded from precision denominator.
- **False-positive rate:** fraction of **legitimate-family** test rows with a positive prediction.
- **Wilson score interval** (`statsmodels.stats.proportion.proportion_confint`, method `wilson`) on that legitimate-family FPR.

**`alert_predicted`** counts both **`alert` and `review`** as detections for these metrics.

**Benign-invalid:** rejection rate = flagged / all benign-invalid test rows.

**Any-anomaly binary metrics** treat malicious + benign-invalid as positive class with a legitimate-family mask for FPR Wilson interval.

**Ranking:** average precision on malicious intent uses `aggregate_percentile` when present; otherwise a fallback score derived from the decision label (see limitations).

**Per-family recall:** computed for malicious rows by `attack_family` on the test set (families with no test rows are omitted).

Latency: median and p95 of `processing_time_ms` per mode on the test run.

---

## 5. Reported numbers (full simulator, checked in)

`results/metrics_summary.json` reflects **`anomaly-guard generate`** without `--small` (~11,500 requests). Primary comparison modes on the **test split** (malicious intent task, **alert** decisions only):

| Mode | Malicious recall | Malicious F1 | Legit FPR (alert) | Rule-evasive slice recall |
|------|------------------|--------------|-------------------|---------------------------|
| `rules_only` | 0.24 | 0.39 | 0.0 | 0.15 |
| `combined_stats` | 0.09 | 0.15 | ~0.064 | 0.0 |
| `rules_plus_stats` | 0.33 | 0.46 | ~0.064 | 0.15 |

**Signature-easy** slice: `rules_only` recall ~0.57; **`rules_plus_stats`** ~0.57. **Size-distribution** slice: statistics modes ~0.50 recall vs rules ~0.0.

**Benign-invalid** alert rate (not 100% saturated): `rules_only` ~11%; `rules_plus_stats` ~19.5%.

Validation targeted **1%** legitimate FPR; **test** legitimate FPR for statistical modes is higher (~6.4% alert)—report actual test rates as in the JSON.

Regenerate:

```bash
anomaly-guard generate --output data/simulated
anomaly-guard evaluate --data data/simulated --output results
```

---

## 6. Current limitations (honest)

The following are **residual limitations** after the experiment-honesty update. Historical issues and the original fix plan are in **[UPDATE_PLAN.md](UPDATE_PLAN.md)** (status: applied).

1. **Partial simulator–rule overlap:** **Signature-easy** attacks still align with `rules.yaml`; use **`attack_slices.rule_evasive`** for the main statistical comparison.

2. **Calibration gap:** The **1% validation FPR budget** does not always hold on the **test** legitimate set (~6% alert FPR for combined/hybrid modes in the checked-in full run).

3. **Splitting drops attack families:** Template-group hashing can place entire malicious templates in non-test partitions; the checked-in summary’s **single-family** recall is a symptom. `held_out_family_meta` exists but is not wired into `experiments.py`.

4. **Small legitimate stencil:** `--small` legitimate traffic cycles a handful of strings, so profiles are tight and **difficult legitimate / shift traffic** can look extreme— inflating statistical FPR (~20–45% on legitimate mask in the demo file) while rules stay at 0% FPR on that mask.

5. **Aggregate scoring:** Length z-scores and composition `sqrt(chi2/n)` are **not calibrated to a common scale**; taking the **max** then one percentile differs from per-component percentile fusion described in design notes. `combined_stats` can duplicate `length_only` when length raw scores dominate numerically.

6. **Calibration vs measured FPR:** Validation targets ~1% FP budget on percentiles, but **`review` counts as detection** in metrics and alert thresholds may sit at the validation maximum—operating point on test legitimate traffic can be much higher than 1%.

7. **Isolation Forest assumptions:** Trained on purely legitimate train vectors with fixed contamination and thresholds; rules can preempt IF decisions.

8. **Metrics presentation:** Malicious-intent `tn` counts do not partition the full test set cleanly when benign-invalid rows exist; Wilson interval applies to the legitimate-family FPR rate. AP fallback when percentiles are missing can inflate separation for rule-saturated modes. PR curve helpers exist but are not exported in the default evaluate plot.

9. **Dashboard:** Baseline inspection is **tabular**, not a comparative visualization of a selected request against profiles.

For **what to change first** (true ablations, decoupled attacks, larger legitimate corpus, metric definitions), follow the ordered plan in [UPDATE_PLAN.md](UPDATE_PLAN.md)—without duplicating that document here.

---

## 7. Reproducibility

```bash
pip install -e ".[dev]"
bash scripts/demo.sh
pytest
streamlit run anomaly_guard/app.py --server.port 8765 --server.address 127.0.0.1
```

Artifacts: `data/simulated/`, `models/baseline.json`, `data/events.db`, `results/metrics_summary.json`.

---

## 8. Summary

Anomaly Guard implements a **transparent pipeline**: contract parsing, explicit rules, median/MAD length outliers, chi-square-derived composition scores with empirical percentile calibration, optional Isolation Forest, SQLite logging, and Streamlit review. Evaluation exports multi-mode metrics with **Wilson intervals on legitimate false-positive rate** and route-stratified splits for baseline training.

The **checked-in `results/` directory is a small-demo artifact**; overlapping rules and generator, non-isolated ablations, and split artifacts mean those numbers **must not** be read as proof that statistics beat rules in general. Use this report for what the code **does**; use [UPDATE_PLAN.md](UPDATE_PLAN.md) for how the experiment should be **fixed** before drawing strong conclusions.
