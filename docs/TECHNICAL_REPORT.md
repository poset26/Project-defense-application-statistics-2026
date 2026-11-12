# Anomaly Guard — Technical Report

This report matches the checked-in run in `results/metrics_summary.json`. The generator wrote **10,700** requests from seed **42**, and every row has a distinct `(method, route, query, body)`. The test split has **450** malicious rows and **705** legitimate-family rows (**271** ordinary, **340** difficult, **94** distribution-shift). An **alert** means a request looks unusual relative to route rules or a legitimate baseline. It is not proof of a successful attack.

Request ids are `req-` plus the generation index. The same seed rewrites the same ids, text, timestamps, and splits. Ordinary test strings do not repeat training strings (0 of 271).

## 1. The operating point

The 1% false-positive budget is fit on **ordinary validation traffic only** (the same `legitimate` group the profiles were trained on). Difficult legitimate text and distribution-shift comments do not choose the cutoff. They are a stress test.

A decision alerts only when the score is **strictly above** that cutoff. Threshold sweeps in `results/threshold_sweep_*.json` use the same comparison (`"comparison": "score > threshold"`). Isolation Forest alerts when its `decision_function` is **strictly below** a lower-tail cutoff. A tie at the cutoff does not alert.

The empirical percentile of a component is the share of calibration scores **strictly below** the raw score. A raw score above every calibration value is 100. A tie at the pool maximum stays below 100. That is why `entropy_only` can alert: its cutoff on this run is **98.76**, not 100, and it alerts on **124/450** attacks. Ordinary test false-positive rate for that mode is **0**.

On that budget, two hybrids are worth reading side by side. **`rules_plus_stats`** is the default. **`rules_plus_full_stats`** is the full statistical stack plus rules.

| | Malicious recall | Ordinary test FPR (n=271) | Difficult FPR (n=340) | Shift FPR (n=94) | Mixed legitimate FPR (n=705) |
|--|------------------|---------------------------|----------------------|------------------|------------------------------|
| `rules_only` | 0.256 (115/450) | 0 | 0 | 0 | 0 |
| `rules_plus_stats` (default) | **0.662** (298/450) | **0.007** (2/271) | 0 | 1.00 (94/94) | **0.136** (96/705) |
| `rules_plus_full_stats` | **0.909** (409/450) | **0.011** (3/271) | 0.497 (169/340) | 1.00 | **0.377** (266/705) |
| `entropy_only` | 0.276 (124/450) | 0 | 0 | 0.309 (29/94) | 0.041 (29/705) |

`rules_plus_full_stats` still alerts on every shifted comment and on about half of the difficult legitimate rows. That is why mixed false-positive rate is 0.377 while ordinary traffic is about 1.1%. The default hybrid keeps difficult text at 0 alerts and spends its mixed false positives on the shift group.

### Base rate

Precision on this file uses a 39.0% attack mixture (450 malicious, 705 legitimate-family). It is not an operational detection rate. Posteriors below are `P(attack | alert)` at the observed mixed false-positive rate, and again at that rate's Wilson upper bound.

| Rates | Posterior at the file prior | At a 1% prior | At a 0.1% prior |
|-------|-----------------------------|---------------|-----------------|
| Full hybrid, mixed FPR 0.377, recall 0.909 | 0.606 | 0.024 | **0.0024** |
| Full hybrid, Wilson upper on that FPR (0.414) | 0.584 | 0.022 | **0.0022** |
| Default hybrid, mixed FPR 0.136, recall 0.662 | 0.756 | 0.047 | **0.0048** |
| Rules, observed FPR 0 | 1 | 1 | 1 |
| Rules, Wilson upper on the mixed FPR (0.542%) | 0.968 | 0.323 | **0.045** |

If production traffic looks like the ordinary training population, quote the ordinary false-positive rate. If production includes the difficult and shifted text in this simulator, an alert from the full hybrid is about a 0.24% chance of being an attack at a 0.1% prior.

Replay and the dashboard load `baseline.settings.json` when the mode matches. Other modes take the cutoff stored for that mode in `metrics_summary.json`. Those cutoffs were fit in this same run.

## 2. Question

Can statistical profiles of legitimate HTTP traffic flag simulated malicious requests that a small signature list misses, and what false-positive rate does that cost once the budget is not allowed to see the stress traffic?

Three slices:

- **Signature-easy** — textbook strings the rules are written to catch.
- **Rule-evasive** — same families, payloads that miss `rules.yaml`.
- **Size / distribution** — long or repetitive fields with no SQL or script token.

Primary metrics count **`alert` only**. **`review`** is its own rate. A Wilson interval is attached to the mixed legitimate false-positive rate. Ordinary, difficult, and shift rates are reported separately.

## 3. What the system does

Three routes: `GET /search`, `POST /login`, `POST /comment`.

1. **Rules** — schema, size limits, and a short pattern list. High severity alerts. Medium severity reviews.
2. **Length** — robust z `0.6745 * |log1p(x) − log median| / MAD`, with the MAD of the log sample.
3. **Chi-square composition** — the decision score is χ²/n on every character class, including classes the training sample never saw, then a validation percentile. Expected counts are nudged by 0.5. Characters in one string are not independent trials, so the teaching p-value is baseline incompatibility, not a Type I error. It is shown only when every expected count is at least 5.
4. **Confidence intervals** (`ci_only`) — a 95% Student-t prediction interval for a new log length, and Wilson intervals for character proportions. The proportion interval uses the length of the request being scored.
5. **Shannon entropy** (`entropy_only`) — robust z against the training median and MAD, then the strict empirical percentile above. Not inside mean fusion.
6. **PAYL, bigrams, OLS residual, Mahalanobis, length-stratified chi-square, Isolation Forest.** `lda_only` is distance to the legitimate mean, not linear discriminant analysis. `payl_only` thresholds the raw byte-histogram distance.

Query names and values are percent-decoded **once**. `%71=hello` is the field `q`. `%25` is a literal percent and is not a malformed escape. A truncated sequence such as `%A` is kept and flagged. JSON bodies are not form-decoded, so `C++` and `50%25` stay as written.

Statistics-only modes record rules and do not let them decide. `full_stats` and `rules_plus_full_stats` aggregate by the **mean** of per-type percentiles. `vote2_stats` uses the **second-highest** type percentile. `combined_stats` uses the max of length and chi-square.

Percentile pools use all non-attack validation traffic. The alert cutoff uses ordinary validation only.

Ranking for precision-recall is the same score the decision uses. `rules_only` ranks by the strongest triggered severity (it is not a constant). `isolation_forest` ranks by the negated forest score, not by another component's percentile. `train` writes that forest beside the baseline as `baseline.isolation.joblib`.

`operating_point_1pct` is a descriptive read of the **mixed** test sweep. It does not move the deployed cutoff. For `rules_plus_full_stats`, that sweep’s nearest point with mixed false-positive rate at most 1% has malicious recall **0.347**, not the deployed **0.909**. The deployed point spends false positives on difficult and shifted traffic in order to keep ordinary traffic near 1%.

## 4. Other modes at the same ordinary cutoff

| Mode | Recall | Ordinary FPR | Mixed FPR | Signature | Evasive | Size |
|------|--------|--------------|-----------|-----------|---------|------|
| `length_only` | 0.113 | 0.007 | 0.136 | 0 | 0.041 | 0.50 |
| `combined_stats` | 0.569 | 0.007 | 0.136 | 0.667 | 0.541 | 0.50 |
| `entropy_only` | 0.276 | 0 | 0.041 | 0.238 | 0.219 | 0.50 |
| `ci_only` | 0.473 | 0.007 | 0.136 | 0.333 | 0.368 | 1.00 |
| `vote2_stats` | 0.816 | 0 | 0.252 | 0.667 | 0.831 | 1.00 |
| `payl_only` | 1.00 | 0.011 | 0.620 | 1.00 | 1.00 | 1.00 |
| `full_stats` | 0.909 | 0.011 | 0.377 | 1.00 | 0.831 | 1.00 |
| `rules_plus_stats` | 0.662 | 0.007 | 0.136 | 1.00 | 0.541 | 0.50 |
| `regression_only` | 0.318 | 0 | 0.133 | 0.333 | 0.248 | 0.50 |
| `isolation_forest` | 0.091 | 0 | 0.001 | 0 | 0 | 0.50 |

PAYL’s ordinary false-positive rate is 3/271. Difficult legitimate and every shifted comment alert, because the raw cutoff that leaves about 1% of ordinary traffic through is far below those groups. Mean fusion (`full_stats`) keeps ordinary FPR at 0.011 and still alerts on half of difficult text and all shifted comments. Entropy’s mixed false-positive rate is 0.041 at recall 0.276, with no ordinary test alerts.

**Contamination.** Replacing 54 of 5,376 legitimate training rows (1%) with train-split attacks and refitting `combined_stats` left recall at 0.569 and the mixed false-positive rate at 0.136. Length medians did not move. A 1% leak does not relocate the median.

## 5. XSS, alert versus review

Support is 84 test XSS rows.

| Mode | XSS alert recall | XSS review rate |
|------|------------------|-----------------|
| `rules_only` | 0.50 | 0.25 |
| `rules_plus_stats` | 0.619 | 0.381 |
| `rules_plus_full_stats` | 1.00 | 0 |

Rules do not fail to notice every signature-like row. `onerror` is medium severity, so some of those decisions are review. The full statistical stack alerts on all 84. Quoting only alert recall for `rules_only` hides the 0.25 review rate.

## 6. What this generator cannot support

Attacks are written in blocks, timestamps step inside those blocks, and the file is shuffled. SPRT, CUSUM, and Hawkes models on that order would fit the generator. They are not implemented. A Kalman filter needs a plant equation these fields do not have.

The `refNNNNN` token makes each row unique. It is not a model of real user text, and it does add a few digits to length and composition. Splits are by request id within the seed, not by an independent content hash.

## 7. Limits

- Ordinary FPR near 1% is not mixed FPR near 1%. Report both.
- The descriptive 1% sweep is on the mixed mask. It is not the deployed cutoff.
- The dashboard is a replay tool, not a packet capture.
- Alerts are not intrusions.
- Benjamini–Hochberg is implemented and tested, and it is not part of any decision.
- Public replay does not load the Isolation Forest file. The CLI does.

## 8. Reproduce

```bash
pip install -e ".[dev]"
anomaly-guard generate --output data/simulated --seed 42
anomaly-guard evaluate --data data/simulated --output results
pytest
streamlit run anomaly_guard/app.py --server.port 8765 --server.address 127.0.0.1
```

`generate` without `--seed` still uses 42. The Evaluation page shows the mode table from `results/metrics_summary.json` (or the copy under `anomaly_guard/publish_assets/` on the hosted demo).
