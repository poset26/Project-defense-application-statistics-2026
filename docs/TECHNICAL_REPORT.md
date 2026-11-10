# Anomaly Guard — Technical Report

This report matches the code and the checked-in full-simulator run in `results/metrics_summary.json` (11,500 generated requests). An **alert** means a request looks unusual relative to route rules or a legitimate baseline. It is not proof of a successful attack. Ground-truth labels record simulator intent.

## 1. The operating point

The 1% false-positive budget is fit on **ordinary validation traffic only** (the same `legitimate` group the profiles were trained on). Difficult legitimate text and distribution-shift comments do not choose the cutoff. They are a stress test.

On that budget, the detector to defend is **`rules_plus_full_stats`**:

| | Malicious recall | Ordinary test FPR (n=270) | Difficult FPR (n=402) | Shift FPR (n=113) | Mixed legitimate FPR |
|--|------------------|---------------------------|----------------------|-------------------|----------------------|
| `rules_only` | 0.24 | 0 | 0 | 0 | 0 |
| `rules_plus_full_stats` | **0.92** | **0** | 0.75 | 1.00 | 0.53 |
| `entropy_only` | 0.18 | 0.011 | 0 | 0.46 | 0.070 |

`rules_plus_full_stats` alert slices: signature-easy 1.00, rule-evasive 0.85, size/distribution 1.00. XSS alert recall is 1.00. Rules alone still have XSS **alert** recall 0 and XSS **review** rate 0.69, because the test rows are medium-severity `onerror` tags and evasive strings.

Ordinary traffic meets the budget. The mixed false-positive rate does not, because difficult text and longer comments were removed from the cutoff and now sit above it. The previous precision of 0.83 was the same model with the shift group allowed to lift the threshold. That number is not the operating point anymore.

### Base rate

Precision on this file uses a 36% attack mixture (450 malicious, 785 legitimate-family). It is not an operational detection rate.

| Rates | Posterior at the file prior | At a 1% prior | At a 0.1% prior |
|-------|-----------------------------|---------------|-----------------|
| Hybrid, mixed FPR 0.53, recall 0.92 | 0.50 | 0.017 | **0.0017** |
| Hybrid if the stream is ordinary traffic (observed FPR 0) | 1 | 1 | 1 |
| Rules, observed FPR 0 | 1 | 1 | 1 |
| Rules, Wilson upper on the mixed FPR (~0.49%) | 0.97 | 0.33 | **0.047** |

If production traffic looks like the ordinary training population, quote the ordinary false-positive rate and its Wilson upper bound, not a posterior of 1. If production includes the difficult and shifted text in this simulator, an alert from the hybrid is about a 0.2% chance of being an attack at a 0.1% prior.

Replay and the dashboard load `baseline.settings.json` when the mode matches. Otherwise they recalibrate on ordinary validation traffic. The cutoff in the report and the cutoff on the screen are the same one.

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
2. **Length** — robust z `0.6745 * |log1p(x) − median| / MAD`, with the MAD of the log sample.
3. **Chi-square composition** — the decision score is χ²/n, then a validation percentile. Expected counts are nudged by 0.25 and rescaled. Characters in one string are not independent trials, so the teaching p-value is baseline incompatibility, not a Type I error. It is shown only when every expected count is at least 5.
4. **Confidence intervals** (`ci_only`) — a 95% Student-t prediction interval for a new log length, and Wilson intervals for character proportions.
5. **Shannon entropy** (`entropy_only`) — median and MAD. Not inside mean fusion.
6. **PAYL, bigrams, OLS residual, Mahalanobis, length-stratified chi-square, Isolation Forest.** `lda_only` is distance to the legitimate mean, not linear discriminant analysis. `payl_only` thresholds the raw byte-histogram distance.

Statistics-only modes record rules and do not let them decide. `full_stats` and `rules_plus_full_stats` aggregate by the **mean** of per-type percentiles. `vote2_stats` uses the **second-highest** type percentile. `combined_stats` uses the max of length and chi-square.

Percentile pools use all non-attack validation traffic. The alert cutoff uses ordinary validation only.

`operating_point_1pct` is a descriptive read of the **mixed** test sweep. It does not move the deployed cutoff. For the hybrid, that sweep’s nearest 1% mixed false-positive point has malicious recall 0.33, not the deployed 0.92. The deployed point spends its false positives on difficult and shifted traffic in order to keep ordinary traffic at 0.

## 4. Other modes at the same ordinary cutoff

| Mode | Recall | Ordinary FPR | Mixed FPR | Signature | Evasive | Size |
|------|--------|--------------|-----------|-----------|---------|------|
| `length_only` | 0.12 | 0.011 | 0.15 | 0 | 0.05 | 0.50 |
| `combined_stats` | 0.12 | 0.011 | 0.20 | 0 | 0.05 | 0.50 |
| `entropy_only` | 0.18 | 0.011 | 0.07 | 0 | 0.17 | 0.50 |
| `ci_only` | 0.51 | 0.015 | 0.40 | 0.50 | 0.52 | 0.50 |
| `vote2_stats` | 0.62 | 0.011 | 0.25 | 0.65 | 0.48 | 1.00 |
| `payl_only` | 0.98 | 0.015 | 0.66 | 1.00 | 0.96 | 1.00 |
| `full_stats` | 0.92 | 0 | 0.53 | 1.00 | 0.85 | 1.00 |
| `rules_plus_stats` | 0.36 | 0.011 | 0.20 | 0.57 | 0.20 | 0.50 |
| `regression_only` | 0 | — | high | 0 | 0 | 0 |

PAYL’s ordinary false-positive rate is about 1.5%. Difficult legitimate and every shifted comment alert, because the raw cutoff that leaves 1% of ordinary traffic through is far below those groups. Mean fusion (`full_stats`) keeps ordinary FPR at 0 and still alerts on 75% of difficult text and all shifted comments. Entropy is the statistical mode with the smallest mixed false-positive rate (0.07), at recall 0.18.

**Contamination.** Replacing 54 of 5,444 legitimate training rows (1%) with train-split attacks and refitting `combined_stats` left recall at 0.12. The mixed false-positive rate moved from 0.20 to 0.15. Length medians did not move. A 1% leak does not relocate the median.

## 5. XSS, alert versus review

| Mode | XSS alert recall | XSS review rate | Support |
|------|------------------|-----------------|---------|
| `rules_only` | 0 | 0.69 | 84 |
| `rules_plus_stats` | 0.14 | 0.86 | 84 |
| `rules_plus_full_stats` | 1.00 | 0 | 84 |

Rules do not fail to notice the signature-like `onerror` rows. Those rules are medium severity, so the decision is review. The lower ordinary-only cutoff makes the full statistical stack alert on the same rows. Quoting only alert recall for `rules_only` hides a 69% review rate.

## 6. What this generator cannot support

Attacks are written in blocks, timestamps step inside those blocks, and the file is shuffled. SPRT, CUSUM, and Hawkes models on that order would fit the generator. They are not implemented. A Kalman filter needs a plant equation these fields do not have.

## 7. Limits

- Ordinary FPR near 1% is not mixed FPR near 1%. Report both.
- The descriptive 1% sweep is on the mixed mask. It is not the deployed cutoff.
- The dashboard is a replay tool, not a packet capture.
- Alerts are not intrusions.
- Benjamini–Hochberg is implemented and tested, and it is not part of any decision.

## 8. Reproduce

```bash
pip install -e ".[dev]"
anomaly-guard generate --output data/simulated
anomaly-guard evaluate --data data/simulated --output results
pytest
streamlit run anomaly_guard/app.py --server.port 8765 --server.address 127.0.0.1
```

The Evaluation page shows the mode table, attack slices, ordinary versus stress false-positive rates, base-rate posteriors, XSS alert recall next to review rate, the contamination refit, and held-out family counts.
