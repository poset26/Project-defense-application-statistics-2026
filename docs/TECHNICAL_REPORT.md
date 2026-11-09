# Anomaly Guard — Technical Report

This report matches the code and the checked-in full-simulator run in `results/metrics_summary.json` (11,500 generated requests). An **alert** means a request looks suspicious or unusual relative to route rules and a legitimate baseline. It is not proof of a successful attack. Ground-truth labels record simulator intent.

## 1. Question

Can statistical profiles of legitimate HTTP traffic flag simulated malicious requests that a small signature list misses, and what false-positive rate does that cost?

The comparison that answers this is not a single accuracy number. It is three slices, scored separately:

- **Signature-easy** — textbook strings the rules are written to catch.
- **Rule-evasive** — same families, payloads that miss `rules.yaml`.
- **Size / distribution** — long or repetitive fields with no SQL or script token.

Primary metrics count **`alert` only**. **`review`** is reported as its own rate. A Wilson interval is attached to the legitimate-family false-positive rate.

## 2. What the system does

Three routes: `GET /search`, `POST /login`, `POST /comment`.

Each request is parsed under byte and field limits, then scored by:

1. **Rules** — schema, size limits, and a short list of suspicious patterns (`rules.yaml`). High severity alerts. Medium severity reviews.
2. **Length** — median and MAD on `log1p(length)`.
3. **Chi-square composition** — character-category counts against the training proportions. A teaching p-value is shown only when every expected count is at least 5, and it is labeled as incompatibility with the baseline, not as an attack probability.
4. **Confidence intervals** (`ci_only`) — a 95% Student-t **prediction interval** for a new `log1p(length)`, and **Wilson intervals** for training character proportions. The score is how far the request sits outside the interval, in interval widths.
5. **PAYL, bigram MAD, OLS residual, Mahalanobis, length-stratified chi-square, Isolation Forest** — ablations. OLS regresses `log1p(length)` on entropy and category fractions.

Statistics-only modes do not let rules change the decision. Hybrid modes (`rules_plus_stats`, `rules_plus_full_stats`) alert on a high-severity pattern or on the statistical score.

**Fusion.** Single-component modes and `combined_stats` use the maximum empirical percentile. `ci_only`, `full_stats`, and `rules_plus_full_stats` use the **mean of per-type percentiles** (the max across fields of that type, then the mean across types). One component can no longer force an alert by itself.

**Calibration.** Profiles are fit on legitimate training traffic only. Alert and review cutoffs are the empirical quantiles of validation legitimate scores at `max_false_positive_rate_target` (default 1% on validation). Test false-positive rates are reported as measured. They are not used to move the cutoff.

`operating_point_1pct` in the metrics file is a **descriptive read of the test threshold sweep**: the sweep point with highest malicious recall whose test legitimate FPR is at most 1%, or the closest point if none qualify. Hybrid sweeps count a rule alert as a detection at every threshold. The deployed cutoff stays the validation cutoff.

## 3. Deployed operating point (validation cutoff, test set)

| Mode | Malicious recall | Precision | Legit alert FPR | Signature-easy | Rule-evasive | Size / distribution |
|------|------------------|-----------|-----------------|----------------|--------------|---------------------|
| `rules_only` | 0.24 | 1.00 | 0.00 | 0.57 | 0.15 | 0.00 |
| `length_only` / `composition_only` | 0.09 | ~0.5 | 0.04–0.05 | 0.00 | 0.00 | 0.50 |
| `ci_only` | 0.28 | 0.71 | 0.066 | 0.38 | 0.15 | 0.50 |
| `payl_only` | 0.64 | 0.72 | 0.144 | 0.65 | 0.51 | 1.00 |
| `full_stats` (mean fusion) | 0.09 | 0.57 | 0.039 | 0.00 | 0.00 | 0.50 |
| `rules_plus_stats` | 0.33 | 0.75 | 0.064 | 0.57 | 0.15 | 0.50 |
| `rules_plus_full_stats` | 0.33 | 0.83 | 0.039 | 0.57 | 0.15 | 0.50 |
| `regression_only` | 0.00 | 0.00 | 0.14 | 0.00 | 0.00 | 0.00 |
| `isolation_forest` | 0.09 | 0.48 | 0.056 | 0.00 | 0.00 | 0.50 |

Wilson 95% intervals on legitimate FPR are in `metrics_summary.json`. For `rules_plus_full_stats` the interval is about 2.8–5.6%.

### Reading the 1% budget on the test sweep

The validation cutoff does not always land at 1% FPR on test. The sweep says:

| Mode | Recall at the sweep point nearest 1% test FPR | FPR at that point | Budget met? |
|------|-----------------------------------------------|-------------------|-------------|
| `length_only` | 0.09 | 0.010 | closest (just over 1%) |
| `full_stats` | 0.09 | 0.010 | closest (just over 1%) |
| `rules_plus_full_stats` | 0.33 | 0.010 | closest (just over 1%); rules supply the recall |
| `ci_only` | 0.28 | 0.066 | no point at or under 1% |
| `payl_only` | 0.64 | 0.144 | no point at or under 1% |

PAYL’s recall does not come down to a 1% false-positive budget. At every threshold in its sweep the legitimate FPR stays about 14%, because a large share of legitimate validation scores already sit at percentile 100, so the cutoff cannot move.

Mean fusion does the opposite. `full_stats` no longer copies PAYL. Its deployed recall matches length and chi-square (the repetitive size slice only), at 3.9% legitimate FPR, and the sweep can place that same 9% recall next to about 1% FPR. The extra models do not add attack recall once they have to agree.

## 4. What statistics add, and what they do not

**They add the size slice.** Rules recall 0 there. Length, chi-square, confidence intervals, and the mean-fused stack recall 0.50 (the repetitive comments; the oversized search family does not alert). PAYL recalls 1.00 on that slice.

**Confidence intervals catch some signature-like oddity without a regex.** `ci_only` recalls 0.38 on signature-easy and 0.15 on rule-evasive, the same evasive recall as the rules, at 6.6% legitimate FPR. On the held-out traversal family its alert recall is 1.0, matching the rules. That is the interval detector doing the job the brief names.

**They do not replace signatures on text attacks once scores are fused.** Mean-fused `full_stats` has 0 recall on signature-easy and rule-evasive. `rules_plus_full_stats` keeps the rules’ 0.33 overall recall and cuts legitimate FPR from 6.4% (`rules_plus_stats`) to 3.9%, with higher precision (0.83). The gain of the full stack, under mean fusion, is fewer false alerts, not more detections.

**OLS regression does not detect this simulator’s attacks** (recall 0, legitimate FPR 0.14). It remains in the ablation so that negative result is visible.

## 5. XSS is a review problem, not a silent one

Test XSS alert recall is 0 in every mode, including rules. That is not because `<script>` fails to match. On this split the test XSS rows are `onerror` image tags (rule `EVENT_HANDLER`, **medium** severity, so **review**) and evasive strings such as `<svg/onload=…>` and `script:alert(1)`.

Held-out family counts (alert vs review), support 84 XSS and 41 oversized:

| Family | `rules_only` alert / review | `ci_only` alert | `full_stats` alert / review | `rules_plus_full_stats` review |
|--------|----------------------------|-----------------|-----------------------------|-------------------------------|
| traversal (84) | 1.00 / 0 | 1.00 | 0 / 0 | rules alert 1.00 |
| xss (84) | 0 / 0.69 | 0 | 0 / 0.42 | review 1.00 |
| oversized (41) | 0 / 0.39 | 0 | 0 / 1.00 | review 1.00 |

Under `full_stats`, median empirical percentiles on test XSS sit in the 70s to mid-90s (`payl` about 95, `length_ci` about 90). They are unusual relative to the comment baseline and still below the alert cutoff, which is why they queue as review rather than alert. Raising them to alert would spend the false-positive budget.

## 6. Limits that remain

- The 1% validation budget is not a guarantee on test. Report the measured rate and the sweep point.
- PAYL cannot be thresholded down to 1% FPR on this data.
- Oversized searches (a long run of one letter) often miss the alert cutoff for length and chi-square; repetitive punctuation comments do not.
- The dashboard compares a selected request with the baseline. It is a replay tool, not a live packet capture.
- Alerts are not intrusions.

## 7. Reproduce

```bash
pip install -e ".[dev]"
anomaly-guard generate --output data/simulated
anomaly-guard evaluate --data data/simulated --output results
pytest
streamlit run anomaly_guard/app.py --server.port 8765 --server.address 127.0.0.1
```

The Evaluation page shows the mode table, the three attack slices, the 1% sweep read, the XSS component profile, and held-out family alert and review rates.
