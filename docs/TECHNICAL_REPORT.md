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
5. **Shannon entropy** (`entropy_only`) — median and MAD of the field entropy. It is not included in mean fusion.
6. **PAYL, bigram MAD, OLS residual, Mahalanobis, length-stratified chi-square, Isolation Forest** — ablations. OLS regresses `log1p(length)` on entropy and category fractions. `lda_only` is Mahalanobis distance to the legitimate mean, not linear discriminant analysis.

The robust length z is `0.6745 * |log1p(x) − median| / MAD`, with the MAD taken on the log sample. An older form divided by `0.6745 * MAD` and approximated the log MAD by `MAD / median`. Percentile calibration is monotone in a constant rescaling, so the constant placement does not by itself change alerts. Using the MAD of the logs can.

**Chi-square.** The procedure that decides is χ²/n, then a validation percentile. Expected counts are nudged by 0.25 and rescaled to the observed total before `chisquare`. Characters inside one string are not independent trials, so the teaching p-value is not a Type I error. It is shown only when every expected count is at least 5, and the text calls it incompatibility with the baseline.

Statistics-only modes do not let rules change the decision. Hybrid modes (`rules_plus_stats`, `rules_plus_full_stats`) alert on a high-severity pattern or on the statistical score.

**Fusion.** Single-component modes and `combined_stats` use the maximum empirical percentile. `ci_only`, `full_stats`, and `rules_plus_full_stats` use the **mean** of per-type percentiles. `vote2_stats` uses the **second-highest** type percentile, so two component types must be unusual. `payl_only` thresholds the **raw** byte-histogram distance, because percentile 100 was tied too often for a 1% budget.

**Calibration.** Profiles are fit on legitimate training traffic only. Alert and review cutoffs are the empirical quantiles of validation legitimate scores at `max_false_positive_rate_target` (default 1% on validation). Test false-positive rates are reported as measured. They are not used to move the cutoff.

`operating_point_1pct` in the metrics file is a **descriptive read of the test threshold sweep**: the sweep point with highest malicious recall whose test legitimate FPR is at most 1%, or the closest point if none qualify. Hybrid sweeps count a rule alert as a detection at every threshold. The deployed cutoff stays the validation cutoff.

## 3. Deployed operating point (validation cutoff, test set)

| Mode | Malicious recall | Precision | Legit alert FPR | Signature-easy | Rule-evasive | Size / distribution |
|------|------------------|-----------|-----------------|----------------|--------------|---------------------|
| `rules_only` | 0.24 | 1.00 | 0.00 | 0.57 | 0.15 | 0.00 |
| `length_only` / `composition_only` | 0.09 | ~0.5 | 0.04–0.05 | 0.00 | 0.00 | 0.50 |
| `ci_only` | 0.28 | 0.71 | 0.066 | 0.38 | 0.15 | 0.50 |
| `payl_only` (raw distance) | 0.64 | 0.72 | 0.144 | 0.65 | 0.51 | 1.00 |
| `entropy_only` | 0.18 | 0.89 | 0.013 | 0.00 | 0.17 | 0.50 |
| `vote2_stats` (second percentile) | 0.59 | 0.63 | 0.197 | 0.65 | 0.43 | 1.00 |
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
| `entropy_only` | 0.18 | 0.0025 | yes |
| `payl_only` (raw distance) | 0.67 | 0.144 | no point at or under 1% on the mixed legitimate mask |

PAYL is now cut on the raw distance (validation threshold about 15.1, not percentile 100). The mixed legitimate false-positive rate is still 14.4%. Every one of those alerts is a distribution-shift comment (113/113). Ordinary test traffic has **zero** PAYL alerts (0/270), and difficult legitimate traffic also has zero. The 1% budget fails because the shift group is about 14% of the legitimate mask and all of it sits above the cutoff. Raw calibration did not create a 1% point on the mixture the metrics call “legitimate.”

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

## 6. Base rate

On this test file there are 450 malicious rows and 785 legitimate-family rows, so the mixture prior is about **36%**. Reported precision is P(attack | alert) at that prior. It is not an operational detection rate.

| Mode | Posterior at file prior | At a 1% prior | At a 0.1% prior |
|------|-------------------------|---------------|-----------------|
| `rules_plus_full_stats` (recall 0.33, FPR 0.039) | 0.83 | 0.078 | **0.008** |
| `payl_only` (0.64, 0.144) | 0.72 | 0.043 | 0.004 |
| `ci_only` (0.28, 0.066) | 0.71 | 0.041 | 0.004 |
| `entropy_only` (0.18, 0.013) | 0.89 | 0.13 | 0.014 |
| `rules_only` observed FPR 0 | 1.00 | 1.00 | 1.00 |
| `rules_only` using Wilson FPR upper (~0.49%) | 0.97 | 0.33 | **0.047** |

The same recall and false-positive rate that look like precision 0.83 on this file give about **0.8%** probability that an alert is an attack when attacks are 0.1% of traffic. `rules_only` has no observed false positive, so the point-estimate posterior is 1. The Wilson upper end is the honest substitute, and at a 0.1% prior it drops the posterior to about 5%.

## 7. False positives are not one population

The legitimate mask mixes three groups. Alert rates on the test split:

| Mode | Ordinary (n=270) | Difficult (n=402) | Distribution shift (n=113) |
|------|------------------|-------------------|----------------------------|
| `rules_only` | 0 | 0 | 0 |
| `rules_plus_full_stats` | 0 | 0 | 0.27 |
| `length_only` / `full_stats` | 0 | 0 | 0.27 |
| `ci_only` | 0 | 0.13 | 0 |
| `payl_only` | 0 | 0 | 1.00 |
| `entropy_only` | 0.007 | 0 | 0.071 |
| `vote2_stats` | 0 | 0.10 | 1.00 |
| `combined_stats` | 0 | 0.10 | 0.071 |

For the hybrid and for PAYL, the published false-positive rate is the shift group (longer comments the baseline was not fit on). Ordinary traffic, the group the profiles were fit on, is clean for those modes. Calibration still sees the shift group, so the threshold is partly set by traffic the model was not trained on. The shift rows are a stress test. They are not the same claim as “1% of normal traffic alerts.”

## 8. Middle fusion, entropy, and one contaminated refit

**Entropy.** `entropy_only` recalls 0.18 overall, 0.17 on rule-evasive text, and 0.50 on the size slice, at 1.3% legitimate FPR. The sweep has a point at or under 1% FPR with that same 0.18 recall. It does not flag signature-easy strings (recall 0). It is the statistical ablation closest to the false-positive budget, and it was kept out of the mean.

**Second-highest percentile.** `vote2_stats` recalls 0.59, including 0.43 on rule-evasive and 0.65 on signature-easy, close to PAYL’s 0.64 / 0.51. The legitimate FPR is 0.20, worse than PAYL, and the cutoff sits at 100, so the 1% budget is not met. Requiring a second agreeing component recovers most of PAYL’s attack recall and does not control the shift group, which is extreme on more than one component at once. Mean fusion remains the strict control (recall falls to 0.09). The middle rule is measured. It is not a better operating point on this file.

**Contamination.** 54 of 5,444 legitimate training rows (1%) were replaced by malicious rows from the train split. `combined_stats` was refit. Validation and test stayed clean. Length medians did not move (a median ignores 1% replacements). Test recall stayed 0.091. Legitimate false-positive rate rose from 0.064 to 0.144. The baseline’s center held. The tail, which sets alerts, did not.

## 9. What this generator cannot support

Attacks are written in blocks, timestamps step inside those blocks, and the file is then shuffled. A CUSUM, SPRT, or Hawkes model on file order or timestamp order would fit that block structure. Those methods are not implemented. A Kalman filter needs a plant equation these HTTP fields do not have.

## 10. Limits that remain

- The 1% validation budget is not a guarantee on test. Report the measured rate and the sweep point.
- PAYL cannot be thresholded down to 1% FPR on this data.
- Oversized searches (a long run of one letter) often miss the alert cutoff for length and chi-square; repetitive punctuation comments do not.
- The dashboard compares a selected request with the baseline. It is a replay tool, not a live packet capture.
- Alerts are not intrusions.

## 11. Reproduce

```bash
pip install -e ".[dev]"
anomaly-guard generate --output data/simulated
anomaly-guard evaluate --data data/simulated --output results
pytest
streamlit run anomaly_guard/app.py --server.port 8765 --server.address 127.0.0.1
```

The Evaluation page shows the mode table, the three attack slices, the 1% sweep read, false positives by legitimate group, the base-rate posteriors, the 1% contamination refit, the XSS component profile, and held-out family alert and review rates.
