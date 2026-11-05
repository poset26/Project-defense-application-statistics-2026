# Update plan: make the experiment able to fail

Status: proposed. This document does not change detector behavior. It records shortcomings of the same kind as “rules already catch every evaluated attack, so the statistical comparison cannot show anything,” and the work required to fix them.

Pulled baseline: `69708ba` (evaluation metric refinements on `main`). Numbers below are from `results/metrics_summary.json` produced by `scripts/demo.sh` on the small simulator (`data/simulated`, 900 requests).

## What is wrong, in one sentence

The published comparison cannot answer “do statistics help?” because the attacks that reach the test set are copies of the rule list, most other attack families never reach the test set, and the “statistics-only” modes still apply those rules.

## Findings

### 1. Rule-based detection is saturated because the attacks are the rules

`anomaly_guard/config/rules.yaml` and `anomaly_guard/simulation/scenarios.py` were written together.

| Family | Generator payload | Rule it is guaranteed to hit |
| --- | --- | --- |
| `sqli` | `' OR '1'='1`, `UNION SELECT`, `OR 1=1` | `SQL_OR_TAUTOLOGY`, `SQL_UNION_SELECT` |
| `xss` | `<script>`, `javascript:`, `onerror=` | `SCRIPT_TAG`, `EVENT_HANDLER` |
| `traversal` | `../`, `%2e%2e` | `PATH_TRAVERSAL` |
| `oversized` | 600 `A`s **plus** `' OR 1=1--` | `SQL_OR_TAUTOLOGY`, and it is also long |

On the checked-in small run, `rules_only` reports precision 1.0, recall 1.0, false-positive rate 0.0. A hybrid cannot beat that on this slice. In a real deployment, signature coverage is incomplete; here it is complete by construction, so length, composition, and Isolation Forest are not tested as detectors of attacks the rules miss.

### 2. The ablations are not ablations

`AnomalyGuard.score` always calls `evaluate_rules`. Then:

- `length_only`, `composition_only`, `combined_stats`, and `rules_plus_stats` all go through `_decision_hybrid`, which returns `alert` on any high-severity pattern before looking at the statistical score (`core/decision.py`).
- Those same modes return `review` on any schema rule, including `length_only` and `composition_only`.
- `isolation_forest` returns the rule decision first whenever rules already say `alert`.
- `alert_predicted` counts both `alert` and `review` as a detection (`evaluation/metrics.py`).

Saved metrics match this. `length_only`, `combined_stats`, and `rules_plus_stats` are the same row: recall 1.0, legitimate false-positive rate about 0.20, F1 about 0.84. Statistics did not change which attacks were caught. They only added false positives on top of the rules. Isolation Forest matches `rules_only` (F1 1.0, false-positive rate 0) because the rules fire first.

### 3. Almost every attack family is dropped before scoring

Malicious rows are assigned a split from `scenario_id:template_id` (`evaluation/splits.py`). `oversized_v1` is a single template, so the whole family lands in one partition. `partition_records` keeps malicious rows only when that partition is `test`. Families hashed into train or validation are deleted from the experiment. They are not used for training either (training is legitimate-only).

`results/metrics_summary.json` therefore has one family: `oversized` at recall 1.0. The “100%” figure is 100% of the one family that both survived the split and contains a literal tautology. `held_out_family_meta` exists and is never called from `evaluation/experiments.py`.

### 4. Benign-invalid traffic is saturated the same way

Every `benign_invalid` variant breaks the route contract on purpose: empty `q`, `page=abc`, empty username, broken JSON, or a 9000-character comment. Every mode reports `rejection_rate: 1.0`. That does not show that statistics separate invalid input from malicious intent. The schema rules do it, and the generators do not emit a schema-valid mistake.

### 5. “Normal” traffic is a handful of repeated strings

Ordinary searches cycle five phrases. Logins are `studentNNN` / `dummy-N`. Comments cycle three sentences. Median, MAD, and character profiles are fit on that stencil. Difficult legitimate text and the longer-comment shift then look extreme even when they are harmless, which is why statistical modes show roughly 20–45% false-positive rate on the legitimate mask while rules show 0%. The design target was on the order of 10,000 varied legitimate requests. The numbers in `results/` come from `anomaly-guard generate --small`.

### 6. Combined statistics do not combine comparable scores

Length uses a robust z-score. Composition uses `sqrt(chi-square / n)` after flooring expected counts at 0.5 and rescaling them (`core/statistics.py`). The aggregate is the **maximum of those raw numbers**, then one percentile. The design asked for a percentile per component, then a fixed aggregate. On this run `combined_stats` copies `length_only` exactly, which is what an incomparable `max()` does when one component has the larger numeric range.

The chi-square p-value is discarded. That part matches the design (p-values are not attack probabilities). What is missing is calibration inside length bands, and a score that is not dressed up as a chi-square after the expected counts have been rewritten.

### 7. The 1% false-positive budget is not the operating point being measured

Calibration sets `alert_threshold_percentile` to **100.0** for every statistical mode in the saved settings. The review threshold sits near 95.4, and review counts as a hit, so the measured false-positive rate is about 20%, not at most 1%. Isolation Forest uses fixed cutoffs (`-0.05`, `-0.12`) and `contamination=0.02` on data that is entirely legitimate, which is the wrong training assumption for novelty detection.

### 8. Ranking metrics and plots overstate separation

If a request has no `aggregate_percentile`, average precision substitutes 100 / 50 / 0 from the decision label (`evaluation/experiments.py`). `rules_only` then gets average precision 1.0 whenever it has no legitimate false positives. `precision_recall_points` is implemented and never plotted. There is no threshold sweep. The dashboard page “Normal vs unusual” prints baseline medians and category probabilities. It does not overlay the selected request on that baseline.

### 9. The malicious confusion counts do not partition the test set

`compute_malicious_attack_metrics` counts false positives only on legitimate-family rows, but true negatives are every non-malicious row that was not flagged, and benign-invalid rows are in neither the false-positive nor a clearly labeled third bucket. For `rules_only`, `37 + 0 + 69 + 0 = 106` while the test slice is 37 attacks + 80 invalid + 69 legitimate-family = 186. The Wilson interval is attached to a rate that is honest; the `tn` field is not a complement of that rate.

## Implementation plan

Work in this order. Do not retune thresholds on the final test split.

### A. Make each detector mode score only what it claims

Files: `anomaly_guard/core/decision.py`, `anomaly_guard/evaluation/metrics.py`, `tests/`.

- `rules_only`: schema rules and suspicious patterns. No statistical score in the decision.
- `length_only`, `composition_only`, `combined_stats`, `isolation_forest`: no pattern short-circuit and no schema override. Schema failures stay visible in the explanation, and a separate column records them, but they do not change the statistical decision.
- `rules_plus_stats`: rules and statistics, with the rule override explicit.
- Primary metrics use `decision == alert` only. Report the review queue as its own rate.
- Per-component empirical percentile on validation legitimate traffic, then aggregate with `max` of percentiles (documented). Calibrate the alert cutoff so validation false-positive rate on legitimate traffic is at most 1%. Record the actual test rate when it misses that target.
- Fit Isolation Forest without pretending the legitimate sample contains 2% outliers. Choose its cutoff on the same validation budget.
- Add a test that fails if a statistics-only mode’s decision changes when the suspicious-pattern rules are removed from the config.

### B. Stop the simulator from answering the rules

Files: `anomaly_guard/simulation/scenarios.py`, `anomaly_guard/simulation/generator.py`, `anomaly_guard/config/rules.yaml`.

Split the malicious catalog into three labeled slices and report them separately:

1. **Signature-easy** — current textbook strings. Rules should do well here. This slice is a check that rules fire, not the main result.
2. **Rule-evasive** — same families, payloads that do not match `rules.yaml` (split tokens, comments, mixed case that the current patterns miss, one-pass encodings, short tautologies that fail the regex). Do not add a new rule for each of these in the same change.
3. **Size- or distribution-only** — oversized and highly repetitive or odd-composition bodies with no SQL or script token. The oversized family must not embed `' OR 1=1`.

Add schema-valid odd traffic to benign-invalid (wrong-but-plausible types that still parse, extra punctuation, mildly malformed numbers that still match the integer pattern) and keep the current contract breaks as their own slice.

Expand ordinary legitimate traffic: more phrases, length variation, unicode and code-like comments inside the route limits, so the baseline is not five strings. Reported experiments use the full generator counts. `--small` stays a smoke test and must not be copied into `results/` as the result.

Build matched pairs: same method, route, headers, and similar length, one benign and one malicious, so a detector cannot win on a generator fingerprint (`client-new-` versus `client-NNN`, shared user agent, payload length alone).

### C. Put every family in the test set on purpose

Files: `anomaly_guard/evaluation/splits.py`, `anomaly_guard/evaluation/experiments.py`.

- Assign train, validation, and test **inside** each attack family instead of giving each template one split and dropping the rest.
- Keep template variants together, but require every family to contribute test rows.
- Call the held-out-family path: fit and calibrate with one family absent from development, evaluate that family only on test, and write `results/held_out_family.json`.
- Do not fit profiles, length bands, or thresholds on test.

### D. Repair the statistical score so a comparison is interpretable

Files: `anomaly_guard/core/statistics.py`, `anomaly_guard/app.py`, `anomaly_guard/evaluation/experiments.py`.

- Keep composition as a discrepancy score. Stop flooring and rescaling in a way that still claims a chi-square statistic. If a teaching p-value is shown, show it only when every expected count is at least 5, and label it as incompatibility with the baseline, not as an attack probability.
- Skip composition with `insufficient_evidence` when the string is short or the length band has too little calibration data. Calibrate composition inside length bands learned from training only.
- Dashboard “Normal vs unusual”: for a chosen request, plot observed length against the profile median and MAD, and observed category fractions against the baseline probabilities.
- Export a precision-recall curve and a threshold sweep per mode, using the statistical percentile or Isolation Forest score, not a number filled in from the decision label.
- Fix `tn` so the published counts partition the evaluated rows. Publish per-family recall with support counts.

### E. Regenerate artifacts and lock the claim

- Re-run generate (full size), train, and evaluate.
- Replace `results/metrics_summary.json`, predictions, the comparison plot, and error analysis.
- The result worth defending is a table with one false-positive budget and three rows (rules only, statistics only, hybrid) on the **rule-evasive** slice, plus the signature-easy slice where rules are expected to win, plus benign-invalid broken out from malicious intent.
- Add a regression test that the rule-evasive slice is not fully caught by `rules_only`. If that recall is 1.0, the simulator and the rules have collapsed together again.

## Out of scope for this update

No neural models, no extra one-class algorithms, no CSIC ingestion, no live packet capture. Those were already deferred until this comparison is honest.
