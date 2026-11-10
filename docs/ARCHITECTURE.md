# Architecture

Anomaly Guard is a batch and replay-oriented detector: requests arrive as structured records, pass through shared core logic, and produce scored decisions stored for inspection in SQLite and Streamlit. There is no inline network proxy in the MVP; traffic is simulated or read from JSONL.

## End-to-end pipeline

```
┌─────────────────┐     ┌──────────┐     ┌──────────┐     ┌─────────────┐
│ Simulator or    │     │  Parser  │     │ Features │     │ Rule engine │
│ requests.jsonl  │────▶│ (bounded)│────▶│ extract  │────▶│ rules.yaml  │
└─────────────────┘     └──────────┘     └──────────┘     └──────┬──────┘
                                                                    │
┌─────────────────┐     ┌──────────┐     ┌──────────┐            │
│ Train split     │────▶│ Baseline │────▶│ Statistics│◀───────────┘
│ (legitimate)    │     │  fit     │     │ score     │
└─────────────────┘     └──────────┘     └──────┬──────┘
                                                │
                     ┌──────────────────────────┴──────────────────────────┐
                     │              Decision engine (AnomalyGuard)          │
                     │  mode: rules_only | length_only | … | isolation_forest│
                     └──────────────────────────┬──────────────────────────┘
                                                │
                     ┌──────────────────────────▼──────────────────────────┐
                     │  ScoreResult: decision, rules, stats, explanations     │
                     └──────────────┬─────────────────────┬──────────────────┘
                                    │                     │
                           ┌────────▼────────┐   ┌────────▼────────┐
                           │ SQLite          │   │ Evaluation /    │
                           │ EventStore      │   │ dashboard       │
                           └─────────────────┘   └─────────────────┘
```

### Parser (`anomaly_guard/core/parser.py`)

Parses each `RequestRecord` against `routes.yaml`: decodes query and body fields, enforces size limits from config, and sets flags (oversized payload, JSON parse errors, malformed encoding, unexpected fields). Output is a `ParsedRequest` used by rules and features.

### Features (`anomaly_guard/core/features.py`)

Per-field numeric summaries: character and byte length, entropy, repeat runs, percent-encoding counts, and **category fractions** over bins `letter`, `digit`, `whitespace`, `quote`, `bracket`, `slash`, `other`. Request-level aggregates feed an optional flat vector for Isolation Forest.

### Rules (`anomaly_guard/core/rules.py`, `config/rules.yaml`)

Two families:

- **Schema rules** — missing required fields, wrong types, length out of contract, bad content-type.
- **Suspicious patterns** — regex-style checks (SQL tautologies, script tags, path traversal fragments, etc.) with severities low / medium / high.

Rules always run inside `AnomalyGuard.score()` for every detector mode; mode only changes **how rule outputs combine** with statistical scores (see Decision engine).

### Statistical baseline (`anomaly_guard/core/statistics.py`)

`StatisticalDetector.fit(normal_records)` builds per-endpoint, per-field profiles:

- **Length profile** — median and MAD. The robust z is `0.6745 * |x − median| / MAD` on `log1p(length)`, using the MAD of the log sample. A Student-t 95% prediction interval is stored on the same log lengths.
- **Composition profile** — character-category proportions. The score is χ²/n from `scipy.stats.chisquare` after a small continuity adjustment. A p-value is attached only when every expected count is at least 5, and the text calls it baseline incompatibility, not P(attack). Characters in one string are not independent trials, so that p-value is not a Type I error.
- **Entropy profile** — median and MAD of Shannon entropy. It is its own ablation (`entropy_only`) and is not folded into mean fusion.
- **Extended profiles** — length-band χ², PAYL byte distance, bigrams, OLS residual, Mahalanobis distance (`lda_only` is that distance, not linear discriminant analysis).

Each raw component is mapped to an empirical percentile on validation traffic. `combined_stats` takes the **max** of the length and composition percentiles. `full_stats` takes the **mean** across component types. `vote2_stats` takes the **second-highest** type percentile. `payl_only` thresholds the **raw** PAYL distance, because percentile 100 is tied too often to meet a 1% budget.

`StatisticalDetector.score()` and `AnomalyGuard.score()` **must not** read ground-truth metadata—only the request record and loaded baseline.

### Decision engine (`anomaly_guard/core/decision.py`)

`AnomalyGuard` is the shared entry point for CLI replay, evaluation, and dashboard.

| Mode | Statistical components used | Decision logic (simplified) |
|------|------------------------------|-----------------------------|
| `rules_only` | None in decision | High-severity rules → alert; medium → review |
| `length_only`, `composition_only`, `entropy_only`, `payl_only`, `ngram_only`, `regression_only`, `lda_only`, `ci_only` | That component only | Statistics decide. Rules are recorded and do not change the label |
| `combined_stats` | Length and composition, max percentile | Statistics only |
| `vote2_stats` | Same components as `full_stats` | Alert from the second-highest type percentile |
| `full_stats` | Length, chi-square, intervals, PAYL, n-grams, OLS, Mahalanobis | Mean of per-type percentiles |
| `rules_plus_stats` / `rules_plus_full_stats` | Length+composition, or the mean of the full stack | High-severity pattern → alert; else the statistical cutoff |
| `isolation_forest` | IF `decision_function` on the feature vector | The forest score alone. `contamination=0.001` |

Decisions:

- **`normal`** — within rules and baseline expectations for the mode.
- **`review`** — elevated schema issues or statistical unusualness below alert cutoff.
- **`alert`** — high-severity pattern match and/or statistical aggregate above calibrated alert percentile (or IF score below threshold).

Explanations list triggered rules. A field is called unusual only on the same scale as the cutoff: a percentile for percentile modes, the raw distance for `payl_only`. A percentile is never compared with a raw-distance threshold.

Optional **Isolation Forest** (`sklearn`, 100 trees, `contamination=0.001`) trains on legitimate train vectors when mode is `isolation_forest`. The alert cutoff is fit on validation scores.

## Request data vs ground truth

| File | Contents | Used at runtime scoring? |
|------|----------|---------------------------|
| `requests.jsonl` | `RequestRecord`: id, timestamp, client, method, route, raw query, headers, body | **Yes** — sole input to `score()` |
| `ground_truth.jsonl` | `GroundTruthMeta`: scenario_group, scenario_id, attack_family, template_id, split | **No** for detection — only evaluation, replay `--with-labels`, dashboard experiment mode |

Scenario groups: `legitimate`, `difficult_legitimate`, `benign_invalid`, `malicious_intent`, `hard_malicious_intent`, `distribution_shift`.

The simulator (`anomaly_guard/simulation/`) writes both files with matching `request_id` keys. `manifest.json` records seed and plan counts.

## Train / validation / test splits

`evaluation/splits.py` assigns each request a split:

- **Malicious** and **benign_invalid** rows: split inside each attack family so every family contributes test rows.
- **Legitimate-family** rows (`legitimate`, `difficult_legitimate`, `distribution_shift`): **route-stratified per request** (`METHOD route:request_id`) so every endpoint appears in both train and validation—needed so baseline calibration sees the same routes as training.

Training baseline: **train** split, **`legitimate`** group only.

Percentile pools: **validation** split, groups `legitimate`, `difficult_legitimate`, and `distribution_shift`. The 1% alert cutoff uses **ordinary** validation traffic only (`scenario_group == legitimate`). Difficult and shifted rows are a stress test. `train` writes `baseline.settings.json` with the mode name. Replay and the dashboard load that file when the mode matches, and recalibrate on ordinary validation traffic when it does not.

Evaluation (`evaluate`): scores **test** split only; metrics join predictions to ground truth by `request_id`.

## Configuration artifacts

| Artifact | Role |
|----------|------|
| `models/baseline.json` | Serialized endpoint profiles + `calibration_scores` |
| `models/baseline.settings.json` | Written by `train`: mode name plus `DecisionSettings`. Replay and the dashboard apply it only when the mode matches |
| `results/models/baseline_<mode>.json` | Per-mode baselines from full ablation run |
| `data/events.db` | SQLite `events` table: request JSON, score JSON, optional ground truth |

## Dashboard mapping

Streamlit app (`anomaly_guard/app.py`):

- **Traffic** — replay JSONL into the event store; filter stored rows.
- **Alert inspector** — escaped request text, rules, statistical components, explanations; optional experiment labels.
- **Normal vs unusual** — baseline medians and a bar chart of observed vs baseline character fractions for a selected request.
- **Evaluation** — saved metrics, attack slices, false positives by traffic group, base-rate posteriors, XSS alert recall next to review rate, contamination summary, and held-out families. Optional single-mode run for `rules_plus_stats`.
- **Model settings** — retrain baseline from disk dataset.

## Evaluation module

`evaluation/experiments.py` trains each mode on the same splits, runs test predictions, writes `predictions_<mode>.jsonl`, `metrics_summary.json`, `detector_comparison.png`, and `error_analysis.json` (sample FN/FP for `rules_plus_stats`).

Metrics (`evaluation/metrics.py`) use **`decision == alert` only**. Review rate is separate. The legitimate false-positive rate has a Wilson interval and is also broken out by ordinary, difficult, and distribution-shift traffic. `base_rate` rows convert recall and that false-positive rate into P(attack | alert) at the file’s prevalence and at 1% and 0.1% priors. `contamination_combined_stats.json` is one refit of length and chi-square after replacing 1% of legitimate training rows with train-split attacks.

Checked-in `results/` come from the full 11,500-request simulator. See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).
