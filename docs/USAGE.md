# Usage

This guide matches the **`anomaly-guard`** console script (`anomaly_guard/cli.py`) and `scripts/demo.sh`. Python **3.11+** is required.

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The package registers the entry point `anomaly-guard` (see `pyproject.toml`). Development dependencies include **pytest**.

## CLI overview

```bash
anomaly-guard {generate|train|replay|evaluate} [options]
```

Defaults assume paths under the repo root: `data/simulated`, `models/baseline.json`, `data/events.db`, `results/`.

### `generate` — simulated dataset

```bash
anomaly-guard generate --output data/simulated [--seed 42] [--small]
```

Writes:

- `requests.jsonl` — request records (no labels)
- `ground_truth.jsonl` — metadata joined by `request_id`
- `manifest.json` — seed and generator plan

`--small` uses a reduced plan (on the order of **900** requests) for quick runs. Omit `--small` for the full default plan from `DEFAULT_PLAN` in `simulation/generator.py`.

Splits are assigned automatically via `assign_splits()` when saving.

### `train` — baseline and calibration

```bash
anomaly-guard train \
  --data data/simulated \
  --output models/baseline.json \
  --mode rules_plus_stats
```

`--mode` is any name in `anomaly_guard.evaluation.experiments.MODES`. The default is `rules_plus_stats`. The full list:

`rules_only`, `length_only`, `composition_only`, `payl_only`, `ngram_only`, `regression_only`, `lda_only`, `ci_only`, `entropy_only`, `vote2_stats`, `combined_stats`, `full_stats`, `rules_plus_stats`, `rules_plus_full_stats`, `isolation_forest`

`lda_only` is Mahalanobis distance to the legitimate mean, not linear discriminant analysis. `payl_only` cuts on the raw byte-histogram distance. `full_stats` and `rules_plus_full_stats` use the mean of per-type percentiles. `vote2_stats` uses the second-highest type percentile.

Behavior:

1. Load requests and ground truth from `--data`.
2. **Fit** statistical profiles on train-split **`legitimate`** traffic only.
3. Build percentile pools on validation traffic from `legitimate`, `difficult_legitimate`, and `distribution_shift`.
4. Set the 1% alert cutoff on **ordinary** validation traffic only (`scenario_group == legitimate`). Difficult text and distribution shift do not choose the threshold.
5. Save baseline JSON to `--output`.
6. Write sibling **`baseline.settings.json`** with the calibrated thresholds and the **mode name**.

For `isolation_forest`, the forest is fit during `train` on the same legitimate train records (`contamination=0.001`). Its cutoff is also fit on ordinary validation scores.

### `replay` — score traffic into SQLite

```bash
anomaly-guard replay \
  --data data/simulated \
  --baseline models/baseline.json \
  --db data/events.db \
  --mode rules_plus_stats \
  [--shuffle] [--seed 0] [--limit N] [--clear] [--with-labels]
```

Loads the baseline from disk. If `baseline.settings.json` was saved for the same `--mode`, that cutoff is used. If the file is missing or was fit for another mode, replay recalibrates on ordinary validation traffic from `--data`. It does not refit the profiles. Each request is scored with `AnomalyGuard.score()` and upserted into `events.db`.

- **`--with-labels`** — stores ground-truth JSON for dashboard experiment views (not used by the scorer).
- **`--clear`** — deletes existing events before replay.
- **`--limit`** — cap number of requests (after optional shuffle).

Replay order uses `simulation/generator.replay_records()` (stable file order unless `--shuffle`).

### `evaluate` — full ablation on test split

```bash
anomaly-guard evaluate --data data/simulated --output results
```

For each detector mode, retrains on train legitimate, builds percentile pools on non-attack validation traffic, sets the cutoff on ordinary validation traffic, scores the **test** split, and writes artifacts under `--output`:

| Output | Description |
|--------|-------------|
| `metrics_summary.json` | Per-mode metrics, ordinary vs stress false-positive rates, base-rate posteriors, XSS alert and review rates, settings, latency |
| `predictions_<mode>.jsonl` | One line per test request with decision and full score dict |
| `models/baseline_<mode>.json` | Baseline snapshot for that run |
| `pr_curve_<mode>.png` | Precision-recall curve for that mode |
| `threshold_sweep_<mode>.json` | Descriptive test sweep. It does not move the deployed cutoff |
| `detector_comparison.png` | Malicious F1, mixed legitimate FPR, and rule-evasive recall by mode |
| `error_analysis.json` | Sample false negatives / false positives (`rules_plus_stats`) |
| `held_out_families.json` | Alert and review rates for traversal, XSS, and oversized |
| `contamination_combined_stats.json` | One `combined_stats` refit after replacing 1% of legitimate training rows with train-split attacks |

Prints a JSON summary of `modes` to stdout.

## Scripted demo

```bash
bash scripts/demo.sh
```

Equivalent to: install dev deps, `generate --small`, `train`, `replay --with-labels --limit 300 --clear`, `evaluate`. Ends with a reminder to start Streamlit on port **8765**.

## Streamlit dashboard

```bash
streamlit run anomaly_guard/app.py --server.port 8765 --server.address 127.0.0.1
```

Open the URL shown in the terminal (typically `http://127.0.0.1:8765`).

| Page | Purpose |
|------|---------|
| **Traffic** | Replay from a data directory into `data/events.db`; optional shuffle, limit, detector mode, experiment labels |
| **Alert inspector** | Inspect alert/review rows: escaped request, rules, statistical components, explanations |
| **Normal vs unusual** | Baseline medians, and a bar chart of observed versus baseline character fractions for a selected stored request |
| **Evaluation** | Saved `results/metrics_summary.json`: mode table, attack slices, false positives by ordinary / difficult / shift, base-rate posteriors, XSS alert recall next to review rate, contamination summary, held-out families. Optional quick run for `rules_plus_stats` only |
| **Model settings** | Retrain from disk on the selected mode. Writes the baseline and `baseline.settings.json`. The 1% cutoff uses ordinary validation traffic |

Traffic replay uses the same settings rule as the CLI: load `baseline.settings.json` when its mode matches, otherwise recalibrate on ordinary validation traffic.

The app caption states that alerts describe unusual validation, not confirmed intrusions.

Default paths in the UI match CLI defaults (`data/simulated`, `models/baseline.json`, `data/events.db`, `results/`).

## Where artifacts land

| Path | Created by |
|------|------------|
| `data/simulated/` (or custom `--output`) | `generate` |
| `data/simulated-full/` | Optional full generate (not run by demo) |
| `models/baseline.json`, `models/baseline.settings.json` | `train` |
| `data/events.db` | `replay` (or dashboard Traffic replay) |
| `results/*` | `evaluate` |

Checked-in `results/` match `anomaly-guard evaluate` on the full simulator (11,500 requests, seed 42), with the 1% cutoff fit on ordinary validation traffic. `scripts/demo.sh` is a small smoke run and is not the source of those numbers. Interpretation is in [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).

## pytest

```bash
pytest
```

Tests live under `tests/` (parser, rules, features, statistics, simulation, metrics). Configuration: `pyproject.toml` `[tool.pytest.ini_options]`.

## Interpreting decisions

- **`alert`** / **`review`** — request deviates from rules and/or learned legitimate profiles for the selected mode; investigate manually.
- Ground truth in the dashboard is for **experiments only**; the detector never reads it during `score()`.

For the operating point and its limits, see [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md). [UPDATE_PLAN.md](UPDATE_PLAN.md) is the historical record of the earlier experiment-honesty fixes, not the current numbers.
