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

`--mode` choices:

`rules_only`, `length_only`, `composition_only`, `combined_stats`, `rules_plus_stats`, `isolation_forest`

Behavior:

1. Load requests and ground truth from `--data`.
2. **Fit** statistical profiles on train-split **`legitimate`** traffic only.
3. **Calibrate** decision percentiles on validation-split legitimate-family traffic.
4. Save baseline JSON to `--output`.
5. Write sibling **`baseline.settings.json`** (same path, `.settings.json` suffix) with calibrated thresholds.

For `isolation_forest`, the Isolation Forest is fit during `train` on the same legitimate train records.

### `replay` — score traffic into SQLite

```bash
anomaly-guard replay \
  --data data/simulated \
  --baseline models/baseline.json \
  --db data/events.db \
  --mode rules_plus_stats \
  [--shuffle] [--seed 0] [--limit N] [--clear] [--with-labels]
```

Loads the baseline from disk (does not refit unless you train separately). Each request is scored with `AnomalyGuard.score()` and upserted into `events.db`.

- **`--with-labels`** — stores ground-truth JSON for dashboard experiment views (not used by the scorer).
- **`--clear`** — deletes existing events before replay.
- **`--limit`** — cap number of requests (after optional shuffle).

Replay order uses `simulation/generator.replay_records()` (stable file order unless `--shuffle`).

### `evaluate` — full ablation on test split

```bash
anomaly-guard evaluate --data data/simulated --output results
```

For each detector mode, retrains on train legitimate, calibrates on validation, scores **test** split, and writes artifacts under `--output`:

| Output | Description |
|--------|-------------|
| `metrics_summary.json` | Per-mode metrics, settings, latency, AP |
| `predictions_<mode>.jsonl` | One line per test request with decision and full score dict |
| `models/baseline_<mode>.json` | Baseline snapshot for that run |
| `detector_comparison.png` | Bar chart: malicious F1 vs legitimate FPR by mode |
| `error_analysis.json` | Sample false negatives / false positives (`rules_plus_stats`) |

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
| **Normal vs unusual** | Browse trained baseline: per-endpoint expanders with **median length, MAD, training counts**, and **composition category_probs** JSON — profile tables only, not overlaid charts |
| **Evaluation** | Show saved `results/metrics_summary.json` and comparison plot; optional quick run for `rules_plus_stats` only |
| **Model settings** | Retrain baseline from disk; show endpoint coverage |

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

Checked-in `results/` in the repository reflects a **small-demo** run; regenerate after changing data or code.

## pytest

```bash
pytest
```

Tests live under `tests/` (parser, rules, features, statistics, simulation, metrics). Configuration: `pyproject.toml` `[tool.pytest.ini_options]`.

## Interpreting decisions

- **`alert`** / **`review`** — request deviates from rules and/or learned legitimate profiles for the selected mode; investigate manually.
- Ground truth in the dashboard is for **experiments only**; the detector never reads it during `score()`.

For evaluation caveats on the bundled metrics file, see [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) and [UPDATE_PLAN.md](UPDATE_PLAN.md).
