# Anomaly Guard

Anomaly Guard is a **local, explainable HTTP request anomaly detector** built as a university security experiment. It replays simulated (or imported) JSONL traffic through a single pipeline: bounded parsing, feature extraction, YAML-driven rules, statistical profiles learned from legitimate traffic, and a decision engine that emits **normal**, **review**, or **alert** with human-readable explanations.

**Important:** An **alert** means a request looks **suspicious or unusual** relative to route contracts and a baseline of legitimate traffic. It is **not** proof that an intrusion succeeded. Ground-truth labels describe **simulator intent**, not real-world exploit outcomes.

## Supported routes

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/search` | Query parameters `q`, optional `page` |
| POST | `/login` | Form fields `username`, `password` |
| POST | `/comment` | Form or JSON fields `text`, optional `author` |

Contracts live in `anomaly_guard/config/routes.yaml`; pattern and schema rules in `anomaly_guard/config/rules.yaml`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Full simulator (~11,500 requests; matches checked-in results/)
anomaly-guard generate --output data/simulated

# Smoke tests only (~740 requests)
# anomaly-guard generate --output data/simulated --small
anomaly-guard train --data data/simulated --output models/baseline.json
anomaly-guard replay --data data/simulated --baseline models/baseline.json \
  --db data/events.db --with-labels --limit 300 --clear
anomaly-guard evaluate --data data/simulated --output results

streamlit run anomaly_guard/app.py --server.port 8765 --server.address 127.0.0.1
```

Full scripted demo (same small dataset):

```bash
bash scripts/demo.sh
```

See [docs/USAGE.md](docs/USAGE.md) for every CLI flag, artifact paths, dashboard pages, and pytest.

## Detector modes

The CLI and library support ablation modes (default production-style mode: **`rules_plus_stats`**):

| Mode | Role |
|------|------|
| `rules_only` | Schema, size limits, and suspicious-pattern rules only |
| `length_only` | log1p length outliers (median / MAD). Rules are recorded and do not decide |
| `composition_only` | Chi-square character-composition discrepancy per field. Rules do not decide |
| `combined_stats` | Length + global χ² (max of the two type percentiles) |
| `ci_only` | 95% prediction interval on log length, plus Wilson intervals on character proportions |
| `ngram_only` | Bigram median/MAD profile |
| `regression_only` | OLS on `log1p(length)` ~ entropy + category fractions (standardized residual) |
| `lda_only` | Mahalanobis distance to the legitimate feature mean. The mode id is historical; this is not linear discriminant analysis |
| `entropy_only` | Robust z of Shannon entropy. Kept out of mean fusion so the ablation stays readable |
| `vote2_stats` | Middle fusion: the second-highest component percentile, so two component types must be unusual |
| `payl_only` | PAYL byte distance. The alert cutoff is a quantile of the **raw** distance |
| `full_stats` | All statistical components. The decision score is the **mean** of per-type percentiles, so one noisy model cannot set the alert by itself |
| `rules_plus_stats` | Hybrid: rules + `combined_stats` |
| `rules_plus_full_stats` | Hybrid: rules + the mean-fused full stack |
| `isolation_forest` | Optional Isolation Forest comparison on its own score |

Training uses **legitimate train-split** requests only. Percentile thresholds are calibrated on **held-out legitimate validation** traffic (including difficult legitimate and distribution-shift groups), targeting a configurable false-positive budget (`max_false_positive_rate_target`, default 1% on validation).

## Repository layout

```
anomaly_guard/
  cli.py              # anomaly-guard CLI
  app.py              # Streamlit dashboard
  core/               # parser, features, rules, statistics, decision
  simulation/         # JSONL traffic generator
  evaluation/         # splits, metrics, experiment runner
  config/             # routes.yaml, rules.yaml
data/
  simulated/          # full JSONL dataset (~11.5k requests; regenerate with generate)
  events.db           # SQLite event store (created by replay)
models/
  baseline.json       # trained profiles + calibration scores
results/              # evaluate output (metrics, predictions, plots)
docs/
  ARCHITECTURE.md     # pipeline and data-flow detail
  USAGE.md            # install and operations
  TECHNICAL_REPORT.md # methodology and honest evaluation limits
  UPDATE_PLAN.md      # proposed fixes for experiment design (do not treat as implemented)
tests/                # pytest suite
scripts/demo.sh       # end-to-end small demo
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — pipeline, request vs ground-truth files, splits, what `score()` may use
- [Usage](docs/USAGE.md) — install, generate, train, replay, evaluate, dashboard, artifacts
- [Technical report](docs/TECHNICAL_REPORT.md) — methodology, metrics definitions, current limitations
- [Update plan](docs/UPDATE_PLAN.md) — known evaluation shortcomings and a fix plan (proposed, not merged into detector code)

## Checked-in results

Files under `results/` match **`anomaly-guard evaluate`** on the full simulator in `data/simulated` (11,500 requests, seed 42). `scripts/demo.sh` is only a small smoke run and is not the source of those numbers. The interpretation, including the base-rate posterior and the limits of this generator, is in [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md).

## Tests

```bash
pytest
```

## License

MIT (student project).
