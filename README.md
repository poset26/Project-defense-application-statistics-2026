# Anomaly Guard

Local, explainable **HTTP request anomaly detector** for a university-style security experiment. It replays simulated (or imported) JSONL traffic through a shared pipeline: bounded parsing → feature extraction → rule engine → statistical baseline → decision engine. Outputs are **alerts with explanations**, not claims that an intrusion succeeded.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Generate simulated dataset + ground truth (use --small for faster runs)
anomaly-guard generate --output data/simulated
anomaly-guard train --data data/simulated --output models/baseline.json
anomaly-guard replay --data data/simulated --baseline models/baseline.json --db data/events.db --with-labels --limit 500
anomaly-guard evaluate --data data/simulated --output results

# Dashboard (localhost)
streamlit run anomaly_guard/app.py --server.port 8765 --server.address 127.0.0.1
```

Or run the full demo:

```bash
bash scripts/demo.sh
```

## Architecture

```
Simulator / JSONL → Parser → Features → Rules ──┐
                                               ├→ Decision → SQLite + Dashboard
                         Baseline → Statistics ┘
```

Supported MVP routes: `GET /search`, `POST /login`, `POST /comment` (see `anomaly_guard/config/routes.yaml`).

## Detector modes (evaluation ablations)

| Mode | Purpose |
|------|---------|
| `rules_only` | Predefined schema + pattern baseline |
| `length_only` | Robust length (median/MAD) outliers |
| `composition_only` | Character distribution discrepancy |
| `combined_stats` | Max of statistical components |
| `rules_plus_stats` | Hybrid (default) |
| `isolation_forest` | Optional multivariate comparison |

Training uses **legitimate train-split traffic only**. Thresholds are calibrated on **held-out legitimate validation** traffic toward a configurable false-positive budget (default 1%).

## Data layout

- `data/simulated/requests.jsonl` — request records (no labels)
- `data/simulated/ground_truth.jsonl` — scenario metadata joined by `request_id`
- `models/baseline.json` — versioned profiles + calibration scores
- `results/` — exported predictions, metrics, plots, error analysis

## Tests

```bash
pytest
```

## Limitations (by design)

Simulator bias, incomplete normal coverage, cold-start endpoints, distribution shift, and attacks that look statistically normal are documented boundaries—not promises of production IDS coverage. Ground truth reflects **generator intent**, not exploit proof.

## License

MIT (student project).
