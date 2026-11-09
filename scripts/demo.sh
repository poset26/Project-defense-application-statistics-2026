#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

pip install -e ".[dev]" -q

echo "== Generate dataset (reported: full plan; use --small only for smoke tests) =="
anomaly-guard generate --output data/simulated

echo "== Train hybrid baseline =="
anomaly-guard train --data data/simulated --output models/baseline.json

echo "== Replay sample into event store =="
anomaly-guard replay --data data/simulated --baseline models/baseline.json \
  --db data/events.db --with-labels --limit 300 --clear

echo "== Run ablation evaluation =="
anomaly-guard evaluate --data data/simulated --output results

echo "Done. Start dashboard: streamlit run anomaly_guard/app.py --server.port 8765 --server.address 127.0.0.1"
