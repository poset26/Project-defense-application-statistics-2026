#!/usr/bin/env bash
# Regenerate committed datasets, models, evaluation exports, and publish bundle.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

pip install -e ".[dev]" -q

echo "== Generate full dataset =="
python3 -m anomaly_guard.cli generate --output data/simulated --seed 42

echo "== Train default hybrid baseline =="
python3 -m anomaly_guard.cli train --data data/simulated --output models/baseline.json --mode rules_plus_stats

echo "== Full ablation evaluation =="
python3 -m anomaly_guard.cli evaluate --data data/simulated --output results

PKG_ASSETS="anomaly_guard/publish_assets"
mkdir -p "$PKG_ASSETS" "$PKG_ASSETS/demo"

cp -f results/metrics_summary.json "$PKG_ASSETS/metrics_summary.json"
cp -f results/detector_comparison.png "$PKG_ASSETS/detector_comparison.png" 2>/dev/null || true

cp -f data/simulated/requests.jsonl "$PKG_ASSETS/demo/requests.jsonl"
cp -f data/simulated/ground_truth.jsonl "$PKG_ASSETS/demo/ground_truth.jsonl"
cp -f models/baseline.json "$PKG_ASSETS/demo/baseline.json"
cp -f models/baseline.settings.json "$PKG_ASSETS/demo/baseline.settings.json"
cp -f data/simulated/manifest.json "$PKG_ASSETS/demo/manifest.json"

echo "Bundled assets updated under $PKG_ASSETS"
