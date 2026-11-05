# Anomaly Guard — Technical Report (MVP)

## Objective

Test whether a **learned legitimate-traffic baseline** (robust length + character composition profiles) improves detection of held-out **malicious-intent** simulated requests compared with rules alone, under a validation-calibrated false-positive budget on legitimate traffic.

## Methodology

1. **Traffic**: Controlled simulator (`anomaly_guard/simulation/`) emits JSONL requests and separate ground-truth metadata (scenario group, attack family, template ID). Labels describe generator intent, not exploit success.
2. **Splits**: Group-aware hashing on `scenario_id:template_id` assigns train / validation / test. Profiles are fit on **train legitimate only**. Thresholds are calibrated on **validation legitimate** (including difficult legitimate and distribution-shift groups).
3. **Detection**: Shared library `AnomalyGuard` implements `fit()` and `score()` without ground truth. Components: bounded parser, ~15-dimensional features per field, YAML-driven schema and pattern rules, MAD-based length scores, chi-style composition discrepancy, optional Isolation Forest on the same feature vector.
4. **Decisions**: `normal` / `review` / `alert` from high-severity patterns and empirical statistical percentiles. Explanations cite rules and profile deviations, not attack probabilities.
5. **Evaluation**: CLI `evaluate` runs ablations (`rules_only`, `length_only`, `composition_only`, `combined_stats`, `rules_plus_stats`, `isolation_forest`), exports predictions, metrics (precision/recall/F1, Wilson FPR interval), plots, and error analysis JSON.

## Limitations

- Simulator bias and matched generator fingerprints (mitigated by scenario design, not eliminated).
- Histogram composition ignores character order; short strings skip composition with `insufficient_evidence`.
- Cold-start routes return `insufficient_baseline`.
- Statistical unusualness ≠ malicious intent; benign-invalid traffic is reported separately from malicious-intent recall.

## Reproducibility

```bash
bash scripts/demo.sh
```

Artifacts land under `data/`, `models/`, and `results/`.
