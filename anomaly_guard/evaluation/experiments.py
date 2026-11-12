"""Reproducible evaluation runner and ablation experiments."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from anomaly_guard.core.decision import AnomalyGuard, DetectorMode, _decision_from_rules, clears_cutoff
from anomaly_guard.core.parser import parse_request
from anomaly_guard.core.rules import evaluate_rules
from anomaly_guard.core.schema import GroundTruthMeta, RequestRecord, read_jsonl
from anomaly_guard.evaluation.metrics import (
    alert_predicted,
    average_precision,
    base_rate_posteriors,
    compute_benign_invalid_metrics,
    compute_binary_metrics,
    compute_malicious_attack_metrics,
    is_positive_any_anomaly,
    is_positive_malicious,
    per_family_recall,
    precision_recall_points,
    ranking_score_from_prediction,
    review_rate_on_legitimate,
    slice_attack_recall,
)
from anomaly_guard.evaluation.splits import (
    partition_records,
    training_normal_records,
    validation_legitimate_records,
    validation_ordinary_records,
)

MODES: list[DetectorMode] = [
    "rules_only",
    "length_only",
    "composition_only",
    "payl_only",
    "ngram_only",
    "regression_only",
    "lda_only",
    "ci_only",
    "entropy_only",
    "vote2_stats",
    "combined_stats",
    "full_stats",
    "rules_plus_stats",
    "rules_plus_full_stats",
    "isolation_forest",
]

PRIMARY_MODES: list[DetectorMode] = [
    "rules_only",
    "combined_stats",
    "ci_only",
    "full_stats",
    "rules_plus_stats",
    "rules_plus_full_stats",
]

HELD_OUT_FAMILIES: tuple[str, ...] = ("traversal", "xss", "oversized")

ATTACK_SLICES = ("signature_easy", "rule_evasive", "size_distribution")


@dataclass
class ExperimentResult:
    mode: DetectorMode
    metrics_malicious: dict[str, Any]
    metrics_benign_invalid: dict[str, Any]
    metrics_any_anomaly: dict[str, Any]
    fpr_on_legitimate_test: float
    review_rate_legitimate: float
    average_precision: float
    per_family: dict[str, Any]
    attack_slices: dict[str, Any]
    latency_ms_median: float
    latency_ms_p95: float
    settings: dict[str, Any]
    operating_point_1pct: dict[str, Any] | None = None
    fpr_by_legitimate_group: dict[str, Any] | None = None
    base_rate: dict[str, Any] | None = None
    fpr_on_ordinary_test: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_dataset(data_dir: Path) -> tuple[list[RequestRecord], list[GroundTruthMeta]]:
    records = [RequestRecord.from_dict(r) for r in read_jsonl(data_dir / "requests.jsonl")]
    meta = [GroundTruthMeta.from_dict(m) for m in read_jsonl(data_dir / "ground_truth.jsonl")]
    return records, meta


def run_mode_experiment(
    records: list[RequestRecord],
    meta: list[GroundTruthMeta],
    mode: DetectorMode,
    model_dir: Path,
) -> tuple[ExperimentResult, list[dict[str, Any]]]:
    train_recs = training_normal_records(records, meta)
    val_recs = validation_legitimate_records(records, meta)
    ordinary_recs = validation_ordinary_records(records, meta)
    parts = partition_records(records, meta)
    test_recs, test_meta = parts["test"]

    guard = AnomalyGuard(mode=mode)
    guard.fit(train_recs)
    guard.calibrate(val_recs, threshold_records=ordinary_recs)
    baseline_path = model_dir / f"baseline_{mode}.json"
    guard.save_baseline(baseline_path)

    predictions: list[dict[str, Any]] = []
    latencies: list[float] = []
    meta_by_id = {m.request_id: m for m in test_meta}

    for rec in test_recs:
        sr = guard.score(rec)
        latencies.append(sr.processing_time_ms)
        gt = meta_by_id[rec.request_id]
        predictions.append(
            {
                "request_id": rec.request_id,
                "decision": sr.decision,
                "mode": mode,
                "scenario_group": gt.scenario_group,
                "attack_family": gt.attack_family,
                "attack_slice": gt.attack_slice,
                "score": sr.to_dict(),
            }
        )

    y_any = [is_positive_any_anomaly(meta_by_id[r.request_id]) for r in test_recs]
    pred_alert = [alert_predicted(p["decision"]) for p in predictions]
    decisions = [p["decision"] for p in predictions]
    legit_mask = [
        meta_by_id[r.request_id].scenario_group
        in ("legitimate", "difficult_legitimate", "distribution_shift")
        for r in test_recs
    ]
    test_meta_ordered = [meta_by_id[r.request_id] for r in test_recs]

    mal_metrics = compute_malicious_attack_metrics(test_meta_ordered, pred_alert)
    invalid_metrics = compute_benign_invalid_metrics(test_meta_ordered, pred_alert)
    any_metrics = compute_binary_metrics(y_any, pred_alert, legit_mask)

    y_mal = [is_positive_malicious(m) for m in test_meta_ordered]
    scores = [ranking_score_from_prediction(p["score"]) for p in predictions]
    ap = average_precision(y_mal, scores)
    fam_recall = per_family_recall(test_meta, {p["request_id"]: p["decision"] for p in predictions})

    slices: dict[str, Any] = {}
    for sl in ATTACK_SLICES:
        sr = slice_attack_recall(test_meta_ordered, pred_alert, sl)
        if sr:
            slices[sl] = sr

    lat = np.array(latencies)
    group_fpr = _fpr_by_legitimate_group(test_meta_ordered, pred_alert)
    n_mal = sum(1 for m in test_meta_ordered if is_positive_malicious(m))
    n_leg = sum(1 for flag in legit_mask if flag)
    base_rate = base_rate_posteriors(
        mal_metrics.recall,
        mal_metrics.fpr,
        mal_metrics.wilson_fpr_high,
        n_mal,
        n_leg,
    )
    result = ExperimentResult(
        mode=mode,
        metrics_malicious=mal_metrics.to_dict(),
        metrics_benign_invalid=invalid_metrics.to_dict(),
        metrics_any_anomaly=any_metrics.to_dict(),
        fpr_on_legitimate_test=mal_metrics.fpr,
        fpr_on_ordinary_test=(group_fpr.get("legitimate") or {}).get("alert_rate"),
        review_rate_legitimate=review_rate_on_legitimate(test_meta_ordered, decisions),
        average_precision=ap,
        per_family=fam_recall,
        attack_slices=slices,
        latency_ms_median=float(np.median(lat)),
        latency_ms_p95=float(np.percentile(lat, 95)),
        settings=asdict(guard.settings),
        fpr_by_legitimate_group=group_fpr,
        base_rate=base_rate,
    )
    _plot_mode_pr(y_mal, scores, mode, model_dir.parent)
    sweep = _threshold_sweep_percentile(
        guard, val_recs, test_meta_ordered, test_recs, mode, model_dir.parent
    )
    result.operating_point_1pct = operating_point_at_fpr(sweep, budget=0.01)
    return result, predictions


def _plot_mode_pr(y_true: list[bool], scores: list[float], mode: str, out: Path) -> None:
    if not any(y_true):
        return
    import matplotlib.pyplot as plt

    p, r = precision_recall_points(y_true, scores)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(r, p)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR curve ({mode})")
    fig.tight_layout()
    fig.savefig(out / f"pr_curve_{mode}.png", dpi=120)
    plt.close(fig)


def _threshold_sweep_percentile(
    guard: AnomalyGuard,
    val_recs: list[RequestRecord],
    test_meta: list[GroundTruthMeta],
    test_recs: list[RequestRecord],
    mode: str,
    out: Path,
) -> None:
    if mode == "rules_only":
        return None
    if mode == "isolation_forest":
        return _threshold_sweep_isolation(guard, val_recs, test_meta, test_recs, mode, out)
    val_pcts = [guard._aggregate_percentile_for_record(r) for r in val_recs]
    val_pcts = [p for p in val_pcts if p is not None]
    if not val_pcts:
        return None
    test_pcts: list[float | None] = []
    meta_by_id = {m.request_id: m for m in test_meta}
    for rec in test_recs:
        test_pcts.append(guard._aggregate_percentile_for_record(rec))
    y_mal = [is_positive_malicious(meta_by_id[r.request_id]) for r in test_recs]
    leg = [
        meta_by_id[r.request_id].scenario_group
        in ("legitimate", "difficult_legitimate", "distribution_shift")
        for r in test_recs
    ]
    hybrid = mode in ("rules_plus_stats", "rules_plus_full_stats")
    rule_alerts = []
    if hybrid:
        for rec in test_recs:
            parsed = parse_request(rec, guard.config)
            rules = evaluate_rules(parsed, guard.config)
            rule_alerts.append(_decision_from_rules(rules, guard.settings) == "alert")
    thresholds = np.linspace(min(val_pcts), max(val_pcts), 25)
    rows = []
    for t in thresholds:
        pred = []
        for i, p in enumerate(test_pcts):
            stat_hit = p is not None and clears_cutoff(p, float(t))
            pred.append(stat_hit or (hybrid and rule_alerts[i]))
        tp = sum(pred[i] and y_mal[i] for i in range(len(pred)))
        fn = sum((not pred[i]) and y_mal[i] for i in range(len(pred)))
        fp = sum(pred[i] and leg[i] for i in range(len(pred)))
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / sum(leg) if sum(leg) else 0.0
        rows.append({
            "threshold_percentile": float(t),
            "recall_malicious": rec,
            "fpr_legitimate": fpr,
            "includes_rule_alerts": hybrid,
            "comparison": "score > threshold",
        })
    with (out / f"threshold_sweep_{mode}.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    return rows


def _isolation_score(guard: AnomalyGuard, rec: RequestRecord) -> float | None:
    if guard._isolation is None or guard.detector.baseline is None:
        return None
    parsed = parse_request(rec, guard.config)
    from anomaly_guard.core.features import extract_features

    feats = extract_features(parsed)
    order = guard.detector.baseline.feature_field_order
    vec = np.array([feats.to_vector(order)], dtype=float)
    return float(guard._isolation.decision_function(vec)[0])


def _threshold_sweep_isolation(
    guard: AnomalyGuard,
    val_recs: list[RequestRecord],
    test_meta: list[GroundTruthMeta],
    test_recs: list[RequestRecord],
    mode: str,
    out: Path,
) -> list[dict[str, Any]] | None:
    """Sweep the forest score with the same lower-is-worse rule as the decision."""
    val_scores = [s for s in (_isolation_score(guard, r) for r in val_recs) if s is not None]
    if not val_scores:
        return None
    test_scores = [_isolation_score(guard, r) for r in test_recs]
    meta_by_id = {m.request_id: m for m in test_meta}
    y_mal = [is_positive_malicious(meta_by_id[r.request_id]) for r in test_recs]
    leg = [
        meta_by_id[r.request_id].scenario_group
        in ("legitimate", "difficult_legitimate", "distribution_shift")
        for r in test_recs
    ]
    thresholds = np.linspace(min(val_scores), max(val_scores), 25)
    rows = []
    for t in thresholds:
        pred = [s is not None and s < float(t) for s in test_scores]
        tp = sum(pred[i] and y_mal[i] for i in range(len(pred)))
        fn = sum((not pred[i]) and y_mal[i] for i in range(len(pred)))
        fp = sum(pred[i] and leg[i] for i in range(len(pred)))
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / sum(leg) if sum(leg) else 0.0
        rows.append({
            "threshold_percentile": float(t),
            "recall_malicious": rec,
            "fpr_legitimate": fpr,
            "includes_rule_alerts": False,
            "score": "isolation_decision_function",
            "comparison": "score < threshold",
        })
    with (out / f"threshold_sweep_{mode}.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    return rows


def _fpr_by_legitimate_group(
    meta: list[GroundTruthMeta], alerts: list[bool]
) -> dict[str, Any]:
    groups = ("legitimate", "difficult_legitimate", "distribution_shift")
    out: dict[str, Any] = {}
    for name in groups:
        idx = [i for i, m in enumerate(meta) if m.scenario_group == name]
        if not idx:
            continue
        hits = sum(1 for i in idx if alerts[i])
        out[name] = {"alerts": hits, "support": len(idx), "alert_rate": hits / len(idx)}
    return out


def operating_point_at_fpr(rows: list[dict[str, Any]] | None, budget: float = 0.01) -> dict[str, Any] | None:
    """Read the sweep. Does not change the validation-fitted threshold.

    Prefer the point with the highest malicious recall whose test legitimate
    FPR is at most `budget`. If every point exceeds the budget, use the
    closest FPR and mark the selection.
    """
    if not rows:
        return None
    under = [r for r in rows if r["fpr_legitimate"] <= budget + 1e-12]
    if under:
        best = max(under, key=lambda r: (r["recall_malicious"], -r["fpr_legitimate"]))
        selection = "highest_recall_with_test_fpr_at_most_budget"
    else:
        best = min(rows, key=lambda r: (abs(r["fpr_legitimate"] - budget), -r["recall_malicious"]))
        selection = "closest_test_fpr_budget_not_met"
    return {
        "budget": budget,
        "selection": selection,
        "note": "Descriptive read of the test sweep. The deployed cutoff is still fit on validation only.",
        **best,
    }


def run_contamination_sensitivity(
    records: list[RequestRecord],
    meta: list[GroundTruthMeta],
    results_dir: Path,
    fraction: float = 0.01,
    seed: int = 42,
) -> dict[str, Any]:
    """Replace a declared fraction of legitimate training rows with train-split attacks.

    Refits combined_stats only. Validation and test stay clean.
    """
    from anomaly_guard.core.decision import AnomalyGuard

    meta_by_id = {m.request_id: m for m in meta}
    train = training_normal_records(records, meta)
    malicious_train = [
        r
        for r in records
        if (m := meta_by_id.get(r.request_id))
        and m.split == "train"
        and m.scenario_group in ("malicious_intent", "hard_malicious_intent")
    ]
    if not train or not malicious_train:
        return {"error": "need legitimate train rows and a malicious train pool"}
    rng = np.random.default_rng(seed)
    n_replace = min(len(train), max(1, int(round(fraction * len(train)))))
    slots = rng.choice(len(train), size=n_replace, replace=False)
    picks = rng.choice(len(malicious_train), size=n_replace, replace=len(malicious_train) < n_replace)
    contaminated = list(train)
    for slot, pick in zip(slots, picks):
        contaminated[int(slot)] = malicious_train[int(pick)]

    val = validation_legitimate_records(records, meta)
    ordinary = validation_ordinary_records(records, meta)
    test_recs, test_meta = partition_records(records, meta)["test"]

    def medians(guard: AnomalyGuard) -> dict[str, float]:
        out: dict[str, float] = {}
        baseline = guard.detector.baseline
        if baseline is None:
            return out
        for key, ep in baseline.endpoints.items():
            for fname, fp in ep.fields.items():
                if fp.length is not None:
                    out[f"{key} {fname}"] = fp.length.median
        return out

    def evaluate(guard: AnomalyGuard) -> dict[str, Any]:
        ordered: list[GroundTruthMeta] = []
        alerts: list[bool] = []
        by_id = {m.request_id: m for m in test_meta}
        for rec in test_recs:
            ordered.append(by_id[rec.request_id])
            alerts.append(alert_predicted(guard.score(rec).decision))
        metrics = compute_malicious_attack_metrics(ordered, alerts)
        return {
            "recall": metrics.recall,
            "precision": metrics.precision,
            "false_positive_rate": metrics.fpr,
            "attack_slices": {
                sl: slice_attack_recall(ordered, alerts, sl) for sl in ATTACK_SLICES
            },
        }

    clean = AnomalyGuard(mode="combined_stats")
    clean.fit(train)
    clean_medians = medians(clean)
    clean.calibrate(val, threshold_records=ordinary)
    dirty = AnomalyGuard(mode="combined_stats")
    dirty.fit(contaminated)
    dirty_medians = medians(dirty)
    dirty.calibrate(val, threshold_records=ordinary)
    deltas = []
    for key in sorted(set(clean_medians) | set(dirty_medians)):
        before = clean_medians.get(key)
        after = dirty_medians.get(key)
        if before is None or after is None:
            continue
        deltas.append(
            {
                "field": key,
                "clean_median_length": before,
                "contaminated_median_length": after,
                "delta": after - before,
            }
        )
    summary = {
        "mode": "combined_stats",
        "fraction": fraction,
        "seed": seed,
        "n_train_legitimate": len(train),
        "n_replaced": int(n_replace),
        "n_malicious_train_pool": len(malicious_train),
        "clean": evaluate(clean),
        "contaminated": evaluate(dirty),
        "length_median_deltas": deltas,
        "note": (
            "One percent of legitimate training rows were replaced by malicious rows from the "
            "train split. The cutoff is still fit on clean validation traffic. This is not an adaptive attacker."
        ),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "contamination_combined_stats.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _xss_component_profile(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Median per-type percentile of full_stats on test XSS rows.

    Explains a zero alert recall when scores sit near the legitimate bulk.
    """
    rows = [
        p
        for p in predictions
        if p.get("experiment_mode") == "full_stats" and p.get("attack_family") == "xss"
    ]
    if not rows:
        return {"support": 0}
    by_type: dict[str, list[float]] = {}
    decisions: dict[str, int] = {}
    for row in rows:
        decisions[row["decision"]] = decisions.get(row["decision"], 0) + 1
        components = ((row.get("score") or {}).get("statistical") or {}).get("components") or []
        seen: dict[str, float] = {}
        for comp in components:
            if comp.get("status") != "ok" or comp.get("percentile") is None:
                continue
            name = str(comp.get("component"))
            seen[name] = max(seen.get(name, 0.0), float(comp["percentile"]))
        for name, pct in seen.items():
            by_type.setdefault(name, []).append(pct)
    medians = {
        name: float(np.median(vals)) for name, vals in sorted(by_type.items())
    }
    return {
        "support": len(rows),
        "decisions": decisions,
        "median_percentile_by_component": medians,
        "note": (
            "Median per-type percentile of full_stats on test XSS rows. "
            "Rules-only can still review a medium-severity handler without alerting. "
            "These percentiles describe the row; the mode cutoff decides the label."
        ),
    }


def run_held_out_family_evaluation(
    data_dir: Path,
    results_dir: Path,
    family: str = "traversal",
    families: list[str] | None = None,
) -> dict[str, Any]:
    """Fit each primary mode once and score every requested attack family on test.

    `family` is kept so older callers still request traversal. Pass `families`
    to score several families without refitting.
    """
    wanted = list(families) if families is not None else [family]
    records, meta = load_dataset(data_dir)
    parts = partition_records(records, meta)
    test_recs, _test_meta = parts["test"]
    meta_by_id = {m.request_id: m for m in meta}
    by_family: dict[str, list[RequestRecord]] = {name: [] for name in wanted}
    for rec in test_recs:
        fam = meta_by_id[rec.request_id].attack_family
        if fam in by_family:
            by_family[fam].append(rec)

    train_recs = training_normal_records(records, meta)
    val_recs = validation_legitimate_records(records, meta)
    ordinary_recs = validation_ordinary_records(records, meta)
    summary: dict[str, Any] = {"families": {}, "modes_fit": list(PRIMARY_MODES)}
    for name in wanted:
        summary["families"][name] = {
            "family": name,
            "support": len(by_family[name]),
            "modes": {},
        }
        if not by_family[name]:
            summary["families"][name]["error"] = "no test rows for family"

    for mode in PRIMARY_MODES:
        guard = AnomalyGuard(mode=mode)
        guard.fit(train_recs)
        guard.calibrate(val_recs, threshold_records=ordinary_recs)
        for name, rows in by_family.items():
            if not rows:
                continue
            hits = 0
            reviews = 0
            for rec in rows:
                decision = guard.score(rec).decision
                if alert_predicted(decision):
                    hits += 1
                elif decision == "review":
                    reviews += 1
            summary["families"][name]["modes"][mode] = {
                "alert_recall": hits / len(rows),
                "alerts": hits,
                "reviews": reviews,
                "review_rate": reviews / len(rows),
            }

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "held_out_families.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    traversal = summary["families"].get("traversal", {"family": "traversal", "error": "not requested"})
    (results_dir / "held_out_family.json").write_text(json.dumps(traversal, indent=2), encoding="utf-8")
    if family in summary["families"] and families is None:
        return summary["families"][family]
    return summary


def run_full_experiment(
    data_dir: Path,
    results_dir: Path,
    modes: list[DetectorMode] | None = None,
) -> dict[str, Any]:
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir = results_dir / "models"
    model_dir.mkdir(exist_ok=True)
    records, meta = load_dataset(data_dir)
    modes = modes or MODES
    summary: dict[str, Any] = {"modes": {}, "generated_at": time.time()}

    all_preds: list[dict[str, Any]] = []
    for mode in modes:
        res, preds = run_mode_experiment(records, meta, mode, model_dir)
        summary["modes"][mode] = res.to_dict()
        for p in preds:
            p["experiment_mode"] = mode
        all_preds.extend(preds)
        with (results_dir / f"predictions_{mode}.jsonl").open("w", encoding="utf-8") as f:
            for row in preds:
                f.write(json.dumps(row) + "\n")

    summary["primary_comparison"] = {
        mode: summary["modes"][mode]
        for mode in PRIMARY_MODES
        if mode in summary["modes"]
    }
    summary["held_out_families"] = run_held_out_family_evaluation(
        data_dir, results_dir, families=list(HELD_OUT_FAMILIES)
    )
    summary["xss_component_profile"] = _xss_component_profile(all_preds)
    summary["contamination_combined_stats"] = run_contamination_sensitivity(records, meta, results_dir)

    with (results_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _plot_comparisons(summary, results_dir)
    _error_analysis(all_preds, results_dir)
    return summary


def _plot_comparisons(summary: dict[str, Any], out: Path) -> None:
    import matplotlib.pyplot as plt

    modes = list(summary["modes"].keys())
    f1s = [summary["modes"][m]["metrics_malicious"]["f1"] for m in modes]
    fprs = [summary["modes"][m]["metrics_malicious"]["false_positive_rate"] for m in modes]
    evasive = [
        summary["modes"][m].get("attack_slices", {}).get("rule_evasive", {}).get("recall", 0)
        for m in modes
    ]

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(modes))
    ax.bar(x - 0.25, f1s, 0.2, label="F1 (all malicious)")
    ax.bar(x - 0.05, fprs, 0.2, label="FPR (legitimate)")
    ax.bar(x + 0.15, evasive, 0.2, label="Recall (rule-evasive slice)")
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=30, ha="right")
    ax.legend()
    ax.set_title("Detector comparison (test split)")
    fig.tight_layout()
    fig.savefig(out / "detector_comparison.png", dpi=120)
    plt.close(fig)


def _error_analysis(preds: list[dict[str, Any]], out: Path) -> None:
    mistakes = []
    for p in preds:
        if p.get("experiment_mode") != "rules_plus_stats":
            continue
        group = p["scenario_group"]
        positive = group in ("malicious_intent", "hard_malicious_intent")
        alerted = alert_predicted(p["decision"])
        if positive and not alerted:
            kind = "false_negative"
        elif not positive and alerted and group in ("legitimate", "difficult_legitimate", "distribution_shift"):
            kind = "false_positive"
        else:
            continue
        explanations = p.get("score", {}).get("explanations", [])[:5]
        mistakes.append(
            {
                "kind": kind,
                "request_id": p["request_id"],
                "scenario_group": group,
                "attack_family": p.get("attack_family"),
                "attack_slice": p.get("attack_slice"),
                "decision": p["decision"],
                "likely_cause": explanations[0] if explanations else "unknown",
                "evidence": explanations,
            }
        )
    with (out / "error_analysis.json").open("w", encoding="utf-8") as f:
        json.dump(mistakes[:200], f, indent=2)
