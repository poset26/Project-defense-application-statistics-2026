"""Reproducible evaluation runner and ablation experiments."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from anomaly_guard.core.decision import AnomalyGuard, DetectorMode
from anomaly_guard.core.schema import GroundTruthMeta, RequestRecord, read_jsonl
from anomaly_guard.evaluation.metrics import (
    alert_predicted,
    average_precision,
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
)

MODES: list[DetectorMode] = [
    "rules_only",
    "length_only",
    "composition_only",
    "combined_stats",
    "rules_plus_stats",
    "isolation_forest",
]

PRIMARY_MODES: list[DetectorMode] = ["rules_only", "combined_stats", "rules_plus_stats"]

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
    parts = partition_records(records, meta)
    test_recs, test_meta = parts["test"]

    guard = AnomalyGuard(mode=mode)
    guard.fit(train_recs)
    guard.calibrate(val_recs)
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
    result = ExperimentResult(
        mode=mode,
        metrics_malicious=mal_metrics.to_dict(),
        metrics_benign_invalid=invalid_metrics.to_dict(),
        metrics_any_anomaly=any_metrics.to_dict(),
        fpr_on_legitimate_test=mal_metrics.fpr,
        review_rate_legitimate=review_rate_on_legitimate(test_meta_ordered, decisions),
        average_precision=ap,
        per_family=fam_recall,
        attack_slices=slices,
        latency_ms_median=float(np.median(lat)),
        latency_ms_p95=float(np.percentile(lat, 95)),
        settings=asdict(guard.settings),
    )
    _plot_mode_pr(y_mal, scores, mode, model_dir.parent)
    _threshold_sweep_percentile(guard, val_recs, test_meta_ordered, test_recs, mode, model_dir.parent)
    return result, predictions


def _plot_mode_pr(y_true: list[bool], scores: list[float], mode: str, out: Path) -> None:
    if not any(y_true):
        return
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
        return
    val_pcts = [guard._aggregate_percentile_for_record(r) for r in val_recs]
    val_pcts = [p for p in val_pcts if p is not None]
    if not val_pcts:
        return
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
    thresholds = np.linspace(min(val_pcts), max(val_pcts), 25)
    rows = []
    for t in thresholds:
        pred = [p is not None and p >= t for p in test_pcts]
        tp = sum(pred[i] and y_mal[i] for i in range(len(pred)))
        fn = sum((not pred[i]) and y_mal[i] for i in range(len(pred)))
        fp = sum(pred[i] and leg[i] for i in range(len(pred)))
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / sum(leg) if sum(leg) else 0.0
        rows.append({"threshold_percentile": float(t), "recall_malicious": rec, "fpr_legitimate": fpr})
    with (out / f"threshold_sweep_{mode}.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def run_held_out_family_evaluation(
    data_dir: Path,
    results_dir: Path,
    family: str = "traversal",
) -> dict[str, Any]:
    records, meta = load_dataset(data_dir)
    parts = partition_records(records, meta)
    test_recs, test_meta = parts["test"]
    meta_by_id = {m.request_id: m for m in meta}
    test_filtered = [
        (r, meta_by_id[r.request_id])
        for r in test_recs
        if meta_by_id[r.request_id].attack_family == family
    ]
    if not test_filtered:
        return {"family": family, "error": "no test rows for family"}

    train_recs = training_normal_records(records, meta)
    val_recs = validation_legitimate_records(records, meta)
    summary: dict[str, Any] = {"family": family, "modes": {}, "support": len(test_filtered)}
    for mode in PRIMARY_MODES:
        guard = AnomalyGuard(mode=mode)
        guard.fit(train_recs)
        guard.calibrate(val_recs)
        hits = 0
        for rec, _m in test_filtered:
            if alert_predicted(guard.score(rec).decision):
                hits += 1
        summary["modes"][mode] = {"recall": hits / len(test_filtered), "alerts": hits}
    path = results_dir / "held_out_family.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
    summary["held_out_traversal"] = run_held_out_family_evaluation(data_dir, results_dir, "traversal")

    with (results_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _plot_comparisons(summary, results_dir)
    _error_analysis(all_preds, results_dir)
    return summary


def _plot_comparisons(summary: dict[str, Any], out: Path) -> None:
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
