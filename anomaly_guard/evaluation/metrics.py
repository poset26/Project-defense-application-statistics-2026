"""Evaluation metrics and threshold analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from statsmodels.stats.proportion import proportion_confint

from anomaly_guard.core.schema import GroundTruthMeta


def is_positive_malicious(meta: GroundTruthMeta) -> bool:
    return meta.scenario_group in ("malicious_intent", "hard_malicious_intent")


def is_positive_any_anomaly(meta: GroundTruthMeta) -> bool:
    return meta.scenario_group in (
        "malicious_intent",
        "hard_malicious_intent",
        "benign_invalid",
    )


@dataclass
class BinaryMetrics:
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    fpr: float
    fnr: float
    wilson_fpr_low: float
    wilson_fpr_high: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "false_positive_rate": self.fpr,
            "false_negative_rate": self.fnr,
            "wilson_fpr_ci": [self.wilson_fpr_low, self.wilson_fpr_high],
        }


def compute_binary_metrics(
    y_true: list[bool],
    y_pred: list[bool],
    legitimate_mask: list[bool] | None = None,
) -> BinaryMetrics:
    yt = np.array(y_true, dtype=int)
    yp = np.array(y_pred, dtype=int)
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    prec = precision_score(yt, yp, zero_division=0)
    rec = recall_score(yt, yp, zero_division=0)
    f1 = f1_score(yt, yp, zero_division=0)
    if legitimate_mask is not None:
        leg = np.array(legitimate_mask, dtype=bool)
        fp_rate = float(np.mean(yp[leg])) if leg.any() else float(fp / max(fp + tn, 1))
        n_leg = int(leg.sum())
        fp_count = int(((yp == 1) & leg).sum())
        low, high = proportion_confint(fp_count, n_leg, method="wilson") if n_leg else (0.0, 0.0)
    else:
        fp_rate = fp / max(fp + tn, 1)
        low, high = proportion_confint(fp, fp + tn, method="wilson") if (fp + tn) else (0.0, 0.0)
    fnr = fn / max(fn + tp, 1)
    return BinaryMetrics(
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn),
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
        fpr=float(fp_rate),
        fnr=float(fnr),
        wilson_fpr_low=float(low),
        wilson_fpr_high=float(high),
    )


def alert_predicted(decision: str) -> bool:
    return decision in ("alert", "review")


def score_to_rank(values: list[float]) -> list[float]:
    arr = np.array(values, dtype=float)
    order = arr.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0, 1, len(arr))
    return ranks.tolist()


def average_precision(y_true: list[bool], scores: list[float]) -> float:
    if not any(y_true):
        return 0.0
    return float(average_precision_score(np.array(y_true, dtype=int), np.array(scores, dtype=float)))


def precision_recall_points(y_true: list[bool], scores: list[float]) -> tuple[list[float], list[float]]:
    p, r, _ = precision_recall_curve(np.array(y_true, dtype=int), np.array(scores, dtype=float))
    return p.tolist(), r.tolist()


def per_family_recall(
    meta: list[GroundTruthMeta],
    preds: dict[str, str],
) -> dict[str, float]:
    by_family: dict[str, list[bool]] = {}
    hits: dict[str, list[bool]] = {}
    for m in meta:
        if not is_positive_malicious(m):
            continue
        fam = m.attack_family or "unknown"
        by_family.setdefault(fam, []).append(True)
        hits.setdefault(fam, []).append(alert_predicted(preds.get(m.request_id, "normal")))
    return {
        fam: float(np.mean(hits[fam])) if hits[fam] else 0.0 for fam in by_family
    }
