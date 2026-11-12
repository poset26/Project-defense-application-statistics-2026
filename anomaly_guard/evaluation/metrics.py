"""Evaluation metrics and threshold analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from anomaly_guard.core.schema import GroundTruthMeta


def _wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval. Matches the usual 95% two-sided bound."""
    if n <= 0:
        return 0.0, 0.0
    p = successes / n
    z2 = z * z
    den = 1.0 + z2 / n
    centre = p + z2 / (2.0 * n)
    margin = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return (centre - margin) / den, (centre + margin) / den


def is_positive_malicious(meta: GroundTruthMeta) -> bool:
    return meta.scenario_group in ("malicious_intent", "hard_malicious_intent")


def is_positive_any_anomaly(meta: GroundTruthMeta) -> bool:
    return meta.scenario_group in (
        "malicious_intent",
        "hard_malicious_intent",
        "benign_invalid",
    )


def is_benign_invalid(meta: GroundTruthMeta) -> bool:
    return meta.scenario_group == "benign_invalid"


def is_legitimate_family(meta: GroundTruthMeta) -> bool:
    return meta.scenario_group in (
        "legitimate",
        "difficult_legitimate",
        "distribution_shift",
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


@dataclass
class BenignInvalidMetrics:
    """How often invalid-but-benign requests are flagged (rejecting them is desired)."""

    total: int
    flagged: int
    rejection_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "flagged": self.flagged,
            "rejection_rate": self.rejection_rate,
        }


def compute_malicious_attack_metrics(
    meta: list[GroundTruthMeta],
    pred_alert: list[bool],
) -> BinaryMetrics:
    """Malicious-intent detection with precision excluding benign_invalid false alarms.

    Recall: flagged / all malicious (unchanged).
    Precision: TP / (TP + FP on legitimate-family traffic only).
    FPR: flagged rate on legitimate-family traffic (benign_invalid excluded).
    """
    legit_mask = [is_legitimate_family(m) for m in meta]
    y_mal = [is_positive_malicious(m) for m in meta]
    yt = np.array(y_mal, dtype=int)
    yp = np.array(pred_alert, dtype=int)
    tp = int(((yt == 1) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    leg = np.array(legit_mask, dtype=bool)
    fp_leg = int(((yp == 1) & leg).sum())
    tn = int(((yp == 0) & leg).sum())
    prec = tp / (tp + fp_leg) if (tp + fp_leg) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    fp_rate = float(np.mean(yp[leg])) if leg.any() else 0.0
    n_leg = int(leg.sum())
    fp_count = int(((yp == 1) & leg).sum())
    low, high = _wilson_interval(fp_count, n_leg) if n_leg else (0.0, 0.0)
    fnr = fn / max(fn + tp, 1)
    return BinaryMetrics(
        tp=tp,
        fp=fp_leg,
        tn=tn,
        fn=fn,
        precision=float(prec),
        recall=float(rec),
        f1=float(f1),
        fpr=float(fp_rate),
        fnr=float(fnr),
        wilson_fpr_low=float(low),
        wilson_fpr_high=float(high),
    )


def compute_benign_invalid_metrics(
    meta: list[GroundTruthMeta],
    pred_alert: list[bool],
) -> BenignInvalidMetrics:
    invalid_mask = [is_benign_invalid(m) for m in meta]
    yp = np.array(pred_alert, dtype=int)
    inv = np.array(invalid_mask, dtype=bool)
    total = int(inv.sum())
    flagged = int((yp & inv).sum()) if total else 0
    rate = float(flagged / total) if total else 0.0
    return BenignInvalidMetrics(total=total, flagged=flagged, rejection_rate=rate)


def compute_binary_metrics(
    y_true: list[bool],
    y_pred: list[bool],
    legitimate_mask: list[bool] | None = None,
) -> BinaryMetrics:
    yt = np.array(y_true, dtype=int)
    yp = np.array(y_pred, dtype=int)
    tp = int(((yt == 1) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    if legitimate_mask is not None:
        leg = np.array(legitimate_mask, dtype=bool)
        fp_rate = float(np.mean(yp[leg])) if leg.any() else float(fp / max(fp + tn, 1))
        n_leg = int(leg.sum())
        fp_count = int(((yp == 1) & leg).sum())
        low, high = _wilson_interval(fp_count, n_leg) if n_leg else (0.0, 0.0)
    else:
        fp_rate = fp / max(fp + tn, 1)
        low, high = _wilson_interval(fp, fp + tn) if (fp + tn) else (0.0, 0.0)
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


def posterior_attack_given_alert(recall: float, fpr: float, prior: float) -> float | None:
    """Bayes P(attack | alert) from recall, legitimate FPR, and a prior.

    Returns None when the denominator is zero (the rates never alert).
    A zero false-positive rate makes the posterior 1 whenever recall and the
    prior are positive. That is the point estimate, not an operational claim.
    """
    if prior < 0.0 or prior > 1.0:
        return None
    numerator = recall * prior
    denominator = numerator + fpr * (1.0 - prior)
    if denominator <= 0.0:
        return None
    return float(numerator / denominator)


def base_rate_posteriors(
    recall: float,
    fpr: float,
    wilson_fpr_high: float,
    n_malicious: int,
    n_legitimate: int,
) -> dict[str, Any]:
    """Posteriors at the file mixture and at rare-event priors.

    The file mixture is malicious / (malicious + legitimate-family). Benign-invalid
    rows are left out of that prevalence, matching the precision denominator.
    """
    mixture = (
        n_malicious / (n_malicious + n_legitimate) if (n_malicious + n_legitimate) else None
    )
    priors = {
        "file_legitimate_mixture": mixture,
        "prior_1pct": 0.01,
        "prior_0_1pct": 0.001,
    }
    rows = []
    for name, prior in priors.items():
        if prior is None:
            continue
        rows.append(
            {
                "name": name,
                "prior": prior,
                "posterior_at_observed_fpr": posterior_attack_given_alert(recall, fpr, prior),
                "posterior_at_wilson_fpr_upper": posterior_attack_given_alert(
                    recall, wilson_fpr_high, prior
                ),
            }
        )
    return {
        "n_malicious": n_malicious,
        "n_legitimate_family": n_legitimate,
        "note": (
            "Precision on this file is not an operational detection rate. "
            "The Wilson-upper column replaces a zero observed FPR with the interval's upper end."
        ),
        "rows": rows,
    }


def alert_predicted(decision: str) -> bool:
    return decision == "alert"


def review_predicted(decision: str) -> bool:
    return decision == "review"


def review_rate_on_legitimate(meta: list[GroundTruthMeta], decisions: list[str]) -> float:
    leg = [is_legitimate_family(m) for m in meta]
    if not any(leg):
        return 0.0
    rev = [review_predicted(d) for d in decisions]
    leg_arr = np.array(leg, dtype=bool)
    rev_arr = np.array(rev, dtype=bool)
    return float(np.mean(rev_arr[leg_arr]))


def score_to_rank(values: list[float]) -> list[float]:
    arr = np.array(values, dtype=float)
    order = arr.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0, 1, len(arr))
    return ranks.tolist()


def average_precision(y_true: list[bool], scores: list[float]) -> float:
    if not any(y_true):
        return 0.0
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(np.array(y_true, dtype=int), np.array(scores, dtype=float)))


def precision_recall_points(y_true: list[bool], scores: list[float]) -> tuple[list[float], list[float]]:
    from sklearn.metrics import precision_recall_curve

    p, r, _ = precision_recall_curve(np.array(y_true, dtype=int), np.array(scores, dtype=float))
    return p.tolist(), r.tolist()


def per_family_recall(
    meta: list[GroundTruthMeta],
    preds: dict[str, str],
) -> dict[str, dict[str, float]]:
    alerts: dict[str, list[bool]] = {}
    reviews: dict[str, list[bool]] = {}
    for m in meta:
        if not is_positive_malicious(m):
            continue
        fam = m.attack_family or "unknown"
        decision = preds.get(m.request_id, "normal")
        alerts.setdefault(fam, []).append(alert_predicted(decision))
        reviews.setdefault(fam, []).append(review_predicted(decision))
    return {
        fam: {
            "recall": float(np.mean(alerts[fam])) if alerts[fam] else 0.0,
            "review_rate": float(np.mean(reviews[fam])) if reviews[fam] else 0.0,
            "support": float(len(alerts[fam])),
        }
        for fam in alerts
    }


def slice_attack_recall(
    meta: list[GroundTruthMeta],
    pred_alert: list[bool],
    attack_slice: str,
) -> dict[str, float] | None:
    indices = [
        i
        for i, m in enumerate(meta)
        if is_positive_malicious(m) and m.attack_slice == attack_slice
    ]
    if not indices:
        return None
    hits = [pred_alert[i] for i in indices]
    return {"recall": float(np.mean(hits)), "support": float(len(indices))}


def ranking_score_from_prediction(score_dict: dict[str, Any]) -> float:
    comp = score_dict.get("component_scores") or {}
    if comp.get("aggregate_percentile") is not None:
        return float(comp["aggregate_percentile"])
    if comp.get("isolation_score") is not None:
        return float(-comp["isolation_score"])
    return 0.0
