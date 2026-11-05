"""Decision engine combining rules and statistical evidence."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
from sklearn.ensemble import IsolationForest

from anomaly_guard.core.features import extract_features
from anomaly_guard.core.parser import parse_request
from anomaly_guard.core.rules import TriggeredRule, evaluate_rules, max_severity
from anomaly_guard.core.schema import AppConfig, DecisionLabel, RequestRecord, load_app_config
from anomaly_guard.core.statistics import StatisticalDetector, StatisticalResult

DetectorMode = Literal[
    "rules_only",
    "length_only",
    "composition_only",
    "combined_stats",
    "rules_plus_stats",
    "isolation_forest",
]


@dataclass
class ScoreResult:
    decision: DecisionLabel
    request_id: str
    method: str
    route: str
    triggered_rules: list[TriggeredRule]
    statistical: StatisticalResult | None
    component_scores: dict[str, float]
    explanations: list[str]
    baseline_version: str
    processing_time_ms: float
    mode: DetectorMode
    isolation_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["triggered_rules"] = [asdict(r) for r in self.triggered_rules]
        if self.statistical:
            d["statistical"] = {
                "baseline_version": self.statistical.baseline_version,
                "aggregate_percentile": self.statistical.aggregate_percentile,
                "insufficient_baseline": self.statistical.insufficient_baseline,
                "processing_time_ms": self.statistical.processing_time_ms,
                "components": [asdict(c) for c in self.statistical.components],
            }
        return d


@dataclass
class DecisionSettings:
    alert_threshold_percentile: float = 95.0
    review_threshold_percentile: float = 85.0
    isolation_alert_threshold: float = -0.05
    isolation_review_threshold: float = -0.12
    max_false_positive_rate_target: float = 0.01
    enable_high_severity_rule_alert: bool = True


class AnomalyGuard:
    """Shared detection library for simulator, dashboard, and evaluation."""

    def __init__(
        self,
        config: AppConfig | None = None,
        mode: DetectorMode = "rules_plus_stats",
        settings: DecisionSettings | None = None,
    ):
        self.config = config or load_app_config()
        self.mode = mode
        self.settings = settings or DecisionSettings()
        self.detector = StatisticalDetector(self.config)
        self._isolation: IsolationForest | None = None
        self._feature_order: list[str] = []

    def fit(self, normal_records: list[RequestRecord]) -> None:
        self.detector.fit(normal_records)
        if self.mode == "isolation_forest":
            self._fit_isolation(normal_records)

    def _fit_isolation(self, records: list[RequestRecord]) -> None:
        if self.detector.baseline is None:
            return
        field_order = self.detector.baseline.feature_field_order
        vectors = []
        for rec in records:
            parsed = parse_request(rec, self.config)
            feats = extract_features(parsed)
            vectors.append(feats.to_vector(field_order))
        if len(vectors) < 10:
            return
        X = np.array(vectors, dtype=float)
        self._isolation = IsolationForest(
            n_estimators=100,
            contamination=0.02,
            random_state=42,
        )
        self._isolation.fit(X)

    def calibrate(self, validation_records: list[RequestRecord]) -> DecisionSettings:
        """Tune percentile thresholds to approximate FP budget on validation legitimate traffic."""
        self.detector.calibrate_percentiles(validation_records)
        scores: list[float] = []
        for rec in validation_records:
            res = self.score(rec)
            pct = res.component_scores.get("aggregate_percentile")
            if pct is not None:
                scores.append(pct)
        if not scores:
            return self.settings
        scores_arr = np.array(sorted(scores))
        target = self.settings.max_false_positive_rate_target
        alert_idx = int(np.ceil((1.0 - target) * len(scores_arr))) - 1
        alert_idx = max(0, min(alert_idx, len(scores_arr) - 1))
        self.settings.alert_threshold_percentile = float(scores_arr[alert_idx])
        review_idx = max(0, alert_idx - max(1, len(scores_arr) // 20))
        self.settings.review_threshold_percentile = float(scores_arr[review_idx])
        return self.settings

    def score(self, record: RequestRecord) -> ScoreResult:
        t0 = time.perf_counter()
        parsed = parse_request(record, self.config)
        rules = evaluate_rules(parsed, self.config)
        explanations: list[str] = []
        component_scores: dict[str, float] = {}

        stat: StatisticalResult | None = None
        if self.mode != "rules_only":
            stat = self._filtered_statistics(record)

        if stat:
            for c in stat.components:
                if c.status == "ok":
                    key = f"{c.component}:{c.field}"
                    component_scores[key] = c.raw_score
                    if c.percentile is not None:
                        component_scores[f"{key}_pct"] = c.percentile
            if stat.aggregate_percentile is not None:
                component_scores["aggregate_percentile"] = stat.aggregate_percentile
            if stat.insufficient_baseline:
                explanations.append(
                    "Insufficient baseline coverage for this endpoint; schema rules and fallback apply."
                )

        isolation_score = None
        if self.mode == "isolation_forest" and self._isolation is not None:
            feats = extract_features(parsed)
            order = self.detector.baseline.feature_field_order if self.detector.baseline else None
            vec = np.array([feats.to_vector(order)], dtype=float)
            isolation_score = float(self._isolation.decision_function(vec)[0])
            component_scores["isolation_score"] = isolation_score

        decision: DecisionLabel = "normal"
        rule_sev = max_severity(rules)

        if self.mode == "rules_only":
            decision = _decision_from_rules(rules, self.settings)
        elif self.mode == "isolation_forest":
            decision = _decision_from_isolation(isolation_score, rules, self.settings)
        else:
            decision = _decision_hybrid(stat, rules, rule_sev, self.settings, self.mode)

        for r in rules:
            explanations.append(f"Rule {r.rule_id} ({r.severity}): {r.explanation}")
        if stat:
            for c in stat.components:
                if c.status == "ok" and c.percentile is not None and c.percentile >= self.settings.review_threshold_percentile:
                    explanations.append(
                        f"Field '{c.field}' {c.component} unusual (empirical percentile {c.percentile:.1f}): {c.detail}"
                    )

        ms = (time.perf_counter() - t0) * 1000
        return ScoreResult(
            decision=decision,
            request_id=record.request_id,
            method=record.method,
            route=record.route,
            triggered_rules=rules,
            statistical=stat,
            component_scores=component_scores,
            explanations=explanations,
            baseline_version=stat.baseline_version if stat else "none",
            processing_time_ms=ms,
            mode=self.mode,
            isolation_score=isolation_score,
        )

    def _filtered_statistics(self, record: RequestRecord) -> StatisticalResult:
        stat = self.detector.score(record)
        if self.mode == "length_only":
            stat.components = [c for c in stat.components if c.component == "length"]
        elif self.mode == "composition_only":
            stat.components = [c for c in stat.components if c.component == "composition"]
        ok = [c.raw_score for c in stat.components if c.status == "ok"]
        if ok:
            agg = max(ok)
            stat.aggregate_percentile = self.detector._empirical_percentile(agg)
        else:
            stat.aggregate_percentile = None
        return stat

    def save_baseline(self, path) -> None:
        self.detector.save(path)

    def load_baseline(self, path) -> None:
        self.detector.load(path)


def _decision_from_rules(rules: list[TriggeredRule], settings: DecisionSettings) -> DecisionLabel:
    if not rules:
        return "normal"
    if settings.enable_high_severity_rule_alert and any(r.severity == "high" for r in rules):
        return "alert"
    if any(r.severity == "medium" for r in rules):
        return "review"
    return "review"


def _decision_hybrid(
    stat: StatisticalResult | None,
    rules: list[TriggeredRule],
    rule_sev: str | None,
    settings: DecisionSettings,
    mode: DetectorMode,
) -> DecisionLabel:
    if settings.enable_high_severity_rule_alert:
        for r in rules:
            if r.severity == "high" and r.kind == "suspicious_pattern":
                return "alert"

    pct = stat.aggregate_percentile if stat else None
    if pct is not None:
        if pct >= settings.alert_threshold_percentile:
            return "alert"
        if pct >= settings.review_threshold_percentile:
            return "review"

    if mode in ("combined_stats", "rules_plus_stats", "length_only", "composition_only"):
        if any(r.kind == "schema" for r in rules):
            return "review"

    if rule_sev == "high":
        return "alert"
    if rule_sev == "medium":
        return "review"
    return "normal"


def _decision_from_isolation(
    score: float | None,
    rules: list[TriggeredRule],
    settings: DecisionSettings,
) -> DecisionLabel:
    dr = _decision_from_rules(rules, settings)
    if dr == "alert":
        return "alert"
    if score is None:
        return dr
    if score <= settings.isolation_alert_threshold:
        return "alert"
    if score <= settings.isolation_review_threshold:
        return "review"
    return dr if dr != "normal" else "normal"
