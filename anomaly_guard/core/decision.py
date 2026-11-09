"""Decision engine combining rules and statistical evidence."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
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

STATISTICS_ONLY_MODES: frozenset[DetectorMode] = frozenset(
    {"length_only", "composition_only", "combined_stats", "isolation_forest"}
)


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
    schema_rule_count: int = 0
    pattern_rule_count: int = 0

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
    alert_threshold_percentile: float = 99.0
    review_threshold_percentile: float = 95.0
    isolation_alert_threshold: float = 0.0
    isolation_review_threshold: float = 0.0
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
            contamination=0.001,
            random_state=42,
        )
        self._isolation.fit(X)

    def calibrate(self, validation_records: list[RequestRecord]) -> DecisionSettings:
        """Tune thresholds on held-out legitimate validation traffic."""
        if self.mode != "rules_only":
            self.detector.calibrate_percentiles(validation_records)

        if self.mode in STATISTICS_ONLY_MODES or self.mode == "rules_plus_stats":
            self._calibrate_percentile_thresholds(validation_records)

        if self.mode == "isolation_forest":
            self._calibrate_isolation(validation_records)

        return self.settings

    def _calibrate_percentile_thresholds(self, validation_records: list[RequestRecord]) -> None:
        percentiles: list[float] = []
        for rec in validation_records:
            pct = self._aggregate_percentile_for_record(rec)
            if pct is not None:
                percentiles.append(pct)
        if not percentiles:
            return
        scores_arr = np.array(sorted(percentiles))
        target = self.settings.max_false_positive_rate_target
        alert_idx = int(np.ceil((1.0 - target) * len(scores_arr))) - 1
        alert_idx = max(0, min(alert_idx, len(scores_arr) - 1))
        self.settings.alert_threshold_percentile = float(scores_arr[alert_idx])
        review_idx = max(0, alert_idx - max(1, len(scores_arr) // 25))
        self.settings.review_threshold_percentile = float(scores_arr[review_idx])

    def _calibrate_isolation(self, validation_records: list[RequestRecord]) -> None:
        if self._isolation is None:
            return
        scores: list[float] = []
        order = self.detector.baseline.feature_field_order if self.detector.baseline else None
        for rec in validation_records:
            parsed = parse_request(rec, self.config)
            feats = extract_features(parsed)
            vec = np.array([feats.to_vector(order)], dtype=float)
            scores.append(float(self._isolation.decision_function(vec)[0]))
        if not scores:
            return
        scores_arr = np.array(sorted(scores))
        target = self.settings.max_false_positive_rate_target
        alert_idx = int(np.floor(target * len(scores_arr)))
        alert_idx = max(0, min(alert_idx, len(scores_arr) - 1))
        review_idx = int(np.floor(target * 2 * len(scores_arr)))
        review_idx = max(0, min(review_idx, len(scores_arr) - 1))
        self.settings.isolation_alert_threshold = float(scores_arr[alert_idx])
        self.settings.isolation_review_threshold = float(scores_arr[review_idx])

    def _aggregate_percentile_for_record(self, record: RequestRecord) -> float | None:
        if self.mode == "rules_only":
            return None
        stat = self._filtered_statistics(record)
        return stat.aggregate_percentile

    def score(self, record: RequestRecord) -> ScoreResult:
        t0 = time.perf_counter()
        parsed = parse_request(record, self.config)
        rules = evaluate_rules(parsed, self.config)
        explanations: list[str] = []
        component_scores: dict[str, float] = {}
        schema_count = sum(1 for r in rules if r.kind == "schema" or r.kind == "size_limit")
        pattern_count = sum(1 for r in rules if r.kind == "suspicious_pattern")
        component_scores["schema_rule_count"] = float(schema_count)
        component_scores["pattern_rule_count"] = float(pattern_count)

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
                    "Insufficient baseline coverage for this endpoint; statistical score unavailable."
                )

        isolation_score = None
        if self.mode == "isolation_forest" and self._isolation is not None:
            feats = extract_features(parsed)
            order = self.detector.baseline.feature_field_order if self.detector.baseline else None
            vec = np.array([feats.to_vector(order)], dtype=float)
            isolation_score = float(self._isolation.decision_function(vec)[0])
            component_scores["isolation_score"] = isolation_score

        if self.mode == "rules_only":
            decision = _decision_from_rules(rules, self.settings)
        elif self.mode == "rules_plus_stats":
            decision = _decision_rules_plus_stats(stat, rules, self.settings)
        elif self.mode == "isolation_forest":
            decision = _decision_from_isolation_only(isolation_score, self.settings)
        elif self.mode in STATISTICS_ONLY_MODES:
            decision = _decision_from_statistics(stat, self.settings)
        else:
            decision = _decision_from_statistics(stat, self.settings)

        for r in rules:
            if self.mode in STATISTICS_ONLY_MODES:
                tag = " (informational; not used in this mode's decision)" if r.kind == "suspicious_pattern" else " (schema note only)"
                explanations.append(f"Rule {r.rule_id} ({r.severity}){tag}: {r.explanation}")
            else:
                explanations.append(f"Rule {r.rule_id} ({r.severity}): {r.explanation}")

        if stat:
            for c in stat.components:
                if (
                    c.status == "ok"
                    and c.percentile is not None
                    and c.percentile >= self.settings.review_threshold_percentile
                ):
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
            schema_rule_count=schema_count,
            pattern_rule_count=pattern_count,
        )

    def _filtered_statistics(self, record: RequestRecord) -> StatisticalResult:
        stat = self.detector.score(record)
        if self.mode == "length_only":
            stat.components = [c for c in stat.components if c.component == "length"]
        elif self.mode == "composition_only":
            stat.components = [c for c in stat.components if c.component == "composition"]
        pcts = [c.percentile for c in stat.components if c.status == "ok" and c.percentile is not None]
        stat.aggregate_percentile = max(pcts) if pcts else None
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
    if any(r.severity == "low" for r in rules):
        return "review"
    return "normal"


def _decision_from_statistics(
    stat: StatisticalResult | None,
    settings: DecisionSettings,
) -> DecisionLabel:
    pct = stat.aggregate_percentile if stat else None
    if pct is None:
        return "normal"
    if pct >= settings.alert_threshold_percentile:
        return "alert"
    if pct >= settings.review_threshold_percentile:
        return "review"
    return "normal"


def _decision_rules_plus_stats(
    stat: StatisticalResult | None,
    rules: list[TriggeredRule],
    settings: DecisionSettings,
) -> DecisionLabel:
    if settings.enable_high_severity_rule_alert:
        for r in rules:
            if r.severity == "high" and r.kind == "suspicious_pattern":
                return "alert"
    stat_decision = _decision_from_statistics(stat, settings)
    if stat_decision == "alert":
        return "alert"
    if stat_decision == "review":
        return "review"
    if any(r.kind == "schema" or r.kind == "size_limit" for r in rules):
        return "review"
    rule_sev = max_severity(rules)
    if rule_sev == "high":
        return "alert"
    if rule_sev == "medium":
        return "review"
    return "normal"


def _decision_from_isolation_only(
    score: float | None,
    settings: DecisionSettings,
) -> DecisionLabel:
    if score is None:
        return "normal"
    if score <= settings.isolation_alert_threshold:
        return "alert"
    if score <= settings.isolation_review_threshold:
        return "review"
    return "normal"
