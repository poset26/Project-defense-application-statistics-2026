"""Decision engine combining rules and statistical evidence."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from sklearn.ensemble import IsolationForest

from anomaly_guard.core.features import extract_features
from anomaly_guard.core.parser import parse_request
from anomaly_guard.core.rules import TriggeredRule, evaluate_rules, max_severity
from anomaly_guard.core.schema import AppConfig, DecisionLabel, RequestRecord, load_app_config
from anomaly_guard.core.statistics import ComponentScore, StatisticalDetector, StatisticalResult

DetectorMode = Literal[
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

STATISTICS_ONLY_MODES: frozenset[DetectorMode] = frozenset(
    {
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
        "isolation_forest",
    }
)

MEAN_AGGREGATE_MODES: frozenset[DetectorMode] = frozenset(
    {"full_stats", "rules_plus_full_stats", "ci_only"}
)
SECOND_AGGREGATE_MODES: frozenset[DetectorMode] = frozenset({"vote2_stats"})
RAW_DISTANCE_MODES: frozenset[DetectorMode] = frozenset({"payl_only"})

MODE_ACTIVE_COMPONENTS: dict[DetectorMode, frozenset[str] | None] = {
    "rules_only": frozenset(),
    "length_only": frozenset({"length"}),
    "composition_only": frozenset({"composition"}),
    "payl_only": frozenset({"payl"}),
    "ngram_only": frozenset({"ngram"}),
    "regression_only": frozenset({"regression"}),
    "lda_only": frozenset({"mahalanobis"}),
    "ci_only": frozenset({"length_ci", "proportion_ci"}),
    "entropy_only": frozenset({"entropy"}),
    "vote2_stats": frozenset(
        {
            "length",
            "composition",
            "length_stratified",
            "payl",
            "ngram",
            "regression",
            "mahalanobis",
            "length_ci",
            "proportion_ci",
        }
    ),
    "combined_stats": frozenset({"length", "composition"}),
    "full_stats": frozenset(
        {
            "length",
            "composition",
            "length_stratified",
            "payl",
            "ngram",
            "regression",
            "mahalanobis",
            "length_ci",
            "proportion_ci",
        }
    ),
    "rules_plus_stats": frozenset({"length", "composition"}),
    "rules_plus_full_stats": frozenset(
        {
            "length",
            "composition",
            "length_stratified",
            "payl",
            "ngram",
            "regression",
            "mahalanobis",
            "length_ci",
            "proportion_ci",
        }
    ),
    "isolation_forest": None,
}


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
    threshold_scale: str = "percentile"
    threshold_population: str = "ordinary_validation"


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

    def calibrate(
        self,
        validation_records: list[RequestRecord],
        threshold_records: list[RequestRecord] | None = None,
    ) -> DecisionSettings:
        """Map scores on held-out non-attack traffic. Set the cutoff on one population.

        `validation_records` build the empirical percentile pools. The alert
        cutoff uses `threshold_records` when that list is passed, so a stress
        group such as distribution shift does not choose the 1% budget.
        """
        cutoff_records = threshold_records if threshold_records is not None else validation_records
        self.settings.threshold_population = (
            "ordinary_validation" if threshold_records is not None else "same_as_percentile_pool"
        )
        if self.mode != "rules_only":
            self.detector.calibrate_percentiles(validation_records)

        hybrid_modes: frozenset[DetectorMode] = frozenset({"rules_plus_stats", "rules_plus_full_stats"})
        if self.mode in STATISTICS_ONLY_MODES or self.mode in hybrid_modes:
            self._calibrate_percentile_thresholds(cutoff_records)

        if self.mode == "isolation_forest":
            self._calibrate_isolation(cutoff_records)

        return self.settings

    def save_settings(self, path) -> None:
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"mode": self.mode, **asdict(self.settings)}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_settings(self, path) -> bool:
        """Apply a saved cutoff when it was fit for this detector mode."""
        from pathlib import Path

        path = Path(path)
        if not path.exists():
            return False
        data = json.loads(path.read_text(encoding="utf-8"))
        saved_mode = data.get("mode")
        if saved_mode and saved_mode != self.mode:
            return False
        for key, value in data.items():
            if key == "mode":
                continue
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
        return True

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
        if self.mode in RAW_DISTANCE_MODES:
            self.settings.threshold_scale = "raw_distance"
        elif self.mode in SECOND_AGGREGATE_MODES:
            self.settings.threshold_scale = "second_percentile"
        else:
            self.settings.threshold_scale = "percentile"

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
                if self.mode in MEAN_AGGREGATE_MODES:
                    explanations.append(
                        "Statistical aggregate is the mean of per-type empirical percentiles "
                        "(one value per component, max across fields), not the max."
                    )
                elif self.mode in SECOND_AGGREGATE_MODES:
                    explanations.append(
                        "Statistical aggregate is the second-highest component-type percentile: "
                        "at least two component types must be unusual."
                    )
                elif self.mode in RAW_DISTANCE_MODES:
                    explanations.append(
                        "PAYL alert cutoff is a quantile of the raw byte-histogram distance, "
                        "not an empirical percentile. Percentile 100 was tied too often to meet a 1% budget."
                    )
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
        elif self.mode in ("rules_plus_stats", "rules_plus_full_stats"):
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
                if c.status == "ok" and component_exceeds_review(c, self.settings):
                    if self.settings.threshold_scale == "raw_distance":
                        explanations.append(
                            f"Field '{c.field}' {c.component} unusual "
                            f"(raw distance {c.raw_score:.3f}): {c.detail}"
                        )
                    else:
                        explanations.append(
                            f"Field '{c.field}' {c.component} unusual "
                            f"(empirical percentile {c.percentile:.1f}): {c.detail}"
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
        allowed = MODE_ACTIVE_COMPONENTS.get(self.mode)
        if allowed is not None:
            stat.components = [c for c in stat.components if c.component in allowed]
        if self.mode in RAW_DISTANCE_MODES:
            raws = [c.raw_score for c in stat.components if c.status == "ok"]
            stat.aggregate_percentile = max(raws) if raws else None
            stat.aggregate_kind = "raw_distance"
        else:
            how = (
                "mean"
                if self.mode in MEAN_AGGREGATE_MODES
                else "second"
                if self.mode in SECOND_AGGREGATE_MODES
                else "max"
            )
            stat.aggregate_percentile = aggregate_component_percentiles(stat.components, how=how)
            stat.aggregate_kind = how
        return stat

    def save_baseline(self, path) -> None:
        self.detector.save(path)

    def load_baseline(self, path) -> None:
        self.detector.load(path)


def aggregate_component_percentiles(
    components: list[ComponentScore], how: str = "max"
) -> float | None:
    """Collapse field scores to one percentile per component type.

    `how` is `max`, `mean`, or `second` (the second-highest type, so two types
    must clear a threshold).
    """
    by_type: dict[str, float] = {}
    for comp in components:
        if comp.status != "ok" or comp.percentile is None:
            continue
        prev = by_type.get(comp.component)
        by_type[comp.component] = comp.percentile if prev is None else max(prev, comp.percentile)
    if not by_type:
        return None
    values = sorted(by_type.values(), reverse=True)
    if how == "mean":
        return float(sum(values) / len(values))
    if how == "second":
        return float(values[1]) if len(values) >= 2 else 0.0
    return float(values[0])


def component_exceeds_review(component: ComponentScore, settings: DecisionSettings) -> bool:
    """Review narration uses the same scale as the cutoff.

    A raw PAYL distance near 15 must not be compared with a 0–100 percentile.
    """
    if settings.threshold_scale == "raw_distance":
        return component.raw_score >= settings.review_threshold_percentile
    if component.percentile is None:
        return False
    return component.percentile >= settings.review_threshold_percentile


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
