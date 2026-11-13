"""Evaluation metric definitions."""

from anomaly_guard.core.schema import GroundTruthMeta
from anomaly_guard.evaluation.metrics import (
    compute_benign_invalid_metrics,
    compute_malicious_attack_metrics,
)


def _meta(group: str, rid: str = "1") -> GroundTruthMeta:
    return GroundTruthMeta(
        request_id=rid,
        scenario_group=group,  # type: ignore[arg-type]
        scenario_id="s",
        attack_family=None,
        template_id="t",
        split="test",
    )


def test_malicious_precision_excludes_benign_invalid_flags():
    meta = [
        _meta("malicious_intent", "m1"),
        _meta("benign_invalid", "b1"),
        _meta("legitimate", "l1"),
    ]
    pred = [True, True, False]
    m = compute_malicious_attack_metrics(meta, pred)
    assert m.tp == 1 and m.fn == 0 and m.recall == 1.0
    assert m.fp == 0 and m.precision == 1.0
    assert m.fpr == 0.0


def test_rules_only_ranking_is_not_constant_and_forest_ignores_other_scores():
    from anomaly_guard.evaluation.metrics import ranking_score_from_prediction

    quiet = {"mode": "rules_only", "triggered_rules": [], "component_scores": {}}
    loud = {
        "mode": "rules_only",
        "triggered_rules": [{"severity": "high", "rule_id": "SQL"}],
        "component_scores": {},
    }
    assert ranking_score_from_prediction(loud) > ranking_score_from_prediction(quiet)
    forest = {
        "mode": "isolation_forest",
        "triggered_rules": [],
        "component_scores": {"aggregate_percentile": 99.0, "isolation_score": -0.42},
    }
    assert ranking_score_from_prediction(forest) == 0.42


def test_hybrid_ranking_boosts_rule_hits_above_stats():
    from anomaly_guard.evaluation.metrics import ranking_score_from_prediction

    low_stat_rule = {
        "mode": "rules_plus_stats",
        "triggered_rules": [{"severity": "high", "rule_id": "SQL"}],
        "component_scores": {"aggregate_percentile": 40.0},
    }
    stats_only = {
        "mode": "combined_stats",
        "triggered_rules": [],
        "component_scores": {"aggregate_percentile": 40.0},
    }
    assert ranking_score_from_prediction(low_stat_rule) > ranking_score_from_prediction(stats_only)


def test_benign_invalid_rejection_rate():
    meta = [_meta("benign_invalid", f"b{i}") for i in range(4)]
    meta.append(_meta("legitimate", "l1"))
    pred = [True, True, False, True, False]
    inv = compute_benign_invalid_metrics(meta, pred)
    assert inv.total == 4
    assert inv.flagged == 3
    assert inv.rejection_rate == 0.75
