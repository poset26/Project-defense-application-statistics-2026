"""Detector mode isolation tests."""

from anomaly_guard.core.decision import AnomalyGuard, STATISTICS_ONLY_MODES
from anomaly_guard.core.schema import RequestRecord, load_app_config
from anomaly_guard.core.rules import evaluate_rules
from anomaly_guard.core.parser import parse_request


def _sqli_login() -> RequestRecord:
    return RequestRecord(
        request_id="1",
        timestamp="t",
        client_id="c",
        method="POST",
        route="/login",
        raw_query="",
        headers={"content-type": "application/x-www-form-urlencoded"},
        body="username=admin' OR 1=1--&password=x",
    )


def _evasive_login() -> RequestRecord:
    return RequestRecord(
        request_id="2",
        timestamp="t",
        client_id="c",
        method="POST",
        route="/login",
        raw_query="",
        headers={"content-type": "application/x-www-form-urlencoded"},
        body="username=ad'||'min&password=x",
    )


def _train_rows(n: int = 30) -> list[RequestRecord]:
    rows = []
    for i in range(n):
        rows.append(
            RequestRecord(
                request_id=f"t{i}",
                timestamp="t",
                client_id="c",
                method="GET",
                route="/search",
                raw_query=f"q=query{i % 10}&page=1",
                headers={},
                body="",
            )
        )
    return rows


def test_length_only_ignores_pattern_rules_in_decision():
    config = load_app_config()
    rec = _sqli_login()
    parsed = parse_request(rec, config)
    assert any(r.kind == "suspicious_pattern" for r in evaluate_rules(parsed, config))

    guard = AnomalyGuard(mode="length_only")
    guard.fit(_train_rows())
    guard.calibrate(_train_rows(15))
    result = guard.score(rec)
    assert result.pattern_rule_count >= 1
    assert result.decision in ("normal", "review", "alert")
    if result.statistical and result.statistical.aggregate_percentile is not None:
        if result.statistical.aggregate_percentile < guard.settings.alert_threshold_percentile:
            assert result.decision != "alert" or result.statistical.aggregate_percentile >= guard.settings.alert_threshold_percentile


def test_statistics_only_decision_unchanged_when_patterns_would_fire():
    rec = _sqli_login()
    guard = AnomalyGuard(mode="composition_only")
    guard.fit(_train_rows(40))
    guard.calibrate(_train_rows(20))
    with_patterns = guard.score(rec).decision

    config = type(guard.config)(
        limits=guard.config.limits,
        routes=guard.config.routes,
        rules=type(guard.config.rules)(suspicious_patterns=[], schema_violation_severity="medium"),
        config_hash="empty-patterns",
    )
    guard2 = AnomalyGuard(config=config, mode="composition_only", settings=guard.settings)
    guard2.detector = guard.detector
    without_patterns = guard2.score(rec).decision
    assert with_patterns == without_patterns


def test_rule_evasive_not_always_signature_hit():
    config = load_app_config()
    rec = _evasive_login()
    parsed = parse_request(rec, config)
    patterns = [r for r in evaluate_rules(parsed, config) if r.kind == "suspicious_pattern"]
    assert len(patterns) == 0


def test_rules_only_alerts_on_signature_sqli():
    guard = AnomalyGuard(mode="rules_only")
    guard.fit(_train_rows())
    result = guard.score(_sqli_login())
    assert result.decision in ("alert", "review")
