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


def test_ordinary_validation_keeps_only_the_fit_population():
    from anomaly_guard.core.schema import GroundTruthMeta
    from anomaly_guard.evaluation.splits import validation_ordinary_records

    recs = [_train_rows(1)[0]]
    groups = ["legitimate", "distribution_shift", "difficult_legitimate"]
    rows = []
    meta = []
    for i, group in enumerate(groups):
        row = _train_rows(1)[0]
        row = type(row)(
            request_id=str(i),
            timestamp=row.timestamp,
            client_id=row.client_id,
            method=row.method,
            route=row.route,
            raw_query=row.raw_query,
            headers=row.headers,
            body=row.body,
        )
        rows.append(row)
        meta.append(
            GroundTruthMeta(
                request_id=str(i),
                scenario_group=group,  # type: ignore
                scenario_id="x",
                split="validation",
            )
        )
    kept = validation_ordinary_records(rows, meta)
    assert [r.request_id for r in kept] == ["0"]
    del recs


def test_raw_distance_review_does_not_use_the_percentile():
    from anomaly_guard.core.decision import DecisionSettings, component_exceeds_review
    from anomaly_guard.core.statistics import ComponentScore

    settings = DecisionSettings(
        review_threshold_percentile=15.0,
        threshold_scale="raw_distance",
    )
    mild = ComponentScore("payl", "q", 2.0, 90.0, "ok", "")
    far = ComponentScore("payl", "q", 20.0, 40.0, "ok", "")
    assert component_exceeds_review(mild, settings) is False
    assert component_exceeds_review(far, settings) is True


def test_settings_roundtrip_rejects_another_mode(tmp_path):
    from anomaly_guard.core.decision import AnomalyGuard

    guard = AnomalyGuard(mode="rules_plus_stats")
    guard.settings.alert_threshold_percentile = 100.0
    path = tmp_path / "baseline.settings.json"
    guard.save_settings(path)
    other = AnomalyGuard(mode="payl_only")
    assert other.load_settings(path) is False
    assert other.settings.alert_threshold_percentile == 99.0
    same = AnomalyGuard(mode="rules_plus_stats")
    assert same.load_settings(path) is True
    assert same.settings.alert_threshold_percentile == 100.0


def test_isolation_forest_persists_across_load(tmp_path):
    guard = AnomalyGuard(mode="isolation_forest")
    train = _train_rows(40)
    guard.fit(train)
    guard.calibrate(train[:20], threshold_records=train[:20])
    assert guard._isolation is not None
    path = tmp_path / "baseline.json"
    guard.save_baseline(path)
    guard.save_settings(path.with_suffix(".settings.json"))
    loaded = AnomalyGuard(mode="isolation_forest")
    loaded.load_baseline(path)
    loaded.load_settings(path.with_suffix(".settings.json"))
    assert loaded._isolation is not None
    fresh = guard.score(train[0]).isolation_score
    again = loaded.score(train[0]).isolation_score
    assert fresh is not None and again is not None
    assert abs(fresh - again) < 1e-9


def test_rules_only_alerts_on_signature_sqli():
    guard = AnomalyGuard(mode="rules_only")
    guard.fit(_train_rows())
    result = guard.score(_sqli_login())
    assert result.decision in ("alert", "review")
