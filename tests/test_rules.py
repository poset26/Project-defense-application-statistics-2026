"""Rule engine tests."""

from anomaly_guard.core.parser import parse_request
from anomaly_guard.core.rules import evaluate_rules
from anomaly_guard.core.schema import RequestRecord, load_app_config


def test_apostrophe_not_high_severity_alone():
    config = load_app_config()
    rec = RequestRecord(
        request_id="1",
        timestamp="t",
        client_id="c",
        method="GET",
        route="/search",
        raw_query="q=O'Brien",
        headers={},
        body="",
    )
    parsed = parse_request(rec, config)
    rules = evaluate_rules(parsed, config)
    assert not any(r.rule_id == "SCRIPT_TAG" for r in rules)
    assert not any(r.severity == "high" for r in rules)


def test_sqli_pattern_triggers():
    config = load_app_config()
    rec = RequestRecord(
        request_id="2",
        timestamp="t",
        client_id="c",
        method="POST",
        route="/login",
        raw_query="",
        headers={"content-type": "application/x-www-form-urlencoded"},
        body="username=admin' OR 1=1--&password=x",
    )
    parsed = parse_request(rec, config)
    rules = evaluate_rules(parsed, config)
    assert any(r.kind == "suspicious_pattern" for r in rules)
