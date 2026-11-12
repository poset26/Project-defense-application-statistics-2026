"""Tests for bounded parsing."""

from anomaly_guard.core.parser import parse_request
from anomaly_guard.core.rules import evaluate_rules
from anomaly_guard.core.schema import RequestRecord, load_app_config


def test_search_parse_basic():
    config = load_app_config()
    rec = RequestRecord(
        request_id="1",
        timestamp="2026-01-01T00:00:00+00:00",
        client_id="c1",
        method="GET",
        route="/search",
        raw_query="q=hello&page=2",
        headers={},
        body="",
    )
    parsed = parse_request(rec, config)
    assert parsed.field_text("q") == "hello"
    assert parsed.field_text("page") == "2"


def test_query_name_percent_decode_maps_to_contract_field():
    config = load_app_config()
    rec = RequestRecord(
        request_id="qname",
        timestamp="2026-01-01T00:00:00+00:00",
        client_id="c1",
        method="GET",
        route="/search",
        raw_query="%71=hello",
        headers={},
        body="",
    )
    parsed = parse_request(rec, config)
    rules = evaluate_rules(parsed, config)
    assert parsed.field_text("q") == "hello"
    assert "%71" not in parsed.flags.unexpected_fields
    assert parsed.flags.unexpected_fields == []
    assert not any(r.rule_id == "UNEXPECTED_FIELD" for r in rules)
    assert not any(r.rule_id == "MISSING_FIELD" and r.field == "q" for r in rules)


def test_percent_25_is_a_literal_percent_and_truncated_sequence_is_malformed():
    config = load_app_config()
    literal = RequestRecord(
        request_id="pct",
        timestamp="2026-01-01T00:00:00+00:00",
        client_id="c1",
        method="GET",
        route="/search",
        raw_query="q=50%25",
        headers={},
        body="",
    )
    parsed = parse_request(literal, config)
    assert parsed.field_text("q") == "50%"
    assert parsed.flags.malformed_encoding is False

    broken = RequestRecord(
        request_id="trunc",
        timestamp="2026-01-01T00:00:00+00:00",
        client_id="c1",
        method="GET",
        route="/search",
        raw_query="q=%A",
        headers={},
        body="",
    )
    parsed_bad = parse_request(broken, config)
    assert parsed_bad.flags.malformed_encoding is True
    assert parsed_bad.field_text("q") == "%A"


def test_single_pass_decode_no_double():
    config = load_app_config()
    rec = RequestRecord(
        request_id="2",
        timestamp="2026-01-01T00:00:00+00:00",
        client_id="c1",
        method="GET",
        route="/search",
        raw_query="q=%252F",
        headers={},
        body="",
    )
    parsed = parse_request(rec, config)
    assert parsed.field_text("q") == "%2F"
