"""Tests for bounded parsing."""

from anomaly_guard.core.parser import parse_request
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
