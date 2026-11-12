"""Hosted demo store and mode detection."""

from anomaly_guard.hosted import is_cursor_hosted
from anomaly_guard.storage import MemoryEventStore
from anomaly_guard.core.decision import ScoreResult
from anomaly_guard.core.schema import GroundTruthMeta, RequestRecord


def _rec() -> RequestRecord:
    return RequestRecord(
        request_id="abc",
        timestamp="t",
        client_id="c",
        method="GET",
        route="/search",
        raw_query="q=hi",
        headers={},
        body="",
    )


def _score() -> ScoreResult:
    return ScoreResult(
        decision="normal",
        request_id="abc",
        method="GET",
        route="/search",
        triggered_rules=[],
        statistical=None,
        component_scores={},
        explanations=[],
        baseline_version="v0",
        processing_time_ms=1.0,
        mode="rules_plus_stats",
    )


def test_memory_event_store_roundtrip():
    store = MemoryEventStore()
    store.upsert_event(
        _rec(),
        _score(),
        GroundTruthMeta.from_dict({
            "request_id": "abc",
            "scenario_id": "legit-1",
            "split": "test",
            "scenario_group": "legitimate",
            "attack_family": None,
            "attack_slice": None,
        }),
    )
    rows = store.search(decision="normal", route="/search", limit=10)
    assert len(rows) == 1
    assert store.get("abc") is not None
    store.clear()
    assert store.search() == []


def test_is_cursor_hosted_env(monkeypatch):
    monkeypatch.delenv("CURSOR_APP_ID", raising=False)
    assert is_cursor_hosted() is False
    monkeypatch.setenv("CURSOR_APP_ID", "app_test")
    assert is_cursor_hosted() is True
