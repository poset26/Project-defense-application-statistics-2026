"""OLS residual scoring must be stable under training row order."""

from anomaly_guard.core.advanced_stats import OLS_RESIDUAL_STD_FLOOR, score_ols_residual
from anomaly_guard.core.decision import AnomalyGuard
from anomaly_guard.core.schema import RequestRecord


def _search(q: str, rid: str = "r") -> RequestRecord:
    return RequestRecord(
        request_id=rid,
        timestamp="2026-01-01T00:00:00+00:00",
        client_id="c1",
        method="GET",
        route="/search",
        raw_query=f"q={q.replace(' ', '+')}&page=1",
        headers={},
        body="",
    )


def _regression_scores(order: list[RequestRecord], probes: list[RequestRecord]) -> list[float]:
    guard = AnomalyGuard(mode="regression_only")
    guard.fit(order)
    guard.calibrate(order, threshold_records=order)
    scores = []
    for probe in probes:
        result = guard.score(probe)
        assert result.statistical is not None
        raw = [c.raw_score for c in result.statistical.components if c.component == "regression"]
        scores.append(raw[0] if raw else 0.0)
    return scores


def test_regression_scores_match_after_reversing_training_order():
    constant = [_search("same words here", f"c{i}") for i in range(10)]
    near = [_search(f"routing notes item {i % 3}", f"n{i}") for i in range(10)]
    train = constant + near
    probes = [
        _search("same words here", "p-const"),
        _search("routing notes item 1", "p-near"),
        _search("!" * 80, "p-far"),
    ]
    forward = _regression_scores(train, probes)
    backward = _regression_scores(list(reversed(train)), probes)
    for left, right in zip(forward, backward):
        assert abs(left - right) < 1e-6
    assert forward[0] < 5.0
    assert forward[2] > forward[0]


def test_near_zero_residual_uses_floor():
    from anomaly_guard.core.features import RequestFeatures

    class _Field:
        char_length = 12
        entropy = 1.0
        distinct_char_fraction = 0.2
        category_fractions = {"digit": 0.0, "other": 0.0}

    class _Feats:
        field_features = {"q": _Field()}

    class _Profile:
        ols_coef = [2.5649493574615367, 0.0, 0.0, 0.0, 0.0]
        ols_residual_std = 1e-16

    score, _detail = score_ols_residual(_Field(), _Profile())
    assert score is not None
    assert score < 50
    assert OLS_RESIDUAL_STD_FLOOR >= 1e-4
    del RequestFeatures, _Feats
