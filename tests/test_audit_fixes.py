"""Fixes from the external audit: composition bins, JSON text, cutoff ties, length center."""

import math

import numpy as np

from anomaly_guard.core.decision import _cutoff_with_rate_cap
from anomaly_guard.core.features import CATEGORY_BINS
from anomaly_guard.core.parser import parse_request
from anomaly_guard.core.schema import RequestRecord, load_app_config
from anomaly_guard.core.statistics import (
    CompositionProfile,
    LengthProfile,
    _composition_score,
    _proportion_ci_score,
    _robust_length_score,
)


def test_json_body_is_not_form_decoded():
    rec = RequestRecord(
        request_id="j",
        timestamp="t",
        client_id="c",
        method="POST",
        route="/comment",
        raw_query="",
        headers={"content-type": "application/json"},
        body='{"text": "C++ and 50%25", "author": "ada"}',
    )
    parsed = parse_request(rec, load_app_config())
    assert parsed.field_text("text") == "C++ and 50%25"


def test_composition_sees_characters_absent_from_training():
    profile = CompositionProfile(
        category_probs={"letter": 0.9, "whitespace": 0.1},
        merged_categories=["letter", "whitespace"],
        training_chars=1000,
    )
    letters = "kubernetes networking notes"
    quotes = "'" * len(letters)
    script = "<script>alert(1)</script>"
    letter_score, _, _, _ = _composition_score(letters, profile)
    quote_score, quote_status, _, _ = _composition_score(quotes, profile)
    script_score, _, _, _ = _composition_score(script, profile)
    assert letter_score is not None and quote_score is not None and script_score is not None
    assert math.isfinite(quote_score) and quote_status == "ok"
    assert quote_score > letter_score * 10
    assert script_score > letter_score * 5
    assert "quote" in CATEGORY_BINS


def test_proportion_interval_uses_request_length_not_training_pool():
    profile = CompositionProfile(
        category_probs={"letter": 0.8, "whitespace": 0.2},
        merged_categories=["letter", "whitespace"],
        training_chars=141_000,
    )
    text = "kubernetes networking"
    score, detail = _proportion_ci_score(text, profile)
    assert score is not None
    assert score < 5.0
    assert "Wilson" in detail


def test_length_robust_z_uses_log_median_not_mean():
    profile = LengthProfile(
        median=20.0,
        mad=2.0,
        tolerance=1.0,
        log_mean=5.0,
        log_median=3.0,
        log_mad=0.2,
        log_std=0.4,
        log_n=50,
    )
    raw, detail = _robust_length_score(20.0, profile)
    expected = 0.6745 * abs(math.log1p(20.0) - 3.0) / 0.2
    assert abs(raw - expected) < 1e-6
    assert "median 3" in detail


def test_cutoff_ties_do_not_exceed_one_percent():
    scores = np.array([1.0] * 50 + [9.0] * 50, dtype=float)
    cutoff = _cutoff_with_rate_cap(np.sort(scores), 0.01)
    flagged = int(np.sum(scores > cutoff))
    assert flagged / len(scores) <= 0.01


def test_percentile_100_is_only_for_scores_above_the_pool():
    from anomaly_guard.core.schema import load_app_config
    from anomaly_guard.core.statistics import StatisticalDetector

    det = StatisticalDetector(load_app_config())
    det._calibration_pools = {"entropy": [1.0, 2.0, 2.0]}
    tied = det._empirical_percentile(2.0, "entropy")
    above = det._empirical_percentile(2.5, "entropy")
    assert tied is not None and tied < 100
    assert above == 100.0


def test_cutoff_equality_does_not_alert_or_review():
    from anomaly_guard.core.decision import DecisionSettings, _decision_from_statistics, clears_cutoff
    from anomaly_guard.core.statistics import StatisticalResult

    assert clears_cutoff(99.0, 99.0) is False
    assert clears_cutoff(99.0, 95.0) is True
    settings = DecisionSettings(alert_threshold_percentile=99.0, review_threshold_percentile=95.0)
    exact_alert = StatisticalResult(
        baseline_version="t",
        components=[],
        aggregate_percentile=99.0,
        insufficient_baseline=False,
        processing_time_ms=0.0,
    )
    exact_review = StatisticalResult(
        baseline_version="t",
        components=[],
        aggregate_percentile=95.0,
        insufficient_baseline=False,
        processing_time_ms=0.0,
    )
    assert _decision_from_statistics(exact_alert, settings) == "review"
    assert _decision_from_statistics(exact_review, settings) == "normal"


def test_entropy_only_can_alert_above_a_tied_percentile():
    from anomaly_guard.core.decision import AnomalyGuard, clears_cutoff
    from anomaly_guard.core.schema import RequestRecord

    rows = []
    for i in range(40):
        rows.append(
            RequestRecord(
                request_id=f"t{i}",
                timestamp="t",
                client_id="c",
                method="GET",
                route="/search",
                raw_query=f"q=ordinary+words+about+routing+{i}&page=1",
                headers={},
                body="",
            )
        )
    guard = AnomalyGuard(mode="entropy_only")
    guard.fit(rows)
    guard.calibrate(rows[:25], threshold_records=rows[:25])
    attack = RequestRecord(
        request_id="atk",
        timestamp="t",
        client_id="c",
        method="GET",
        route="/search",
        raw_query="q=" + ("!" * 160),
        headers={},
        body="",
    )
    result = guard.score(attack)
    assert guard.settings.alert_threshold_percentile < 100
    assert result.statistical is not None
    assert result.statistical.aggregate_percentile == 100
    assert clears_cutoff(
        result.statistical.aggregate_percentile,
        guard.settings.alert_threshold_percentile,
    )
    assert result.decision == "alert"
    flagged = sum(guard.score(r).decision == "alert" for r in rows[:25])
    assert flagged / 25 <= guard.settings.max_false_positive_rate_target


def test_composition_worked_example_matches_scorer():
    import math

    import numpy as np

    from anomaly_guard.core.features import CATEGORY_BINS
    from anomaly_guard.core.stat_lite import chisquare
    from anomaly_guard.core.statistics import CompositionProfile, _composition_score

    text = "abcdefghij" * 3
    n = len(text)
    probs = {cat: 0.0 for cat in CATEGORY_BINS}
    probs["letter"] = 1.0
    profile = CompositionProfile(
        category_probs=probs,
        merged_categories=["letter"],
        training_chars=1000,
        min_total_count=8,
    )
    score, status, _detail, _p = _composition_score(text, profile)
    observed = [float(n if cat == "letter" else 0) for cat in CATEGORY_BINS]
    expected = [probs[cat] * n + 0.5 for cat in CATEGORY_BINS]
    obs_arr = np.array(observed)
    exp_arr = np.array(expected)
    exp_arr = np.maximum(exp_arr * (obs_arr.sum() / exp_arr.sum()), 1e-6)
    stat, _ = chisquare(f_obs=obs_arr, f_exp=exp_arr)
    assert status == "ok"
    assert score is not None
    assert math.isclose(score, float(stat) / n, rel_tol=1e-9)

