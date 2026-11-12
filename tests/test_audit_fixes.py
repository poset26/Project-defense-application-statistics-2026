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
