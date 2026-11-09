"""Advanced statistical components (OLS regression, PAYL, n-gram)."""

import math

import numpy as np

from anomaly_guard.core.advanced_stats import (
    FieldExtendedProfile,
    benjamini_hochberg,
    fit_endpoint_extended,
    score_ols_residual,
)
from anomaly_guard.core.features import extract_features
from anomaly_guard.core.parser import parse_request
from anomaly_guard.core.schema import RequestRecord, load_app_config


def _login(username: str, rid: str = "1") -> RequestRecord:
    return RequestRecord(
        request_id=rid,
        timestamp="t",
        client_id="c",
        method="POST",
        route="/login",
        raw_query="",
        headers={"content-type": "application/x-www-form-urlencoded"},
        body=f"username={username}&password=secret12",
    )


def test_ols_fits_and_flags_long_outlier():
    config = load_app_config()
    train = [_login(f"user{i}", str(i)) for i in range(40)]
    items = []
    vectors = []
    for rec in train:
        parsed = parse_request(rec, config)
        feats = extract_features(parsed)
        items.append((feats, parsed))
        vectors.append(feats.to_vector(["username", "password"]))
    ext = fit_endpoint_extended(items, ["username", "password"], vectors)
    coef = ext.fields["username"].ols_coef
    assert coef is not None
    assert len(coef) == 5

    outlier_rec = _login("a" * 200)
    parsed = parse_request(outlier_rec, config)
    ff = extract_features(parsed).field_features["username"]
    raw, detail = score_ols_residual(ff, ext.fields["username"])
    assert raw is not None
    assert raw > 0.5
    assert "OLS" in detail


def test_regression_only_mode_uses_component():
    from anomaly_guard.core.decision import AnomalyGuard

    train = [_login(f"user{i}", str(i)) for i in range(50)]
    guard = AnomalyGuard(mode="regression_only")
    guard.fit(train)
    guard.calibrate(train[:25])
    res = guard.score(_login("x" * 180))
    comps = {c.component for c in (res.statistical.components if res.statistical else [])}
    assert "regression" in comps


def test_length_prediction_interval_flags_long_field():
    from anomaly_guard.core.decision import AnomalyGuard

    train = []
    for i in range(40):
        train.append(
            RequestRecord(
                request_id=f"s{i}",
                timestamp="t",
                client_id="c",
                method="GET",
                route="/search",
                raw_query=f"q={'ab' * (3 + (i % 12))}&page=1",
                headers={},
                body="",
            )
        )
    guard = AnomalyGuard(mode="ci_only")
    guard.fit(train)
    guard.calibrate(train[:20])
    long = RequestRecord(
        request_id="long",
        timestamp="t",
        client_id="c",
        method="GET",
        route="/search",
        raw_query="q=" + ("Z" * 400),
        headers={},
        body="",
    )
    res = guard.score(long)
    comps = res.statistical.components if res.statistical else []
    ci = [c for c in comps if c.component == "length_ci" and c.field == "q"]
    assert ci
    assert ci[0].raw_score > 0
    assert "prediction interval" in ci[0].detail


def test_mean_aggregate_is_not_the_max():
    from anomaly_guard.core.decision import aggregate_component_percentiles
    from anomaly_guard.core.statistics import ComponentScore

    comps = [
        ComponentScore("payl", "q", 3.0, 99.0, "ok", ""),
        ComponentScore("composition", "q", 0.1, 40.0, "ok", ""),
        ComponentScore("length_ci", "q", 0.0, 10.0, "ok", ""),
    ]
    assert aggregate_component_percentiles(comps, how="max") == 99.0
    mean = aggregate_component_percentiles(comps, how="mean")
    assert aggregate_component_percentiles(comps, how="second") == 40.0
    assert mean is not None
    assert abs(mean - (99 + 40 + 10) / 3) < 1e-9


def test_rare_prior_posterior_is_far_below_file_precision():
    from anomaly_guard.evaluation.metrics import posterior_attack_given_alert

    posterior = posterior_attack_given_alert(0.33, 0.039, 0.001)
    assert posterior is not None
    assert 0.007 < posterior < 0.01
    assert posterior_attack_given_alert(0.24, 0.0, 0.001) == 1.0


def test_entropy_only_flags_a_repeated_character_string():
    from anomaly_guard.core.decision import AnomalyGuard

    train = []
    for i in range(40):
        phrase = f"ordinary search phrase number {i} about routing"
        train.append(
            RequestRecord(
                request_id=f"e{i}",
                timestamp="t",
                client_id="c",
                method="GET",
                route="/search",
                raw_query=f"q={phrase}&page=1",
                headers={},
                body="",
            )
        )
    guard = AnomalyGuard(mode="entropy_only")
    guard.fit(train)
    guard.calibrate(train[:20])
    odd = RequestRecord(
        request_id="rep",
        timestamp="t",
        client_id="c",
        method="GET",
        route="/search",
        raw_query="q=" + ("A" * 80),
        headers={},
        body="",
    )
    res = guard.score(odd)
    ents = [c for c in (res.statistical.components if res.statistical else []) if c.component == "entropy"]
    assert ents
    assert ents[0].raw_score > 0.5


def test_payl_cutoff_is_a_raw_distance():
    from anomaly_guard.core.decision import AnomalyGuard

    train = []
    for i in range(30):
        train.append(
            RequestRecord(
                request_id=f"p{i}",
                timestamp="t",
                client_id="c",
                method="GET",
                route="/search",
                raw_query=f"q=query about item {i % 9}&page=1",
                headers={},
                body="",
            )
        )
    guard = AnomalyGuard(mode="payl_only")
    guard.fit(train)
    guard.calibrate(train[:15])
    assert guard.settings.threshold_scale == "raw_distance"


def test_operating_point_prefers_budget():
    from anomaly_guard.evaluation.experiments import operating_point_at_fpr

    rows = [
        {"threshold_percentile": 90, "recall_malicious": 0.4, "fpr_legitimate": 0.02},
        {"threshold_percentile": 95, "recall_malicious": 0.25, "fpr_legitimate": 0.009},
        {"threshold_percentile": 99, "recall_malicious": 0.1, "fpr_legitimate": 0.0},
    ]
    point = operating_point_at_fpr(rows, budget=0.01)
    assert point is not None
    assert point["recall_malicious"] == 0.25
    assert point["selection"] == "highest_recall_with_test_fpr_at_most_budget"


def test_benjamini_hochberg_rejects_low_p_values():
    pvals = [0.001, 0.04, 0.2, 0.5]
    rej = benjamini_hochberg(pvals, q=0.05)
    assert rej[0] is True
    assert sum(rej) >= 1
