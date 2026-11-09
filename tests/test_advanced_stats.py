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


def test_benjamini_hochberg_rejects_low_p_values():
    pvals = [0.001, 0.04, 0.2, 0.5]
    rej = benjamini_hochberg(pvals, q=0.05)
    assert rej[0] is True
    assert sum(rej) >= 1
