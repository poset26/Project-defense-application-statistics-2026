"""Baseline and zero-MAD handling."""

from anomaly_guard.core.schema import RequestRecord
from anomaly_guard.core.statistics import LengthProfile, StatisticalDetector, _robust_length_score


def test_robust_z_constant_is_in_the_numerator():
    from anomaly_guard.core.statistics import MAD_SCALE, _robust_z

    z, detail = _robust_z(10.0 + 2.0 / MAD_SCALE, 10.0, 2.0)
    assert abs(z - 1.0) < 1e-9
    assert "0.6745" in detail


def test_zero_mad_tolerance():
    lp = LengthProfile(median=10.0, mad=0.0, tolerance=1.0)
    score, _ = _robust_length_score(10.5, lp)
    assert score == 0.0
    score2, _ = _robust_length_score(15.0, lp)
    assert score2 > 0


def test_fit_and_score():
    detector = StatisticalDetector()
    records = []
    for i in range(20):
        records.append(
            RequestRecord(
                request_id=str(i),
                timestamp="t",
                client_id="c",
                method="GET",
                route="/search",
                raw_query=f"q=query{i % 5}",
                headers={},
                body="",
            )
        )
    detector.fit(records)
    detector.calibrate_percentiles(records[:10])
    res = detector.score(records[0])
    assert res.baseline_version != "none"
