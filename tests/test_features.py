"""Feature extraction tests."""

from anomaly_guard.core.features import CATEGORY_BINS, extract_features
from anomaly_guard.core.parser import parse_request
from anomaly_guard.core.schema import RequestRecord, load_app_config


def test_feature_vector_length():
    config = load_app_config()
    rec = RequestRecord(
        request_id="1",
        timestamp="t",
        client_id="c",
        method="GET",
        route="/search",
        raw_query="q=test",
        headers={},
        body="",
    )
    parsed = parse_request(rec, config)
    feats = extract_features(parsed)
    vec = feats.to_vector()
    names = feats.feature_names()
    assert len(vec) == len(names)
    assert "q" in feats.field_features
    fr = feats.field_features["q"].category_fractions
    assert abs(sum(fr.values()) - 1.0) < 0.01
    assert set(fr.keys()) == set(CATEGORY_BINS)
