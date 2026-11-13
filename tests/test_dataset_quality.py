"""Simulator quality: route contracts and content-family splits."""

from anomaly_guard.evaluation.splits import assign_splits
from anomaly_guard.simulation.dataset_quality import payload_overlap_report, schema_violations
from anomaly_guard.simulation.generator import DEFAULT_PLAN, generate_dataset


def test_full_generated_dataset_satisfies_route_contracts():
    records, meta = generate_dataset(DEFAULT_PLAN)
    violations = schema_violations(records, meta)
    assert violations == [], violations[:5]
    assert not any("ctx=" in rec.raw_query for rec in records)


def test_ordinary_test_splits_in_distribution_and_unseen_content():
    plan = DEFAULT_PLAN.__class__(
        legitimate=240,
        difficult_legitimate=20,
        benign_invalid=0,
        malicious_intent=24,
        hard_malicious_intent=8,
        distribution_shift=8,
        seed=3,
    )
    records, meta = generate_dataset(plan)
    meta = assign_splits(meta, records)
    report = payload_overlap_report(records, meta)
    assert report["ordinary_test_count"] > 0
    assert report["exact_content_overlap_count"] == 0
    assert report["in_distribution_ordinary_test_count"] > 0
    assert report["unseen_content_ordinary_test_count"] > 0
