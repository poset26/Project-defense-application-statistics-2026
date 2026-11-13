"""End-to-end smoke tests."""

from pathlib import Path

from anomaly_guard.evaluation.splits import (
    assign_splits,
    partition_records,
    training_normal_records,
)
from anomaly_guard.simulation.generator import DEFAULT_PLAN, generate_dataset, save_dataset


def test_generate_and_split(tmp_path: Path):
    plan = DEFAULT_PLAN.__class__(
        legitimate=50,
        difficult_legitimate=10,
        benign_invalid=0,
        malicious_intent=10,
        hard_malicious_intent=5,
        distribution_shift=5,
        seed=1,
    )
    records, meta = generate_dataset(plan)
    meta = assign_splits(meta, records)
    save_dataset(tmp_path, records, meta)
    train = training_normal_records(records, meta)
    assert len(train) > 0
    from anomaly_guard.evaluation.splits import validation_legitimate_records

    val = validation_legitimate_records(records, meta)
    train_routes = {f"{r.method} {r.route}" for r in train}
    val_routes = {f"{r.method} {r.route}" for r in val}
    assert train_routes & val_routes, "train and validation must share at least one route"
    assert (tmp_path / "requests.jsonl").exists()


def test_seed_replays_splits_and_ordinary_test_text_is_new():
    plan = DEFAULT_PLAN.__class__(
        legitimate=90,
        difficult_legitimate=12,
        benign_invalid=0,
        malicious_intent=24,
        hard_malicious_intent=8,
        distribution_shift=8,
        seed=3,
    )
    first_records, first_meta = generate_dataset(plan)
    second_records, second_meta = generate_dataset(plan)
    assert [r.request_id for r in first_records] == [r.request_id for r in second_records]
    assert [(r.raw_query, r.body) for r in first_records] == [
        (r.raw_query, r.body) for r in second_records
    ]
    assert all(r.request_id.startswith("req-") for r in first_records)
    contents = {(r.method, r.route, r.raw_query, r.body) for r in first_records}
    assert len(contents) == len(first_records)
    split_a = assign_splits(first_meta, first_records)
    split_b = assign_splits(second_meta, second_records)
    assert [(m.request_id, m.split) for m in split_a] == [(m.request_id, m.split) for m in split_b]
    train = training_normal_records(first_records, split_a)
    test_recs, test_meta = partition_records(first_records, split_a)["test"]
    train_text = {(r.method, r.route, r.raw_query, r.body) for r in train}
    ordinary = [r for r, m in zip(test_recs, test_meta) if m.scenario_group == "legitimate"]
    assert ordinary
    assert all((r.method, r.route, r.raw_query, r.body) not in train_text for r in ordinary)
    from anomaly_guard.simulation.dataset_quality import payload_overlap_report

    report = payload_overlap_report(first_records, split_a)
    assert report["exact_content_overlap_count"] == 0
    assert report["in_distribution_ordinary_test_count"] > 0
    assert report["unseen_content_ordinary_test_count"] > 0
