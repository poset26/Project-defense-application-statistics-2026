"""End-to-end smoke tests."""

from pathlib import Path

from anomaly_guard.evaluation.splits import assign_splits, training_normal_records
from anomaly_guard.simulation.generator import DEFAULT_PLAN, generate_dataset, save_dataset


def test_generate_and_split(tmp_path: Path):
    plan = DEFAULT_PLAN.__class__(
        legitimate=50,
        difficult_legitimate=10,
        benign_invalid=10,
        malicious_intent=10,
        hard_malicious_intent=5,
        distribution_shift=5,
        seed=1,
    )
    records, meta = generate_dataset(plan)
    meta = assign_splits(meta)
    save_dataset(tmp_path, records, meta)
    train = training_normal_records(records, meta)
    assert len(train) > 0
    assert (tmp_path / "requests.jsonl").exists()
