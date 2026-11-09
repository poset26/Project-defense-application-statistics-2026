"""Regression: rules must not saturate on rule-evasive slice."""

from pathlib import Path

from anomaly_guard.core.decision import AnomalyGuard
from anomaly_guard.core.schema import GroundTruthMeta, RequestRecord, read_jsonl
from anomaly_guard.evaluation.metrics import alert_predicted, slice_attack_recall
from anomaly_guard.evaluation.splits import assign_splits, partition_records, training_normal_records, validation_legitimate_records
from anomaly_guard.simulation.generator import DEFAULT_PLAN, generate_dataset, save_dataset


def test_rules_only_not_full_recall_on_rule_evasive(tmp_path: Path):
    plan = DEFAULT_PLAN.__class__(
        legitimate=200,
        difficult_legitimate=40,
        benign_invalid=30,
        malicious_intent=80,
        hard_malicious_intent=20,
        distribution_shift=30,
        seed=7,
    )
    records, meta = generate_dataset(plan)
    meta = assign_splits(meta, records)
    save_dataset(tmp_path, records, meta)

    parts = partition_records(records, meta)
    test_recs, test_meta = parts["test"]
    train = training_normal_records(records, meta)
    val = validation_legitimate_records(records, meta)

    guard = AnomalyGuard(mode="rules_only")
    guard.fit(train)
    guard.calibrate(val)

    pred = [alert_predicted(guard.score(r).decision) for r in test_recs]
    evasive = slice_attack_recall(test_meta, pred, "rule_evasive")
    assert evasive is not None and evasive["support"] >= 5
    assert evasive["recall"] < 1.0, "rules_only must not catch every rule-evasive attack"
