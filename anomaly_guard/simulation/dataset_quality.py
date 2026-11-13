"""Dataset integrity checks for the traffic simulator."""

from __future__ import annotations

from anomaly_guard.core.parser import parse_request
from anomaly_guard.core.rules import evaluate_rules
from anomaly_guard.core.schema import GroundTruthMeta, RequestRecord, load_app_config
from anomaly_guard.simulation.payload_family import content_family_from_meta, content_key_parts


def content_key(rec: RequestRecord) -> tuple[str, str, str, str]:
    return content_key_parts(rec.method, rec.route, rec.raw_query, rec.body)


def schema_violations(
    records: list[RequestRecord],
    meta: list[GroundTruthMeta],
) -> list[dict[str, str]]:
    """Rows that trigger schema or size rules.

    Benign-invalid rows are labeled invalid on purpose. Attack rows may exceed
    field limits on purpose. Ordinary, difficult, and shifted traffic must not.
    """
    config = load_app_config()
    meta_by_id = {m.request_id: m for m in meta}
    violations: list[dict[str, str]] = []
    for rec in records:
        m = meta_by_id.get(rec.request_id)
        if m and m.scenario_group in {"benign_invalid", "malicious_intent", "hard_malicious_intent"}:
            continue
        parsed = parse_request(rec, config)
        rules = evaluate_rules(parsed, config)
        schema_hits = [r for r in rules if r.kind in {"schema", "size_limit"}]
        if schema_hits:
            violations.append(
                {
                    "request_id": rec.request_id,
                    "route": rec.route,
                    "rules": ";".join(r.rule_id for r in schema_hits),
                }
            )
    return violations


def payload_overlap_report(
    records: list[RequestRecord],
    meta: list[GroundTruthMeta],
) -> dict[str, object]:
    """Exact-string overlap versus content-family overlap on ordinary test traffic.

    Content families are the template topic slots written into ``cf:``. Holdout
    families never enter training. In-distribution ordinary test rows reuse a
    family that training also saw, with different wording.
    """
    from anomaly_guard.evaluation.splits import partition_records, training_normal_records

    meta_by_id = {m.request_id: m for m in meta}
    train = training_normal_records(records, meta)
    test_recs, _test_meta = partition_records(records, meta)["test"]
    train_families = {
        content_family_from_meta(meta_by_id[r.request_id])
        for r in train
        if content_family_from_meta(meta_by_id[r.request_id])
    }
    ordinary_test = [
        (r, meta_by_id[r.request_id])
        for r in test_recs
        if meta_by_id[r.request_id].scenario_group == "legitimate"
    ]
    in_distribution = []
    unseen = []
    for rec, row in ordinary_test:
        family = content_family_from_meta(row)
        if family and family in train_families and not is_holdout_row(row):
            in_distribution.append(rec.request_id)
        else:
            unseen.append(rec.request_id)
    train_text = {content_key(r) for r in train}
    exact_overlap = sum(1 for rec, _row in ordinary_test if content_key(rec) in train_text)
    n = len(ordinary_test)
    return {
        "ordinary_test_count": n,
        "train_ordinary_count": len(train),
        "exact_content_overlap_count": exact_overlap,
        "exact_content_overlap_rate": exact_overlap / n if n else 0.0,
        "in_distribution_ordinary_test_count": len(in_distribution),
        "unseen_content_ordinary_test_count": len(unseen),
    }


def is_holdout_row(row: GroundTruthMeta) -> bool:
    from anomaly_guard.simulation.payload_family import is_holdout_content

    return is_holdout_content(row)
