"""Train/validation/test splits with group-aware assignment."""

from __future__ import annotations

import hashlib
from collections import defaultdict

from anomaly_guard.core.schema import GroundTruthMeta, RequestRecord, ScenarioGroup

TRAIN_GROUPS: set[ScenarioGroup] = {"legitimate"}
VALIDATION_GROUPS: set[ScenarioGroup] = {"legitimate", "difficult_legitimate", "distribution_shift"}
TEST_GROUPS: set[ScenarioGroup] = set(
    [
        "legitimate",
        "difficult_legitimate",
        "benign_invalid",
        "malicious_intent",
        "hard_malicious_intent",
        "distribution_shift",
    ]
)


def _bucket(key: str, train_ratio: float, val_ratio: float) -> str:
    h = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 10000
    train_cut = int(train_ratio * 10000)
    val_cut = train_cut + int(val_ratio * 10000)
    if h < train_cut:
        return "train"
    if h < val_cut:
        return "validation"
    return "test"


def assign_splits(
    meta: list[GroundTruthMeta],
    train_ratio: float = 0.6,
    val_ratio: float = 0.15,
) -> list[GroundTruthMeta]:
    """Group-aware: template_id + scenario_id keep variants in one partition."""
    out: list[GroundTruthMeta] = []
    for m in meta:
        group_key = f"{m.scenario_id}:{m.template_id or 'na'}"
        split = _bucket(group_key, train_ratio, val_ratio)
        if m.scenario_group in TRAIN_GROUPS and split == "test" and m.scenario_group == "legitimate":
            split = _bucket(group_key + ":leg", 0.7, 0.15)
        updated = GroundTruthMeta(
            request_id=m.request_id,
            scenario_group=m.scenario_group,
            scenario_id=m.scenario_id,
            attack_family=m.attack_family,
            template_id=m.template_id,
            split=split,  # type: ignore
            notes=m.notes,
        )
        out.append(updated)
    return out


def partition_records(
    records: list[RequestRecord],
    meta: list[GroundTruthMeta],
) -> dict[str, tuple[list[RequestRecord], list[GroundTruthMeta]]]:
    meta_by_id = {m.request_id: m for m in meta}
    parts: dict[str, tuple[list[RequestRecord], list[GroundTruthMeta]]] = {
        "train": ([], []),
        "validation": ([], []),
        "test": ([], []),
    }
    for rec in records:
        m = meta_by_id.get(rec.request_id)
        if not m or not m.split:
            continue
        if m.scenario_group == "legitimate" and m.split == "train":
            parts["train"][0].append(rec)
            parts["train"][1].append(m)
        elif m.split == "validation" and m.scenario_group in ("legitimate", "difficult_legitimate", "distribution_shift"):
            parts["validation"][0].append(rec)
            parts["validation"][1].append(m)
        elif m.split == "test":
            parts["test"][0].append(rec)
            parts["test"][1].append(m)
    return parts


def training_normal_records(
    records: list[RequestRecord], meta: list[GroundTruthMeta]
) -> list[RequestRecord]:
    meta_by_id = {m.request_id: m for m in meta}
    return [
        r
        for r in records
        if (m := meta_by_id.get(r.request_id))
        and m.split == "train"
        and m.scenario_group == "legitimate"
    ]


def validation_legitimate_records(
    records: list[RequestRecord], meta: list[GroundTruthMeta]
) -> list[RequestRecord]:
    meta_by_id = {m.request_id: m for m in meta}
    return [
        r
        for r in records
        if (m := meta_by_id.get(r.request_id))
        and m.split == "validation"
        and m.scenario_group in ("legitimate", "difficult_legitimate", "distribution_shift")
    ]


def held_out_family_meta(meta: list[GroundTruthMeta], family: str = "traversal") -> list[GroundTruthMeta]:
    """Mark test records whose attack family was absent from train/validation development."""
    dev_families = {
        m.attack_family
        for m in meta
        if m.split in ("train", "validation") and m.attack_family
    }
    if family in dev_families:
        return []
    return [m for m in meta if m.split == "test" and m.attack_family == family]
