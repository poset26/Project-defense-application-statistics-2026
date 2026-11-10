"""Train/validation/test splits with group-aware assignment."""

from __future__ import annotations

import hashlib
from collections import defaultdict

from anomaly_guard.core.schema import GroundTruthMeta, RequestRecord, ScenarioGroup

TRAIN_GROUPS: set[ScenarioGroup] = {"legitimate"}
VALIDATION_GROUPS: set[ScenarioGroup] = {"legitimate", "difficult_legitimate", "distribution_shift"}

_ROUTE_STRATIFIED_GROUPS: set[ScenarioGroup] = {
    "legitimate",
    "difficult_legitimate",
    "distribution_shift",
}

_ATTACK_SPLITTABLE: set[ScenarioGroup] = {
    "malicious_intent",
    "hard_malicious_intent",
    "benign_invalid",
}

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


def _split_indices(n: int, train_ratio: float, val_ratio: float) -> list[str]:
    """Round-robin style splits so every bucket gets rows when n is large enough."""
    if n == 0:
        return []
    test_ratio = 1.0 - train_ratio - val_ratio
    n_train = max(0, int(round(n * train_ratio)))
    n_val = max(0, int(round(n * val_ratio)))
    n_test = n - n_train - n_val
    if n >= 3:
        n_test = max(1, n_test)
        if n_train + n_val + n_test > n:
            n_train = max(0, n - n_val - n_test)
    out = ["train"] * n_train + ["validation"] * n_val + ["test"] * n_test
    while len(out) < n:
        out.append("test")
    return out[:n]


def assign_splits(
    meta: list[GroundTruthMeta],
    records: list[RequestRecord] | None = None,
    train_ratio: float = 0.6,
    val_ratio: float = 0.15,
) -> list[GroundTruthMeta]:
    """Assign train/validation/test splits.

    Normal-traffic groups are stratified per request and route.
    Attack and benign-invalid families are split within each family so every
    family contributes test rows when enough samples exist.
    """
    record_by_id = {r.request_id: r for r in records} if records else {}
    family_idxs: dict[str, list[int]] = defaultdict(list)

    for i, m in enumerate(meta):
        if m.scenario_group in _ATTACK_SPLITTABLE:
            fam = m.attack_family or m.scenario_id
            family_idxs[f"{m.scenario_group}:{fam}"].append(i)

    family_split: dict[int, str] = {}
    for _key, indices in family_idxs.items():
        indices_sorted = sorted(indices, key=lambda j: meta[j].request_id)
        splits = _split_indices(len(indices_sorted), train_ratio, val_ratio)
        for j, idx in enumerate(indices_sorted):
            family_split[idx] = splits[j]

    out: list[GroundTruthMeta] = []
    for i, m in enumerate(meta):
        if i in family_split:
            split = family_split[i]
        elif m.scenario_group in _ROUTE_STRATIFIED_GROUPS and m.request_id in record_by_id:
            rec = record_by_id[m.request_id]
            group_key = f"{rec.method.upper()} {rec.route}:{m.request_id}"
            split = _bucket(group_key, train_ratio, val_ratio)
            if m.scenario_group == "legitimate" and split == "test":
                split = _bucket(group_key + ":leg", 0.7, 0.15)
        else:
            group_key = f"{m.scenario_id}:{m.template_id or 'na'}:{m.request_id}"
            split = _bucket(group_key, train_ratio, val_ratio)

        updated = GroundTruthMeta(
            request_id=m.request_id,
            scenario_group=m.scenario_group,
            scenario_id=m.scenario_id,
            attack_family=m.attack_family,
            template_id=m.template_id,
            split=split,  # type: ignore
            attack_slice=m.attack_slice,
            invalid_slice=m.invalid_slice,
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
        elif m.split == "validation" and m.scenario_group in (
            "legitimate",
            "difficult_legitimate",
            "distribution_shift",
        ):
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


def validation_ordinary_records(
    records: list[RequestRecord], meta: list[GroundTruthMeta]
) -> list[RequestRecord]:
    """Validation rows from the same population the baseline was fit on.

    Difficult legitimate text and distribution-shift rows stay out of the
    false-positive cutoff. They are a stress test, not the 1% budget.
    """
    meta_by_id = {m.request_id: m for m in meta}
    return [
        r
        for r in records
        if (m := meta_by_id.get(r.request_id))
        and m.split == "validation"
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


def held_out_family_indices(
    meta: list[GroundTruthMeta],
    family: str = "traversal",
    held_out_split: str = "test",
) -> list[int]:
    """Indices of test rows for a family absent from train/validation development."""
    dev_families = {
        m.attack_family
        for m in meta
        if m.split in ("train", "validation") and m.attack_family
    }
    if family in dev_families:
        return []
    return [
        i
        for i, m in enumerate(meta)
        if m.split == held_out_split and m.attack_family == family
    ]


def exclude_families_from_development(
    records: list[RequestRecord],
    meta: list[GroundTruthMeta],
    exclude_families: set[str],
) -> tuple[list[RequestRecord], list[GroundTruthMeta]]:
    """Drop malicious rows whose family is in exclude_families from train/val use."""
    meta_by_id = {m.request_id: m for m in meta}
    kept_recs = []
    kept_meta = []
    for r in records:
        m = meta_by_id[r.request_id]
        if (
            m.attack_family in exclude_families
            and m.scenario_group in ("malicious_intent", "hard_malicious_intent")
            and m.split in ("train", "validation")
        ):
            continue
        kept_recs.append(r)
        kept_meta.append(m)
    return kept_recs, kept_meta
