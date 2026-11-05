"""Traffic simulator: generate and replay JSONL request streams."""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

from anomaly_guard.core.schema import GroundTruthMeta, RequestRecord, write_jsonl
from anomaly_guard.simulation.scenarios import ALL_TEMPLATES, ScenarioTemplate


@dataclass
class GenerationPlan:
    legitimate: int = 7000
    difficult_legitimate: int = 1500
    benign_invalid: int = 800
    malicious_intent: int = 1500
    hard_malicious_intent: int = 300
    distribution_shift: int = 400
    seed: int = 42


DEFAULT_PLAN = GenerationPlan(
    legitimate=7000,
    difficult_legitimate=1500,
    benign_invalid=800,
    malicious_intent=1500,
    hard_malicious_intent=300,
    distribution_shift=400,
)


def _group_templates(group: str) -> list[ScenarioTemplate]:
    return [t for t in ALL_TEMPLATES if t.group == group]


def generate_dataset(
    plan: GenerationPlan | None = None,
    base_time: datetime | None = None,
) -> tuple[list[RequestRecord], list[GroundTruthMeta]]:
    plan = plan or DEFAULT_PLAN
    rng = random.Random(plan.seed)
    base = base_time or datetime(2026, 1, 1, tzinfo=timezone.utc)

    counts = {
        "legitimate": plan.legitimate,
        "difficult_legitimate": plan.difficult_legitimate,
        "benign_invalid": plan.benign_invalid,
        "malicious_intent": plan.malicious_intent,
        "hard_malicious_intent": plan.hard_malicious_intent,
        "distribution_shift": plan.distribution_shift,
    }

    records: list[RequestRecord] = []
    meta: list[GroundTruthMeta] = []
    seq = 0
    t_offset = 0

    for group, n in counts.items():
        templates = _group_templates(group)
        if not templates:
            continue
        for i in range(n):
            tmpl = templates[i % len(templates)]
            rid = str(uuid.uuid4())
            ts = (base + timedelta(milliseconds=t_offset)).isoformat()
            t_offset += rng.randint(1, 50)
            rec, m = tmpl.builder(rid, ts, seq)
            records.append(rec)
            meta.append(m)
            seq += 1

    rng.shuffle(records)
    order = {r.request_id: i for i, r in enumerate(records)}
    meta.sort(key=lambda m: order[m.request_id])
    return records, meta


def save_dataset(
    out_dir: Path,
    records: list[RequestRecord],
    meta: list[GroundTruthMeta],
    manifest: dict | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "requests.jsonl", [r.to_dict() for r in records])
    write_jsonl(out_dir / "ground_truth.jsonl", [m.to_dict() for m in meta])
    if manifest:
        import json

        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def replay_records(
    records: list[RequestRecord],
    shuffle: bool = False,
    seed: int = 0,
    limit: int | None = None,
) -> Iterator[RequestRecord]:
    rows = list(records)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)
    if limit:
        rows = rows[:limit]
    yield from rows
