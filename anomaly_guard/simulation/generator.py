"""Traffic simulator: generate and replay JSONL request streams."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

from anomaly_guard.core.schema import GroundTruthMeta, RequestRecord, write_jsonl
from anomaly_guard.simulation.payload_family import (
    content_family_from_meta,
    content_key_parts,
    payload_family_digest,
)
from anomaly_guard.simulation.scenarios import ALL_TEMPLATES, ScenarioTemplate, _unique_phrase, _SEARCH_QUALIFIERS, _SEARCH_DETAILS, _SEARCH_SCOPE


@dataclass
class GenerationPlan:
    legitimate: int = 7000
    difficult_legitimate: int = 1500
    benign_invalid: int = 0
    malicious_intent: int = 1500
    hard_malicious_intent: int = 300
    distribution_shift: int = 400
    seed: int = 42


DEFAULT_PLAN = GenerationPlan()

_GROUP_WEIGHTS: dict[str, list[ScenarioTemplate]] = {}


def _disambiguate(rec: RequestRecord, seq: int) -> RequestRecord:
    """Keep a colliding row inside its route contract with a real extra clause."""
    phrase, _slot = _unique_phrase(seq, _SEARCH_QUALIFIERS, _SEARCH_DETAILS, _SEARCH_SCOPE, ("row",))
    clause = f"{phrase.replace(' ', '+')}+case{seq}"
    if rec.method.upper() == "GET":
        query = rec.raw_query
        if "q=" in query:
            query = query.replace("q=", f"q={clause}+", 1)
        else:
            query = f"{query}&q={clause}" if query else f"q={clause}"
        return RequestRecord(
            request_id=rec.request_id,
            timestamp=rec.timestamp,
            client_id=rec.client_id,
            method=rec.method,
            route=rec.route,
            raw_query=query,
            headers=dict(rec.headers),
            body=rec.body,
        )
    body = rec.body
    if body.startswith("{") and '"text"' in body:
        body = body.replace('"text": "', f'"text": "{phrase} ', 1)
    elif "text=" in body:
        body = body.replace("text=", f"text={clause}+", 1)
    elif "username=" in body:
        body = body.replace("username=", f"username={clause}", 1)
    else:
        body = f"{body}&note={clause}" if body else f"text={clause}"
    return RequestRecord(
        request_id=rec.request_id,
        timestamp=rec.timestamp,
        client_id=rec.client_id,
        method=rec.method,
        route=rec.route,
        raw_query=rec.raw_query,
        headers=dict(rec.headers),
        body=body,
    )


def content_key(rec: RequestRecord) -> tuple[str, str, str, str]:
    return content_key_parts(rec.method, rec.route, rec.raw_query, rec.body)


def _init_weights() -> None:
    global _GROUP_WEIGHTS
    if _GROUP_WEIGHTS:
        return
    for t in ALL_TEMPLATES:
        _GROUP_WEIGHTS.setdefault(t.group, []).append(t)


def _pick_template(group: str, seq: int) -> ScenarioTemplate:
    _init_weights()
    templates = _GROUP_WEIGHTS.get(group, [])
    if not templates:
        raise ValueError(f"No templates for group {group}")
    return templates[seq % len(templates)]


def _annotate(rec: RequestRecord, meta: GroundTruthMeta) -> GroundTruthMeta:
    """Tag the underlying content family. Hold out every 8th family slot for test only."""
    family = content_family_from_meta(meta) or (meta.template_id or meta.scenario_id)
    slot = 0
    tail = family.rsplit(":", 1)[-1]
    if tail.isdigit():
        slot = int(tail)
    holdout = 1 if meta.scenario_group == "legitimate" and slot % 8 == 7 else 0
    digest = payload_family_digest(content_key(rec))
    return GroundTruthMeta(
        request_id=meta.request_id,
        scenario_group=meta.scenario_group,
        scenario_id=meta.scenario_id,
        attack_family=meta.attack_family,
        template_id=meta.template_id,
        split=meta.split,
        attack_slice=meta.attack_slice,
        invalid_slice=meta.invalid_slice,
        notes=f"pf:{digest} cf:{family} holdout:{holdout}",
    )


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
    seen: set[tuple[str, str, str, str]] = set()

    for group, n in counts.items():
        if n <= 0:
            continue
        for i in range(n):
            tmpl = _pick_template(group, i)
            rid = f"req-{seq:07d}"
            ts = (base + timedelta(milliseconds=t_offset)).isoformat()
            t_offset += rng.randint(1, 50)
            rec, m = tmpl.builder(rid, ts, seq)
            key = content_key(rec)
            if key in seen:
                rec = _disambiguate(rec, seq)
                key = content_key(rec)
            if key in seen:
                raise RuntimeError(f"Duplicate request content at {rid}")
            seen.add(key)
            records.append(rec)
            meta.append(_annotate(rec, m))
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
