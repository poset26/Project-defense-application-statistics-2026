"""Traffic simulator: generate and replay JSONL request streams."""

from __future__ import annotations

import json
import random
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
    benign_invalid: int = 0
    malicious_intent: int = 1500
    hard_malicious_intent: int = 300
    distribution_shift: int = 400
    seed: int = 42


DEFAULT_PLAN = GenerationPlan()

# Weighted distribution within each scenario group (cycles by index % len(templates))
_GROUP_WEIGHTS: dict[str, list[ScenarioTemplate]] = {}


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


def _content_key(rec: RequestRecord) -> tuple[str, str, str, str]:
    return (rec.method.upper(), rec.route, rec.raw_query, rec.body)


def _stamp_pair(raw: str, key: str, token: str, sep: str) -> str:
    """Append `token` to an existing `key` value, or add that pair."""
    parts = raw.split("&") if raw else []
    out: list[str] = []
    found = False
    for part in parts:
        if not part:
            continue
        name, _, value = part.partition("=")
        if name == key and not found:
            value = f"{value}{sep}{token}" if value else token
            out.append(f"{name}={value}")
            found = True
        else:
            out.append(part)
    if not found:
        out.append(f"{key}={token}")
    return "&".join(out)


def _stamp_json_text(body: str, token: str) -> str:
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return body + token
    if isinstance(obj, dict):
        if "text" in obj:
            obj["text"] = f"{obj['text']} {token}"
        elif "author" in obj:
            obj["author"] = f"{obj['author']}-{token}"
        else:
            obj["text"] = token
        return json.dumps(obj, ensure_ascii=False)
    return body + token


def _with_unique_content(rec: RequestRecord, seq: int) -> RequestRecord:
    """Give every row its own query or body so train and test cannot share a string.

    The token is appended to a field the route already allows. Attack payloads
    stay in place as a prefix; only a short reference is added.
    """
    token = f"ref{seq:05d}"
    headers = dict(rec.headers)
    content_type = ""
    for key, value in headers.items():
        if key.lower() == "content-type":
            content_type = value.split(";")[0].strip().lower()
            break
    query = rec.raw_query
    body = rec.body
    if rec.method.upper() == "GET":
        query = _stamp_pair(query, "q", token, "+")
    elif content_type == "application/json":
        body = _stamp_json_text(body, token)
    elif rec.route == "/login":
        body = _stamp_pair(body, "password", token, "")
    else:
        body = _stamp_pair(body, "text", token, "+")
    return RequestRecord(
        request_id=rec.request_id,
        timestamp=rec.timestamp,
        client_id=rec.client_id,
        method=rec.method,
        route=rec.route,
        raw_query=query,
        headers=headers,
        body=body,
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
            # Sequential ids follow the seed's generation order. uuid4() did not.
            rid = f"req-{seq:07d}"
            ts = (base + timedelta(milliseconds=t_offset)).isoformat()
            t_offset += rng.randint(1, 50)
            rec, m = tmpl.builder(rid, ts, seq)
            rec = _with_unique_content(rec, seq)
            key = _content_key(rec)
            if key in seen:
                raise RuntimeError(f"Duplicate request content at {rid}")
            seen.add(key)
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
