"""Payload-family helpers for splits and dataset QA."""

from __future__ import annotations

import hashlib
import re

from anomaly_guard.core.schema import GroundTruthMeta

PAYLOAD_FAMILY_RE = re.compile(r"(?:^|\s)pf:([0-9a-f]+)")
CONTENT_FAMILY_RE = re.compile(r"(?:^|\s)cf:(\S+)")
HOLDOUT_RE = re.compile(r"(?:^|\s)holdout:1(?:\s|$)")


def content_key_parts(method: str, route: str, raw_query: str, body: str) -> tuple[str, str, str, str]:
    return (method.upper(), route, raw_query, body)


def payload_family_digest(base_key: tuple[str, str, str, str]) -> str:
    raw = "|".join(base_key)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def payload_family_from_meta(meta: GroundTruthMeta) -> str | None:
    match = PAYLOAD_FAMILY_RE.search(meta.notes or "")
    return match.group(1) if match else None


def content_family_from_meta(meta: GroundTruthMeta) -> str | None:
    match = CONTENT_FAMILY_RE.search(meta.notes or "")
    return match.group(1) if match else None


def is_holdout_content(meta: GroundTruthMeta) -> bool:
    return HOLDOUT_RE.search(meta.notes or "") is not None
