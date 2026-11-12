"""Bounded HTTP request parsing and canonicalization."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import unquote_plus

from anomaly_guard.core.schema import AppConfig, RequestRecord, RouteSpec


@dataclass
class ParsedField:
    name: str
    values: list[str]
    raw_values: list[str]
    decoded_values: list[str]
    source: str


@dataclass
class ParseFlags:
    oversized_query: bool = False
    oversized_body: bool = False
    too_many_parameters: bool = False
    unsupported_content_type: bool = False
    json_parse_error: bool = False
    malformed_encoding: bool = False
    unknown_route: bool = False
    duplicate_keys: list[str] = field(default_factory=list)
    unexpected_fields: list[str] = field(default_factory=list)
    skipped_body: bool = False


@dataclass
class ParsedRequest:
    record: RequestRecord
    route_spec: RouteSpec | None
    fields: dict[str, ParsedField]
    flags: ParseFlags
    content_type: str
    query_byte_length: int
    body_byte_length: int
    parameter_count: int

    def field_text(self, name: str) -> str:
        pf = self.fields.get(name)
        if not pf or not pf.decoded_values:
            return ""
        return pf.decoded_values[0]

    def all_field_strings(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for name, pf in self.fields.items():
            out[name] = " | ".join(pf.decoded_values) if pf.decoded_values else ""
        out["raw_query"] = self.record.raw_query
        out["raw_body"] = self.record.body[:5000]
        return out


def _split_query_pairs(raw: str) -> list[tuple[str, str]]:
    """Split query string without decoding (single decode applied later)."""
    if not raw:
        return []
    pairs: list[tuple[str, str]] = []
    for part in raw.split("&"):
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
        else:
            k, v = part, ""
        pairs.append((k, v))
    return pairs


def _decode_once(value: str) -> tuple[str, bool]:
    """Single-pass percent decoding; detect malformed sequences."""
    malformed = False
    try:
        decoded = unquote_plus(value, errors="strict")
    except Exception:
        malformed = True
        decoded = unquote_plus(value, errors="replace")
    if "%" in decoded and re.search(r"%[0-9A-Fa-f]{0,1}[^0-9A-Fa-f]|%$", decoded):
        malformed = True
    return decoded, malformed


def parse_request(record: RequestRecord, config: AppConfig) -> ParsedRequest:
    limits = config.limits
    flags = ParseFlags()
    route_spec = config.route_key.get((record.method.upper(), record.route))

    query_bytes = len(record.raw_query.encode("utf-8", errors="replace"))
    body_bytes = len(record.body.encode("utf-8", errors="replace"))
    if query_bytes > limits.max_query_bytes:
        flags.oversized_query = True
    if body_bytes > limits.max_body_bytes:
        flags.oversized_body = True

    content_type = ""
    for k, v in record.headers.items():
        if k.lower() == "content-type":
            content_type = v.split(";")[0].strip().lower()
            break

    fields: dict[str, ParsedField] = {}
    parameter_count = 0

    if record.method.upper() == "GET":
        pairs = _split_query_pairs(record.raw_query)
        parameter_count = len(pairs)
        if parameter_count > limits.max_parameters:
            flags.too_many_parameters = True
            pairs = pairs[: limits.max_parameters]

        by_name: dict[str, list[tuple[str, str]]] = {}
        for raw_k, raw_v in pairs:
            by_name.setdefault(raw_k, []).append((raw_k, raw_v))

        for name, vals in by_name.items():
            if len(vals) > 1:
                flags.duplicate_keys.append(name)
            raw_values = [v for _, v in vals]
            decoded_values: list[str] = []
            for rv in raw_values:
                dec, bad = _decode_once(rv)
                if bad:
                    flags.malformed_encoding = True
                decoded_values.append(dec)
            fields[name] = ParsedField(
                name=name,
                values=decoded_values,
                raw_values=raw_values,
                decoded_values=decoded_values,
                source="query",
            )

    elif record.body and not flags.oversized_body:
        if content_type == "application/json":
            try:
                obj = json.loads(record.body)
            except json.JSONDecodeError:
                flags.json_parse_error = True
                obj = None
            if isinstance(obj, dict):
                parameter_count = len(obj)
                for name, val in obj.items():
                    if isinstance(val, (dict, list)):
                        sval = json.dumps(val, ensure_ascii=False)
                    else:
                        sval = str(val)
                    # JSON strings are already decoded. Form percent-decoding
                    # turns "C++" into "C  " and "%25" into "%".
                    fields[str(name)] = ParsedField(
                        name=str(name),
                        values=[sval],
                        raw_values=[sval],
                        decoded_values=[sval],
                        source="json",
                    )
        elif content_type in ("application/x-www-form-urlencoded", "") or not content_type:
            pairs = _split_query_pairs(record.body)
            parameter_count = len(pairs)
            if parameter_count > limits.max_parameters:
                flags.too_many_parameters = True
                pairs = pairs[: limits.max_parameters]
            by_name: dict[str, list[tuple[str, str]]] = {}
            for raw_k, raw_v in pairs:
                by_name.setdefault(raw_k, []).append((raw_k, raw_v))
            for name, vals in by_name.items():
                if len(vals) > 1:
                    flags.duplicate_keys.append(name)
                raw_values = [v for _, v in vals]
                decoded_values = []
                for rv in raw_values:
                    dec, bad = _decode_once(rv)
                    if bad:
                        flags.malformed_encoding = True
                    decoded_values.append(dec)
                fields[name] = ParsedField(
                    name=name,
                    values=decoded_values,
                    raw_values=raw_values,
                    decoded_values=decoded_values,
                    source="form",
                )
        else:
            flags.unsupported_content_type = True
            flags.skipped_body = True
    elif flags.oversized_body:
        flags.skipped_body = True

    if route_spec is None:
        flags.unknown_route = True
    else:
        expected = {f.name for f in route_spec.fields}
        present = set(fields.keys())
        for name in present - expected:
            flags.unexpected_fields.append(name)

    return ParsedRequest(
        record=record,
        route_spec=route_spec,
        fields=fields,
        flags=flags,
        content_type=content_type,
        query_byte_length=query_bytes,
        body_byte_length=body_bytes,
        parameter_count=parameter_count,
    )
