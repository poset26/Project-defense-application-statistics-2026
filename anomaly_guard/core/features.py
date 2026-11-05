"""Numeric feature extraction from parsed requests."""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from anomaly_guard.core.parser import ParsedRequest

# Mutually exclusive Unicode category bins (decoded text)
CATEGORY_BINS = ("letter", "digit", "whitespace", "quote", "bracket", "slash", "other")


@dataclass
class FieldFeatures:
    field_name: str
    char_length: int
    byte_length: int
    percent_escape_count: int
    raw_to_decoded_len_diff: int
    longest_repeat_run: int
    distinct_char_fraction: float
    entropy: float
    category_fractions: dict[str, float]
    is_empty: bool


@dataclass
class RequestFeatures:
    query_byte_length: int
    body_byte_length: int
    parameter_count: int
    repeated_key_count: int
    unexpected_field_count: int
    json_parse_failure: int
    malformed_encoding: int
    field_features: dict[str, FieldFeatures] = field(default_factory=dict)

    def to_vector(self, field_order: list[str] | None = None) -> list[float]:
        """Flat numeric vector for optional multivariate detector."""
        base = [
            float(self.query_byte_length),
            float(self.body_byte_length),
            float(self.parameter_count),
            float(self.repeated_key_count),
            float(self.unexpected_field_count),
            float(self.json_parse_failure),
            float(self.malformed_encoding),
        ]
        names = field_order or sorted(self.field_features.keys())
        for fname in names:
            ff = self.field_features.get(fname)
            per_field = 8 + len(CATEGORY_BINS)
            if ff is None:
                base.extend([0.0] * per_field)
                continue
            base.extend(
                [
                    float(ff.char_length),
                    float(ff.byte_length),
                    float(ff.percent_escape_count),
                    float(ff.raw_to_decoded_len_diff),
                    float(ff.longest_repeat_run),
                    float(ff.distinct_char_fraction),
                    float(ff.entropy),
                    float(ff.is_empty),
                ]
                + [float(ff.category_fractions.get(c, 0.0)) for c in CATEGORY_BINS]
            )
        return base

    def feature_names(self, field_order: list[str] | None = None) -> list[str]:
        names = field_order or sorted(self.field_features.keys())
        out = [
            "query_bytes",
            "body_bytes",
            "param_count",
            "repeated_keys",
            "unexpected_fields",
            "json_fail",
            "malformed_encoding",
        ]
        for fname in names:
            prefix = f"{fname}_"
            out.extend(
                [
                    prefix + "char_len",
                    prefix + "byte_len",
                    prefix + "pct_esc",
                    prefix + "raw_dec_diff",
                    prefix + "repeat_run",
                    prefix + "distinct_frac",
                    prefix + "entropy",
                    prefix + "empty",
                ]
                + [prefix + cat for cat in CATEGORY_BINS]
            )
        return out


def _count_percent_escapes(raw: str) -> int:
    import re

    return len(re.findall(r"%[0-9A-Fa-f]{2}", raw))


def _longest_repeat_run(s: str) -> int:
    if not s:
        return 0
    best = 1
    cur = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(s)
    ent = 0.0
    for c in counts.values():
        p = c / n
        ent -= p * math.log2(p)
    return ent


def _category_fractions(decoded: str) -> dict[str, float]:
    if not decoded:
        return {c: 0.0 for c in CATEGORY_BINS}
    counts = {c: 0 for c in CATEGORY_BINS}
    for ch in decoded:
        if ch in "'\"`":
            counts["quote"] += 1
        elif ch in "()[]{}":
            counts["bracket"] += 1
        elif ch in "/\\":
            counts["slash"] += 1
        elif ch.isspace():
            counts["whitespace"] += 1
        elif ch.isdigit():
            counts["digit"] += 1
        elif unicodedata.category(ch).startswith("L"):
            counts["letter"] += 1
        else:
            counts["other"] += 1
    total = len(decoded)
    return {k: v / total for k, v in counts.items()}


def extract_features(parsed: ParsedRequest) -> RequestFeatures:
    flags = parsed.flags
    field_feats: dict[str, FieldFeatures] = {}

    if parsed.route_spec:
        for spec in parsed.route_spec.fields:
            pf = parsed.fields.get(spec.name)
            if pf is None:
                field_feats[spec.name] = FieldFeatures(
                    field_name=spec.name,
                    char_length=0,
                    byte_length=0,
                    percent_escape_count=0,
                    raw_to_decoded_len_diff=0,
                    longest_repeat_run=0,
                    distinct_char_fraction=0.0,
                    entropy=0.0,
                    category_fractions={c: 0.0 for c in CATEGORY_BINS},
                    is_empty=True,
                )
                continue
            raw_joined = " ".join(pf.raw_values)
            dec_joined = " ".join(pf.decoded_values)
            char_len = len(dec_joined)
            byte_len = len(dec_joined.encode("utf-8", errors="replace"))
            distinct = len(set(dec_joined)) / max(char_len, 1)
            field_feats[spec.name] = FieldFeatures(
                field_name=spec.name,
                char_length=char_len,
                byte_length=byte_len,
                percent_escape_count=_count_percent_escapes(raw_joined),
                raw_to_decoded_len_diff=len(raw_joined) - char_len,
                longest_repeat_run=_longest_repeat_run(dec_joined),
                distinct_char_fraction=distinct,
                entropy=_shannon_entropy(dec_joined),
                category_fractions=_category_fractions(dec_joined),
                is_empty=char_len == 0,
            )

    return RequestFeatures(
        query_byte_length=parsed.query_byte_length,
        body_byte_length=parsed.body_byte_length,
        parameter_count=parsed.parameter_count,
        repeated_key_count=len(flags.duplicate_keys),
        unexpected_field_count=len(flags.unexpected_fields),
        json_parse_failure=int(flags.json_parse_error),
        malformed_encoding=int(flags.malformed_encoding),
        field_features=field_feats,
    )
