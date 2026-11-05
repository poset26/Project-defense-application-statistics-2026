"""Application-contract and suspicious-pattern rule engine."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from anomaly_guard.core.parser import ParsedRequest
from anomaly_guard.core.schema import AppConfig, RouteFieldSpec

Severity = Literal["low", "medium", "high"]
RuleKind = Literal["schema", "suspicious_pattern", "size_limit"]


@dataclass
class TriggeredRule:
    rule_id: str
    kind: RuleKind
    severity: Severity
    explanation: str
    field: str | None = None


def _severity_rank(s: Severity) -> int:
    return {"low": 1, "medium": 2, "high": 3}[s]


def max_severity(rules: list[TriggeredRule]) -> Severity | None:
    if not rules:
        return None
    return max(rules, key=lambda r: _severity_rank(r.severity)).severity


def evaluate_rules(parsed: ParsedRequest, config: AppConfig) -> list[TriggeredRule]:
    triggered: list[TriggeredRule] = []
    limits = config.limits
    flags = parsed.flags

    if flags.oversized_query:
        triggered.append(
            TriggeredRule(
                rule_id="SIZE_QUERY",
                kind="size_limit",
                severity="medium",
                explanation="Query string exceeds configured byte limit",
                field="raw_query",
            )
        )
    if flags.oversized_body:
        triggered.append(
            TriggeredRule(
                rule_id="SIZE_BODY",
                kind="size_limit",
                severity="medium",
                explanation="Request body exceeds configured byte limit (body not fully parsed)",
                field="raw_body",
            )
        )
    if flags.too_many_parameters:
        triggered.append(
            TriggeredRule(
                rule_id="TOO_MANY_PARAMS",
                kind="schema",
                severity="medium",
                explanation="Parameter count exceeds configured maximum",
            )
        )

    sev = config.rules.schema_violation_severity

    if flags.unknown_route:
        triggered.append(
            TriggeredRule(
                rule_id="UNKNOWN_ROUTE",
                kind="schema",
                severity=sev,
                explanation="HTTP method and path do not match a supported application route",
            )
        )
        return triggered

    route = parsed.route_spec
    assert route is not None

    if parsed.record.method == "POST" and route.content_types:
        ct = parsed.content_type or "application/x-www-form-urlencoded"
        if ct not in route.content_types and not flags.unsupported_content_type:
            if parsed.body.strip():
                triggered.append(
                    TriggeredRule(
                        rule_id="CONTENT_TYPE",
                        kind="schema",
                        severity=sev,
                        explanation=f"Content-Type '{ct}' not permitted for this route",
                    )
                )
        if flags.unsupported_content_type:
            triggered.append(
                TriggeredRule(
                    rule_id="UNSUPPORTED_CONTENT",
                    kind="schema",
                    severity=sev,
                    explanation="Body content type is not supported in this MVP",
                )
            )

    if flags.json_parse_error:
        triggered.append(
            TriggeredRule(
                rule_id="JSON_PARSE",
                kind="schema",
                severity=sev,
                explanation="JSON body could not be parsed",
            )
        )

    for fname in flags.unexpected_fields:
        triggered.append(
            TriggeredRule(
                rule_id="UNEXPECTED_FIELD",
                kind="schema",
                severity=sev,
                explanation=f"Field '{fname}' is not defined in the route contract",
                field=fname,
            )
        )

    for spec in route.fields:
        _check_field_contract(parsed, spec, sev, triggered)

    strings = parsed.all_field_strings()
    for pr in config.rules.suspicious_patterns:
        for fname in pr.fields:
            text = strings.get(fname, "")
            if not text:
                continue
            target = text
            if pr.match_decoded and fname in parsed.fields:
                target = parsed.field_text(fname)
            try:
                if re.search(pr.pattern, target):
                    triggered.append(
                        TriggeredRule(
                            rule_id=pr.id,
                            kind="suspicious_pattern",
                            severity=pr.severity,
                            explanation=pr.explanation,
                            field=fname if fname not in ("raw_query", "raw_body") else fname,
                        )
                    )
                    break
            except re.error:
                continue

    return triggered


def _check_field_contract(
    parsed: ParsedRequest,
    spec: RouteFieldSpec,
    sev: Severity,
    triggered: list[TriggeredRule],
) -> None:
    pf = parsed.fields.get(spec.name)
    present = pf is not None and any(v for v in (pf.decoded_values if pf else []))
    if spec.required and not present:
        triggered.append(
            TriggeredRule(
                rule_id="MISSING_FIELD",
                kind="schema",
                severity=sev,
                explanation=f"Required field '{spec.name}' is missing or empty",
                field=spec.name,
            )
        )
        return
    if not present:
        return

    val = parsed.field_text(spec.name)
    char_len = len(val)
    byte_len = len(val.encode("utf-8", errors="replace"))
    if char_len < spec.min_length:
        triggered.append(
            TriggeredRule(
                rule_id="FIELD_TOO_SHORT",
                kind="schema",
                severity=sev,
                explanation=f"Field '{spec.name}' is shorter than minimum length {spec.min_length}",
                field=spec.name,
            )
        )
    if char_len > spec.max_length or byte_len > spec.max_length * 4:
        triggered.append(
            TriggeredRule(
                rule_id="FIELD_TOO_LONG",
                kind="schema",
                severity=sev,
                explanation=f"Field '{spec.name}' exceeds maximum allowed length",
                field=spec.name,
            )
        )

    if spec.type == "integer":
        try:
            iv = int(val.strip())
        except ValueError:
            triggered.append(
                TriggeredRule(
                    rule_id="INVALID_INTEGER",
                    kind="schema",
                    severity=sev,
                    explanation=f"Field '{spec.name}' must be an integer",
                    field=spec.name,
                )
            )
            return
        if spec.min_value is not None and iv < spec.min_value:
            triggered.append(
                TriggeredRule(
                    rule_id="INTEGER_RANGE",
                    kind="schema",
                    severity=sev,
                    explanation=f"Field '{spec.name}' below minimum {spec.min_value}",
                    field=spec.name,
                )
            )
        if spec.max_value is not None and iv > spec.max_value:
            triggered.append(
                TriggeredRule(
                    rule_id="INTEGER_RANGE",
                    kind="schema",
                    severity=sev,
                    explanation=f"Field '{spec.name}' above maximum {spec.max_value}",
                    field=spec.name,
                )
            )
