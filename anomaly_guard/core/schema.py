"""Request record schema, labels, and configuration loading."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml

DecisionLabel = Literal["normal", "review", "alert"]
ScenarioGroup = Literal[
    "legitimate",
    "difficult_legitimate",
    "benign_invalid",
    "malicious_intent",
    "hard_malicious_intent",
    "distribution_shift",
]


@dataclass
class RequestRecord:
    """JSONL request record (no ground truth)."""

    request_id: str
    timestamp: str
    client_id: str
    method: str
    route: str
    raw_query: str
    headers: dict[str, str]
    body: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RequestRecord:
        return cls(
            request_id=str(data["request_id"]),
            timestamp=str(data["timestamp"]),
            client_id=str(data["client_id"]),
            method=str(data["method"]).upper(),
            route=str(data["route"]),
            raw_query=str(data.get("raw_query", "")),
            headers={k.lower(): str(v) for k, v in (data.get("headers") or {}).items()},
            body=str(data.get("body", "")),
        )


@dataclass
class GroundTruthMeta:
    """Separate metadata joined by request_id."""

    request_id: str
    scenario_group: ScenarioGroup
    scenario_id: str
    attack_family: str | None = None
    template_id: str | None = None
    split: Literal["train", "validation", "test"] | None = None
    attack_slice: str | None = None
    invalid_slice: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GroundTruthMeta:
        return cls(
            request_id=str(data["request_id"]),
            scenario_group=data["scenario_group"],
            scenario_id=str(data["scenario_id"]),
            attack_family=data.get("attack_family"),
            template_id=data.get("template_id"),
            split=data.get("split"),
            attack_slice=data.get("attack_slice"),
            invalid_slice=data.get("invalid_slice"),
            notes=str(data.get("notes", "")),
        )


@dataclass
class RouteFieldSpec:
    name: str
    location: str
    required: bool
    type: str
    min_length: int = 0
    max_length: int = 16384
    min_value: int | None = None
    max_value: int | None = None


@dataclass
class RouteSpec:
    method: str
    path: str
    content_types: list[str]
    fields: list[RouteFieldSpec]


@dataclass
class AppLimits:
    max_body_bytes: int
    max_query_bytes: int
    max_parameters: int
    max_field_length: int


@dataclass
class SuspiciousPatternRule:
    id: str
    severity: Literal["low", "medium", "high"]
    explanation: str
    pattern: str
    fields: list[str]
    match_decoded: bool = False


@dataclass
class RulesConfig:
    suspicious_patterns: list[SuspiciousPatternRule]
    schema_violation_severity: Literal["low", "medium", "high"]


@dataclass
class AppConfig:
    limits: AppLimits
    routes: list[RouteSpec]
    rules: RulesConfig
    config_hash: str = ""

    @property
    def route_key(self) -> dict[tuple[str, str], RouteSpec]:
        return {(r.method.upper(), r.path): r for r in self.routes}


def _file_hash(*paths: Path) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def load_app_config(config_dir: Path | None = None) -> AppConfig:
    config_dir = config_dir or Path(__file__).resolve().parent.parent / "config"
    routes_path = config_dir / "routes.yaml"
    rules_path = config_dir / "rules.yaml"
    routes_raw = yaml.safe_load(routes_path.read_text())
    rules_raw = yaml.safe_load(rules_path.read_text())

    limits = AppLimits(**routes_raw["limits"])
    routes: list[RouteSpec] = []
    for r in routes_raw["routes"]:
        fields = [RouteFieldSpec(**f) for f in r["fields"]]
        routes.append(
            RouteSpec(
                method=r["method"].upper(),
                path=r["path"],
                content_types=list(r.get("content_types") or []),
                fields=fields,
            )
        )

    patterns = [
        SuspiciousPatternRule(
            id=p["id"],
            severity=p["severity"],
            explanation=p["explanation"],
            pattern=p["pattern"],
            fields=list(p["fields"]),
            match_decoded=p.get("match_decoded", False),
        )
        for p in rules_raw["suspicious_patterns"]
    ]
    rules = RulesConfig(
        suspicious_patterns=patterns,
        schema_violation_severity=rules_raw.get("schema_violation_severity", "medium"),
    )
    return AppConfig(
        limits=limits,
        routes=routes,
        rules=rules,
        config_hash=_file_hash(routes_path, rules_path),
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
