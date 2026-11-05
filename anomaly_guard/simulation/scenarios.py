"""Scenario definitions and payload templates for traffic simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from anomaly_guard.core.schema import GroundTruthMeta, RequestRecord, ScenarioGroup


@dataclass
class ScenarioTemplate:
    scenario_id: str
    group: ScenarioGroup
    attack_family: str | None
    template_id: str
    builder: Callable[[str, str, int], tuple[RequestRecord, GroundTruthMeta]]


def _headers_form() -> dict[str, str]:
    return {"content-type": "application/x-www-form-urlencoded", "user-agent": "AnomalyGuardSim/1.0"}


def _headers_json() -> dict[str, str]:
    return {"content-type": "application/json", "user-agent": "AnomalyGuardSim/1.0"}


def build_search_legitimate(rid: str, ts: str, seq: int) -> tuple[RequestRecord, GroundTruthMeta]:
    queries = ["python tutorial", "weather london", "anomaly detection", "http parsing", "university project"]
    q = queries[seq % len(queries)]
    page = (seq % 5) + 1
    rec = RequestRecord(
        request_id=rid,
        timestamp=ts,
        client_id=f"client-{seq % 50:03d}",
        method="GET",
        route="/search",
        raw_query=f"q={q.replace(' ', '+')}&page={page}",
        headers={"user-agent": "AnomalyGuardSim/1.0"},
        body="",
    )
    meta = GroundTruthMeta(
        request_id=rid,
        scenario_group="legitimate",
        scenario_id="search_ordinary",
        attack_family=None,
        template_id="leg_search_v1",
    )
    return rec, meta


def build_search_difficult(rid: str, ts: str, seq: int) -> tuple[RequestRecord, GroundTruthMeta]:
    queries = [
        "O'Brien cafe",
        "naïve approach café",
        "emoji test 😀🎉",
        "日本語 search",
        "It's a valid query with apostrophe",
        "discussion of <html> tags in text not attack",
    ]
    q = queries[seq % len(queries)]
    rec = RequestRecord(
        request_id=rid,
        timestamp=ts,
        client_id=f"client-{seq % 50:03d}",
        method="GET",
        route="/search",
        raw_query=f"q={q.replace(' ', '+')}",
        headers={"user-agent": "AnomalyGuardSim/1.0"},
        body="",
    )
    meta = GroundTruthMeta(
        request_id=rid,
        scenario_group="difficult_legitimate",
        scenario_id="search_difficult",
        attack_family=None,
        template_id="diff_search_v1",
    )
    return rec, meta


def build_login_legitimate(rid: str, ts: str, seq: int) -> tuple[RequestRecord, GroundTruthMeta]:
    user = f"student{(seq % 100):03d}"
    body = f"username={user}&password=dummy-{seq % 20}"
    rec = RequestRecord(
        request_id=rid,
        timestamp=ts,
        client_id=f"client-{seq % 50:03d}",
        method="POST",
        route="/login",
        raw_query="",
        headers=_headers_form(),
        body=body,
    )
    meta = GroundTruthMeta(
        request_id=rid,
        scenario_group="legitimate",
        scenario_id="login_ordinary",
        attack_family=None,
        template_id="leg_login_v1",
    )
    return rec, meta


def build_comment_legitimate(rid: str, ts: str, seq: int) -> tuple[RequestRecord, GroundTruthMeta]:
    texts = [
        "Great article, thanks!",
        "I disagree but appreciate the write-up.",
        "Could you clarify section 2?",
    ]
    text = texts[seq % len(texts)]
    if seq % 3 == 0:
        body = f"text={text.replace(' ', '+')}"
        headers = _headers_form()
    else:
        import json

        body = json.dumps({"text": text, "author": f"user{seq % 30}"})
        headers = _headers_json()
    rec = RequestRecord(
        request_id=rid,
        timestamp=ts,
        client_id=f"client-{seq % 50:03d}",
        method="POST",
        route="/comment",
        raw_query="",
        headers=headers,
        body=body,
    )
    meta = GroundTruthMeta(
        request_id=rid,
        scenario_group="legitimate",
        scenario_id="comment_ordinary",
        attack_family=None,
        template_id="leg_comment_v1",
    )
    return rec, meta


def build_benign_invalid(rid: str, ts: str, seq: int) -> tuple[RequestRecord, GroundTruthMeta]:
    variants = [
        ("GET", "/search", "q=&page=abc", "", {}),
        ("GET", "/search", "page=9999", "", {}),
        ("POST", "/login", "", "username=&password=test", _headers_form()),
        ("POST", "/comment", "", "{not json", _headers_json()),
        ("POST", "/comment", "", "text=" + "x" * 9000, _headers_form()),
    ]
    method, route, rq, body, hdrs = variants[seq % len(variants)]
    rec = RequestRecord(
        request_id=rid,
        timestamp=ts,
        client_id=f"client-{seq % 50:03d}",
        method=method,
        route=route,
        raw_query=rq,
        headers=hdrs or {"user-agent": "AnomalyGuardSim/1.0"},
        body=body,
    )
    meta = GroundTruthMeta(
        request_id=rid,
        scenario_group="benign_invalid",
        scenario_id="client_mistakes",
        attack_family=None,
        template_id="invalid_v1",
    )
    return rec, meta


def build_sql_injection(rid: str, ts: str, seq: int) -> tuple[RequestRecord, GroundTruthMeta]:
    payloads = [
        "' OR '1'='1",
        "1; DROP TABLE users--",
        "admin' UNION SELECT null,null--",
        "x' OR 1=1--",
    ]
    p = payloads[seq % len(payloads)]
    rec = RequestRecord(
        request_id=rid,
        timestamp=ts,
        client_id=f"client-{seq % 50:03d}",
        method="POST",
        route="/login",
        raw_query="",
        headers=_headers_form(),
        body=f"username={p}&password=x",
    )
    meta = GroundTruthMeta(
        request_id=rid,
        scenario_group="malicious_intent",
        scenario_id="sqli_login",
        attack_family="sqli",
        template_id=f"sqli_t{seq % len(payloads)}",
    )
    return rec, meta


def build_xss(rid: str, ts: str, seq: int) -> tuple[RequestRecord, GroundTruthMeta]:
    payloads = [
        "<script>alert(1)</script>",
        "javascript:alert(document.cookie)",
        "<img src=x onerror=alert(1)>",
    ]
    p = payloads[seq % len(payloads)].replace(" ", "+")
    rec = RequestRecord(
        request_id=rid,
        timestamp=ts,
        client_id=f"client-{seq % 50:03d}",
        method="POST",
        route="/comment",
        raw_query="",
        headers=_headers_form(),
        body=f"text={p}",
    )
    meta = GroundTruthMeta(
        request_id=rid,
        scenario_group="malicious_intent",
        scenario_id="xss_comment",
        attack_family="xss",
        template_id=f"xss_t{seq % len(payloads)}",
    )
    return rec, meta


def build_traversal(rid: str, ts: str, seq: int) -> tuple[RequestRecord, GroundTruthMeta]:
    payloads = ["../../../etc/passwd", "..%2f..%2fetc/passwd", "....//....//etc/passwd"]
    p = payloads[seq % len(payloads)]
    rec = RequestRecord(
        request_id=rid,
        timestamp=ts,
        client_id=f"client-{seq % 50:03d}",
        method="GET",
        route="/search",
        raw_query=f"q={p.replace('/', '%2F')}",
        headers={"user-agent": "AnomalyGuardSim/1.0"},
        body="",
    )
    meta = GroundTruthMeta(
        request_id=rid,
        scenario_group="malicious_intent",
        scenario_id="traversal_search",
        attack_family="traversal",
        template_id=f"trav_t{seq % len(payloads)}",
    )
    return rec, meta


def build_oversized_attack(rid: str, ts: str, seq: int) -> tuple[RequestRecord, GroundTruthMeta]:
    blob = "A" * 600 + "' OR 1=1--"
    rec = RequestRecord(
        request_id=rid,
        timestamp=ts,
        client_id=f"client-{seq % 50:03d}",
        method="GET",
        route="/search",
        raw_query=f"q={blob}",
        headers={"user-agent": "AnomalyGuardSim/1.0"},
        body="",
    )
    meta = GroundTruthMeta(
        request_id=rid,
        scenario_group="malicious_intent",
        scenario_id="oversized_search",
        attack_family="oversized",
        template_id="oversized_v1",
    )
    return rec, meta


def build_hard_malicious(rid: str, ts: str, seq: int) -> tuple[RequestRecord, GroundTruthMeta]:
    """Short payloads resembling legitimate length/composition."""
    payloads = ["1 OR 1=1", "uni/**/on select", "script:alert(1)", "..%2e%2f"]
    routes = ["/login", "/search", "/comment", "/search"]
    methods = ["POST", "GET", "POST", "GET"]
    idx = seq % len(payloads)
    p = payloads[idx]
    method, route = methods[idx], routes[idx]
    if method == "POST" and route == "/login":
        body = f"username={p}&password=x"
        rq = ""
        headers = _headers_form()
    elif method == "POST":
        body = f"text={p}"
        rq = ""
        headers = _headers_form()
    else:
        body = ""
        rq = f"q={p}"
        headers = {"user-agent": "AnomalyGuardSim/1.0"}
    rec = RequestRecord(
        request_id=rid,
        timestamp=ts,
        client_id=f"client-{seq % 50:03d}",
        method=method,
        route=route,
        raw_query=rq,
        headers=headers,
        body=body,
    )
    meta = GroundTruthMeta(
        request_id=rid,
        scenario_group="hard_malicious_intent",
        scenario_id="hard_short",
        attack_family="evasive",
        template_id=f"hard_t{idx}",
    )
    return rec, meta


def build_distribution_shift(rid: str, ts: str, seq: int) -> tuple[RequestRecord, GroundTruthMeta]:
    long_comment = "Valid but longer comment: " + ("lorem ipsum dolor sit amet. " * (5 + seq % 10))
    rec = RequestRecord(
        request_id=rid,
        timestamp=ts,
        client_id=f"client-new-{seq % 20:03d}",
        method="POST",
        route="/comment",
        raw_query="",
        headers=_headers_json(),
        body='{"text": ' + __import__("json").dumps(long_comment) + ', "author": "new_locale_user"}',
    )
    meta = GroundTruthMeta(
        request_id=rid,
        scenario_group="distribution_shift",
        scenario_id="longer_comments",
        attack_family=None,
        template_id="shift_v1",
    )
    return rec, meta


ALL_TEMPLATES: list[ScenarioTemplate] = [
    ScenarioTemplate("search_ordinary", "legitimate", None, "leg_search_v1", build_search_legitimate),
    ScenarioTemplate("login_ordinary", "legitimate", None, "leg_login_v1", build_login_legitimate),
    ScenarioTemplate("comment_ordinary", "legitimate", None, "leg_comment_v1", build_comment_legitimate),
    ScenarioTemplate("search_difficult", "difficult_legitimate", None, "diff_search_v1", build_search_difficult),
    ScenarioTemplate("client_mistakes", "benign_invalid", None, "invalid_v1", build_benign_invalid),
    ScenarioTemplate("sqli_login", "malicious_intent", "sqli", "sqli", build_sql_injection),
    ScenarioTemplate("xss_comment", "malicious_intent", "xss", "xss", build_xss),
    ScenarioTemplate("traversal_search", "malicious_intent", "traversal", "traversal", build_traversal),
    ScenarioTemplate("oversized_search", "malicious_intent", "oversized", "oversized", build_oversized_attack),
    ScenarioTemplate("hard_short", "hard_malicious_intent", "evasive", "hard", build_hard_malicious),
    ScenarioTemplate("longer_comments", "distribution_shift", None, "shift", build_distribution_shift),
]

TEMPLATE_BY_ID = {t.scenario_id: t for t in ALL_TEMPLATES}
