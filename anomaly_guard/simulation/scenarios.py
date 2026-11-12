"""Scenario definitions and payload templates for traffic simulation."""

from __future__ import annotations

import json
import random
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


_LEGIT_SEARCH = [
    "python tutorial",
    "weather london",
    "anomaly detection",
    "http parsing",
    "university project",
    "linear algebra exercises",
    "best pasta recipe",
    "public transport map",
    "history of computing",
    "differential equations",
    "podcast recommendations",
    "bike repair tips",
    "medieval literature",
    "kubernetes networking",
    "noise cancellation headphones",
]


def build_search_legitimate(rid: str, ts: str, seq: int) -> tuple[RequestRecord, GroundTruthMeta]:
    q = _LEGIT_SEARCH[seq % len(_LEGIT_SEARCH)]
    if seq % 7 == 0:
        q = q + " " + ("details " * (seq % 4))
    page = (seq % 12) + 1
    rec = RequestRecord(
        request_id=rid,
        timestamp=ts,
        client_id=f"client-{seq % 120:03d}",
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
        "SELECT is a SQL keyword but this is prose about databases",
        "path/to/file in a tutorial",
    ]
    q = queries[seq % len(queries)]
    rec = RequestRecord(
        request_id=rid,
        timestamp=ts,
        client_id=f"client-{seq % 120:03d}",
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
    user = f"student{(seq % 500):03d}"
    body = f"username={user}&password=dummy-{seq % 40}"
    rec = RequestRecord(
        request_id=rid,
        timestamp=ts,
        client_id=f"client-{seq % 120:03d}",
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
        "See also `printf` and similar APIs in C.",
        "We use union types in TypeScript for API responses.",
        "Repeated chars ok: .... and ---- in markdown examples.",
    ]
    text = texts[seq % len(texts)]
    if seq % 5 == 0:
        text = text + " " + ("More context. " * (1 + seq % 8))
    if seq % 3 == 0:
        body = f"text={text.replace(' ', '+')}"
        headers = _headers_form()
    else:
        body = json.dumps({"text": text, "author": f"user{seq % 80}"})
        headers = _headers_json()
    rec = RequestRecord(
        request_id=rid,
        timestamp=ts,
        client_id=f"client-{seq % 120:03d}",
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


def build_benign_invalid_contract(rid: str, ts: str, seq: int) -> tuple[RequestRecord, GroundTruthMeta]:
    variants = [
        ("GET", "/search", "q=&page=2", "", {}),
        ("GET", "/search", "page=abc", "", {}),
        ("POST", "/login", "", "username=&password=test", _headers_form()),
        ("POST", "/comment", "", "{not json", _headers_json()),
        ("POST", "/comment", "", "text=" + "x" * 9000, _headers_form()),
    ]
    method, route, rq, body, hdrs = variants[seq % len(variants)]
    rec = RequestRecord(
        request_id=rid,
        timestamp=ts,
        client_id=f"client-{seq % 120:03d}",
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
        template_id="invalid_contract",
        invalid_slice="contract_break",
    )
    return rec, meta


def build_benign_invalid_schema_valid(rid: str, ts: str, seq: int) -> tuple[RequestRecord, GroundTruthMeta]:
    """Parses cleanly but odd / weakly validated values."""
    variants: list[tuple[str, str, str, str, dict[str, str]]] = [
        ("GET", "/search", "q=!!!???&page=1", "", {"user-agent": "AnomalyGuardSim/1.0"}),
        ("GET", "/search", "q=" + "z" * 200 + "&page=2", "", {"user-agent": "AnomalyGuardSim/1.0"}),
        ("POST", "/login", "", "username=guest!!!&password=dummy-1", _headers_form()),
        ("POST", "/comment", "", f"text={'#' * 40} not really a comment", _headers_form()),
        ("POST", "/comment", "", json.dumps({"text": "..\\..\\notes.txt", "author": "x"}), _headers_json()),
    ]
    method, route, rq, body, hdrs = variants[seq % len(variants)]
    rec = RequestRecord(
        request_id=rid,
        timestamp=ts,
        client_id=f"client-{seq % 120:03d}",
        method=method,
        route=route,
        raw_query=rq,
        headers=hdrs,
        body=body,
    )
    meta = GroundTruthMeta(
        request_id=rid,
        scenario_group="benign_invalid",
        scenario_id="schema_valid_odd",
        attack_family=None,
        template_id="invalid_odd",
        invalid_slice="schema_valid_odd",
    )
    return rec, meta


def build_sqli_signature(rid: str, ts: str, seq: int) -> tuple[RequestRecord, GroundTruthMeta]:
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
        client_id=f"client-{seq % 120:03d}",
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
        template_id=f"sqli_sig_{seq % len(payloads)}",
        attack_slice="signature_easy",
    )
    return rec, meta


def build_sqli_evasive(rid: str, ts: str, seq: int) -> tuple[RequestRecord, GroundTruthMeta]:
    payloads = [
        "ad'||'min",
        "x' %7c%7c '1'='1",
        "1uni/**/on select 2",
        "admin'--",
        "1 OR 2>1",
    ]
    p = payloads[seq % len(payloads)]
    rec = RequestRecord(
        request_id=rid,
        timestamp=ts,
        client_id=f"client-{seq % 120:03d}",
        method="POST",
        route="/login",
        raw_query="",
        headers=_headers_form(),
        body=f"username={p}&password=x",
    )
    meta = GroundTruthMeta(
        request_id=rid,
        scenario_group="malicious_intent",
        scenario_id="sqli_evasive",
        attack_family="sqli",
        template_id=f"sqli_eva_{seq % len(payloads)}",
        attack_slice="rule_evasive",
    )
    return rec, meta


def build_xss_signature(rid: str, ts: str, seq: int) -> tuple[RequestRecord, GroundTruthMeta]:
    payloads = [
        "<script>alert(1)</script>",
        "javascript:alert(document.cookie)",
        "<img src=x onerror=alert(1)>",
    ]
    p = payloads[seq % len(payloads)].replace(" ", "+")
    rec = RequestRecord(
        request_id=rid,
        timestamp=ts,
        client_id=f"client-{seq % 120:03d}",
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
        template_id=f"xss_sig_{seq % len(payloads)}",
        attack_slice="signature_easy",
    )
    return rec, meta


def build_xss_evasive(rid: str, ts: str, seq: int) -> tuple[RequestRecord, GroundTruthMeta]:
    payloads = [
        "<img/src=x>",
        "<svg/onload=alert(1)>",
        "script:alert(1)",
        "&#x3c;script&#x3e;",
    ]
    p = payloads[seq % len(payloads)].replace(" ", "+")
    rec = RequestRecord(
        request_id=rid,
        timestamp=ts,
        client_id=f"client-{seq % 120:03d}",
        method="POST",
        route="/comment",
        raw_query="",
        headers=_headers_form(),
        body=f"text={p}",
    )
    meta = GroundTruthMeta(
        request_id=rid,
        scenario_group="malicious_intent",
        scenario_id="xss_evasive",
        attack_family="xss",
        template_id=f"xss_eva_{seq % len(payloads)}",
        attack_slice="rule_evasive",
    )
    return rec, meta


def build_traversal_signature(rid: str, ts: str, seq: int) -> tuple[RequestRecord, GroundTruthMeta]:
    payloads = ["../../../etc/passwd", "..%2f..%2fetc/passwd", "....//....//etc/passwd"]
    p = payloads[seq % len(payloads)]
    rec = RequestRecord(
        request_id=rid,
        timestamp=ts,
        client_id=f"client-{seq % 120:03d}",
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
        template_id=f"trav_sig_{seq % len(payloads)}",
        attack_slice="signature_easy",
    )
    return rec, meta


def build_traversal_evasive(rid: str, ts: str, seq: int) -> tuple[RequestRecord, GroundTruthMeta]:
    payloads = ["..%252f..%252fsecret", "....//....//", "%2e%2e%2f%2e%2e%2fwin.ini"]
    p = payloads[seq % len(payloads)]
    rec = RequestRecord(
        request_id=rid,
        timestamp=ts,
        client_id=f"client-{seq % 120:03d}",
        method="GET",
        route="/search",
        raw_query=f"q={p}",
        headers={"user-agent": "AnomalyGuardSim/1.0"},
        body="",
    )
    meta = GroundTruthMeta(
        request_id=rid,
        scenario_group="malicious_intent",
        scenario_id="traversal_evasive",
        attack_family="traversal",
        template_id=f"trav_eva_{seq % len(payloads)}",
        attack_slice="rule_evasive",
    )
    return rec, meta


def build_size_only_search(rid: str, ts: str, seq: int) -> tuple[RequestRecord, GroundTruthMeta]:
    blob = "Z" * (450 + (seq % 100))
    rec = RequestRecord(
        request_id=rid,
        timestamp=ts,
        client_id=f"client-{seq % 120:03d}",
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
        template_id="size_search",
        attack_slice="size_distribution",
    )
    return rec, meta


def build_size_only_comment(rid: str, ts: str, seq: int) -> tuple[RequestRecord, GroundTruthMeta]:
    text = ("!" * 20 + " ") * (30 + seq % 20)
    body = json.dumps({"text": text, "author": "bot"})
    rec = RequestRecord(
        request_id=rid,
        timestamp=ts,
        client_id=f"client-{seq % 120:03d}",
        method="POST",
        route="/comment",
        raw_query="",
        headers=_headers_json(),
        body=body,
    )
    meta = GroundTruthMeta(
        request_id=rid,
        scenario_group="malicious_intent",
        scenario_id="repetitive_comment",
        attack_family="repetitive",
        template_id="size_comment",
        attack_slice="size_distribution",
    )
    return rec, meta


def build_hard_malicious(rid: str, ts: str, seq: int) -> tuple[RequestRecord, GroundTruthMeta]:
    payloads = ["1 OR 2=1", "uni%0aon", "scr\u0069pt:alert(1)", "..%2e"]
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
        client_id=f"client-{seq % 120:03d}",
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
        attack_slice="rule_evasive",
    )
    return rec, meta


def build_matched_pair_benign(rid: str, ts: str, seq: int) -> tuple[RequestRecord, GroundTruthMeta]:
    topic = _LEGIT_SEARCH[seq % len(_LEGIT_SEARCH)]
    pad = "x" * (50 + (seq % 30))
    q = f"{topic} {pad}"
    rec = RequestRecord(
        request_id=rid,
        timestamp=ts,
        client_id=f"client-pair-{seq % 40:03d}",
        method="GET",
        route="/search",
        raw_query=f"q={q.replace(' ', '+')}&page=1",
        headers={"user-agent": "AnomalyGuardSim/1.0"},
        body="",
    )
    meta = GroundTruthMeta(
        request_id=rid,
        scenario_group="legitimate",
        scenario_id="matched_pair",
        attack_family=None,
        template_id="pair_benign",
    )
    return rec, meta


def build_matched_pair_malicious(rid: str, ts: str, seq: int) -> tuple[RequestRecord, GroundTruthMeta]:
    topic = _LEGIT_SEARCH[seq % len(_LEGIT_SEARCH)]
    pad = "x" * (50 + (seq % 30))
    inject = "uni%0aon" if seq % 2 else "1 OR 2=1"
    q = f"{topic} {pad} {inject}"
    rec = RequestRecord(
        request_id=rid,
        timestamp=ts,
        client_id=f"client-pair-{seq % 40:03d}",
        method="GET",
        route="/search",
        raw_query=f"q={q.replace(' ', '+')}&page=1",
        headers={"user-agent": "AnomalyGuardSim/1.0"},
        body="",
    )
    meta = GroundTruthMeta(
        request_id=rid,
        scenario_group="malicious_intent",
        scenario_id="matched_pair",
        attack_family="evasive",
        template_id="pair_mal",
        attack_slice="rule_evasive",
    )
    return rec, meta


def build_distribution_shift(rid: str, ts: str, seq: int) -> tuple[RequestRecord, GroundTruthMeta]:
    long_comment = "Valid but longer comment: " + ("lorem ipsum dolor sit amet. " * (5 + seq % 10))
    rec = RequestRecord(
        request_id=rid,
        timestamp=ts,
        client_id=f"client-{seq % 120:03d}",
        method="POST",
        route="/comment",
        raw_query="",
        headers=_headers_json(),
        body='{"text": ' + json.dumps(long_comment) + ', "author": "new_locale_user"}',
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
    ScenarioTemplate("search_ordinary", "legitimate", None, "leg_search", build_search_legitimate),
    ScenarioTemplate("login_ordinary", "legitimate", None, "leg_login", build_login_legitimate),
    ScenarioTemplate("comment_ordinary", "legitimate", None, "leg_comment", build_comment_legitimate),
    ScenarioTemplate("matched_pair_benign", "legitimate", None, "pair_benign", build_matched_pair_benign),
    ScenarioTemplate("search_difficult", "difficult_legitimate", None, "diff_search", build_search_difficult),
    ScenarioTemplate("sqli_sig", "malicious_intent", "sqli", "sqli_sig", build_sqli_signature),
    ScenarioTemplate("sqli_eva", "malicious_intent", "sqli", "sqli_eva", build_sqli_evasive),
    ScenarioTemplate("xss_sig", "malicious_intent", "xss", "xss_sig", build_xss_signature),
    ScenarioTemplate("xss_eva", "malicious_intent", "xss", "xss_eva", build_xss_evasive),
    ScenarioTemplate("trav_sig", "malicious_intent", "traversal", "trav_sig", build_traversal_signature),
    ScenarioTemplate("trav_eva", "malicious_intent", "traversal", "trav_eva", build_traversal_evasive),
    ScenarioTemplate("size_search", "malicious_intent", "oversized", "size_search", build_size_only_search),
    ScenarioTemplate("size_comment", "malicious_intent", "repetitive", "size_comment", build_size_only_comment),
    ScenarioTemplate("pair_mal", "malicious_intent", "evasive", "pair_mal", build_matched_pair_malicious),
    ScenarioTemplate("hard_short", "hard_malicious_intent", "evasive", "hard", build_hard_malicious),
    ScenarioTemplate("longer_comments", "distribution_shift", None, "shift", build_distribution_shift),
]

TEMPLATE_BY_ID = {t.scenario_id: t for t in ALL_TEMPLATES}
