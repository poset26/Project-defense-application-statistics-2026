"""Read-only Streamlit dashboard for Cursor publish (session memory, bundled assets)."""

from __future__ import annotations

import html
import json
import threading
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from anomaly_guard.core.decision import AnomalyGuard
from anomaly_guard.core.schema import GroundTruthMeta, RequestRecord
from anomaly_guard.evaluation.experiments import MODES
from anomaly_guard.evaluation.splits import validation_legitimate_records, validation_ordinary_records
from anomaly_guard.dashboard_ui import (
    explain_legitimate_group_columns,
    explain_mode_metrics_columns,
    page_formulas,
    page_home,
    plot_decision_counts,
    plot_mode_metrics,
    explain_composition_chart,
    field_display_name,
    render_decision_conclusion,
    render_request_preview,
    shell_header,
    sidebar_theme_toggle,
)
from anomaly_guard.simulation.generator import replay_records
from anomaly_guard.storage import MemoryEventStore

_PKG = Path(__file__).resolve().parent
_ASSETS = _PKG / "publish_assets"
_DEMO = _ASSETS / "demo"
BASELINE = _DEMO / "baseline.json"
MAX_REPLAY = 20
REPLAY_COOLDOWN_SEC = 30
_REPLAY_LOCK = threading.Lock()
_NEXT_REPLAY_AT = 0.0


def _escape(s: str) -> str:
    return html.escape(s, quote=True)


def _load_demo_slice(limit: int) -> tuple[list[RequestRecord], list[GroundTruthMeta]]:
    """Read only the first `limit` demo rows so a click cannot scan the whole file."""
    records: list[RequestRecord] = []
    with (_DEMO / "requests.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if len(records) >= limit:
                break
            line = line.strip()
            if line:
                records.append(RequestRecord.from_dict(json.loads(line)))
    wanted = {r.request_id for r in records}
    meta: list[GroundTruthMeta] = []
    with (_DEMO / "ground_truth.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if len(meta) >= len(wanted):
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("request_id") in wanted:
                meta.append(GroundTruthMeta.from_dict(row))
    return records, meta


def _public_modes() -> list[str]:
    """Modes that have a saved cutoff. Isolation Forest has no saved model here."""
    return [m for m in MODES if m != "isolation_forest"]


def _apply_saved_cutoff(guard: AnomalyGuard, mode: str) -> None:
    settings_path = BASELINE.with_suffix(".settings.json")
    if settings_path.exists() and guard.load_settings(settings_path):
        return
    summary_path = _ASSETS / "metrics_summary.json"
    saved: dict = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        saved = ((summary.get("modes") or {}).get(mode) or {}).get("settings") or {}
    if not saved:
        return
    for key, value in saved.items():
        if hasattr(guard.settings, key):
            setattr(guard.settings, key, value)


@st.cache_resource
def _cached_guard(mode: str) -> AnomalyGuard:
    guard = AnomalyGuard(mode=mode)  # type: ignore[arg-type]
    guard.load_baseline(BASELINE)
    _apply_saved_cutoff(guard, mode)
    return guard


def _acquire_replay_slot() -> str | None:
    """One replay at a time, with a gap shared by every visitor on this process."""
    global _NEXT_REPLAY_AT
    if not _REPLAY_LOCK.acquire(blocking=False):
        return "Another replay is already running. Wait a moment and try again."
    now = time.monotonic()
    if now < _NEXT_REPLAY_AT:
        wait = int(_NEXT_REPLAY_AT - now) + 1
        _REPLAY_LOCK.release()
        return f"Shared hosting allows one replay every {REPLAY_COOLDOWN_SEC}s. Try again in {wait}s."
    return None


def _release_replay_slot() -> None:
    global _NEXT_REPLAY_AT
    _NEXT_REPLAY_AT = time.monotonic() + REPLAY_COOLDOWN_SEC
    _REPLAY_LOCK.release()


def _store() -> MemoryEventStore:
    if "ag_event_store" not in st.session_state:
        st.session_state.ag_event_store = MemoryEventStore()
    return st.session_state.ag_event_store


def page_traffic() -> None:
    st.header("Traffic")
    st.caption("Bundled sample traffic only. Replay affects your browser session, not other visitors.")
    modes = _public_modes()
    mode = st.selectbox("Detector mode", modes, index=modes.index("rules_plus_stats"))
    shuffle = st.checkbox("Shuffle replay", value=False)
    limit = int(
        st.number_input(
            "Replay limit",
            min_value=1,
            max_value=MAX_REPLAY,
            value=MAX_REPLAY,
        )
    )
    experiment_mode = st.checkbox("Experiment mode (store ground-truth labels)", value=True)

    if st.button("Replay traffic"):
        if not BASELINE.exists():
            st.error("Demo baseline missing from deployment package.")
            return
        blocked = _acquire_replay_slot()
        if blocked:
            st.warning(blocked)
            return
        try:
            records, meta_list = _load_demo_slice(limit)
            meta_by_id = {m.request_id: m for m in meta_list}
            guard = _cached_guard(mode)
            if guard.settings.alert_threshold_percentile <= 0 and not BASELINE.with_suffix(".settings.json").exists():
                val = validation_legitimate_records(records, meta_list)
                ordinary = validation_ordinary_records(records, meta_list)
                guard.calibrate(val, threshold_records=ordinary)
            store = _store()
            store.clear()
            counts = {"normal": 0, "review": 0, "alert": 0}
            parse_fail = 0
            for rec in replay_records(records, shuffle=shuffle, limit=limit):
                from anomaly_guard.core.parser import parse_request
                from anomaly_guard.core.schema import load_app_config

                parsed = parse_request(rec, load_app_config())
                if parsed.flags.json_parse_error or parsed.flags.malformed_encoding:
                    parse_fail += 1
                score = guard.score(rec)
                counts[score.decision] = counts.get(score.decision, 0) + 1
                gt = meta_by_id.get(rec.request_id) if experiment_mode else None
                store.upsert_event(rec, score, gt)
            st.session_state.ag_last_counts = counts
            st.success(f"Processed {sum(counts.values())} requests. Parsing flags: {parse_fail}")
            st.json(counts)
        finally:
            _release_replay_slot()

    if st.session_state.get("ag_last_counts"):
        plot_decision_counts(st.session_state.ag_last_counts, st.session_state.get("ag_dark", True))

    store = _store()
    st.subheader("Stored events")
    route_filter = st.selectbox("Route filter", ["", "/search", "/login", "/comment"])
    decision_filter = st.selectbox("Decision filter", ["", "normal", "review", "alert"])
    rows = store.search(
        decision=decision_filter or None,
        route=route_filter or None,
        limit=100,
    )
    if rows:
        df = pd.DataFrame(
            [
                {
                    "request_id": r["request_id"][:8],
                    "route": r["route"],
                    "decision": r["decision"],
                    "timestamp": r["timestamp"],
                }
                for r in rows
            ]
        )
        st.dataframe(df, use_container_width=True)
        st.caption("Events by route")
        st.bar_chart(df["route"].value_counts())
    else:
        st.info("No events yet. Replay traffic to populate the store.")


def page_alert_inspector() -> None:
    st.header("Alert inspector")
    store = _store()
    alerts = store.search(decision="alert", limit=50) + store.search(decision="review", limit=50)
    if not alerts:
        st.info("No alerts or reviews in the event store.")
        return
    selected = st.selectbox("Select request", [a["request_id"] for a in alerts])
    row = store.get(selected)
    if not row:
        return
    req = json.loads(row["request_json"])
    score = json.loads(row["score_json"])
    st.markdown("**Safe rendered request (escaped)**")
    st.code(
        _escape(
            f"{req['method']} {req['route']}\nQuery: {req.get('raw_query','')}\n"
            f"Body: {req.get('body','')[:2000]}"
        ),
        language="text",
    )
    st.markdown(f"**Decision:** `{score['decision']}`")
    if row.get("ground_truth_json"):
        gt = json.loads(row["ground_truth_json"])
        st.markdown(f"*Experiment label:* `{gt['scenario_group']}` / `{gt.get('attack_family')}`")
    for r in score.get("triggered_rules", []):
        st.write(f"- **{r['rule_id']}** ({r['severity']}): {r['explanation']}")
    for c in (score.get("statistical") or {}).get("components", []):
        st.write(f"- {c['component']} / {c['field']}: {c['detail']} (percentile={c.get('percentile')})")
    for e in score.get("explanations", []):
        st.write(f"- {e}")


def page_normal_vs_unusual() -> None:
    st.header("Normal versus unusual")
    if not BASELINE.exists():
        st.warning("Demo baseline is not available.")
        return
    store = _store()
    events = store.search(limit=200)
    req_ids = [e["request_id"] for e in events]
    selected = st.selectbox("Compare request (from event store)", [""] + req_ids) if req_ids else ""
    data = json.loads(BASELINE.read_text(encoding="utf-8"))
    st.write(f"Version **{data.get('version')}**, training count **{data.get('training_count')}**")

    if selected:
        row = store.get(selected)
        if row:
            from anomaly_guard.core.features import CATEGORY_BINS, _category_fractions
            from anomaly_guard.core.parser import parse_request
            from anomaly_guard.core.schema import load_app_config

            req_dict = json.loads(row["request_json"])
            render_request_preview(req_dict)

            score = json.loads(row["score_json"])
            gt = json.loads(row["ground_truth_json"]) if row.get("ground_truth_json") else None
            render_decision_conclusion(score, gt)

            req = RequestRecord.from_dict(req_dict)
            parsed = parse_request(req, load_app_config())
            key = f"{req.method.upper()} {req.route}"
            ep = data.get("endpoints", {}).get(key, {})
            for fname, fp in ep.get("fields", {}).items():
                text = parsed.field_text(fname)
                length = fp.get("length") or {}
                st.markdown(f"**`{fname}`** — {field_display_name(fname, req.route)}")
                st.write(
                    f"Length: observed={len(text)}, baseline median={length.get('median', 0):.1f}, "
                    f"MAD={length.get('mad', 0):.1f}"
                )
                base_probs = (fp.get("composition") or {}).get("category_probs") or {}
                if text:
                    explain_composition_chart()
                    obs_frac = _category_fractions(text)
                    labels = sorted(set(base_probs) | set(CATEGORY_BINS))
                    st.bar_chart(
                        pd.DataFrame(
                            {
                                "baseline": [base_probs.get(c, 0) for c in labels],
                                "observed": [obs_frac.get(c, 0) for c in labels],
                            },
                            index=labels,
                        )
                    )

    for ep_key, ep in data.get("endpoints", {}).items():
        with st.expander(ep_key):
            for fname, fp in ep.get("fields", {}).items():
                length = fp.get("length") or {}
                st.write(
                    f"**{fname}**: median len={length.get('median')}, MAD={length.get('mad')}, "
                    f"training n={fp.get('training_count')}, status={fp.get('status')}"
                )
                comp = fp.get("composition")
                if comp:
                    st.json(comp.get("category_probs"))


def _render_metrics_summary(summary: dict) -> None:
    modes = summary.get("modes", {})
    rows = []
    for mode, m in modes.items():
        mm = m.get("metrics_malicious", {})
        rows.append(
            {
                "mode": mode,
                "precision (attack)": mm.get("precision"),
                "recall (attack)": mm.get("recall"),
                "f1 (attack)": mm.get("f1"),
                "fpr (legit)": mm.get("false_positive_rate"),
                "AP": m.get("average_precision"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    explain_mode_metrics_columns()
    st.subheader("Confusion counts (alert vs not, test split)")
    st.caption("True positive is a malicious request that alerted. False positive is a legitimate-family request that alerted.")
    confusion = []
    for mode, m in modes.items():
        mm = m.get("metrics_malicious") or {}
        confusion.append({
            "mode": mode,
            "tp": mm.get("tp"),
            "fp": mm.get("fp"),
            "tn": mm.get("tn"),
            "fn": mm.get("fn"),
        })
    st.dataframe(pd.DataFrame(confusion), use_container_width=True)
    group_rows = []
    for mode, m in modes.items():
        groups = m.get("fpr_by_legitimate_group") or {}
        if not groups:
            continue
        group_rows.append(
            {
                "mode": mode,
                "ordinary": (groups.get("legitimate") or {}).get("alert_rate"),
                "difficult": (groups.get("difficult_legitimate") or {}).get("alert_rate"),
            }
        )
    if group_rows:
        st.subheader("False positives by legitimate group")
        st.dataframe(pd.DataFrame(group_rows), use_container_width=True)
        explain_legitimate_group_columns()


def page_evaluation() -> None:
    st.header("Evaluation")
    st.caption("Pre-computed metrics from the development CLI run (read-only).")
    metrics_path = _ASSETS / "metrics_summary.json"
    if metrics_path.exists():
        summary = json.loads(metrics_path.read_text(encoding="utf-8"))
        _render_metrics_summary(summary)
        plot_mode_metrics(summary, st.session_state.get("ag_dark", True))
    else:
        st.info("Bundled metrics summary is not available.")
    plot = _ASSETS / "detector_comparison.png"
    if plot.exists():
        st.image(str(plot), caption="Detector comparison (precision–recall style export)")


def page_try_request() -> None:
    st.header("Try a request")
    st.caption("Score one request you type. Nothing is stored on the server.")
    modes = _public_modes()
    mode = st.selectbox("Detector mode", modes, index=modes.index("rules_plus_stats"), key="try_mode")
    method = st.selectbox("Method", ["GET", "POST"])
    route = st.selectbox("Route", ["/search", "/login", "/comment"])
    raw_query = st.text_input("Query string", "q=hello")
    body = st.text_area("Body", '{"text": "hello", "author": "ada"}' if route == "/comment" else "")
    if st.button("Score this request"):
        guard = _cached_guard(mode)
        rec = RequestRecord(
            request_id="manual",
            timestamp="t",
            client_id="manual",
            method=method,
            route=route,
            raw_query=raw_query,
            headers={"content-type": "application/json" if body.strip().startswith("{") else "application/x-www-form-urlencoded"},
            body=body,
        )
        score = guard.score(rec)
        render_decision_conclusion(score.to_dict(), None)
        render_request_preview(rec.to_dict())


def page_model_settings() -> None:
    st.header("Model settings")
    st.info("Published demo is read-only. Train with the CLI on your machine.")
    if BASELINE.exists():
        data = json.loads(BASELINE.read_text(encoding="utf-8"))
        st.write("Profile coverage (bundled demo baseline):")
        for k, ep in data.get("endpoints", {}).items():
            st.write(f"- {k}: {ep.get('training_count')} requests")


def main() -> None:
    shell_header(is_public=True)
    sidebar_theme_toggle()
    pages = {
        "Home": lambda: page_home(assets_dir=_ASSETS, is_public=True),
        "Formulas": page_formulas,
        "Try a request": page_try_request,
        "Traffic": page_traffic,
        "Alert inspector": page_alert_inspector,
        "Normal vs unusual": page_normal_vs_unusual,
        "Evaluation": page_evaluation,
        "Model settings": page_model_settings,
    }
    page_names = list(pages.keys())
    choice = st.sidebar.radio("View", page_names, index=0, help="Start on Home for the theory overview.")
    pages[choice]()
