"""Read-only Streamlit dashboard for Cursor publish (session memory, bundled assets)."""

from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from anomaly_guard.core.decision import AnomalyGuard
from anomaly_guard.core.schema import GroundTruthMeta, RequestRecord, read_jsonl
from anomaly_guard.evaluation.experiments import MODES
from anomaly_guard.evaluation.splits import validation_legitimate_records, validation_ordinary_records
from anomaly_guard.simulation.generator import replay_records
from anomaly_guard.storage import MemoryEventStore

_PKG = Path(__file__).resolve().parent
_ASSETS = _PKG / "publish_assets"
_DEMO = _ASSETS / "demo"
BASELINE = _DEMO / "baseline.json"
MAX_REPLAY = 200


def _escape(s: str) -> str:
    return html.escape(s, quote=True)


def _store() -> MemoryEventStore:
    if "ag_event_store" not in st.session_state:
        st.session_state.ag_event_store = MemoryEventStore()
    return st.session_state.ag_event_store


def page_traffic() -> None:
    st.header("Traffic")
    st.caption("Bundled sample traffic only. Replay affects your browser session, not other visitors.")
    mode = st.selectbox("Detector mode", list(MODES), index=list(MODES).index("rules_plus_stats"))
    shuffle = st.checkbox("Shuffle replay", value=False)
    limit = int(
        st.number_input(
            "Replay limit",
            min_value=1,
            max_value=MAX_REPLAY,
            value=min(100, MAX_REPLAY),
        )
    )
    experiment_mode = st.checkbox("Experiment mode (store ground-truth labels)", value=True)

    if st.button("Replay traffic"):
        if not BASELINE.exists():
            st.error("Demo baseline missing from deployment package.")
            return
        records = [RequestRecord.from_dict(r) for r in read_jsonl(_DEMO / "requests.jsonl")]
        meta_list = [GroundTruthMeta.from_dict(m) for m in read_jsonl(_DEMO / "ground_truth.jsonl")]
        meta_by_id = {m.request_id: m for m in meta_list}
        guard = AnomalyGuard(mode=mode)  # type: ignore[arg-type]
        guard.load_baseline(BASELINE)
        settings_path = BASELINE.with_suffix(".settings.json")
        if settings_path.exists():
            guard.load_settings(settings_path)
        else:
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
        st.success(f"Processed {sum(counts.values())} requests. Parsing flags: {parse_fail}")
        st.json(counts)

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
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "request_id": r["request_id"][:8],
                        "route": r["route"],
                        "decision": r["decision"],
                        "timestamp": r["timestamp"],
                    }
                    for r in rows
                ]
            ),
            use_container_width=True,
        )
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

            req = RequestRecord.from_dict(json.loads(row["request_json"]))
            parsed = parse_request(req, load_app_config())
            key = f"{req.method.upper()} {req.route}"
            ep = data.get("endpoints", {}).get(key, {})
            for fname, fp in ep.get("fields", {}).items():
                text = parsed.field_text(fname)
                length = fp.get("length") or {}
                st.write(
                    f"**{fname}** length: observed={len(text)}, baseline median={length.get('median', 0):.1f}, "
                    f"MAD={length.get('mad', 0):.1f}"
                )
                base_probs = (fp.get("composition") or {}).get("category_probs") or {}
                if text:
                    obs_frac = _category_fractions(text)
                    labels = sorted(set(base_probs) | set(CATEGORY_BINS))
                    fig, ax = plt.subplots(figsize=(6, 3))
                    xpos = np.arange(len(labels))
                    ax.bar(xpos - 0.2, [base_probs.get(c, 0) for c in labels], 0.4, label="baseline")
                    ax.bar(xpos + 0.2, [obs_frac.get(c, 0) for c in labels], 0.4, label="observed")
                    ax.set_xticks(xpos)
                    ax.set_xticklabels(labels, rotation=45, ha="right")
                    ax.legend()
                    ax.set_title(f"{fname} character composition")
                    st.pyplot(fig)
                    plt.close(fig)

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
        inv = m.get("metrics_benign_invalid", {})
        rows.append(
            {
                "mode": mode,
                "precision (attack)": mm.get("precision"),
                "recall (attack)": mm.get("recall"),
                "f1 (attack)": mm.get("f1"),
                "fpr (legit)": mm.get("false_positive_rate"),
                "invalid rejection": inv.get("rejection_rate"),
                "AP": m.get("average_precision"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
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
                "distribution shift": (groups.get("distribution_shift") or {}).get("alert_rate"),
            }
        )
    if group_rows:
        st.subheader("False positives by legitimate group")
        st.dataframe(pd.DataFrame(group_rows), use_container_width=True)


def page_evaluation() -> None:
    st.header("Evaluation")
    st.caption("Pre-computed metrics from the development CLI run (read-only).")
    metrics_path = _ASSETS / "metrics_summary.json"
    if metrics_path.exists():
        _render_metrics_summary(json.loads(metrics_path.read_text(encoding="utf-8")))
    else:
        st.info("Bundled metrics summary is not available.")
    plot = _ASSETS / "detector_comparison.png"
    if plot.exists():
        st.image(str(plot), caption="Detector comparison")


def page_model_settings() -> None:
    st.header("Model settings")
    st.info("Published demo is read-only. Train with the CLI on your machine.")
    if BASELINE.exists():
        data = json.loads(BASELINE.read_text(encoding="utf-8"))
        st.write("Profile coverage (bundled demo baseline):")
        for k, ep in data.get("endpoints", {}).items():
            st.write(f"- {k}: {ep.get('training_count')} requests")


def main() -> None:
    st.set_page_config(page_title="Anomaly Guard", layout="wide")
    st.title("Anomaly Guard")
    st.caption(
        "Public demo — explainable HTTP anomaly scores on a small bundled sample. "
        "Not a production WAF."
    )
    st.warning("Read-only: per-session replay only. No shared database or model writes.")
    pages = {
        "Traffic": page_traffic,
        "Alert inspector": page_alert_inspector,
        "Normal vs unusual": page_normal_vs_unusual,
        "Evaluation": page_evaluation,
        "Model settings": page_model_settings,
    }
    choice = st.sidebar.radio("View", list(pages.keys()))
    pages[choice]()
