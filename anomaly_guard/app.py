"""Streamlit dashboard: traffic, alerts, evaluation, settings."""

from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from anomaly_guard.core.decision import AnomalyGuard, DetectorMode
from anomaly_guard.core.schema import GroundTruthMeta, RequestRecord, read_jsonl
from anomaly_guard.evaluation.experiments import MODES, load_dataset, run_mode_experiment
from anomaly_guard.evaluation.splits import training_normal_records, validation_legitimate_records
from anomaly_guard.simulation.generator import replay_records
from anomaly_guard.storage import EventStore

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "simulated"
DEFAULT_BASELINE = ROOT / "models" / "baseline.json"
DEFAULT_DB = ROOT / "data" / "events.db"
RESULTS_DIR = ROOT / "results"


def _escape(s: str) -> str:
    return html.escape(s, quote=True)


@st.cache_resource
def _store() -> EventStore:
    return EventStore(DEFAULT_DB)


def page_traffic() -> None:
    st.header("Traffic")
    data_dir = Path(st.text_input("Data directory", str(DEFAULT_DATA)))
    baseline = Path(st.text_input("Baseline path", str(DEFAULT_BASELINE)))
    mode = st.selectbox("Detector mode", list(MODES), index=list(MODES).index("rules_plus_stats"))
    shuffle = st.checkbox("Shuffle replay", value=False)
    limit = st.number_input("Limit (0 = all)", min_value=0, value=200)
    experiment_mode = st.checkbox("Experiment mode (store ground-truth labels)", value=True)

    if st.button("Replay traffic"):
        if not (data_dir / "requests.jsonl").exists():
            st.error("Missing requests.jsonl — run CLI generate first.")
            return
        records = [RequestRecord.from_dict(r) for r in read_jsonl(data_dir / "requests.jsonl")]
        meta_list = [GroundTruthMeta.from_dict(m) for m in read_jsonl(data_dir / "ground_truth.jsonl")]
        meta_by_id = {m.request_id: m for m in meta_list}
        guard = AnomalyGuard(mode=mode)  # type: ignore
        if baseline.exists():
            guard.load_baseline(baseline)
        else:
            train = training_normal_records(records, meta_list)
            val = validation_legitimate_records(records, meta_list)
            guard.fit(train)
            guard.calibrate(val)
        store = _store()
        store.clear()
        lim = None if limit == 0 else int(limit)
        counts = {"normal": 0, "review": 0, "alert": 0}
        parse_fail = 0
        for rec in replay_records(records, shuffle=shuffle, limit=lim):
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
    else:
        st.info("No events yet. Replay traffic to populate the store.")


def page_alert_inspector() -> None:
    st.header("Alert inspector")
    store = _store()
    alerts = store.search(decision="alert", limit=50) + store.search(decision="review", limit=50)
    if not alerts:
        st.info("No alerts or reviews in the event store.")
        return
    ids = [a["request_id"] for a in alerts]
    selected = st.selectbox("Select request", ids)
    row = store.get(selected)
    if not row:
        return
    req = json.loads(row["request_json"])
    score = json.loads(row["score_json"])
    st.markdown("**Safe rendered request (escaped)**")
    st.code(
        _escape(f"{req['method']} {req['route']}\nQuery: {req.get('raw_query','')}\nBody: {req.get('body','')[:2000]}"),
        language="text",
    )
    st.markdown(f"**Decision:** `{score['decision']}`")
    if row.get("ground_truth_json"):
        gt = json.loads(row["ground_truth_json"])
        st.markdown(f"*Experiment label:* `{gt['scenario_group']}` / `{gt.get('attack_family')}`")
    st.subheader("Rules")
    for r in score.get("triggered_rules", []):
        st.write(f"- **{r['rule_id']}** ({r['severity']}): {r['explanation']}")
    st.subheader("Statistical components")
    stat = score.get("statistical") or {}
    for c in stat.get("components", []):
        st.write(f"- {c['component']} / {c['field']}: {c['detail']} (percentile={c.get('percentile')})")
    st.subheader("Explanations")
    for e in score.get("explanations", []):
        st.write(f"- {e}")


def page_normal_vs_unusual() -> None:
    st.header("Normal versus unusual")
    baseline_path = Path(st.text_input("Baseline JSON", str(DEFAULT_BASELINE)))
    store = _store()
    events = store.search(limit=200)
    req_ids = [e["request_id"] for e in events] if events else []
    selected = st.selectbox("Compare request (from event store)", [""] + req_ids) if req_ids else ""
    if not baseline_path.exists():
        st.warning("Train a baseline first (CLI or Model settings page).")
        return
    data = json.loads(baseline_path.read_text(encoding="utf-8"))
    st.write(f"Version **{data.get('version')}**, training count **{data.get('training_count')}**")

    if selected:
        row = store.get(selected)
        if row:
            from anomaly_guard.core.decision import AnomalyGuard
            from anomaly_guard.core.features import CATEGORY_BINS, _category_fractions
            from anomaly_guard.core.parser import parse_request
            from anomaly_guard.core.schema import RequestRecord, load_app_config

            req = RequestRecord.from_dict(json.loads(row["request_json"]))
            config = load_app_config()
            parsed = parse_request(req, config)
            guard = AnomalyGuard()
            if baseline_path.exists():
                guard.load_baseline(baseline_path)
            st.subheader("Selected request vs baseline")
            key = f"{req.method.upper()} {req.route}"
            ep = data.get("endpoints", {}).get(key, {})
            for fname, fp in ep.get("fields", {}).items():
                text = parsed.field_text(fname)
                length = fp.get("length") or {}
                obs_len = len(text)
                med = length.get("median", 0)
                mad = length.get("mad", 0)
                st.write(
                    f"**{fname}** length: observed={obs_len}, baseline median={med:.1f}, MAD={mad:.1f}"
                )
                comp = fp.get("composition") or {}
                base_probs = comp.get("category_probs") or {}
                if text:
                    obs_frac = _category_fractions(text)
                    labels = sorted(set(base_probs) | set(CATEGORY_BINS))
                    import matplotlib.pyplot as plt

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


def page_evaluation() -> None:
    st.header("Evaluation")
    data_dir = Path(st.text_input("Dataset", str(DEFAULT_DATA)))
    if st.button("Run quick evaluation (rules_plus_stats only)"):
        records, meta = load_dataset(data_dir)
        with st.spinner("Running..."):
            res, _ = run_mode_experiment(records, meta, "rules_plus_stats", RESULTS_DIR / "models")
        st.json(res.to_dict())
    metrics_path = RESULTS_DIR / "metrics_summary.json"
    if metrics_path.exists():
        summary = json.loads(metrics_path.read_text(encoding="utf-8"))
        st.subheader("Saved experiment summary")
        modes = summary.get("modes", {})
        rows = []
        for mode, m in modes.items():
            mm = m.get("metrics_malicious", {})
            inv = m.get("metrics_benign_invalid", {})
            rows.append({
                "mode": mode,
                "precision (attack)": mm.get("precision"),
                "recall (attack)": mm.get("recall"),
                "f1 (attack)": mm.get("f1"),
                "fpr (legit)": mm.get("false_positive_rate"),
                "invalid rejection": inv.get("rejection_rate"),
                "AP": m.get("average_precision"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.subheader("Attack slices and the 1% false-positive read")
        st.caption(
            "Slice numbers are alert recall. The 1% columns are a descriptive read of the "
            "test threshold sweep. They do not replace the validation-fitted cutoff."
        )
        slice_rows = []
        for mode, m in modes.items():
            sl = m.get("attack_slices") or {}
            op = m.get("operating_point_1pct") or {}
            slice_rows.append(
                {
                    "mode": mode,
                    "signature_easy": (sl.get("signature_easy") or {}).get("recall"),
                    "rule_evasive": (sl.get("rule_evasive") or {}).get("recall"),
                    "size_distribution": (sl.get("size_distribution") or {}).get("recall"),
                    "recall at test FPR ≤ 1%": op.get("recall_malicious"),
                    "FPR at that point": op.get("fpr_legitimate"),
                    "sweep selection": op.get("selection"),
                }
            )
        st.dataframe(pd.DataFrame(slice_rows), use_container_width=True)
        xss = summary.get("xss_component_profile")
        if xss:
            st.subheader("XSS under full_stats")
            st.json(xss)
        held = summary.get("held_out_families")
        if held:
            st.subheader("Held-out families (alert recall)")
            held_rows = []
            for fam, block in (held.get("families") or {}).items():
                for mode_name, stats in (block.get("modes") or {}).items():
                    held_rows.append(
                        {
                            "family": fam,
                            "support": block.get("support"),
                            "mode": mode_name,
                            "alert recall": stats.get("alert_recall"),
                            "review rate": stats.get("review_rate"),
                        }
                    )
            if held_rows:
                st.dataframe(pd.DataFrame(held_rows), use_container_width=True)
        plot = RESULTS_DIR / "detector_comparison.png"
        if plot.exists():
            st.image(str(plot), caption="Detector comparison")
    else:
        st.info("Run `anomaly-guard evaluate` to export full ablation results.")


def page_model_settings() -> None:
    st.header("Model settings")
    data_dir = Path(st.text_input("Training data directory", str(DEFAULT_DATA)))
    mode: DetectorMode = st.selectbox(
        "Mode", list(MODES), index=list(MODES).index("rules_plus_stats")
    )  # type: ignore
    out = Path(st.text_input("Baseline output", str(DEFAULT_BASELINE)))
    if st.button("Train baseline (explicit retrain)"):
        records = [RequestRecord.from_dict(r) for r in read_jsonl(data_dir / "requests.jsonl")]
        meta = [GroundTruthMeta.from_dict(m) for m in read_jsonl(data_dir / "ground_truth.jsonl")]
        train = training_normal_records(records, meta)
        val = validation_legitimate_records(records, meta)
        guard = AnomalyGuard(mode=mode)
        guard.fit(train)
        settings = guard.calibrate(val)
        out.parent.mkdir(parents=True, exist_ok=True)
        guard.save_baseline(out)
        st.success(f"Saved baseline v{guard.detector.baseline.version} from {len(train)} samples")
        st.json(settings.__dict__)
    if out.exists():
        data = json.loads(out.read_text(encoding="utf-8"))
        st.write("Profile coverage:")
        for k, ep in data.get("endpoints", {}).items():
            st.write(f"- {k}: {ep.get('training_count')} requests")


def main() -> None:
    st.set_page_config(page_title="Anomaly Guard", layout="wide")
    st.title("Anomaly Guard")
    st.caption("Local explainable HTTP request anomaly detector — alerts describe unusual validation, not confirmed intrusions.")
    pages = {
        "Traffic": page_traffic,
        "Alert inspector": page_alert_inspector,
        "Normal vs unusual": page_normal_vs_unusual,
        "Evaluation": page_evaluation,
        "Model settings": page_model_settings,
    }
    choice = st.sidebar.radio("View", list(pages.keys()))
    pages[choice]()


if __name__ == "__main__":
    main()
