"""Full local Streamlit dashboard (SQLite, train, eval). Set ANOMALY_GUARD_LOCAL=1."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from anomaly_guard.core.decision import AnomalyGuard, DetectorMode
from anomaly_guard.core.schema import GroundTruthMeta, RequestRecord, read_jsonl
from anomaly_guard.evaluation.experiments import MODES, load_dataset, run_mode_experiment
from anomaly_guard.evaluation.splits import (
    training_normal_records,
    validation_legitimate_records,
    validation_ordinary_records,
)
from anomaly_guard.dashboard_ui import (
    explain_legitimate_group_columns,
    explain_mode_metrics_columns,
    page_formulas,
    page_home,
    explain_composition_chart,
    field_display_name,
    render_decision_conclusion,
    render_request_preview,
    shell_header,
    sidebar_theme_toggle,
)
from anomaly_guard.hosted import is_cursor_hosted
from anomaly_guard.simulation.generator import replay_records
from anomaly_guard.storage import EventStore, MemoryEventStore

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "simulated"
DEFAULT_BASELINE = ROOT / "models" / "baseline.json"
DEFAULT_DB = ROOT / "data" / "events.db"
RESULTS_DIR = ROOT / "results"
MAX_REPLAY_REQUESTS = 5_000
HOSTED_MAX_REPLAY_REQUESTS = 500
_WRITE_ROOTS = (ROOT / "models", ROOT / "results", ROOT / "data")
StoreT = Union[EventStore, MemoryEventStore]


def _escape(s: str) -> str:
    return html.escape(s, quote=True)


def _sandbox_path(label: str, user_path: str, *, must_exist: bool = False) -> Path | None:
    """Resolve a path and require it to stay inside the project root."""
    try:
        raw = Path(user_path).expanduser()
        resolved = (ROOT / raw).resolve() if not raw.is_absolute() else raw.resolve()
        resolved.relative_to(ROOT.resolve())
    except (ValueError, OSError):
        st.error(f"{label} must stay inside the project directory.")
        return None
    if must_exist and not resolved.exists():
        st.error(f"{label} not found.")
        return None
    return resolved


def _sandbox_write_path(label: str, user_path: str) -> Path | None:
    resolved = _sandbox_path(label, user_path)
    if resolved is None:
        return None
    if not any(resolved == root or root in resolved.parents for root in _WRITE_ROOTS):
        st.error(f"{label} must be under models/, results/, or data/.")
        return None
    return resolved


def _hosted_demo() -> bool:
    return False


@st.cache_resource
def _sqlite_store() -> EventStore:
    return EventStore(DEFAULT_DB)


def _store() -> StoreT:
    if _hosted_demo():
        if "ag_event_store" not in st.session_state:
            st.session_state.ag_event_store = MemoryEventStore()
        return st.session_state.ag_event_store
    return _sqlite_store()


def _replay_cap() -> int:
    return HOSTED_MAX_REPLAY_REQUESTS if _hosted_demo() else MAX_REPLAY_REQUESTS


def page_traffic() -> None:
    st.header("Traffic")
    if _hosted_demo():
        data_dir = DEFAULT_DATA
        baseline = DEFAULT_BASELINE
        st.caption("Hosted demo uses bundled data and baseline. Replay affects only your session.")
    else:
        data_dir = _sandbox_path("Data directory", st.text_input("Data directory", str(DEFAULT_DATA)))
        baseline = _sandbox_path("Baseline path", st.text_input("Baseline path", str(DEFAULT_BASELINE)))
        if data_dir is None or baseline is None:
            return
    cap = _replay_cap()
    mode = st.selectbox("Detector mode", list(MODES), index=list(MODES).index("rules_plus_stats"))
    shuffle = st.checkbox("Shuffle replay", value=False)
    limit = int(
        st.number_input(
            "Replay limit",
            min_value=1,
            max_value=cap,
            value=min(200, cap),
            help=f"At most {cap} requests per replay.",
        )
    )
    experiment_mode = st.checkbox("Experiment mode (store ground-truth labels)", value=True)

    if st.button("Replay traffic"):
        if not (data_dir / "requests.jsonl").exists():
            st.error("Missing requests.jsonl — run CLI generate first.")
            return
        records = [RequestRecord.from_dict(r) for r in read_jsonl(data_dir / "requests.jsonl")]
        meta_list = [GroundTruthMeta.from_dict(m) for m in read_jsonl(data_dir / "ground_truth.jsonl")]
        meta_by_id = {m.request_id: m for m in meta_list}
        guard = AnomalyGuard(mode=mode)  # type: ignore
        val = validation_legitimate_records(records, meta_list)
        ordinary = validation_ordinary_records(records, meta_list)
        if not baseline.exists():
            if _hosted_demo():
                st.error("Bundled baseline is missing. Redeploy from a build that includes models/.")
                return
            train = training_normal_records(records, meta_list)
            guard.fit(train)
            guard.calibrate(val, threshold_records=ordinary)
        else:
            guard.load_baseline(baseline)
            if guard.load_settings(baseline.with_suffix(".settings.json")):
                st.caption(
                    f"Using saved cutoff {guard.settings.alert_threshold_percentile:.4g} "
                    f"({guard.settings.threshold_scale}, {guard.settings.threshold_population})."
                )
            else:
                guard.calibrate(val, threshold_records=ordinary)
                st.caption(
                    "Saved cutoff was missing or fit for another mode. "
                    "Recalibrated the 1% budget on ordinary validation traffic."
                )
        store = _store()
        store.clear()
        lim = limit
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
    baseline_path = _sandbox_path(
        "Baseline JSON", st.text_input("Baseline JSON", str(DEFAULT_BASELINE)), must_exist=False
    )
    if baseline_path is None:
        return
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

            req_dict = json.loads(row["request_json"])
            render_request_preview(req_dict)

            score = json.loads(row["score_json"])
            gt = json.loads(row["ground_truth_json"]) if row.get("ground_truth_json") else None
            render_decision_conclusion(score, gt)

            req = RequestRecord.from_dict(req_dict)
            config = load_app_config()
            parsed = parse_request(req, config)
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
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(6, 3))
                    xpos = np.arange(len(labels))
                    ax.bar(xpos - 0.2, [base_probs.get(c, 0) for c in labels], 0.4, label="baseline")
                    ax.bar(xpos + 0.2, [obs_frac.get(c, 0) for c in labels], 0.4, label="observed")
                    ax.set_xticks(xpos)
                    ax.set_xticklabels(labels, rotation=45, ha="right")
                    ax.legend()
                    ax.set_title(f"{fname}: character-class mix")
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
    if _hosted_demo():
        st.caption("Hosted demo is read-only. Metrics below are from the last CLI evaluation export.")
    else:
        data_dir = _sandbox_path("Dataset", st.text_input("Dataset", str(DEFAULT_DATA)))
        if data_dir is None:
            return
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
        explain_mode_metrics_columns()
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
        st.subheader("False positives by legitimate group")
        st.caption(
            "The 1% alert cutoff is fit on ordinary validation traffic only. "
            "Difficult legitimate rows are a stress test. They are not the budget."
        )
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
            st.dataframe(pd.DataFrame(group_rows), use_container_width=True)
            explain_legitimate_group_columns()
        st.subheader("P(attack | alert) at three priors")
        st.caption(
            "The file mixture is malicious divided by malicious plus legitimate-family test rows. "
            "A 0.1% prior is the operational base rate this simulator does not have. "
            "The Wilson column uses the upper end of the legitimate false-positive interval."
        )
        posterior_rows = []
        for mode, m in modes.items():
            for row in (m.get("base_rate") or {}).get("rows") or []:
                posterior_rows.append(
                    {
                        "mode": mode,
                        "prior": row.get("name"),
                        "P(attack|alert)": row.get("posterior_at_observed_fpr"),
                        "using Wilson FPR upper": row.get("posterior_at_wilson_fpr_upper"),
                    }
                )
        if posterior_rows:
            st.dataframe(pd.DataFrame(posterior_rows), use_container_width=True)
        contam = summary.get("contamination_combined_stats")
        if contam and "clean" in contam:
            st.subheader("Contamination refit (combined_stats, 1% of train)")
            st.write(contam.get("note"))
            st.json(
                {
                    "clean_recall": contam["clean"].get("recall"),
                    "clean_fpr": contam["clean"].get("false_positive_rate"),
                    "contaminated_recall": contam["contaminated"].get("recall"),
                    "contaminated_fpr": contam["contaminated"].get("false_positive_rate"),
                    "n_replaced": contam.get("n_replaced"),
                }
            )
        st.subheader("XSS alert recall and review rate")
        st.caption(
            "Alert recall of zero is not a silent miss when the review rate is high. "
            "Medium-severity onerror tags are reviews, not alerts."
        )
        xss_rows = []
        for mode, m in modes.items():
            fam = (m.get("per_family") or {}).get("xss") or {}
            if not fam:
                continue
            xss_rows.append(
                {
                    "mode": mode,
                    "alert recall": fam.get("recall"),
                    "review rate": fam.get("review_rate"),
                    "support": fam.get("support"),
                }
            )
        if xss_rows:
            st.dataframe(pd.DataFrame(xss_rows), use_container_width=True)
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
    if _hosted_demo():
        st.info(
            "Hosted demo cannot retrain or overwrite baselines. "
            "Use the CLI locally (`anomaly-guard train`) to refresh models."
        )
        out = DEFAULT_BASELINE
    else:
        data_dir = _sandbox_path(
            "Training data directory", st.text_input("Training data directory", str(DEFAULT_DATA))
        )
        mode: DetectorMode = st.selectbox(
            "Mode", list(MODES), index=list(MODES).index("rules_plus_stats")
        )  # type: ignore
        out = _sandbox_write_path("Baseline output", st.text_input("Baseline output", str(DEFAULT_BASELINE)))
        if data_dir is None or out is None:
            return
        if st.button("Train baseline (explicit retrain)"):
            records = [RequestRecord.from_dict(r) for r in read_jsonl(data_dir / "requests.jsonl")]
            meta = [GroundTruthMeta.from_dict(m) for m in read_jsonl(data_dir / "ground_truth.jsonl")]
            train = training_normal_records(records, meta)
            val = validation_legitimate_records(records, meta)
            ordinary = validation_ordinary_records(records, meta)
            guard = AnomalyGuard(mode=mode)
            guard.fit(train)
            settings = guard.calibrate(val, threshold_records=ordinary)
            out.parent.mkdir(parents=True, exist_ok=True)
            guard.save_baseline(out)
            guard.save_settings(out.with_suffix(".settings.json"))
            st.success(f"Saved baseline v{guard.detector.baseline.version} from {len(train)} samples")
            st.json(settings.__dict__)
    if out.exists():
        data = json.loads(out.read_text(encoding="utf-8"))
        st.write("Profile coverage:")
        for k, ep in data.get("endpoints", {}).items():
            st.write(f"- {k}: {ep.get('training_count')} requests")


def main() -> None:
    shell_header(is_public=False)
    sidebar_theme_toggle()
    assets = ROOT / "anomaly_guard" / "publish_assets"
    if not (assets / "detector_comparison.png").exists():
        assets = ROOT / "results"
    pages = {
        "Home": lambda: page_home(assets_dir=assets if assets.exists() else None, is_public=False),
        "Formulas": page_formulas,
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
