"""Shared Streamlit UI: home, formulas, theme, and chart helpers."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

_DARK_PALETTE = {
    "bg": "#0e1117",
    "panel": "#262730",
    "text": "#fafafa",
    "accent": "#7c9cbf",
    "alert": "#ff6b6b",
    "review": "#ffd166",
    "normal": "#6bcb77",
}


def sidebar_theme_toggle() -> bool:
    if "ag_dark" not in st.session_state:
        st.session_state.ag_dark = True
    dark = st.sidebar.toggle("Dark mode", value=st.session_state.ag_dark)
    st.session_state.ag_dark = dark
    _inject_theme_overrides(dark)
    return dark


def _inject_theme_overrides(dark: bool) -> None:
    if dark:
        st.markdown(
            """
            <style>
            .stApp { background-color: #0e1117; color: #fafafa; }
            [data-testid="stSidebar"] { background-color: #262730; }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            .stApp { background-color: #ffffff; color: #31333f; }
            </style>
            """,
            unsafe_allow_html=True,
        )


def plot_pipeline_diagram(dark: bool) -> None:
    del dark
    st.markdown(
        "**Pipeline:** HTTP request → parse and features → rule engine → "
        "statistical baseline → fuse and threshold → **normal / review / alert**"
    )


def plot_decision_counts(counts: dict[str, int], dark: bool) -> None:
    del dark
    if not any(counts.values()):
        return
    st.caption("Replay decision mix")
    st.bar_chart(pd.DataFrame({"count": counts}))


def plot_mode_metrics(summary: dict[str, Any], dark: bool) -> None:
    del dark
    modes = summary.get("modes") or {}
    if not modes:
        return
    rows = []
    for name, block in modes.items():
        mm = block.get("metrics_malicious") or {}
        rows.append(
            {
                "mode": name,
                "attack recall": mm.get("recall") or 0.0,
                "legit FPR": mm.get("false_positive_rate") or 0.0,
            }
        )
    st.caption("Detector modes (bundled evaluation export)")
    st.bar_chart(pd.DataFrame(rows).set_index("mode"))


def page_home(*, assets_dir: Path | None, is_public: bool) -> None:
    st.header("Home — what Anomaly Guard does")
    st.markdown(
        """
        **Anomaly Guard** scores simulated HTTP requests with two complementary layers:

        1. **Signature rules** catch known attack patterns (SQLi, XSS, traversal, oversize input).
        2. **Statistical baselines** compare each request to profiles learned from *ordinary legitimate*
           training traffic: field length, character composition, optional PAYL / n-gram / Mahalanobis signals.

        The system does **not** claim a confirmed intrusion on every alert. It reports *unusual validation*
        relative to the baseline, with explanations you can audit.
        """
    )
    dark = st.session_state.get("ag_dark", True)
    plot_pipeline_diagram(dark)

    st.markdown(
        """
        ```mermaid
        flowchart LR
          A[Request log] --> B[Parser]
          B --> C[Rules]
          B --> D[Baseline stats]
          C --> E[Fusion]
          D --> E
          E --> F{Threshold}
          F --> G[normal]
          F --> H[review]
          F --> I[alert]
        ```
        """
    )

    st.subheader("Calibration and operating point")
    st.markdown(
        """
        - **Train** only on `legitimate` split rows (same population the baseline describes).
        - **Percentile pools** use all validation legitimate-family rows (including difficult / shift).
        - **Alert cutoff** is fit on **ordinary** validation legitimate only (~1% false-alert budget).
        - Decisions: **normal** (within budget), **review** (elevated but below alert), **alert** (at or above cutoff
          or critical rule hit).
        """
    )

    if assets_dir:
        plot_path = assets_dir / "detector_comparison.png"
        if plot_path.exists():
            st.subheader("Research snapshot")
            st.image(str(plot_path), caption="Mode comparison from offline evaluation (CLI export)")


def page_formulas() -> None:
    st.header("Statistical formulas in use")
    st.markdown(
        r"""
        ### Field length (robust $z$ on $\log(1+\ell)$)

        For observed length $\ell$, baseline median $\tilde{m}$ and MAD on the log scale:

        $$
        z_{\mathrm{len}} = 0.6745 \cdot \frac{\bigl|\log(1+\ell) - \tilde{m}\bigr|}{\mathrm{MAD}}
        $$

        The constant $0.6745$ maps MAD to a normal-scale robust dispersion (consistent with the implementation).

        ### Character composition (multinomial / $\chi^2$ style)

        Categories $c \in \{\mathrm{alpha}, \mathrm{digit}, \mathrm{space}, \ldots\}$ with baseline probabilities
        $\hat{p}_c$ and counts $n_c$ on length $L$:

        $$
        \chi^2_{\mathrm{comp}} = \sum_c \frac{(n_c - L\hat{p}_c)^2}{L\hat{p}_c + \varepsilon}
        $$

        Cell contributions drive which symbols look most unlike training.

        ### Mahalanobis distance (feature vector $\mathbf{x}$)

        $$
        d_M = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})}
        $$

        Used in `lda_only` / fused modes when the baseline stores $\boldsymbol{\mu}$ and $\Sigma$.

        ### Percentile calibration

        Each component yields a **validation percentile**: the share of calibration scores strictly below the raw score, in percent. A raw score above every calibration value is 100. Ties at the pool maximum stay below 100, which is what lets entropy alert. Modes aggregate with

        $$
        p_{\mathrm{agg}} = \max_j p_j \quad\text{or}\quad \frac{1}{k}\sum_j p_j \quad\text{or}\quad \text{2nd-largest } p_j
        $$

        The **alert threshold** is fit on ordinary validation traffic for a 1% false-alert budget. A decision alerts only when the aggregate is **strictly above** that cutoff. The threshold sweep uses the same comparison.

        ### Wilson upper bound (legitimate FPR reporting)

        For $k$ alerts in $n$ ordinary legitimate validation rows:

        $$
        \mathrm{WilsonUpper}(k,n) = \frac{\hat{p} + \frac{z^2}{2n} + z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + z^2/n}
        $$

        On the evaluation export this is the **upper end of a Wilson score interval** for the
        **observed alert rate** among **ordinary legitimate validation** rows (same $k$, $n$
        population as calibration for the 1% budget), not a post-hoc test-set recount.

        ### Posterior probability (reporting only)

        With attack prior $\pi$, malicious recall $\mathrm{TPR}$, and legitimate false-positive
        rate $\mathrm{FPR}$ on held-out test traffic:

        $$
        P(\mathrm{attack}\mid\mathrm{alert}) \approx \frac{\pi \cdot \mathrm{TPR}}{\pi \cdot \mathrm{TPR} + (1-\pi)\cdot \mathrm{FPR}}
        $$

        This is a **posterior probability**, not posterior odds. TPR/FPR come from held-out test
        metrics; priors are scenario assumptions, not measured live.
        """
    )


_FIELD_HELP: dict[str, str] = {
    "text": "Comment body (`text` on POST /comment).",
    "author": "Display name or handle (`author` on POST /comment).",
    "q": "Search query string (GET /search).",
    "username": "Login username (POST /login).",
    "password": "Login password (POST /login).",
}

_COMPOSITION_HELP = (
    "Each bar is the **fraction of characters** in that symbol class: letter, digit, whitespace, "
    "quote, bracket, slash, or other. **Baseline** = training on ordinary legitimate traffic; "
    "**observed** = this request after parsing."
)


def field_display_name(field_name: str, route: str) -> str:
    hint = _FIELD_HELP.get(field_name, f"Parsed input field `{field_name}` for `{route}`.")
    return hint


def explain_composition_chart() -> None:
    st.caption(_COMPOSITION_HELP)


def render_request_preview(req: dict[str, Any]) -> None:
    st.subheader("Request")
    body = str(req.get("body") or "")
    if len(body) > 3000:
        body = body[:3000] + "\n… [truncated for display]"
    text = html.escape(
        f"{req.get('method', 'GET')} {req.get('route', '')}\n"
        f"Query: {req.get('raw_query', '')}\n"
        f"Body: {body}",
        quote=True,
    )
    st.code(text, language="text")
    st.caption("Rendered with HTML escaping. This is what you are comparing to the baseline.")


def render_decision_conclusion(score: dict[str, Any], ground_truth: dict[str, Any] | None) -> None:
    """Show detector outcome and reasoning before baseline comparison charts."""
    decision = str(score.get("decision", "unknown"))
    headline = {
        "normal": "Normal — within expected variation",
        "review": "Review — unusual but below alert cutoff",
        "alert": "Alert — exceeds unusualness threshold or critical rule",
    }.get(decision, decision)

    if ground_truth:
        sg = str(ground_truth.get("scenario_group", ""))
        if "malicious" in sg or sg == "hard_malicious_intent":
            sim = "Simulated **attack** traffic"
        elif sg == "benign_invalid":
            sim = "Simulated **invalid** request (not an attack pattern)"
        else:
            sim = "Simulated **benign** traffic"
        st.subheader("Conclusion")
        st.markdown(f"**Detector:** {headline} (`{decision}`)")
        st.markdown(f"**Ground truth (demo labels only):** {sim} — `{sg}`")
    else:
        st.subheader("Conclusion")
        st.markdown(f"**Detector:** {headline} (`{decision}`)")

    stat = score.get("statistical") or {}
    agg = stat.get("aggregate_percentile")
    if agg is not None:
        st.markdown(f"**Fused unusualness:** {float(agg):.1f}th percentile vs validation pool")

    st.markdown("**What drove the decision**")
    rules = score.get("triggered_rules") or []
    if rules:
        for r in rules:
            st.write(f"- **Rule {r.get('rule_id')}** ({r.get('severity')}): {r.get('explanation')}")
    comps = sorted(
        (stat.get("components") or []),
        key=lambda c: float(c.get("percentile") or 0),
        reverse=True,
    )[:4]
    for c in comps:
        pct = c.get("percentile")
        pct_s = f"{float(pct):.1f}th pct" if pct is not None else "n/a"
        st.write(
            f"- **{c.get('component')}** on `{c.get('field') or '—'}`: {c.get('detail')} ({pct_s})"
        )
    for e in score.get("explanations") or []:
        if e not in {r.get("explanation") for r in rules}:
            st.write(f"- {e}")

    st.divider()
    st.subheader("Analysis vs baseline")


def explain_mode_metrics_columns() -> None:
    st.markdown(
        """
        **Column guide (test split, alert decision unless noted)**

        - **mode** — Detector variant in the ablation (rules-only, single stat, or fused).
        - **precision (attack)** — Of requests labeled attack that got **alert**, how many were truly attacks.
        - **recall (attack)** — Fraction of attack test rows that received an **alert** (main detection rate).
        - **f1 (attack)** — Harmonic mean of attack precision and recall on **alert**.
        - **fpr (legit)** — Fraction of legitimate-family test rows (ordinary + difficult + shift) that got **alert**.
        - **AP** — Ranking-based average precision on the test split. Hybrid modes use
          max(statistic percentile, rule severity); see **AP (stat component)** for stats-only.
        - **AP (stat component)** — Statistic percentile only (hybrid modes). Shown when exported.
        """
    )


def explain_legitimate_group_columns() -> None:
    st.markdown(
        """
        **Column guide**

        - **mode** — Same detector variant as the summary table above.
        - **ordinary** — Alert rate on test **ordinary** legitimate traffic (closest to training distribution).
        - **difficult** — Alert rate on **difficult_legitimate** test rows (still benign, awkward text).
        """
    )


def shell_header(*, is_public: bool) -> None:
    st.set_page_config(page_title="Anomaly Guard", layout="wide", initial_sidebar_state="expanded")
    st.title("Anomaly Guard")
    if is_public:
        st.caption("Explainable HTTP anomaly detection — public read-only demo on bundled sample traffic.")
        st.warning("Session-scoped replay only. No shared database or model writes.")
    else:
        st.caption(
            "Explainable HTTP request anomaly detector — alerts describe unusual validation, not confirmed intrusions."
        )
