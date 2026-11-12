"""Shared Streamlit UI: home, formulas, theme, and chart helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
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


def matplotlib_style(dark: bool) -> None:
    if dark:
        plt.style.use("dark_background")
        plt.rcParams.update({
            "figure.facecolor": _DARK_PALETTE["bg"],
            "axes.facecolor": _DARK_PALETTE["panel"],
            "text.color": _DARK_PALETTE["text"],
            "axes.labelcolor": _DARK_PALETTE["text"],
            "xtick.color": _DARK_PALETTE["text"],
            "ytick.color": _DARK_PALETTE["text"],
        })
    else:
        plt.style.use("default")


def plot_pipeline_diagram(dark: bool) -> None:
    matplotlib_style(dark)
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [
        (0.3, "HTTP\nrequest", "#4a5568"),
        (2.2, "Parse +\nfeatures", "#5a67d8"),
        (4.1, "Rule\nengine", "#d69e2e"),
        (6.0, "Statistical\nbaseline", "#38b2ac"),
        (7.9, "Fuse +\nthreshold", "#ed64a6"),
        (9.0, "normal /\nreview / alert", "#48bb78"),
    ]
    for x, label, color in boxes:
        ax.add_patch(plt.Rectangle((x, 0.8), 1.5, 1.2, facecolor=color, alpha=0.85, edgecolor="white"))
        ax.text(x + 0.75, 1.4, label, ha="center", va="center", fontsize=9, color="white", weight="bold")
    for x in range(4):
        ax.annotate(
            "",
            xy=(boxes[x + 1][0], 1.4),
            xytext=(boxes[x][0] + 1.5, 1.4),
            arrowprops=dict(arrowstyle="->", color=_DARK_PALETTE["text"] if dark else "#333", lw=1.5),
        )
    ax.set_title("Anomaly Guard decision pipeline", fontsize=11, pad=8)
    st.pyplot(fig)
    plt.close(fig)


def plot_decision_counts(counts: dict[str, int], dark: bool) -> None:
    if not any(counts.values()):
        return
    matplotlib_style(dark)
    labels = list(counts.keys())
    values = [counts[k] for k in labels]
    colors = [_DARK_PALETTE.get(k, _DARK_PALETTE["accent"]) for k in labels]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(values, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
    ax.set_title("Replay decision mix")
    st.pyplot(fig)
    plt.close(fig)


def plot_mode_metrics(summary: dict[str, Any], dark: bool) -> None:
    modes = summary.get("modes") or {}
    if not modes:
        return
    matplotlib_style(dark)
    names, recalls, fprs = [], [], []
    for name, block in modes.items():
        mm = block.get("metrics_malicious") or {}
        names.append(name.replace("_", "\n"))
        recalls.append(mm.get("recall") or 0.0)
        fprs.append((block.get("metrics_malicious") or {}).get("false_positive_rate") or 0.0)
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w / 2, recalls, w, label="Attack recall", color=_DARK_PALETTE["accent"])
    ax.bar(x + w / 2, fprs, w, label="Legit FPR", color=_DARK_PALETTE["alert"])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Detector modes (bundled evaluation export)")
    st.pyplot(fig)
    plt.close(fig)


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

        Each component yields a **validation percentile** $p_j \in [0,100]$. Modes aggregate with

        $$
        p_{\mathrm{agg}} = \max_j p_j \quad\text{or}\quad \frac{1}{k}\sum_j p_j \quad\text{or}\quad \text{2nd-largest } p_j
        $$

        The **alert threshold** is the validation percentile at target false-alert rate (1% on ordinary traffic).

        ### Wilson upper bound (legitimate FPR reporting)

        For $k$ alerts in $n$ ordinary legitimate validation rows:

        $$
        \mathrm{WilsonUpper}(k,n) = \frac{\hat{p} + \frac{z^2}{2n} + z\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + z^2/n}
        $$

        Used when discussing conservative false-positive rates in evaluation exports.

        ### Posterior odds (reporting only)

        With attack prior $\pi$ and alert rate $\mathrm{FPR}$ on legitimate traffic:

        $$
        P(\mathrm{attack}\mid\mathrm{alert}) \approx \frac{\pi \cdot \mathrm{TPR}}{\pi \cdot \mathrm{TPR} + (1-\pi)\cdot \mathrm{FPR}}
        $$

        (TPR/FPR taken from held-out test metrics; priors are scenario assumptions, not measured live.)
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
