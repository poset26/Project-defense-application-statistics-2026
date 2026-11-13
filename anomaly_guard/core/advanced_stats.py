"""PAYL, n-gram, OLS residual, and Mahalanobis profiles for interpretable detection."""

from __future__ import annotations

# Near-singular OLS fits can yield ~1e-16 residual std; dividing by that amplifying noise.
OLS_RESIDUAL_STD_FLOOR = 1e-3
OLS_ABSOLUTE_RESID_TOLERANCE = 1e-9

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from anomaly_guard.core.features import CATEGORY_BINS, RequestFeatures, _category_fractions
from anomaly_guard.core.parser import ParsedRequest

MAD_SCALE = 0.6745
PAYL_SMOOTH = 0.001
NGRAM_K = 2
NGRAM_TOP = 200


@dataclass
class LengthBandComposition:
    band_index: int
    low: float
    high: float
    category_probs: dict[str, float]
    merged_categories: list[str]


@dataclass
class PaylProfile:
    mean_freq: list[float]
    std_freq: list[float]


@dataclass
class FieldExtendedProfile:
    length_band_compositions: list[LengthBandComposition]
    length_band_edges: list[float]
    payl: PaylProfile | None = None
    ngram_median: dict[str, float] = field(default_factory=dict)
    ngram_mad: dict[str, float] = field(default_factory=dict)
    ols_coef: list[float] | None = None
    ols_residual_std: float = 1.0


@dataclass
class EndpointExtendedProfile:
    method: str
    route: str
    fields: dict[str, FieldExtendedProfile] = field(default_factory=dict)
    feature_mean: list[float] | None = None
    feature_cov_inv: list[list[float]] | None = None


def _byte_histogram(text: str) -> np.ndarray:
    h = np.zeros(256, dtype=float)
    raw = text.encode("utf-8", errors="replace")
    if not raw:
        return h
    for b in raw:
        h[b] += 1.0
    return h / len(raw)


def _bigram_counts(text: str) -> dict[str, int]:
    raw = text.encode("utf-8", errors="replace")
    counts: dict[str, int] = {}
    if len(raw) < NGRAM_K:
        return counts
    for i in range(len(raw) - NGRAM_K + 1):
        g = raw[i : i + NGRAM_K].hex()
        counts[g] = counts.get(g, 0) + 1
    return counts


def _length_band_index(edges: list[float], length: float) -> int:
    if not edges:
        return 0
    for i in range(len(edges) - 1):
        if edges[i] <= length < edges[i + 1]:
            return i
    return len(edges) - 2


def fit_endpoint_extended(
    items: list[tuple[RequestFeatures, ParsedRequest]],
    field_names: list[str],
    feature_vectors: list[list[float]],
) -> EndpointExtendedProfile:
    ep = EndpointExtendedProfile(method="", route="", fields={})
    if not items:
        return ep

    if len(feature_vectors) >= 5:
        X = np.array(feature_vectors, dtype=float)
        mu = np.mean(X, axis=0)
        cov = np.cov(X.T)
        if cov.ndim == 0:
            cov = np.array([[float(cov) + PAYL_SMOOTH]])
        cov = cov + np.eye(cov.shape[0]) * PAYL_SMOOTH
        try:
            inv = np.linalg.inv(cov)
            ep.feature_mean = mu.tolist()
            ep.feature_cov_inv = inv.tolist()
        except np.linalg.LinAlgError:
            pass

    for fname in field_names:
        lengths: list[float] = []
        texts: list[str] = []
        rows_x: list[list[float]] = []
        rows_y: list[float] = []
        payl_hists: list[np.ndarray] = []
        ngram_docs: list[dict[str, int]] = []

        for feats, parsed in items:
            ff = feats.field_features.get(fname)
            if ff is None or ff.is_empty:
                continue
            text = parsed.field_text(fname)
            lengths.append(float(ff.char_length))
            texts.append(text)
            y = math.log1p(ff.char_length)
            rows_y.append(y)
            rows_x.append(
                [
                    1.0,
                    ff.entropy,
                    ff.distinct_char_fraction,
                    ff.category_fractions.get("digit", 0.0),
                    ff.category_fractions.get("other", 0.0),
                ]
            )
            payl_hists.append(_byte_histogram(text))
            ngram_docs.append(_bigram_counts(text))

        fep = FieldExtendedProfile(length_band_edges=[], length_band_compositions=[])
        if len(lengths) < 5:
            ep.fields[fname] = fep
            continue

        log_lens = [math.log1p(x) for x in lengths]
        qs = [0.0, 0.25, 0.5, 0.75, 1.0]
        edges = [float(np.quantile(lengths, q)) for q in qs]
        edges[-1] = edges[-1] + 1.0
        fep.length_band_edges = edges

        for bi in range(len(edges) - 1):
            lo, hi = edges[bi], edges[bi + 1]
            band_texts = [t for t, ln in zip(texts, lengths) if lo <= ln < hi]
            if len(band_texts) < 3:
                continue
            cat_counts = {c: 0 for c in CATEGORY_BINS}
            for t in band_texts:
                fr = _category_fractions(t)
                for c in CATEGORY_BINS:
                    cat_counts[c] += int(round(fr[c] * max(len(t), 1)))
            total = sum(cat_counts.values()) or 1
            probs = {k: v / total for k, v in cat_counts.items()}
            fep.length_band_compositions.append(
                LengthBandComposition(
                    band_index=bi,
                    low=lo,
                    high=hi,
                    category_probs=probs,
                    merged_categories=list(CATEGORY_BINS),
                )
            )

        if payl_hists:
            stack = np.stack(payl_hists, axis=0)
            mean_f = np.mean(stack, axis=0)
            std_f = np.std(stack, axis=0) + PAYL_SMOOTH
            fep.payl = PaylProfile(mean_freq=mean_f.tolist(), std_freq=std_f.tolist())

        agg: dict[str, list[int]] = {}
        for doc in ngram_docs:
            for g, c in doc.items():
                agg.setdefault(g, []).append(c)
        for g, vals in sorted(agg.items(), key=lambda x: -len(x[1]))[:NGRAM_TOP]:
            arr = np.array(vals, dtype=float)
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med)))
            fep.ngram_median[g] = med
            fep.ngram_mad[g] = mad if mad > 0 else 0.5

        if len(rows_y) >= 8:
            paired = list(zip(rows_x, rows_y))
            X = np.array([row[0] for row in paired], dtype=float)
            y = np.array([row[1] for row in paired], dtype=float)
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
            fep.ols_coef = beta.tolist()
            std = float(np.std(resid, ddof=1)) if len(resid) > 1 else 1.0
            fep.ols_residual_std = max(std, OLS_RESIDUAL_STD_FLOOR)

        ep.fields[fname] = fep
    return ep


def score_stratified_composition(
    text: str,
    fep: FieldExtendedProfile,
    char_len: float,
) -> tuple[float | None, str]:
    if len(text) < 8 or not fep.length_band_compositions:
        return None, "no length band baseline"
    bi = _length_band_index(fep.length_band_edges, char_len)
    band = next((b for b in fep.length_band_compositions if b.band_index == bi), None)
    if band is None:
        return None, "length band without composition profile"
    from anomaly_guard.core.stat_lite import chisquare

    n = len(text)
    obs_frac = _category_fractions(text)
    obs = {c: int(round(obs_frac.get(c, 0) * n)) for c in CATEGORY_BINS}
    diff = n - sum(obs.values())
    if diff:
        obs["other"] = obs.get("other", 0) + diff
    expected, observed = [], []
    for cat in CATEGORY_BINS:
        p = band.category_probs.get(cat, 0.0)
        expected.append(p * n + 0.5)
        observed.append(float(obs.get(cat, 0)))
    exp = np.array(expected, dtype=float)
    obs_arr = np.array(observed, dtype=float)
    if float(exp.sum()) <= 0 or float(obs_arr.sum()) <= 0:
        return 1e6, "composition counts were not finite"
    exp = np.maximum(exp * (obs_arr.sum() / exp.sum()), 1e-6)
    stat, pval = chisquare(f_obs=obs_arr, f_exp=exp)
    if not np.isfinite(stat):
        stat = 1e6
        pval = 0.0
    contrib = ((obs_arr - exp) ** 2 / np.maximum(exp, 0.25)).tolist()
    top = sorted(
        zip(CATEGORY_BINS, contrib), key=lambda x: -x[1]
    )[:2]
    detail = f"band {bi} [{band.low:.0f},{band.high:.0f}): χ²/n={stat/n:.4f}; top shifts {top}"
    if all(e >= 5 for e in exp):
        detail += f"; p={float(pval):.4g} (baseline incompatibility)"
    return float(stat / max(n, 1)), detail


def score_payl(text: str, payl: PaylProfile) -> tuple[float, str]:
    h = _byte_histogram(text)
    mean = np.array(payl.mean_freq)
    std = np.array(payl.std_freq)
    z = (h - mean) / std
    dist = float(np.sqrt(np.sum(z**2) / 256.0))
    return dist, f"PAYL Mahalanobis-style byte distance {dist:.4f}"


def score_ngram(text: str, fep: FieldExtendedProfile) -> tuple[float, str]:
    counts = _bigram_counts(text)
    if not counts or not fep.ngram_median:
        return 0.0, "no bigrams or empty n-gram profile"
    zs: list[float] = []
    for g, c in counts.items():
        if g not in fep.ngram_median:
            zs.append(min(3.0, float(c)))
            continue
        med = fep.ngram_median[g]
        mad = max(fep.ngram_mad.get(g, 0.5), 0.5)
        zs.append(MAD_SCALE * abs(c - med) / mad)
    raw = max(zs) if zs else 0.0
    return raw, f"robust n-gram z max {raw:.3f} over {len(counts)} grams"


def score_ols_residual(ff: Any, fep: FieldExtendedProfile) -> tuple[float | None, str]:
    if fep.ols_coef is None:
        return None, "no OLS length model"
    x = np.array(
        [
            1.0,
            ff.entropy,
            ff.distinct_char_fraction,
            ff.category_fractions.get("digit", 0.0),
            ff.category_fractions.get("other", 0.0),
        ],
        dtype=float,
    )
    beta = np.array(fep.ols_coef)
    y_hat = float(x @ beta)
    y = math.log1p(ff.char_length)
    raw_resid = abs(y - y_hat)
    if raw_resid <= OLS_ABSOLUTE_RESID_TOLERANCE:
        return (
            0.0,
            "OLS residual within numerical tolerance; treated as 0",
        )
    scale = max(float(fep.ols_residual_std), OLS_RESIDUAL_STD_FLOOR)
    resid = raw_resid / scale
    return (
        resid,
        f"OLS standardized residual {resid:.3f} on log1p(length)~entropy+digit+other "
        f"(σ_floor={OLS_RESIDUAL_STD_FLOOR})",
    )


def score_mahalanobis(vec: list[float], ep: EndpointExtendedProfile) -> tuple[float | None, str]:
    if ep.feature_mean is None or ep.feature_cov_inv is None:
        return None, "insufficient multivariate training"
    x = np.array(vec, dtype=float)
    mu = np.array(ep.feature_mean)
    inv = np.array(ep.feature_cov_inv)
    d = x - mu
    dist = float(np.sqrt(max(d @ inv @ d, 0.0)))
    return dist, f"Mahalanobis distance on feature vector {dist:.4f}"


def extended_to_dict(ep: EndpointExtendedProfile) -> dict[str, Any]:
    fields = {}
    for fn, fep in ep.fields.items():
        fields[fn] = {
            "length_band_edges": fep.length_band_edges,
            "length_band_compositions": [asdict(b) for b in fep.length_band_compositions],
            "payl": asdict(fep.payl) if fep.payl else None,
            "ngram_median": fep.ngram_median,
            "ngram_mad": fep.ngram_mad,
            "ols_coef": fep.ols_coef,
            "ols_residual_std": fep.ols_residual_std,
        }
    return {
        "fields": fields,
        "feature_mean": ep.feature_mean,
        "feature_cov_inv": ep.feature_cov_inv,
    }


def extended_from_dict(d: dict[str, Any]) -> EndpointExtendedProfile:
    ep = EndpointExtendedProfile(
        method="", route="",
        feature_mean=d.get("feature_mean"),
        feature_cov_inv=d.get("feature_cov_inv"),
    )
    for fn, fd in d.get("fields", {}).items():
        payl = fd.get("payl")
        bands = [LengthBandComposition(**b) for b in fd.get("length_band_compositions", [])]
        ep.fields[fn] = FieldExtendedProfile(
            length_band_edges=list(fd.get("length_band_edges") or []),
            length_band_compositions=bands,
            payl=PaylProfile(**payl) if payl else None,
            ngram_median=dict(fd.get("ngram_median") or {}),
            ngram_mad=dict(fd.get("ngram_mad") or {}),
            ols_coef=fd.get("ols_coef"),
            ols_residual_std=float(fd.get("ols_residual_std") or 1.0),
        )
    return ep


def benjamini_hochberg(p_values: list[float], q: float = 0.05) -> list[bool]:
    """Return which hypotheses are rejected under BH-FDR."""
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    rejected = [False] * m
    max_i = -1
    for rank, idx in enumerate(order, start=1):
        p = p_values[idx]
        if p <= (rank / m) * q:
            max_i = rank
    if max_i >= 0:
        for rank, idx in enumerate(order[:max_i], start=1):
            rejected[idx] = True
    return rejected
