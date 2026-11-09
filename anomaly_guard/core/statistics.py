"""Statistical baseline profiles and detector."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.stats import chisquare, t as student_t

import math

from anomaly_guard.core.advanced_stats import (
    extended_from_dict,
    extended_to_dict,
    fit_endpoint_extended,
    score_mahalanobis,
    score_ngram,
    score_ols_residual,
    score_payl,
    score_stratified_composition,
)
from anomaly_guard.core.features import CATEGORY_BINS, RequestFeatures, extract_features
from anomaly_guard.core.parser import ParsedRequest, parse_request
from anomaly_guard.core.schema import AppConfig, RequestRecord, load_app_config, utc_now_iso

MAD_SCALE = 0.6745


@dataclass
class LengthProfile:
    median: float
    mad: float
    tolerance: float
    discrete_values: list[int] | None = None
    log_mean: float | None = None
    log_std: float | None = None
    log_n: int = 0
    ci_low: float | None = None
    ci_high: float | None = None


@dataclass
class CompositionProfile:
    category_probs: dict[str, float]
    merged_categories: list[str]
    min_total_count: int = 5
    training_chars: int = 0


@dataclass
class FieldProfile:
    field_name: str
    length: LengthProfile | None = None
    composition: CompositionProfile | None = None
    training_count: int = 0
    status: Literal["ok", "insufficient_evidence", "empty_only"] = "ok"


@dataclass
class EndpointProfile:
    method: str
    route: str
    fields: dict[str, FieldProfile] = field(default_factory=dict)
    training_count: int = 0


@dataclass
class BaselineArtifact:
    version: str
    created_at: str
    config_hash: str
    training_count: int
    endpoints: dict[str, EndpointProfile]
    feature_field_order: list[str]
    notes: str = ""

    def endpoint_key(self, method: str, route: str) -> str:
        return f"{method.upper()} {route}"


@dataclass
class ComponentScore:
    component: str
    field: str | None
    raw_score: float
    percentile: float | None
    status: str
    detail: str
    p_value: float | None = None


@dataclass
class StatisticalResult:
    baseline_version: str
    components: list[ComponentScore]
    aggregate_percentile: float | None
    insufficient_baseline: bool
    processing_time_ms: float


def _robust_length_score(observed: float, profile: LengthProfile, use_log: bool = True) -> tuple[float, str]:
    if not use_log:
        return _robust_length_score_raw(observed, profile)
    obs = math.log1p(max(observed, 0.0))
    med = math.log1p(max(profile.median, 0.0))
    if profile.mad > 0:
        mad = max(profile.mad / max(profile.median, 1.0), 1e-3)
    else:
        mad = profile.mad
    log_profile = LengthProfile(
        median=med,
        mad=mad,
        tolerance=max(0.05, med * 0.05),
        discrete_values=(
            [math.log1p(max(x, 0)) for x in profile.discrete_values]
            if profile.discrete_values is not None and len(profile.discrete_values) <= 3
            else None
        ),
    )
    raw, detail = _robust_length_score_raw(obs, log_profile)
    return raw, f"log1p(length): {detail}"


def _robust_length_score_raw(observed: float, profile: LengthProfile) -> tuple[float, str]:
    if profile.discrete_values is not None and len(profile.discrete_values) <= 3:
        if int(round(observed)) in profile.discrete_values:
            return 0.0, "matches discrete training values"
        return 3.0, "length outside discrete training set"
    if profile.mad <= 0:
        diff = abs(observed - profile.median)
        if diff <= profile.tolerance:
            return 0.0, f"within tolerance {profile.tolerance} of median {profile.median}"
        return diff / max(profile.tolerance, 1.0), f"deviation from fixed median {profile.median}"
    z = abs(observed - profile.median) / (MAD_SCALE * profile.mad)
    return float(z), f"robust z vs median {profile.median:.1f}, MAD {profile.mad:.1f}"


def _fit_log_length_interval(lengths: list[float], alpha: float = 0.05) -> tuple[float, float, float | None, float | None, int]:
    """95% prediction interval for a new log1p(length), from the training sample."""
    logs = np.log1p(np.maximum(np.array(lengths, dtype=float), 0.0))
    n = int(logs.size)
    mean = float(np.mean(logs)) if n else 0.0
    std = float(np.std(logs, ddof=1)) if n > 1 else 0.0
    if n < 8 or std <= 1e-8:
        return mean, std, None, None, n
    crit = float(student_t.ppf(1.0 - alpha / 2.0, n - 1))
    half = crit * std * math.sqrt(1.0 + 1.0 / n)
    return mean, std, mean - half, mean + half, n


def _length_ci_score(observed: float, profile: LengthProfile) -> tuple[float | None, str]:
    if profile.ci_low is None or profile.ci_high is None or not profile.log_std:
        return None, "no prediction interval (short or constant training lengths)"
    y = math.log1p(max(observed, 0.0))
    lo, hi = profile.ci_low, profile.ci_high
    if lo <= y <= hi:
        return 0.0, f"log1p(length) {y:.3f} inside 95% prediction interval [{lo:.3f}, {hi:.3f}]"
    if y > hi:
        dist = (y - hi) / profile.log_std
        side = "above"
    else:
        dist = (lo - y) / profile.log_std
        side = "below"
    return float(dist), (
        f"log1p(length) {y:.3f} {side} 95% prediction interval [{lo:.3f}, {hi:.3f}] "
        f"(n={profile.log_n})"
    )


def _wilson_interval(successes: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = min(max(successes / n, 0.0), 1.0)
    den = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / den
    margin = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return max(0.0, center - margin), min(1.0, center + margin)


def _proportion_ci_score(text: str, profile: CompositionProfile) -> tuple[float | None, str]:
    if profile.training_chars < 30 or len(text) < 8:
        return None, "too little text for a proportion interval"
    from anomaly_guard.core.features import _category_fractions

    obs = _category_fractions(text)
    worst = 0.0
    worst_detail = "all category proportions inside training Wilson intervals"
    for cat, p in profile.category_probs.items():
        if cat == "other_merged":
            continue
        k = p * profile.training_chars
        lo, hi = _wilson_interval(k, float(profile.training_chars))
        p_obs = float(obs.get(cat, 0.0))
        width = max(hi - lo, 1e-3)
        if p_obs < lo:
            dist = (lo - p_obs) / width
            if dist > worst:
                worst = dist
                worst_detail = (
                    f"{cat} proportion {p_obs:.3f} below Wilson interval [{lo:.3f}, {hi:.3f}]"
                )
        elif p_obs > hi:
            dist = (p_obs - hi) / width
            if dist > worst:
                worst = dist
                worst_detail = (
                    f"{cat} proportion {p_obs:.3f} above Wilson interval [{lo:.3f}, {hi:.3f}]"
                )
    return float(worst), worst_detail


def _merge_rare_categories(counts: dict[str, int], min_count: int = 3) -> tuple[dict[str, int], list[str]]:
    merged: dict[str, int] = {}
    rare = 0
    kept: list[str] = []
    for cat, c in counts.items():
        if c >= min_count:
            merged[cat] = c
            kept.append(cat)
        else:
            rare += c
    if rare:
        merged["other_merged"] = merged.get("other_merged", 0) + rare
        if "other_merged" not in kept:
            kept.append("other_merged")
    return merged, kept


def _composition_score(text: str, profile: CompositionProfile) -> tuple[float | None, str, str, float | None]:
    if len(text) < 8:
        return None, "insufficient_evidence", "string too short for composition test", None
    from anomaly_guard.core.features import _category_fractions

    obs_frac = _category_fractions(text)
    n = len(text)
    obs_counts: dict[str, int] = {}
    for cat in CATEGORY_BINS:
        obs_counts[cat] = int(round(obs_frac.get(cat, 0) * n))
    diff = n - sum(obs_counts.values())
    if diff:
        obs_counts["other"] = obs_counts.get("other", 0) + diff

    obs_merged, _ = _merge_rare_categories(obs_counts, min_count=1)
    expected: list[float] = []
    observed: list[float] = []
    for cat in profile.merged_categories:
        p = profile.category_probs.get(cat, 0.0)
        expected.append(p * n + 0.25)
        observed.append(float(obs_merged.get(cat, 0)))

    if min(expected) < 1.0 or n < profile.min_total_count:
        return None, "insufficient_evidence", "expected counts too small for composition test", None

    obs_arr = np.array(observed, dtype=float)
    exp_arr = np.array(expected, dtype=float)
    exp_arr = exp_arr * (obs_arr.sum() / exp_arr.sum())

    stat, p_value = chisquare(f_obs=obs_arr, f_exp=exp_arr)
    score = float(stat / max(n, 1))
    teaching_p: float | None = None
    if all(e >= 5 for e in expected) and all(o >= 0 for o in observed):
        teaching_p = float(p_value)
    contribs = ((obs_arr - exp_arr) ** 2 / np.maximum(exp_arr, 0.25)).tolist()
    top = sorted(zip(profile.merged_categories, contribs), key=lambda x: -x[1])[:2]
    detail = f"distribution discrepancy {score:.4f} (n={n}); top cells {top}"
    if teaching_p is not None:
        detail += f"; p={teaching_p:.4g} (baseline incompatibility, not attack probability)"
    return score, "ok", detail, teaching_p


class StatisticalDetector:
    """fit(normal_records) and score(request) — no ground truth in score()."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_app_config()
        self.baseline: BaselineArtifact | None = None
        self._calibration_pools: dict[str, list[float]] = {}
        self._calibration_aggregate: list[float] = []
        self._extended_endpoints: dict[str, dict[str, Any]] = {}

    def fit(self, normal_records: list[RequestRecord]) -> BaselineArtifact:
        endpoints: dict[str, EndpointProfile] = {}
        field_order: set[str] = set()

        grouped: dict[str, list[tuple[RequestFeatures, ParsedRequest]]] = {}
        for rec in normal_records:
            parsed = parse_request(rec, self.config)
            feats = extract_features(parsed)
            if parsed.route_spec is None:
                continue
            key = f"{rec.method.upper()} {rec.route}"
            grouped.setdefault(key, []).append((feats, parsed))
            for spec in parsed.route_spec.fields:
                field_order.add(spec.name)

        for key, items in grouped.items():
            method, route = key.split(" ", 1)
            ep = EndpointProfile(method=method, route=route, training_count=len(items))
            route_spec = self.config.route_key[(method, route)]
            for spec in route_spec.fields:
                lengths: list[float] = []
                cat_counts: dict[str, int] = {c: 0 for c in CATEGORY_BINS}
                non_empty = 0
                for feats, parsed in items:
                    ff = feats.field_features.get(spec.name)
                    if ff is None or ff.is_empty:
                        continue
                    non_empty += 1
                    lengths.append(float(ff.char_length))
                    text = parsed.field_text(spec.name)
                    from anomaly_guard.core.features import _category_fractions

                    fr = _category_fractions(text)
                    for c in CATEGORY_BINS:
                        cat_counts[c] += int(round(fr[c] * len(text)))

                fp = FieldProfile(field_name=spec.name, training_count=non_empty)
                if non_empty < 5:
                    fp.status = "insufficient_evidence"
                elif non_empty == 0:
                    fp.status = "empty_only"
                else:
                    arr = np.array(lengths)
                    med = float(np.median(arr))
                    mad = float(np.median(np.abs(arr - med)))
                    tol = max(1.0, med * 0.05)
                    discrete = None
                    if mad <= 0 and len(set(int(x) for x in lengths)) <= 3:
                        discrete = sorted(set(int(x) for x in lengths))
                    log_mean, log_std, ci_low, ci_high, log_n = _fit_log_length_interval(lengths)
                    fp.length = LengthProfile(
                        median=med,
                        mad=mad,
                        tolerance=tol,
                        discrete_values=discrete,
                        log_mean=log_mean,
                        log_std=log_std,
                        log_n=log_n,
                        ci_low=ci_low,
                        ci_high=ci_high,
                    )
                    merged_counts, merged_cats = _merge_rare_categories(cat_counts, min_count=3)
                    total = sum(merged_counts.values()) or 1
                    probs = {k: v / total for k, v in merged_counts.items()}
                    fp.composition = CompositionProfile(
                        category_probs=probs,
                        merged_categories=merged_cats,
                        training_chars=int(total),
                    )
                ep.fields[spec.name] = fp
            endpoints[key] = ep

        forder = sorted(field_order)
        self._extended_endpoints = {}
        for key, items in grouped.items():
            method, route = key.split(" ", 1)
            route_spec = self.config.route_key[(method, route)]
            fnames = [spec.name for spec in route_spec.fields]
            vectors = [feats.to_vector(forder) for feats, _ in items]
            ext = fit_endpoint_extended(items, fnames, vectors)
            ext.method = method
            ext.route = route
            self._extended_endpoints[key] = extended_to_dict(ext)

        self.baseline = BaselineArtifact(
            version=f"v{int(time.time())}",
            created_at=utc_now_iso(),
            config_hash=self.config.config_hash,
            training_count=len(normal_records),
            endpoints=endpoints,
            feature_field_order=sorted(field_order),
        )
        return self.baseline

    def calibrate_percentiles(self, validation_records: list[RequestRecord]) -> dict[str, list[float]]:
        """Collect per-component raw scores from legitimate validation traffic."""
        self._calibration_pools = {}
        for rec in validation_records:
            res = self._score_internal(rec, collect_for_calibration=True)
            for c in res.components:
                if c.status != "ok":
                    continue
                self._calibration_pools.setdefault(c.component, []).append(c.raw_score)
        for comp, pool in self._calibration_pools.items():
            pool.sort()
        agg: list[float] = []
        for rec in validation_records:
            res = self._score_internal(rec, collect_for_calibration=False)
            if res.aggregate_percentile is not None:
                agg.append(res.aggregate_percentile)
        self._calibration_aggregate = sorted(agg)
        out = dict(self._calibration_pools)
        out["aggregate"] = self._calibration_aggregate
        return out

    def _empirical_percentile(self, raw: float, component: str) -> float | None:
        pool = self._calibration_pools.get(component, [])
        if not pool:
            return None
        arr = np.array(pool)
        return float(np.mean(arr <= raw) * 100.0)

    def _score_internal(
        self, record: RequestRecord, collect_for_calibration: bool = False
    ) -> StatisticalResult:
        t0 = time.perf_counter()
        if self.baseline is None:
            return StatisticalResult(
                baseline_version="none",
                components=[],
                aggregate_percentile=None,
                insufficient_baseline=True,
                processing_time_ms=0.0,
            )

        parsed = parse_request(record, self.config)
        feats = extract_features(parsed)
        key = self.baseline.endpoint_key(record.method, record.route)
        ep = self.baseline.endpoints.get(key)
        components: list[ComponentScore] = []

        if ep is None:
            ms = (time.perf_counter() - t0) * 1000
            return StatisticalResult(
                baseline_version=self.baseline.version,
                components=[],
                aggregate_percentile=None,
                insufficient_baseline=True,
                processing_time_ms=ms,
            )

        ext_data = self._extended_endpoints.get(key)
        ext_prof = extended_from_dict(ext_data) if ext_data else None
        field_order = self.baseline.feature_field_order
        if ext_prof and ext_prof.feature_mean is not None:
            vec = feats.to_vector(field_order)
            raw_m, detail_m = score_mahalanobis(vec, ext_prof)
            if raw_m is not None:
                pct_m = None if collect_for_calibration else self._empirical_percentile(raw_m, "mahalanobis")
                components.append(
                    ComponentScore(
                        component="mahalanobis",
                        field=None,
                        raw_score=raw_m,
                        percentile=pct_m,
                        status="ok",
                        detail=detail_m,
                    )
                )

        for fname, ff in feats.field_features.items():
            fp = ep.fields.get(fname)
            if fp is None:
                continue
            if ff.is_empty:
                components.append(
                    ComponentScore(
                        component="length",
                        field=fname,
                        raw_score=0.0,
                        percentile=None,
                        status="missing",
                        detail="empty field",
                    )
                )
                continue
            if fp.length and fp.status == "ok":
                raw, detail = _robust_length_score(float(ff.char_length), fp.length)
                pct = None if collect_for_calibration else self._empirical_percentile(raw, "length")
                components.append(
                    ComponentScore(
                        component="length",
                        field=fname,
                        raw_score=raw,
                        percentile=pct,
                        status="ok",
                        detail=detail,
                    )
                )
            if fp.composition and fp.status == "ok":
                text = parsed.field_text(fname)
                comp_raw, st, detail, teaching_p = _composition_score(text, fp.composition)
                if comp_raw is None:
                    components.append(
                        ComponentScore(
                            component="composition",
                            field=fname,
                            raw_score=0.0,
                            percentile=None,
                            status=st,
                            detail=detail,
                            p_value=teaching_p,
                        )
                    )
                else:
                    pct = None if collect_for_calibration else self._empirical_percentile(comp_raw, "composition")
                    components.append(
                        ComponentScore(
                            component="composition",
                            field=fname,
                            raw_score=comp_raw,
                            percentile=pct,
                            status="ok",
                            detail=detail,
                            p_value=teaching_p,
                        )
                    )

            if fp.length and fp.status == "ok":
                ci_raw, ci_detail = _length_ci_score(float(ff.char_length), fp.length)
                if ci_raw is not None:
                    pct_ci = None if collect_for_calibration else self._empirical_percentile(ci_raw, "length_ci")
                    components.append(
                        ComponentScore(
                            component="length_ci",
                            field=fname,
                            raw_score=ci_raw,
                            percentile=pct_ci,
                            status="ok",
                            detail=ci_detail,
                        )
                    )
            if fp.composition and fp.status == "ok":
                prop_raw, prop_detail = _proportion_ci_score(parsed.field_text(fname), fp.composition)
                if prop_raw is not None:
                    pct_prop = None if collect_for_calibration else self._empirical_percentile(
                        prop_raw, "proportion_ci"
                    )
                    components.append(
                        ComponentScore(
                            component="proportion_ci",
                            field=fname,
                            raw_score=prop_raw,
                            percentile=pct_prop,
                            status="ok",
                            detail=prop_detail,
                        )
                    )

            if ext_prof:
                fep = ext_prof.fields.get(fname)
                if fep is None:
                    continue
                text = parsed.field_text(fname)
                strat_raw, strat_detail = score_stratified_composition(text, fep, float(ff.char_length))
                if strat_raw is not None:
                    pct_s = None if collect_for_calibration else self._empirical_percentile(
                        strat_raw, "length_stratified"
                    )
                    components.append(
                        ComponentScore(
                            component="length_stratified",
                            field=fname,
                            raw_score=strat_raw,
                            percentile=pct_s,
                            status="ok",
                            detail=strat_detail,
                        )
                    )
                if fep.payl:
                    payl_raw, payl_detail = score_payl(text, fep.payl)
                    pct_p = None if collect_for_calibration else self._empirical_percentile(payl_raw, "payl")
                    components.append(
                        ComponentScore(
                            component="payl",
                            field=fname,
                            raw_score=payl_raw,
                            percentile=pct_p,
                            status="ok",
                            detail=payl_detail,
                        )
                    )
                ngram_raw, ngram_detail = score_ngram(text, fep)
                if ngram_raw > 0 or fep.ngram_median:
                    pct_n = None if collect_for_calibration else self._empirical_percentile(ngram_raw, "ngram")
                    components.append(
                        ComponentScore(
                            component="ngram",
                            field=fname,
                            raw_score=ngram_raw,
                            percentile=pct_n,
                            status="ok",
                            detail=ngram_detail,
                        )
                    )
                reg_raw, reg_detail = score_ols_residual(ff, fep)
                if reg_raw is not None:
                    pct_r = None if collect_for_calibration else self._empirical_percentile(reg_raw, "regression")
                    components.append(
                        ComponentScore(
                            component="regression",
                            field=fname,
                            raw_score=reg_raw,
                            percentile=pct_r,
                            status="ok",
                            detail=reg_detail,
                        )
                    )

        pcts = [c.percentile for c in components if c.status == "ok" and c.percentile is not None]
        agg_pct = max(pcts) if pcts and not collect_for_calibration else None

        ms = (time.perf_counter() - t0) * 1000
        return StatisticalResult(
            baseline_version=self.baseline.version,
            components=components,
            aggregate_percentile=agg_pct,
            insufficient_baseline=False,
            processing_time_ms=ms,
        )

    def score(self, record: RequestRecord) -> StatisticalResult:
        return self._score_internal(record, collect_for_calibration=False)

    def save(self, path: Path) -> None:
        if self.baseline is None:
            raise ValueError("No baseline to save")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = _baseline_to_dict(self.baseline)
        payload["calibration_pools"] = self._calibration_pools
        payload["calibration_aggregate"] = self._calibration_aggregate
        payload["extended_endpoints"] = self._extended_endpoints
        # Legacy keys for older baselines
        payload["calibration_length"] = self._calibration_pools.get("length", [])
        payload["calibration_composition"] = self._calibration_pools.get("composition", [])
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.baseline = _baseline_from_dict(payload)
        pools = payload.get("calibration_pools")
        if pools:
            self._calibration_pools = {k: list(v) for k, v in pools.items()}
        else:
            self._calibration_pools = {}
            if payload.get("calibration_length") or payload.get("calibration_scores"):
                self._calibration_pools["length"] = list(
                    payload.get("calibration_length") or payload.get("calibration_scores") or []
                )
            if payload.get("calibration_composition"):
                self._calibration_pools["composition"] = list(payload["calibration_composition"])
        self._calibration_aggregate = list(payload.get("calibration_aggregate") or [])
        self._extended_endpoints = dict(payload.get("extended_endpoints") or {})


def _baseline_to_dict(b: BaselineArtifact) -> dict[str, Any]:
    eps = {}
    for k, ep in b.endpoints.items():
        fields = {}
        for fn, fp in ep.fields.items():
            fields[fn] = {
                "field_name": fp.field_name,
                "training_count": fp.training_count,
                "status": fp.status,
                "length": asdict(fp.length) if fp.length else None,
                "composition": asdict(fp.composition) if fp.composition else None,
            }
        eps[k] = {
            "method": ep.method,
            "route": ep.route,
            "training_count": ep.training_count,
            "fields": fields,
        }
    return {
        "version": b.version,
        "created_at": b.created_at,
        "config_hash": b.config_hash,
        "training_count": b.training_count,
        "endpoints": eps,
        "feature_field_order": b.feature_field_order,
        "notes": b.notes,
    }


def _baseline_from_dict(d: dict[str, Any]) -> BaselineArtifact:
    endpoints: dict[str, EndpointProfile] = {}
    for k, epd in d["endpoints"].items():
        fields = {}
        for fn, fpd in epd["fields"].items():
            lp = fpd.get("length")
            cp = fpd.get("composition")
            fields[fn] = FieldProfile(
                field_name=fn,
                training_count=fpd.get("training_count", 0),
                status=fpd.get("status", "ok"),
                length=LengthProfile(**lp) if lp else None,
                composition=CompositionProfile(**cp) if cp else None,
            )
        endpoints[k] = EndpointProfile(
            method=epd["method"],
            route=epd["route"],
            fields=fields,
            training_count=epd.get("training_count", 0),
        )
    return BaselineArtifact(
        version=d["version"],
        created_at=d["created_at"],
        config_hash=d["config_hash"],
        training_count=d["training_count"],
        endpoints=endpoints,
        feature_field_order=list(d.get("feature_field_order") or []),
        notes=str(d.get("notes", "")),
    )
