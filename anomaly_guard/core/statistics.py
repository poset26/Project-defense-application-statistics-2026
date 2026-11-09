"""Statistical baseline profiles and detector."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.stats import chisquare

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


@dataclass
class CompositionProfile:
    category_probs: dict[str, float]
    merged_categories: list[str]
    min_total_count: int = 5


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


@dataclass
class StatisticalResult:
    baseline_version: str
    components: list[ComponentScore]
    aggregate_percentile: float | None
    insufficient_baseline: bool
    processing_time_ms: float


def _robust_length_score(observed: float, profile: LengthProfile) -> tuple[float, str]:
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
    detail = f"distribution discrepancy {score:.4f} (n={n})"
    if teaching_p is not None:
        detail += f"; baseline incompatibility p={teaching_p:.4g} (not attack probability)"
    return score, "ok", detail, teaching_p


class StatisticalDetector:
    """fit(normal_records) and score(request) — no ground truth in score()."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_app_config()
        self.baseline: BaselineArtifact | None = None
        self._calibration_length: list[float] = []
        self._calibration_composition: list[float] = []
        self._calibration_aggregate: list[float] = []

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
                    fp.length = LengthProfile(
                        median=med, mad=mad, tolerance=tol, discrete_values=discrete
                    )
                    merged_counts, merged_cats = _merge_rare_categories(cat_counts, min_count=3)
                    total = sum(merged_counts.values()) or 1
                    probs = {k: v / total for k, v in merged_counts.items()}
                    fp.composition = CompositionProfile(
                        category_probs=probs,
                        merged_categories=merged_cats,
                    )
                ep.fields[spec.name] = fp
            endpoints[key] = ep

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
        self._calibration_length = []
        self._calibration_composition = []
        for rec in validation_records:
            res = self._score_internal(rec, collect_for_calibration=True)
            for c in res.components:
                if c.status != "ok":
                    continue
                if c.component == "length":
                    self._calibration_length.append(c.raw_score)
                elif c.component == "composition":
                    self._calibration_composition.append(c.raw_score)
        self._calibration_length.sort()
        self._calibration_composition.sort()
        agg: list[float] = []
        for rec in validation_records:
            res = self._score_internal(rec, collect_for_calibration=False)
            if res.aggregate_percentile is not None:
                agg.append(res.aggregate_percentile)
        self._calibration_aggregate = sorted(agg)
        return {
            "length": self._calibration_length,
            "composition": self._calibration_composition,
            "aggregate": self._calibration_aggregate,
        }

    def _empirical_percentile(self, raw: float, component: str) -> float | None:
        pool = (
            self._calibration_length
            if component == "length"
            else self._calibration_composition
            if component == "composition"
            else self._calibration_aggregate
        )
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
                comp_raw, st, detail, _tp = _composition_score(text, fp.composition)
                if comp_raw is None:
                    components.append(
                        ComponentScore(
                            component="composition",
                            field=fname,
                            raw_score=0.0,
                            percentile=None,
                            status=st,
                            detail=detail,
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
        payload["calibration_length"] = self._calibration_length
        payload["calibration_composition"] = self._calibration_composition
        payload["calibration_aggregate"] = self._calibration_aggregate
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.baseline = _baseline_from_dict(payload)
        self._calibration_length = list(payload.get("calibration_length") or payload.get("calibration_scores") or [])
        self._calibration_composition = list(payload.get("calibration_composition") or [])
        self._calibration_aggregate = list(payload.get("calibration_aggregate") or [])


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
