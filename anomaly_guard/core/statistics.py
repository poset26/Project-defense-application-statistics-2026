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


def _composition_score(text: str, profile: CompositionProfile) -> tuple[float | None, str, str]:
    if len(text) < 4:
        return None, "insufficient_evidence", "string too short for composition test"
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
        expected.append(max(p * n, 0.5))
        observed.append(float(obs_merged.get(cat, 0)))

    if sum(expected) < profile.min_total_count:
        return None, "insufficient_evidence", "expected counts too small"

    obs_arr = np.array(observed, dtype=float)
    exp_arr = np.array(expected, dtype=float)
    obs_sum = obs_arr.sum()
    exp_sum = exp_arr.sum()
    if exp_sum > 0 and abs(obs_sum - exp_sum) > 1e-6:
        exp_arr = exp_arr * (obs_sum / exp_sum)

    stat, _p = chisquare(f_obs=obs_arr, f_exp=exp_arr)
    score = float(np.sqrt(max(stat, 0.0) / max(n, 1)))
    return score, "ok", f"chi-style discrepancy score {score:.3f} (n={n})"


class StatisticalDetector:
    """fit(normal_records) and score(request) — no ground truth in score()."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_app_config()
        self.baseline: BaselineArtifact | None = None
        self._calibration_scores: list[float] = []

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

    def calibrate_percentiles(self, validation_records: list[RequestRecord]) -> list[float]:
        """Collect aggregate statistical scores from legitimate validation traffic."""
        scores: list[float] = []
        for rec in validation_records:
            res = self._score_internal(rec, collect_for_calibration=True)
            if res.aggregate_percentile is None and res.components:
                raw_vals = [c.raw_score for c in res.components if c.status == "ok"]
                if raw_vals:
                    scores.append(max(raw_vals))
            elif res.components:
                raw_vals = [c.raw_score for c in res.components if c.status == "ok"]
                if raw_vals:
                    scores.append(max(raw_vals))
        self._calibration_scores = sorted(scores)
        return self._calibration_scores

    def _empirical_percentile(self, raw: float) -> float | None:
        if not self._calibration_scores:
            return None
        arr = np.array(self._calibration_scores)
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
                pct = None if collect_for_calibration else self._empirical_percentile(raw)
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
                comp_raw, st, detail = _composition_score(text, fp.composition)
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
                    pct = None if collect_for_calibration else self._empirical_percentile(comp_raw)
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

        ok_scores = [c.raw_score for c in components if c.status == "ok"]
        agg_raw = max(ok_scores) if ok_scores else None
        agg_pct = None
        if agg_raw is not None and not collect_for_calibration:
            agg_pct = self._empirical_percentile(agg_raw)

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
        payload["calibration_scores"] = self._calibration_scores
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.baseline = _baseline_from_dict(payload)
        self._calibration_scores = list(payload.get("calibration_scores") or [])


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
