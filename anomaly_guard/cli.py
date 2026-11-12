"""Command-line interface: generate, train, replay, evaluate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anomaly_guard.core.decision import AnomalyGuard, DetectorMode
from anomaly_guard.core.schema import GroundTruthMeta, RequestRecord, read_jsonl
from anomaly_guard.evaluation.experiments import MODES, run_full_experiment
from anomaly_guard.evaluation.splits import (
    assign_splits,
    training_normal_records,
    validation_legitimate_records,
    validation_ordinary_records,
)
from anomaly_guard.simulation.generator import DEFAULT_PLAN, generate_dataset, replay_records, save_dataset
from anomaly_guard.storage import EventStore


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def cmd_generate(args: argparse.Namespace) -> None:
    plan = DEFAULT_PLAN
    if args.small:
        plan = type(plan)(
            legitimate=400,
            difficult_legitimate=80,
            benign_invalid=0,
            malicious_intent=120,
            hard_malicious_intent=40,
            distribution_shift=40,
            seed=args.seed,
        )
    records, meta = generate_dataset(plan)
    meta = assign_splits(meta, records)
    out = Path(args.output)
    save_dataset(
        out,
        records,
        meta,
        manifest={"seed": plan.seed, "counts": len(records), "plan": plan.__dict__},
    )
    print(f"Wrote {len(records)} requests to {out}")


def cmd_train(args: argparse.Namespace) -> None:
    data_dir = Path(args.data)
    records = [RequestRecord.from_dict(r) for r in read_jsonl(data_dir / "requests.jsonl")]
    meta = [GroundTruthMeta.from_dict(m) for m in read_jsonl(data_dir / "ground_truth.jsonl")]
    train_recs = training_normal_records(records, meta)
    val_recs = validation_legitimate_records(records, meta)
    ordinary_recs = validation_ordinary_records(records, meta)
    mode: DetectorMode = args.mode
    guard = AnomalyGuard(mode=mode)
    guard.fit(train_recs)
    settings = guard.calibrate(val_recs, threshold_records=ordinary_recs)
    out = Path(args.output)
    guard.save_baseline(out)
    settings_path = out.with_suffix(".settings.json")
    guard.save_settings(settings_path)
    print(f"Trained on {len(train_recs)} legitimate records; baseline -> {out}")
    print(f"Calibrated settings -> {settings_path}")
    print(f"Alert cutoff {settings.alert_threshold_percentile} on {settings.threshold_population}")


def cmd_replay(args: argparse.Namespace) -> None:
    data_dir = Path(args.data)
    records = [RequestRecord.from_dict(r) for r in read_jsonl(data_dir / "requests.jsonl")]
    meta_list = [GroundTruthMeta.from_dict(m) for m in read_jsonl(data_dir / "ground_truth.jsonl")]
    meta_by_id = {m.request_id: m for m in meta_list}
    guard = AnomalyGuard(mode=args.mode)
    baseline = Path(args.baseline)
    guard.load_baseline(baseline)
    if not guard.load_settings(baseline.with_suffix(".settings.json")):
        guard.calibrate(
            validation_legitimate_records(records, meta_list),
            threshold_records=validation_ordinary_records(records, meta_list),
        )
        print("Saved cutoff did not match this mode; recalibrated on ordinary validation traffic.")
    store = EventStore(Path(args.db))
    if args.clear:
        store.clear()
    n = 0
    for rec in replay_records(records, shuffle=args.shuffle, seed=args.seed, limit=args.limit):
        score = guard.score(rec)
        gt = meta_by_id.get(rec.request_id) if args.with_labels else None
        store.upsert_event(rec, score, gt)
        n += 1
    print(f"Replayed {n} requests into {args.db}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    summary = run_full_experiment(Path(args.data), Path(args.output))
    print(json.dumps(summary["modes"], indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="anomaly-guard", description="Anomaly Guard CLI")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Generate simulated JSONL dataset")
    g.add_argument("--output", default=str(_project_root() / "data" / "simulated"))
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--small", action="store_true", help="Smaller dataset for quick runs")
    g.set_defaults(func=cmd_generate)

    t = sub.add_parser("train", help="Train baseline on legitimate training split")
    t.add_argument("--data", default=str(_project_root() / "data" / "simulated"))
    t.add_argument("--output", default=str(_project_root() / "models" / "baseline.json"))
    t.add_argument("--mode", default="rules_plus_stats", choices=list(MODES))
    t.set_defaults(func=cmd_train)

    r = sub.add_parser("replay", help="Score requests and store events")
    r.add_argument("--data", default=str(_project_root() / "data" / "simulated"))
    r.add_argument("--baseline", default=str(_project_root() / "models" / "baseline.json"))
    r.add_argument("--db", default=str(_project_root() / "data" / "events.db"))
    r.add_argument("--mode", default="rules_plus_stats")
    r.add_argument("--shuffle", action="store_true")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--limit", type=int, default=None)
    r.add_argument("--clear", action="store_true")
    r.add_argument("--with-labels", action="store_true", help="Store ground truth for experiment UI")
    r.set_defaults(func=cmd_replay)

    e = sub.add_parser("evaluate", help="Run ablation experiments")
    e.add_argument("--data", default=str(_project_root() / "data" / "simulated"))
    e.add_argument("--output", default=str(_project_root() / "results"))
    e.set_defaults(func=cmd_evaluate)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
