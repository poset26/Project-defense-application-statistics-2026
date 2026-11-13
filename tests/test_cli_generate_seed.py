"""CLI generate must honor --seed on full and small plans."""

import json
from pathlib import Path

from anomaly_guard.cli import cmd_generate


def test_generate_honors_seed_without_small_flag(tmp_path: Path):
    args_a = type("A", (), {"output": str(tmp_path / "a"), "seed": 123, "small": False})()
    args_b = type("B", (), {"output": str(tmp_path / "b"), "seed": 999, "small": False})()
    cmd_generate(args_a)
    cmd_generate(args_b)
    manifest_a = json.loads((tmp_path / "a" / "manifest.json").read_text(encoding="utf-8"))
    manifest_b = json.loads((tmp_path / "b" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_a["seed"] == 123
    assert manifest_b["seed"] == 999
    assert manifest_a["plan"]["seed"] == 123
