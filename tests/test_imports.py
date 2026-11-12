"""Smoke imports for CLI and dashboard entry points."""


def test_splits_exports_ordinary_records():
    from anomaly_guard.evaluation import validation_ordinary_records

    assert callable(validation_ordinary_records)


def test_app_module_importable():
    import anomaly_guard.app  # noqa: F401


def test_publish_entrypoint_importable():
    import anomaly_guard.app_public  # noqa: F401


def test_root_publish_entrypoint():
    from pathlib import Path

    text = Path("app.py").read_text(encoding="utf-8")
    assert "app_public" in text
    assert "main()" not in text.split("if __name__")[0]
