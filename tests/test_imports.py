"""Smoke imports for CLI and dashboard entry points."""


def test_splits_exports_ordinary_records():
    from anomaly_guard.evaluation import validation_ordinary_records

    assert callable(validation_ordinary_records)


def test_app_module_importable():
    import anomaly_guard.app  # noqa: F401


def test_publish_entrypoint_importable():
    import app  # noqa: F401
