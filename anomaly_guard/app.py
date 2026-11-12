"""Streamlit entry: public read-only dashboard, or full local UI with ANOMALY_GUARD_LOCAL=1."""

from __future__ import annotations

import os


def main() -> None:
    if os.environ.get("ANOMALY_GUARD_LOCAL") == "1":
        from anomaly_guard.app_local import main as run

        run()
    else:
        from anomaly_guard.app_public import main as run

        run()


if __name__ == "__main__":
    main()
