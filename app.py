"""Cursor publish entrypoint — read-only public dashboard."""

from anomaly_guard.app_public import main

__all__ = ["main"]

if __name__ == "__main__":
    main()
