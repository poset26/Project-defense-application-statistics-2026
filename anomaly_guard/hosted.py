"""Detect Cursor hosted (published) runtime."""

from __future__ import annotations

import os


def is_cursor_hosted() -> bool:
    """True when the app runs on Cursor published hosting (not local Streamlit)."""
    return bool(os.environ.get("CURSOR_APP_ID"))
