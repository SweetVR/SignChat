"""Where SignChat keeps its settings, profiles and logs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from . import APP_NAME

PACKAGE_DIR = Path(__file__).resolve().parent


def app_dir() -> Path:
    """Per-user data folder (created on first use).

    Windows: %APPDATA%\\SignChat   Other systems: ~/.signchat
    SIGNCHAT_HOME overrides both (used by the tests).
    """
    override = os.environ.get("SIGNCHAT_HOME")
    if override:
        base = Path(override)
    elif sys.platform.startswith("win") and os.environ.get("APPDATA"):
        base = Path(os.environ["APPDATA"]) / APP_NAME
    else:
        base = Path.home() / ".signchat"
    base.mkdir(parents=True, exist_ok=True)
    return base


def profiles_dir() -> Path:
    p = app_dir() / "profiles"
    p.mkdir(parents=True, exist_ok=True)
    return p


def logs_dir() -> Path:
    p = app_dir() / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def steamvr_manifest() -> Path:
    return PACKAGE_DIR / "steamvr" / "actions.json"
