import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Keep settings and profiles of every test in a temporary folder."""
    monkeypatch.setenv("SIGNCHAT_HOME", str(tmp_path / "home"))
    yield tmp_path / "home"
