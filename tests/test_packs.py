import json

import numpy as np
import pytest

from signchat import packs
from signchat.profile import Profile, Take


def _profile(name="Mine"):
    prof = Profile(name)
    rng = np.random.default_rng(1)
    prof.add(Take(label="A", kind="static", source="steamvr", hand="right", t=np.arange(10) / 30,
                  points=rng.normal(0, 0.05, (10, 21, 3)), quality="partial"))
    prof.add(Take(label="HELLO", kind="sign", source="steamvr", hand="right", t=np.arange(40) / 30,
                  points=rng.normal(0, 0.05, (40, 2, 21, 3)), present=np.ones((40, 2), bool), quality="partial"))
    prof.meta["signs"] = {"HELLO": {"text": "hello", "kind": "word"}, "BYE": {"text": "bye"}}
    return prof


def test_create_install_and_load(tmp_path):
    path = packs.create_pack(_profile(), ["A", "HELLO"], "Everyday ASL", author="Sam", tracking="Index",
                             dest=tmp_path / "p.npz")
    info = packs.install_file(path)
    assert info.id == "everyday-asl" and info.letters == ["A"] and info.signs == ["HELLO"]
    assert info.author == "Sam" and info.quality == "partial" and info.source == "steamvr"
    pack = packs.load_packs([info.id])[0]
    assert all(t.id.startswith("everyday-asl:") for t in pack.takes)
    assert list(pack.vocab) == ["HELLO"]                 # only signs that are in the pack
    assert packs.remove(info.id) and packs.list_installed() == []


def test_damaged_or_wrong_files_are_refused(tmp_path):
    bad = _profile()
    bad.takes[1].label = "<script>"
    path = bad.save(tmp_path / "bad.npz")
    with pytest.raises(packs.PackError):
        packs.install_file(path)
    weird = _profile()
    weird.takes[0].points = np.full((10, 21, 3), np.nan)
    with pytest.raises(packs.PackError):
        packs.install_file(weird.save(tmp_path / "nan.npz"))
    (tmp_path / "junk.npz").write_bytes(b"not a pack")
    with pytest.raises(packs.PackError):
        packs.install_file(tmp_path / "junk.npz")


def test_community_index(monkeypatch):
    with pytest.raises(packs.PackError):
        packs._get("http://insecure.example.org/x", 10)     # https only
    data = {"format": 1, "packs": [{"id": "x", "name": "X pack", "url": "https://example.org/x.npz",
                                    "tracking": "Index"}, {"name": "no url"}]}
    monkeypatch.setattr(packs, "_get", lambda url, limit, timeout=15.0: json.dumps(data).encode())
    entries = packs.fetch_index("https://example.org/index.json")
    assert [e.name for e in entries] == ["X pack"] and entries[0].tracking == "Index"


def test_repo_index_is_valid_json():
    from pathlib import Path
    data = json.loads((Path(__file__).resolve().parents[1] / "packs" / "index.json").read_text())
    assert isinstance(data["packs"], list)
