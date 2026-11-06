"""Template packs: recordings made by someone else, so you can start signing
without recording everything yourself.

A pack is a profile file (same .npz format) with a "pack" section in its
metadata.  Your own recordings always win: if you record a letter or sign
yourself, the pack's recordings of it are no longer used.  Packs work best
when they were recorded with the same kind of hand tracking you use
(Index controllers vs. camera-based hand tracking).

Packs only contain numbers (hand positions), never code, and are loaded with
pickling disabled.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .paths import app_dir
from .profile import Profile, Take
from .vocab import VocabEntry, validate_gloss

log = logging.getLogger(__name__)

PACK_SUFFIX = ".npz"
MAX_PACK_BYTES = 60 * 1024 * 1024
MAX_TAKES = 6000
MAX_FRAMES = 1200
MAX_INDEX_BYTES = 512 * 1024
_ID_RE = re.compile(r"[^a-z0-9_-]+")
LETTER_RE = re.compile(r"^([A-Z]|_rest|_space|_backspace|_send|_clear)$")


def packs_dir() -> Path:
    p = app_dir() / "packs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def make_id(name: str) -> str:
    return _ID_RE.sub("-", name.strip().lower()).strip("-")[:48] or f"pack-{int(time.time())}"


@dataclass
class PackInfo:
    id: str
    name: str
    author: str = ""
    description: str = ""
    version: str = "1"
    tracking: str = ""            # what it was recorded with, in words ("Valve Index controllers")
    quality: str = ""             # most common SteamVR skeleton quality: full / partial / estimated
    source: str = "steamvr"
    letters: list = field(default_factory=list)
    signs: list = field(default_factory=list)
    n_takes: int = 0
    path: Path | None = None

    def summary(self) -> str:
        parts = []
        if self.letters:
            parts.append(f"{len(self.letters)} letters")
        if self.signs:
            parts.append(f"{len(self.signs)} words/phrases")
        return ", ".join(parts) or "empty"


@dataclass
class Pack:
    info: PackInfo
    takes: list[Take]
    vocab: dict[str, VocabEntry]


class PackError(Exception):
    pass


def _validate(prof: Profile) -> None:
    if len(prof.takes) > MAX_TAKES:
        raise PackError("This pack has too many recordings.")
    for t in prof.takes:
        if t.kind == "sign":
            if validate_gloss(t.label):
                raise PackError(f"Bad sign name in pack: {t.label!r}")
            ok = t.points.ndim == 4 and t.points.shape[1:] == (2, 21, 3)
        else:
            if not LETTER_RE.match(t.label):
                raise PackError(f"Bad letter in pack: {t.label!r}")
            ok = t.points.ndim == 3 and t.points.shape[1:] == (21, 3)
        if not ok or len(t.points) != len(t.t) or len(t.t) > MAX_FRAMES or len(t.t) < 2:
            raise PackError("A recording in this pack is damaged.")
        if not (np.all(np.isfinite(t.points)) and np.all(np.isfinite(t.t))):
            raise PackError("A recording in this pack is damaged.")
        if t.present is not None and t.present.shape != (len(t.t), 2):
            raise PackError("A recording in this pack is damaged.")


def _info_from(prof: Profile, path: Path | None, pack_id: str) -> PackInfo:
    meta = prof.meta.get("pack") if isinstance(prof.meta.get("pack"), dict) else {}
    qualities = Counter(t.quality for t in prof.takes if t.quality)
    sources = Counter(t.source for t in prof.takes)
    return PackInfo(
        id=pack_id, name=str(meta.get("name") or prof.name)[:60], author=str(meta.get("author", ""))[:60],
        description=str(meta.get("description", ""))[:400], version=str(meta.get("version", "1"))[:20],
        tracking=str(meta.get("tracking", ""))[:80],
        quality=qualities.most_common(1)[0][0] if qualities else "",
        source=sources.most_common(1)[0][0] if sources else "steamvr",
        letters=sorted({t.label for t in prof.takes if t.kind != "sign"}),
        signs=sorted({t.label for t in prof.takes if t.kind == "sign"}),
        n_takes=len(prof.takes), path=path)


def read_pack(path: Path | str, pack_id: str | None = None) -> Pack:
    path = Path(path)
    if path.stat().st_size > MAX_PACK_BYTES:
        raise PackError("This file is too big to be a template pack.")
    try:
        prof = Profile.load(path)
    except Exception as exc:
        raise PackError(f"This isn't a SignChat template pack or profile ({exc}).") from exc
    _validate(prof)
    info = _info_from(prof, path, pack_id or path.stem)
    vocab = {}
    for gloss, d in (prof.meta.get("signs") or {}).items():
        if isinstance(d, dict) and not validate_gloss(gloss):
            vocab[gloss] = VocabEntry.from_dict(gloss, d)
    for t in prof.takes:                     # takes from other packs must never share an id with yours
        t.id = f"{info.id}:{t.id}"
    return Pack(info, prof.takes, vocab)


def list_installed() -> list[PackInfo]:
    out = []
    for p in sorted(packs_dir().glob(f"*{PACK_SUFFIX}")):
        try:
            out.append(read_pack(p).info)
        except (PackError, OSError) as exc:
            log.warning("skipping pack %s: %s", p.name, exc)
    return out


def load_packs(ids: list[str]) -> list[Pack]:
    out = []
    for pid in ids:
        p = packs_dir() / f"{pid}{PACK_SUFFIX}"
        if not p.exists():
            continue
        try:
            out.append(read_pack(p, pid))
        except (PackError, OSError) as exc:
            log.warning("could not load pack %s: %s", pid, exc)
    return out


def install_file(src: Path | str) -> PackInfo:
    """Copy a pack (or someone's exported profile) into the packs folder."""
    pack = read_pack(src)
    pid = make_id(pack.info.name)
    dest = packs_dir() / f"{pid}{PACK_SUFFIX}"
    if Path(src).resolve() != dest.resolve():
        shutil.copyfile(src, dest)
    return read_pack(dest, pid).info


def remove(pack_id: str) -> bool:
    p = packs_dir() / f"{pack_id}{PACK_SUFFIX}"
    if p.exists():
        p.unlink()
        return True
    return False


def create_pack(profile: Profile, labels: list[str] | None, name: str, author: str = "", description: str = "",
                tracking: str = "", source: str | None = None, dest: Path | str | None = None) -> Path:
    """Save some of your recordings as a pack to share."""
    takes = [t for t in profile.takes if (labels is None or t.label in labels)
             and (source is None or t.source == source)]
    if not takes:
        raise PackError("There are no recordings to put in the pack.")
    prof = Profile(make_id(name))
    prof.takes = list(takes)
    signs = profile.meta.get("signs") or {}
    prof.meta = {"pack": {"name": name, "author": author, "description": description, "version": "1",
                          "tracking": tracking, "created": time.time()},
                 "signs": {g: d for g, d in signs.items() if any(t.label == g for t in takes)}}
    dest = Path(dest) if dest else packs_dir() / f"{make_id(name)}{PACK_SUFFIX}"
    return prof.save(dest)


# ---------------------------------------------------------------- community packs
@dataclass
class IndexEntry:
    id: str
    name: str
    url: str
    author: str = ""
    description: str = ""
    tracking: str = ""
    size: int = 0
    contents: str = ""


def _get(url: str, limit: int, timeout: float = 15.0) -> bytes:
    if not url.lower().startswith("https://"):
        raise PackError("Only https:// links are allowed.")
    req = urllib.request.Request(url, headers={"User-Agent": "SignChat"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read(limit + 1)
    if len(data) > limit:
        raise PackError("The download is too big.")
    return data


def fetch_index(url: str) -> list[IndexEntry]:
    try:
        data = json.loads(_get(url, MAX_INDEX_BYTES).decode("utf-8"))
    except PackError:
        raise
    except Exception as exc:
        raise PackError(f"Couldn't load the pack list: {exc}") from exc
    items = data.get("packs", data) if isinstance(data, dict) else data
    out = []
    for d in items if isinstance(items, list) else []:
        if not isinstance(d, dict) or not d.get("url") or not d.get("name"):
            continue
        out.append(IndexEntry(id=make_id(str(d.get("id") or d["name"])), name=str(d["name"])[:60],
                              url=str(d["url"]), author=str(d.get("author", ""))[:60],
                              description=str(d.get("description", ""))[:400],
                              tracking=str(d.get("tracking", ""))[:80], size=int(d.get("size", 0) or 0),
                              contents=str(d.get("contents", ""))[:120]))
    return out


def download(entry: IndexEntry) -> PackInfo:
    data = _get(entry.url, MAX_PACK_BYTES, timeout=60.0)
    tmp = packs_dir() / f".{entry.id}.download"
    tmp.write_bytes(data)
    try:
        read_pack(tmp)                          # validate before installing
        dest = packs_dir() / f"{entry.id}{PACK_SUFFIX}"
        tmp.replace(dest)
        return read_pack(dest, entry.id).info
    finally:
        if tmp.exists():
            tmp.unlink()
