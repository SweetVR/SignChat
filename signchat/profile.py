"""A person's recorded examples ("takes"), saved as one .npz file.

Raw landmark points are stored (not features), so newer versions of SignChat
can re-learn from old recordings.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .paths import profiles_dir

PROFILE_VERSION = 1


@dataclass
class Take:
    label: str
    kind: str                 # "static" or "motion"
    source: str               # "steamvr", "webcam" or "sim"
    hand: str                 # "left" or "right"
    t: np.ndarray             # (n,) seconds
    points: np.ndarray        # (n, 21, 3) viewer-space landmarks
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created: float = field(default_factory=time.time)


class Profile:
    def __init__(self, name: str = "default"):
        self.name = name
        self.takes: list[Take] = []

    @property
    def path(self) -> Path:
        return profiles_dir() / f"{self.name}.npz"

    def add(self, take: Take) -> None:
        self.takes.append(take)

    def takes_for(self, source: str) -> list[Take]:
        return [t for t in self.takes if t.source == source]

    def counts(self, source: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for t in self.takes_for(source):
            out[t.label] = out.get(t.label, 0) + 1
        return out

    def remove_label(self, label: str, source: str) -> int:
        before = len(self.takes)
        self.takes = [t for t in self.takes if not (t.label == label and t.source == source)]
        return before - len(self.takes)

    def remove_last(self, label: str, source: str) -> bool:
        for i in range(len(self.takes) - 1, -1, -1):
            t = self.takes[i]
            if t.label == label and t.source == source:
                del self.takes[i]
                return True
        return False

    # ----------------------------------------------------------- storage
    def save(self, path: Path | str | None = None) -> Path:
        path = Path(path) if path else self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"version": PROFILE_VERSION, "name": self.name, "takes": []}
        arrays = {}
        for i, t in enumerate(self.takes):
            meta["takes"].append({"id": t.id, "label": t.label, "kind": t.kind, "source": t.source,
                                  "hand": t.hand, "created": t.created})
            arrays[f"t{i}"] = np.asarray(t.t, np.float64)
            arrays[f"p{i}"] = np.asarray(t.points, np.float32)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, meta=np.array(json.dumps(meta)), **arrays)
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "Profile":
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            prof = cls(meta.get("name") or path.stem)
            for i, m in enumerate(meta["takes"]):
                prof.takes.append(Take(label=m["label"], kind=m["kind"], source=m["source"], hand=m["hand"],
                                       t=np.array(data[f"t{i}"]), points=np.array(data[f"p{i}"]),
                                       id=m["id"], created=m["created"]))
        return prof

    @classmethod
    def load_or_create(cls, name: str = "default") -> "Profile":
        p = profiles_dir() / f"{name}.npz"
        if p.exists():
            try:
                return cls.load(p)
            except Exception:  # keep a broken file for inspection, start fresh
                p.rename(p.with_suffix(f".broken-{int(time.time())}.npz"))
        return cls(name)
