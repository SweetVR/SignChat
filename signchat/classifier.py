"""Per-user handshape classifier.

Instead of a big pre-trained network, SignChat remembers the examples each
person records and compares new frames to them (k-nearest neighbours).
"Training" is instant, works with a couple of seconds of data per letter,
adapts to the person's hand, hardware and signing style, and copes with
community-adapted handshapes for controllers that can't do every letter.

A frame that is far from every recorded example is rejected (label None),
so random hand movement doesn't type letters.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

from .features import FEATURE_DIM, GROUP_ORDER, GROUP_SLICES, GROUP_SIZES, SHAPE_MASK

REST_LABEL = "_rest"
DEFAULT_GROUP_WEIGHTS = {"shape": 1.0, "contacts": 1.0, "angles": 1.0, "orientation": 1.2}


@dataclass
class Prediction:
    label: str | None        # accepted label, None when unsure
    raw_label: str | None    # best guess before the "too far from any example" check
    conf: float              # share of neighbour votes for the best label (0..1)
    ratio: float             # distance to the closest example / that sign's normal spread
    second: str | None = None

    @property
    def is_sign(self) -> bool:
        return self.label is not None and self.label != REST_LABEL


@dataclass
class EvalReport:
    overall: float
    per_label: dict[str, dict] = field(default_factory=dict)
    untestable: list[str] = field(default_factory=list)

    def worst(self, n: int = 6) -> list[tuple[str, float, str | None]]:
        rows = []
        for label, info in self.per_label.items():
            conf = info["confused_with"][0][0] if info["confused_with"] else None
            rows.append((label, info["acc"], conf))
        rows.sort(key=lambda r: r[1])
        return [r for r in rows[:n] if r[1] < 0.999]


class SignModel:
    def __init__(self, k: int = 7, reject_ratio: float = 3.0, group_weights: dict | None = None,
                 max_per_label: int = 400):
        self.k = k
        self.reject_ratio = reject_ratio
        self.group_weights = dict(DEFAULT_GROUP_WEIGHTS, **(group_weights or {}))
        self.max_per_label = max_per_label
        self.labels: list[str] = []
        self.Z = np.zeros((0, FEATURE_DIM), np.float32)
        self.y = np.zeros(0, int)
        self.take = np.zeros(0, int)
        self.mean = np.zeros(FEATURE_DIM)
        self.w = np.ones(FEATURE_DIM)
        self.class_scale = np.zeros(0)

    # ------------------------------------------------------------------ fit
    def fit(self, data: list[tuple[str, str, np.ndarray]]) -> "SignModel":
        """data: list of (label, take_id, features[n, FEATURE_DIM])."""
        data = [(lab, tid, np.asarray(X, np.float32)) for lab, tid, X in data if len(X)]
        if not data:
            raise ValueError("no examples to learn from")
        self.labels = sorted({lab for lab, _, _ in data})
        lab_index = {lab: i for i, lab in enumerate(self.labels)}
        take_ids = {}
        Xs, ys, ts = [], [], []
        by_label = defaultdict(list)
        for lab, tid, X in data:
            by_label[lab].append((tid, X))
        for lab, takes in by_label.items():
            X = np.concatenate([x for _, x in takes])
            t = np.concatenate([np.full(len(x), take_ids.setdefault(tid, len(take_ids))) for tid, x in takes])
            if len(X) > self.max_per_label:
                keep = np.unique(np.linspace(0, len(X) - 1, self.max_per_label).round().astype(int))
                X, t = X[keep], t[keep]
            Xs.append(X)
            ys.append(np.full(len(X), lab_index[lab]))
            ts.append(t)
        X = np.concatenate(Xs).astype(np.float64)
        self.y = np.concatenate(ys)
        self.take = np.concatenate(ts)

        self.mean = X.mean(axis=0)
        std = X.std(axis=0)
        w = np.zeros(FEATURE_DIM)
        for g in GROUP_ORDER:
            sl = GROUP_SLICES[g]
            s = std[sl]
            floor = 0.1 * float(np.median(s)) + 1e-4
            w[sl] = self.group_weights[g] / np.sqrt(GROUP_SIZES[g]) / np.maximum(s, floor)
        self.w = w
        self.Z = ((X - self.mean) * w).astype(np.float32)
        self._zsq = (self.Z.astype(np.float64) ** 2).sum(axis=1)
        self._Zs = np.ascontiguousarray(self.Z[:, SHAPE_MASK])
        self._zs_sq = (self._Zs.astype(np.float64) ** 2).sum(axis=1)
        self.class_scale = self._class_scales()
        return self

    def _class_scales(self) -> np.ndarray:
        scales = np.full(len(self.labels), np.nan)
        for ci in range(len(self.labels)):
            idx = np.where(self.y == ci)[0]
            if len(idx) < 2:
                continue
            Zc = self.Z[idx].astype(np.float64)
            tc = self.take[idx]
            sq = (Zc ** 2).sum(axis=1)
            d2 = sq[:, None] + sq[None, :] - 2.0 * Zc @ Zc.T
            if len(np.unique(tc)) > 1:
                d2[tc[:, None] == tc[None, :]] = np.inf
            else:
                ii = np.arange(len(idx))
                d2[np.abs(ii[:, None] - ii[None, :]) <= 3] = np.inf
            nn = d2.min(axis=1)
            nn = nn[np.isfinite(nn)]
            if len(nn):
                scales[ci] = float(np.median(np.sqrt(np.maximum(nn, 0.0))))
        finite = scales[np.isfinite(scales) & (scales > 0)]
        med = float(np.median(finite)) if len(finite) else 1.0
        scales = np.where(np.isfinite(scales) & (scales > 0), scales, med)
        return np.maximum(scales, 0.35 * med)

    # -------------------------------------------------------------- predict
    @property
    def ready(self) -> bool:
        return len(self.labels) > 0

    @property
    def median_scale(self) -> float:
        return float(np.median(self.class_scale)) if len(self.class_scale) else 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, np.float64) - self.mean) * self.w

    def _knn(self, z: np.ndarray, mask: np.ndarray | None = None):
        d2 = self._zsq - 2.0 * (self.Z @ z.astype(np.float32)).astype(np.float64) + float(z @ z)
        if mask is not None:
            d2 = np.where(mask, d2, np.inf)
        k = min(self.k, int(np.isfinite(d2).sum()))
        if k <= 0:
            return np.zeros(0, int), np.zeros(0)
        idx = np.argpartition(d2, k - 1)[:k]
        return idx, np.sqrt(np.maximum(d2[idx], 0.0))

    def _decide(self, idx: np.ndarray, d: np.ndarray) -> Prediction:
        if len(idx) == 0:
            return Prediction(None, None, 0.0, float("inf"))
        weights = 1.0 / (d + 1e-6)
        votes = np.bincount(self.y[idx], weights=weights, minlength=len(self.labels))
        order = np.argsort(votes)[::-1]
        best = int(order[0])
        conf = float(votes[best] / votes.sum())
        d_best = float(d[self.y[idx] == best].min())
        ratio = d_best / float(self.class_scale[best])
        second = self.labels[int(order[1])] if len(order) > 1 and votes[order[1]] > 0 else None
        raw = self.labels[best]
        label = raw if ratio <= self.reject_ratio else None
        return Prediction(label, raw, conf, ratio, second)

    def predict(self, x: np.ndarray) -> Prediction:
        if not self.ready:
            return Prediction(None, None, 0.0, float("inf"))
        z = self.transform(x)
        return self._decide(*self._knn(z))

    def predict_shape_label(self, x: np.ndarray) -> str | None:
        """Closest sign by hand shape alone (ignores which way the hand points).
        Used for moving letters, where the hand turns while the shape stays."""
        if not self.ready:
            return None
        z = self.transform(x)[SHAPE_MASK]
        d2 = self._zs_sq - 2.0 * (self._Zs @ z.astype(np.float32)).astype(np.float64) + float(z @ z)
        k = min(self.k, len(d2))
        idx = np.argpartition(d2, k - 1)[:k]
        votes = np.bincount(self.y[idx], weights=1.0 / (np.sqrt(np.maximum(d2[idx], 0.0)) + 1e-6),
                            minlength=len(self.labels))
        return self.labels[int(np.argmax(votes))]

    def shape_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Distance between two feature vectors using hand shape only (no orientation)."""
        za, zb = self.transform(a), self.transform(b)
        return float(np.linalg.norm((za - zb)[SHAPE_MASK]))

    # ------------------------------------------------------------- evaluate
    def evaluate(self, frames_per_take: int = 24) -> EvalReport:
        """Leave-one-take-out check: classify each recording using only the
        *other* recordings. Shows which letters get confused with which."""
        per = defaultdict(lambda: {"n": 0, "ok": 0, "unknown": 0, "conf": Counter()})
        untestable = []
        for ci, lab in enumerate(self.labels):
            if len(np.unique(self.take[self.y == ci])) < 2:
                untestable.append(lab)
        for tid in np.unique(self.take):
            rows = np.where(self.take == tid)[0]
            lab = self.labels[int(self.y[rows[0]])]
            if lab in untestable:
                continue
            pick = rows[np.unique(np.linspace(0, len(rows) - 1, min(frames_per_take, len(rows))).round().astype(int))]
            mask = self.take != tid
            for r in pick:
                p = self._decide(*self._knn(self.Z[r].astype(np.float64), mask))
                info = per[lab]
                info["n"] += 1
                if p.label == lab:
                    info["ok"] += 1
                elif p.label is None:
                    info["unknown"] += 1
                else:
                    info["conf"][p.label] += 1
        per_label, tot_ok, tot_n = {}, 0, 0
        for lab, info in per.items():
            n = max(info["n"], 1)
            per_label[lab] = {
                "acc": info["ok"] / n,
                "unknown": info["unknown"] / n,
                "n": info["n"],
                "confused_with": [(o, c / n) for o, c in info["conf"].most_common(3)],
            }
            tot_ok += info["ok"]
            tot_n += info["n"]
        return EvalReport(tot_ok / tot_n if tot_n else 0.0, per_label, sorted(untestable))
