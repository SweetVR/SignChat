"""Letters that move: J, Z (and any other motion sign a user records).

When the hand moves and then stops, the path of the fingertip that moved the
most is compared with the person's recorded examples using dynamic time
warping (DTW), which tolerates doing the motion faster, slower or a bit
bigger/smaller.  The hand shape during the movement must also match, so
ordinary transitions between letters don't trigger J or Z.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import SHAPE_MASK
from .hand import TIPS

N_POINTS = 32


def resample_path(xy: np.ndarray, n: int = N_POINTS) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) == 1:
        return np.repeat(xy, n, axis=0)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-9:
        return np.repeat(xy[:1], n, axis=0)
    targets = np.linspace(0.0, total, n)
    return np.stack([np.interp(targets, cum, xy[:, 0]), np.interp(targets, cum, xy[:, 1])], axis=1)


def normalize_path(xy: np.ndarray) -> np.ndarray:
    """Start at the origin, scale so the bigger side of the bounding box is 1."""
    xy = np.asarray(xy, dtype=np.float64)
    xy = xy - xy[0]
    ext = max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), 1e-9)
    return resample_path(xy / ext)


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    n, m = cost.shape
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        ci = cost[i - 1]
        prev = D[i - 1]
        row = D[i]
        for j in range(1, m + 1):
            row[j] = ci[j - 1] + min(prev[j], row[j - 1], prev[j - 1])
    return float(D[n, m] / (n + m))


def path_length(xy: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) > 1 else 0.0


def find_motion_segment(times: np.ndarray, speeds: np.ndarray, on: float = 2.2, off: float = 1.2,
                        quiet_s: float = 0.08, min_s: float = 0.12, pts_seq: np.ndarray | None = None):
    """(start, end) indices of the biggest movement in a recording, or None."""
    segs, moving, start, quiet = [], False, 0, None
    for i, s in enumerate(speeds):
        if not moving:
            if s > on:
                moving, start, quiet = True, i, None
        elif s < off:
            if quiet is None:
                quiet = i
            if times[i] - times[quiet] >= quiet_s:
                segs.append((start, quiet))
                moving, quiet = False, None
        else:
            quiet = None
    if moving:
        segs.append((start, len(speeds) - 1))
    segs = [(a, b) for a, b in segs if times[b] - times[a] >= min_s]
    if not segs:
        return None

    def size(seg):
        a, b = seg
        if pts_seq is None:
            return times[b] - times[a]
        return max(path_length(pts_seq[a:b + 1, t, :2]) for t in TIPS)

    a, b = max(segs, key=size)
    return max(0, a - 2), min(len(speeds) - 1, b + 2)


@dataclass
class MotionTemplate:
    label: str
    take_id: str
    tip: int                 # landmark index of the fingertip that draws the letter
    path: np.ndarray         # (N_POINTS, 2) normalised path
    shape: np.ndarray        # mean feature vector during the movement
    start_shape: np.ndarray  # feature vector at the start of the movement
    path_len: float          # in palm lengths
    duration: float
    wrist_ratio: float = 1.0  # how much the wrist travels compared with the fingertip
    frames: np.ndarray | None = None  # feature vectors of every frame of the movement
    start_label: str | None = None  # static letter with the same hand shape (e.g. "I" for J)


@dataclass
class MotionMatch:
    label: str
    dtw: float
    shape_ratio: float
    start_t: float
    end_t: float
    template: MotionTemplate | None = None


def build_template(label: str, take_id: str, times: np.ndarray, pts_seq: np.ndarray,
                   feats_seq: np.ndarray, speeds: np.ndarray, **seg_kw) -> MotionTemplate | None:
    seg = find_motion_segment(times, speeds, pts_seq=pts_seq, **seg_kw)
    if seg is None:
        return None
    a, b = seg
    pts = pts_seq[a:b + 1]
    if len(pts) < 4:
        return None
    palm = float(np.median(np.linalg.norm(pts[:, 9] - pts[:, 0], axis=1))) or 1e-6
    lengths = [path_length(pts[:, t, :2]) for t in TIPS]
    tip = TIPS[int(np.argmax(lengths))]
    feats = np.asarray(feats_seq[a:b + 1])
    return MotionTemplate(
        label=label, take_id=take_id, tip=tip,
        path=normalize_path(pts[:, tip, :2]),
        shape=feats.mean(axis=0),
        start_shape=feats[:3].mean(axis=0),
        path_len=max(lengths) / palm,
        duration=float(times[b] - times[a]),
        wrist_ratio=path_length(pts[:, 0, :2]) / max(max(lengths), 1e-9),
        frames=feats,
    )


class MotionMatcher:
    """Matches a finished movement against the recorded motion signs.

    A match needs all of:
      * a similar fingertip path (DTW), judged against how consistent the
        person's own recordings of that letter are
      * a similar amount of movement, with the hand travelling as a whole
        (not just fingers curling, which is what happens between letters)
      * the letter's hand shape held through most of the movement
    """

    def __init__(self, templates: list[MotionTemplate], model=None, dtw_tol: float = 0.30,
                 shape_tol: float = 2.5, min_path_frac: float = 0.45, shape_frac: float = 0.75,
                 agree_frac: float = 0.5):
        self.templates = [t for t in templates if t is not None]
        self.model = model
        self.dtw_tol = dtw_tol
        self.shape_tol = shape_tol
        self.min_path_frac = min_path_frac
        self.shape_frac = shape_frac
        self.agree_frac = agree_frac
        self._motion_shapes: dict[str, np.ndarray] = {}
        if self._has_model():
            for t in self.templates:
                t.start_label = model.predict_shape_label(t.start_shape)
            for label in self.labels:
                frames = [t.frames for t in self.templates if t.label == label and t.frames is not None]
                if frames:
                    self._motion_shapes[label] = model.transform(np.concatenate(frames))[:, SHAPE_MASK]
        self.thresholds = {}
        for label in self.labels:
            same = [t for t in self.templates if t.label == label]
            pair = [dtw_distance(a.path, b.path) for i, a in enumerate(same) for b in same[i + 1:]]
            if pair:
                self.thresholds[label] = float(np.clip(3.0 * np.median(pair), 0.10, dtw_tol))
            else:
                self.thresholds[label] = 0.7 * dtw_tol

    def _has_model(self) -> bool:
        return self.model is not None and self.model.ready

    @property
    def labels(self) -> list[str]:
        return sorted({t.label for t in self.templates})

    def match(self, times: np.ndarray, pts_seq: np.ndarray, feats_seq: np.ndarray) -> MotionMatch | None:
        if not self.templates or len(pts_seq) < 4:
            return None
        feats_seq = np.asarray(feats_seq)
        palm = float(np.median(np.linalg.norm(pts_seq[:, 9] - pts_seq[:, 0], axis=1))) or 1e-6
        duration = float(times[-1] - times[0])
        wrist_len = path_length(pts_seq[:, 0, :2])
        if self._has_model():
            zf = self.model.transform(feats_seq)[:, SHAPE_MASK]
            scale = self.model.median_scale
        best = None
        for t in self.templates:
            if not (0.3 * t.duration <= duration <= 3.5 * t.duration + 0.3):
                continue
            xy = pts_seq[:, t.tip, :2]
            tip_len = path_length(xy)
            if tip_len / palm < self.min_path_frac * t.path_len:
                continue
            if wrist_len / max(tip_len, 1e-9) < 0.4 * t.wrist_ratio:
                continue
            shape_ratio = 0.0
            if self._has_model():
                zt = self.model.transform(t.shape)[SHAPE_MASK]
                ratios = np.linalg.norm(zf - zt, axis=1) / scale
                shape_ratio = float(np.median(ratios))
                if shape_ratio > self.shape_tol or float(np.mean(ratios <= self.shape_tol)) < self.shape_frac:
                    continue
            d = dtw_distance(normalize_path(xy), t.path)
            score = d / self.thresholds[t.label]
            if best is None or score < best[0]:
                best = (score, MotionMatch(t.label, d, shape_ratio, float(times[0]), float(times[-1]), t))
        if best is None or best[0] > 1.0:
            return None
        m = best[1]
        if self._has_model() and m.label in self._motion_shapes:
            # Is the hand shape during the movement more like this letter's recorded
            # shape than like any still letter? (Rejects shape changes between letters.)
            d_motion = _min_dist(zf, self._motion_shapes[m.label])
            d_static = _min_dist(zf, self.model._Zs.astype(np.float64))
            if float(np.mean(d_motion <= 1.15 * d_static)) < self.agree_frac:
                return None
        return m


def _min_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For every row of a, the distance to the closest row of b."""
    d2 = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2.0 * a @ b.T
    return np.sqrt(np.maximum(d2.min(axis=1), 0.0))
