"""Whole-word and phrase signs.

Recording
---------
Each recording of a sign becomes a *template*: 30 times a second it stores
both hands' position (relative to your head, in a frame that doesn't tilt
when you look up or down), hand shape and orientation, plus where the hands
are relative to each other.  It is trimmed to the part where the hands sign
(raising the hands into position and dropping them afterwards is cut off).
If your other hand isn't part of the sign, the template ignores it.

Recognising
-----------
While you sign, every template is compared with what your hands just did,
using a streaming version of dynamic time warping (the SPRING algorithm,
Sakurai et al. 2007).  A template can match anywhere in the stream, faster
(up to 2x) or slower than you recorded it, and the first and last few frames
may be skipped (signs blend into each other in fluent signing).  Each sign
gets its own acceptance threshold, worked out from how much your recordings
of that sign differ from each other.  When a sign matches and nothing better
turns up a moment later, it is typed.  A longer sign or phrase that matches
later replaces the shorter signs it contains
("NICE" "MEET" "YOU" -> "nice to meet you").
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import islice

import numpy as np

from .features import GROUP_SLICES, extract_features, extract_features_batch

SIGN_FPS = 30.0
SIGN_DT = 1.0 / SIGN_FPS
PALM_IDX = [0, 5, 9, 13, 17]
SPACE_Y = -0.60            # palm more than this far below eye level (m) = hand lowered, not signing
VEL_LAG = 3                # velocity is measured over 3 frames (0.1 s), backwards, so it works live
VEL_SCALE = 0.5            # m/s
VEL_WEIGHT = 0.8

# ---------------------------------------------------------------- features
HAND_DIM = 40              # palm position 3, contacts 12, joint angles 19, orientation 6
DIM = 2 * HAND_DIM + 6     # dominant hand, other hand, other-minus-dominant position, dominant hand velocity
BLOCK_SLICES = (slice(0, 3), slice(3, 34), slice(34, 40),        # dominant: position, shape, orientation
                slice(40, 43), slice(43, 74), slice(74, 80),     # other hand
                slice(80, 83),                                   # hands relative to each other
                slice(83, 86))                                   # how the dominant hand is moving
BLOCK_NAMES = ("d_pos", "d_shape", "d_ori", "n_pos", "n_shape", "n_ori", "rel", "d_vel")
BLOCK_WEIGHTS = np.array([1.6, 1.0, 0.7, 0.9, 0.5, 0.4, 0.8, VEL_WEIGHT])
ONE_HANDED_MASK = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
_HAND_SCALE = [0.08] * 3 + [0.25] * 12 + [0.35] * 15 + [0.15] * 4 + [0.4] * 6
SCALE = np.array(_HAND_SCALE * 2 + [0.08] * 3 + [VEL_SCALE] * 3)
BLOCK_IND = np.zeros((DIM, len(BLOCK_SLICES)))
for _b, _sl in enumerate(BLOCK_SLICES):
    BLOCK_IND[_sl, _b] = 1.0
BLOCK_SIZE = BLOCK_IND.sum(axis=0)

# ---------------------------------------------------------------- matching constants
SKIP_FRACTION = 0.15       # share of a template's frames that may be skipped at its start and at its end
SKIP_PENALTY = 0.45        # cost counted for every skipped template frame
MAX_SLOWDOWN = 3.0         # a sign may be signed up to 3x slower than recorded (2x faster is built in)
MOVE_MIN_PATH = 0.08       # templates whose dominant hand travels at least this far (m) ...
MOVE_RATIO = (0.55, 2.2)   # ... only match when the hand travels 0.55x-2.2x as far
MOVE_FRAC_RATIO = 0.5      # ... and the hand must be moving for at least half as much of the time as recorded
MOVING_SPEED = 0.12        # m/s
REPEAT_GAP_S = 0.4         # the same sign right after itself is the rest of that sign, not a new one
EXTEND_GAIN = 0.9          # a longer match of the same sign must score 10% better to replace the shorter one
DROP_SPEED = 0.3           # m/s downwards: the hands are being lowered
DROP_KEEP_SCORE = 0.75     # a match that ends while the hands are lowered is kept only if it's this good
THRESHOLD_MULT = 2.0       # a sign is accepted up to 2x as far off as your recordings are from each other
THRESHOLD_MULT_PACK = 2.4
THRESHOLD_MIN, THRESHOLD_MAX = 0.35, 0.9


def hand_vector(pts: np.ndarray) -> np.ndarray:
    """40 numbers for one hand (canonical, level points). Raises ValueError for a broken hand."""
    pts = np.asarray(pts, np.float64)
    f = extract_features(pts)
    pos = pts[PALM_IDX].mean(axis=0)
    return np.concatenate([pos, f[GROUP_SLICES["contacts"]], f[GROUP_SLICES["angles"]],
                           f[GROUP_SLICES["orientation"]]]).astype(np.float64)


@lru_cache(maxsize=1)
def rest_vectors() -> tuple[np.ndarray, np.ndarray]:
    """Placeholder hands (relaxed, lowered) used until a hand has been seen."""
    from .synth import LETTER_POSES, REST_POSITION, render
    right = render(LETTER_POSES["_rest"], REST_POSITION)
    return hand_vector(right), hand_vector(right * np.array([-1.0, 1.0, 1.0]))


def mirror(pts: np.ndarray | None) -> np.ndarray | None:
    return None if pts is None else np.asarray(pts, np.float64) * np.array([-1.0, 1.0, 1.0])


def canonical_hands(right: np.ndarray | None, left: np.ndarray | None, dominant: str):
    """Physical hands -> (dominant, other) as seen by a right-handed signer
    (a left-handed signer's hands are mirrored)."""
    if dominant == "left":
        return mirror(left), mirror(right)
    return right, left


class FrameEncoder:
    """Turns (dominant, other) hand points into one feature frame (without the
    velocity block, which LiveEncoder adds). A hand that drops out of tracking
    keeps its last known state."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        d, n = rest_vectors()
        self.d, self.n = d.copy(), n.copy()

    def encode(self, d_pts: np.ndarray | None, n_pts: np.ndarray | None) -> np.ndarray:
        if d_pts is not None:
            try:
                self.d = hand_vector(d_pts)
            except ValueError:
                pass
        if n_pts is not None:
            try:
                self.n = hand_vector(n_pts)
            except ValueError:
                pass
        return np.concatenate([self.d, self.n, self.n[:3] - self.d[:3]])


def palm_velocity(pos: np.ndarray) -> np.ndarray:
    """Velocity (m/s) of a position track at 30 fps, measured backwards over VEL_LAG frames."""
    pos = np.asarray(pos, np.float64)
    v = np.zeros_like(pos)
    for i in range(1, min(VEL_LAG, len(pos))):
        v[i] = (pos[i] - pos[0]) / (i * SIGN_DT)
    if len(pos) > VEL_LAG:
        v[VEL_LAG:] = (pos[VEL_LAG:] - pos[:-VEL_LAG]) / (VEL_LAG * SIGN_DT)
    return v


class LiveEncoder:
    """Irregular tracking updates (physical right/left hands) -> 30 fps feature frames."""

    def __init__(self, dominant: str = "right", hold_s: float = 0.15):
        self.dominant = dominant
        self.hold_s = hold_s
        self.enc = FrameEncoder()
        self.reset()

    def reset(self) -> None:
        self.enc.reset()
        self.next_t: float | None = None
        self.d = self.n = None
        self.d_t = self.n_t = -1e9
        self.recent: deque = deque(maxlen=VEL_LAG + 1)

    def _with_velocity(self, x: np.ndarray) -> np.ndarray:
        self.recent.append(x[0:3])
        k = len(self.recent) - 1
        v = (self.recent[-1] - self.recent[0]) / (k * SIGN_DT) if k else np.zeros(3)
        return np.concatenate([x, v])

    def update(self, t: float, right: np.ndarray | None, left: np.ndarray | None) -> list[tuple[float, np.ndarray]]:
        d, n = canonical_hands(right, left, self.dominant)
        if d is not None:
            self.d, self.d_t = d, t
        if n is not None:
            self.n, self.n_t = n, t
        if self.next_t is None or t - self.next_t > 0.5 or t < self.next_t - 1.0:
            self.next_t = t
        out = []
        while self.next_t <= t + 1e-9:
            g = self.next_t
            dd = self.d if g - self.d_t <= self.hold_s else None
            nn = self.n if g - self.n_t <= self.hold_s else None
            out.append((g, self._with_velocity(self.enc.encode(dd, nn))))
            self.next_t += SIGN_DT
        return out


# ---------------------------------------------------------------- recordings
_HAND_COLS = np.r_[GROUP_SLICES["contacts"], GROUP_SLICES["angles"], GROUP_SLICES["orientation"]]


def _resample(t: np.ndarray, pts: np.ndarray, ok: np.ndarray, grid: np.ndarray):
    """One hand's points at the grid times (linear interpolation; a frame next to a
    tracking gap is used as-is for up to 0.1 s). Returns (points, valid mask)."""
    n = len(t)
    i = np.searchsorted(t, grid)
    lo, hi = np.clip(i - 1, 0, n - 1), np.clip(i, 0, n - 1)
    both = ok[lo] & ok[hi]
    span = t[hi] - t[lo]
    u = np.where(span > 1e-9, (grid - t[lo]) / np.where(span > 1e-9, span, 1.0), 1.0)
    u = np.clip(u, 0.0, 1.0)[:, None, None]
    out = (1 - u) * pts[lo] + u * pts[hi]
    use_hi = ~both & ok[hi] & (np.abs(t[hi] - grid) <= 0.1)
    use_lo = ~both & ~use_hi & ok[lo] & (np.abs(t[lo] - grid) <= 0.1)
    out[use_hi] = pts[hi][use_hi]
    out[use_lo] = pts[lo][use_lo]
    return out, both | use_hi | use_lo


def _hand_vectors(P: np.ndarray, ok: np.ndarray, rest: np.ndarray) -> np.ndarray:
    """hand_vector for every frame; frames without a (usable) hand keep the last known one."""
    feats, valid = extract_features_batch(P)
    valid &= ok
    V = np.concatenate([P[:, PALM_IDX].mean(axis=1), feats[:, _HAND_COLS].astype(np.float64)], axis=1)
    idx = np.where(valid, np.arange(len(P)), -1)
    np.maximum.accumulate(idx, out=idx)
    return np.where((idx >= 0)[:, None], V[np.maximum(idx, 0)], rest[None, :])


@dataclass
class SignSequence:
    t: np.ndarray            # (m,) times at SIGN_FPS
    F: np.ndarray            # (m, DIM) feature frames
    d_present: np.ndarray    # (m,) bool
    n_present: np.ndarray    # (m,) bool


def encode_recording(t, points2, present, dominant: str) -> SignSequence:
    """A two-hand recording -> 30 fps features.

    points2: (n, 2, 21, 3) level points of the physical [right, left] hands
    present: (n, 2) bool, which hands were tracked
    """
    t = np.asarray(t, np.float64)
    points2 = np.asarray(points2, np.float64)
    present = np.asarray(present, bool)
    if len(t) == 0:
        raise ValueError("empty recording")
    grid = np.arange(t[0], t[-1] + 1e-9, SIGN_DT)
    R, r_ok = _resample(t, points2[:, 0], present[:, 0], grid)
    L, l_ok = _resample(t, points2[:, 1], present[:, 1], grid)
    if dominant == "left":
        D, d_ok, N, n_ok = L * np.array([-1.0, 1.0, 1.0]), l_ok, R * np.array([-1.0, 1.0, 1.0]), r_ok
    else:
        D, d_ok, N, n_ok = R, r_ok, L, l_ok
    rest_d, rest_n = rest_vectors()
    Fd = _hand_vectors(D, d_ok, rest_d)
    Fn = _hand_vectors(N, n_ok, rest_n)
    F = np.concatenate([Fd, Fn, Fn[:, :3] - Fd[:, :3], palm_velocity(Fd[:, :3])], axis=1)
    return SignSequence(grid, F, d_ok, n_ok)


def _diff_speed(x: np.ndarray, lag: int = 3) -> np.ndarray:
    """Per-frame rate of change, measured over `lag` frames (robust to tracking jitter)."""
    v = np.zeros(len(x))
    if len(x) > lag:
        v[lag:] = np.linalg.norm(x[lag:] - x[:-lag], axis=1) / (lag * SIGN_DT)
        v[:lag] = v[lag]
    return v


def smooth_path(pos: np.ndarray) -> np.ndarray:
    """A hand path without the tracking jitter: 3-frame average, every 3rd frame."""
    pos = np.asarray(pos, np.float64)
    if len(pos) < 4:
        return pos
    k = np.ones(3) / 3.0
    sm = np.stack([np.convolve(pos[:, c], k, mode="valid") for c in range(pos.shape[1])], axis=1)
    idx = list(range(0, len(sm), 3))
    if idx[-1] != len(sm) - 1:
        idx.append(len(sm) - 1)
    return sm[idx]


def path_length(pos: np.ndarray) -> float:
    p = smooth_path(pos)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def moving_fraction(pos: np.ndarray) -> float:
    """Share of the time the hand is moving (so a long hold followed by a jerk doesn't
    count as the same movement as a sign that moves all the way through)."""
    pos = np.asarray(pos, np.float64)
    if len(pos) < 5:
        return 0.0
    return float((_diff_speed(pos) > MOVING_SPEED).mean())


def activity(F: np.ndarray, space_y: float = SPACE_Y) -> np.ndarray:
    """Per-frame 'the hands are signing' flags: a hand moves or the handshape changes."""
    v_d = _diff_speed(F[:, 0:3])
    v_n = _diff_speed(F[:, 40:43])
    sr_d = _diff_speed(F[:, 3:34] / SCALE[3:34]) / np.sqrt(31)
    sr_n = _diff_speed(F[:, 43:74] / SCALE[43:74]) / np.sqrt(31)
    n_up = F[:, 41] >= space_y
    return (v_d > 0.12) | (sr_d > 1.5) | (n_up & ((v_n > 0.12) | (sr_n > 1.5)))


def _runs(mask: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    runs, start, prev = [], int(idx[0]), int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev > max_gap + 1:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    return runs


@dataclass
class TrimResult:
    a: int
    b: int                  # inclusive
    static: bool = False
    cut_off: bool = False   # the hands were still signing when the recording ended


def _static_window(ok: np.ndarray, length: int) -> TrimResult | None:
    runs = _runs(ok, max_gap=0)
    if not runs:
        return None
    a, b = max(runs, key=lambda r: r[1] - r[0])
    if b - a + 1 < 6:
        return None
    L = min(length, b - a + 1)
    mid = (a + b) // 2
    s = int(np.clip(mid - L // 2, a, b - L + 1))
    return TrimResult(s, s + L - 1, static=True)


def trim_sign(seq: SignSequence, space_y: float = SPACE_Y, head_relative: bool = True,
              pre_s: float = 0.15, post_s: float = 0.15) -> TrimResult | None:
    F = seq.F
    m = len(F)
    in_space = (F[:, 1] >= space_y if head_relative else np.ones(m, bool)) & seq.d_present
    if in_space.sum() < 6:
        return None
    act = activity(F, space_y if head_relative else -1e9) & seq.d_present
    runs = [r for r in _runs(act, max_gap=int(0.25 * SIGN_FPS)) if in_space[r[0]:r[1] + 1].any()]
    if not runs:
        return _static_window(in_space, int(0.6 * SIGN_FPS))
    a, b = runs[0][0], runs[-1][1]
    from_below, to_below = not in_space[a], not in_space[b]
    if from_below:                   # began with raising the hand: start where it reaches signing space
        a = a + int(np.argmax(in_space[a:b + 1]))
    if to_below:                     # ended with dropping the hand: stop where it leaves signing space
        b = b - int(np.argmax(in_space[a:b + 1][::-1]))
    if head_relative:
        pos = F[:, 0:3]
        vy = np.zeros(m)
        vy[3:] = (pos[3:, 1] - pos[:-3, 1]) / (3 * SIGN_DT)
        v = _diff_speed(pos)
        if from_below:               # the rest of the raise: fast, mostly upward movement
            while a < b and vy[a] > 0.25 and vy[a] > 0.7 * v[a]:
                a += 1
        if to_below:                 # the start of the drop: fast, mostly downward movement
            while b > a and vy[b] < -0.25 and -vy[b] > 0.7 * v[b]:
                b -= 1
    cut_off = runs[-1][1] >= m - 3 and bool(in_space[m - 1])
    a = max(0, a - int(round(pre_s * SIGN_FPS)))
    b = min(m - 1, b + int(round(post_s * SIGN_FPS)))
    if b - a + 1 < 6:
        return _static_window(in_space, int(0.6 * SIGN_FPS))
    return TrimResult(a, b, cut_off=cut_off)


@dataclass
class SignTemplate:
    label: str
    take_id: str
    F: np.ndarray            # (m, DIM)
    two_handed: bool
    d_path: float            # how far the dominant hand travels (m)
    duration: float
    static: bool = False
    cut_off: bool = False
    auto_two: bool = False   # what the automatic one/two-hand check said
    source: str = ""         # "" = your own recording, otherwise the template pack's name

    @property
    def length(self) -> int:
        return len(self.F)


def looks_two_handed(F: np.ndarray, n_present: np.ndarray, space_y: float = SPACE_Y) -> bool:
    n_pos = F[:, 40:43]
    if n_present.mean() < 0.5 or np.median(n_pos[:, 1]) < space_y:
        return False
    if path_length(n_pos) >= 0.10:
        return True
    close = np.linalg.norm(F[:, 80:83], axis=1).min() <= 0.15
    return bool(close and np.median(n_pos[:, 1]) >= space_y + 0.10)


def build_template(label: str, take_id: str, t, points2, present, dominant: str, hands: str = "auto",
                   space_y: float = SPACE_Y, head_relative: bool = True, source: str = "",
                   trim: bool = True) -> SignTemplate | None:
    """A recording -> template, or None if no sign was found in it.
    trim=False: the recording is already exactly the sign (keep all of it)."""
    seq = encode_recording(t, points2, present, dominant)
    if trim:
        tr = trim_sign(seq, space_y, head_relative)
    else:
        ok = np.flatnonzero(seq.d_present)
        tr = TrimResult(int(ok[0]), int(ok[-1])) if len(ok) else None
    if tr is None:
        return None
    F = seq.F[tr.a:tr.b + 1]
    if len(F) < 6:
        return None
    auto = looks_two_handed(F, seq.n_present[tr.a:tr.b + 1], space_y if head_relative else -1e9)
    two = {"one": False, "two": True}.get(hands, auto)
    return SignTemplate(label, take_id, F, two, path_length(F[:, 0:3]), len(F) * SIGN_DT,
                        static=tr.static, cut_off=tr.cut_off, auto_two=auto, source=source)


def harmonize_hands(templates: list[SignTemplate], overrides: dict[str, str] | None = None) -> None:
    """All recordings of a sign agree on one- or two-handed (majority; ties count as two-handed)."""
    overrides = overrides or {}
    by: dict[str, list[SignTemplate]] = {}
    for t in templates:
        by.setdefault(t.label, []).append(t)
    for label, ts in by.items():
        mode = overrides.get(label, "auto")
        if mode in ("one", "two"):
            two = mode == "two"
        else:
            two = 2 * sum(t.auto_two for t in ts) >= len(ts)
        for t in ts:
            t.two_handed = two


# ---------------------------------------------------------------- matching
class SignBank:
    """All templates in one place, with an acceptance threshold for every sign."""

    def __init__(self, templates: list[SignTemplate], tolerance: float = 1.0,
                 sensitivity: dict[str, float] | None = None, intra: dict[str, float] | None = None,
                 thresholds: dict[str, float] | None = None):
        if not templates:
            raise ValueError("no sign templates")
        self.templates = list(templates)
        self.labels = sorted({t.label for t in templates})
        self.tolerance = tolerance
        self.sensitivity = dict(sensitivity or {})
        K = len(self.templates)
        self.K = K
        self.lengths = np.array([t.length for t in self.templates])
        self.M = M = int(self.lengths.max())
        rows = np.concatenate([t.F / SCALE for t in self.templates])
        self.rows = np.ascontiguousarray(rows)
        self.row_k = np.concatenate([np.full(t.length, k) for k, t in enumerate(self.templates)])
        self.row_j = np.concatenate([np.arange(t.length) for t in self.templates])
        self.flat_idx = self.row_k * M + self.row_j
        self.row_sq = (self.rows ** 2) @ BLOCK_IND
        W = np.tile(BLOCK_WEIGHTS, (K, 1))
        for k, t in enumerate(self.templates):
            if not t.two_handed:
                W[k] = W[k] * ONE_HANDED_MASK
        self.W = W / W.sum(axis=1, keepdims=True)
        self.row_w = self.W[self.row_k] / np.sqrt(BLOCK_SIZE)
        self.d_path = np.array([t.d_path for t in self.templates])
        self.move_frac = np.array([moving_fraction(t.F[:, 0:3]) for t in self.templates])
        # skipping the first / last few frames of a template
        j = np.arange(M)[None, :]
        m = self.lengths[:, None]
        skip = np.maximum(1, np.round(SKIP_FRACTION * self.lengths)).astype(int)[:, None]
        self.start_cost = np.where(j <= skip, SKIP_PENALTY * j, np.inf) + np.zeros((K, 1))
        self.start_w = j.astype(np.float64) + np.zeros((K, 1))
        left = m - 1 - j
        self.end_cost = np.where((left >= 0) & (left <= skip), SKIP_PENALTY * left, np.inf)
        self.end_w = np.where(left >= 0, left, 0).astype(np.float64)
        # thresholds
        if thresholds is not None:
            self.label_thr = dict(thresholds)
        else:
            if intra is None:
                intra = intra_costs(self.templates)
            self.label_thr = self._thresholds(intra)
        self.thr = np.array([self.label_thr[t.label] for t in self.templates])

    def _thresholds(self, intra: dict[str, float | None]) -> dict[str, float]:
        known = [v for v in intra.values() if v is not None and np.isfinite(v)]
        fallback = float(np.median(known)) * 1.15 if known else 0.3
        own = {t.label for t in self.templates if not t.source}
        out = {}
        for label in self.labels:
            base = intra.get(label)
            if base is None or not np.isfinite(base):
                base = fallback
            # someone else's recordings (a template pack) need more slack than your own
            mult = THRESHOLD_MULT if label in own else THRESHOLD_MULT_PACK
            thr = float(np.clip(mult * base, THRESHOLD_MIN, THRESHOLD_MAX)) * self.tolerance
            out[label] = thr * float(self.sensitivity.get(label, 1.0))
        return out

    def local_costs(self, x: np.ndarray) -> np.ndarray:
        """(K, M) cost of frame x against every template frame (inf past a template's end)."""
        z = np.asarray(x, np.float64) / SCALE
        cross = self.rows @ (BLOCK_IND * z[:, None])
        d2 = self.row_sq + (z ** 2) @ BLOCK_IND - 2.0 * cross
        np.maximum(d2, 0.0, out=d2)
        cost = (np.sqrt(d2) * self.row_w).sum(axis=1)
        C = np.full(self.K * self.M, np.inf)
        C[self.flat_idx] = cost
        return C.reshape(self.K, self.M)

    def subset(self, keep: list[int]) -> "SignBank":
        return SignBank([self.templates[k] for k in keep], self.tolerance, self.sensitivity,
                        thresholds=self.label_thr)


class _DP:
    """Streaming subsequence DTW state for every template at once.

    Every step consumes one stream frame and either stays on the same template
    frame, moves to the next one, or skips one (counted twice).  A path may
    start on any of a template's first few frames and end on any of its last
    few; skipped frames cost SKIP_PENALTY each.  The score of a path is its
    total cost divided by the number of template frames it covers plus the
    number of extra (slow) steps.
    """

    def __init__(self, bank: SignBank):
        K, M = bank.K, bank.M
        self.bank = bank
        self.D = np.full((K, M), np.inf)
        self.W = np.ones((K, M))
        self.S = np.zeros((K, M), np.int64)
        self.rows = np.arange(K)
        self._c = np.empty((4, K, M))
        self._w = np.ones((4, K, M))
        self._s = np.zeros((4, K, M), np.int64)

    def step(self, C: np.ndarray, i: int):
        b = self.bank
        D, W, S = self.D, self.W, self.S
        c, w, s = self._c, self._w, self._s
        np.add(D, C, out=c[0])
        np.add(W, 1.0, out=w[0])
        s[0] = S
        c[1, :, 0] = np.inf
        np.add(D[:, :-1], C[:, 1:], out=c[1, :, 1:])
        np.add(W[:, :-1], 1.0, out=w[1, :, 1:])
        s[1, :, 1:] = S[:, :-1]
        c[2, :, :2] = np.inf
        np.add(D[:, :-2], 2.0 * C[:, 2:], out=c[2, :, 2:])
        np.add(W[:, :-2], 2.0, out=w[2, :, 2:])
        s[2, :, 2:] = S[:, :-2]
        np.add(b.start_cost, C, out=c[3])
        np.add(b.start_w, 1.0, out=w[3])
        s[3] = i
        pick = c.argmin(axis=0)[None]
        self.D = np.take_along_axis(c, pick, 0)[0]
        self.W = np.take_along_axis(w, pick, 0)[0]
        self.S = np.take_along_axis(s, pick, 0)[0]
        avg = (self.D + b.end_cost) / (self.W + b.end_w)
        j = avg.argmin(axis=1)
        r = self.rows
        return avg[r, j], self.S[r, j], b.end_w[r, j].astype(int)

    def forget_before(self, ks: np.ndarray, i_end: int) -> None:
        """Paths of templates ks that started at or before stream frame i_end can't be used again."""
        if len(ks):
            Dk = self.D[ks]
            Dk[self.S[ks] <= i_end] = np.inf
            self.D[ks] = Dk


def scan(bank: SignBank, X: np.ndarray):
    """Best match of every template anywhere inside the sequence X.
    Returns (score, start index, end index) arrays of length K."""
    dp = _DP(bank)
    best = np.full(bank.K, np.inf)
    b_start = np.zeros(bank.K, np.int64)
    b_end = np.zeros(bank.K, np.int64)
    for i, x in enumerate(X):
        avg, start, _ = dp.step(bank.local_costs(x), i)
        n = i - start + 1
        avg = np.where(n <= MAX_SLOWDOWN * bank.lengths + 3, avg, np.inf)
        upd = avg < best
        best[upd], b_start[upd], b_end[upd] = avg[upd], start[upd], i
    return best, b_start, b_end


def intra_cost(templates: list[SignTemplate]) -> float | None:
    """How well recordings of one sign match each other (median of leave-one-out best matches)."""
    if len(templates) < 2:
        return None
    costs = []
    for a, ta in enumerate(templates):
        others = [tb for b, tb in enumerate(templates) if b != a]
        sub = SignBank(others, thresholds={ta.label: 1.0})
        costs.append(float(scan(sub, ta.F)[0].min()))
    return float(np.median(costs))


def intra_costs(templates: list[SignTemplate]) -> dict[str, float | None]:
    by: dict[str, list[SignTemplate]] = {}
    for t in templates:
        by.setdefault(t.label, []).append(t)
    return {label: intra_cost(ts) for label, ts in by.items()}


@dataclass
class Detection:
    label: str
    start_t: float
    end_t: float
    score: float          # cost / threshold; <= 1 means accepted
    cost: float
    threshold: float
    k: int
    i0: int
    i1: int
    end_skip: int = 0     # template frames skipped at the end

    @property
    def span(self) -> int:
        return self.i1 - self.i0 + 1


def _inter(a: Detection, b: Detection) -> int:
    return min(a.i1, b.i1) - max(a.i0, b.i0) + 1


def _overlap(a: Detection, b: Detection) -> bool:
    return _inter(a, b) >= 0.4 * min(a.span, b.span)


def _covers(a: Detection, b: Detection) -> bool:
    """a is a clearly longer sign/phrase that contains most of b."""
    return a.label != b.label and a.span >= 1.4 * b.span and _inter(a, b) >= 0.8 * b.span


def _prefer(new: Detection, old: Detection) -> bool:
    if new.label != old.label:
        if _covers(new, old):
            return True
        if _covers(old, new):
            return False
    elif new.i1 > old.i1:
        # the same sign matched a bit longer (e.g. the final pose was held): only worth
        # waiting for if it's clearly better, otherwise holding still delays typing
        return new.score < EXTEND_GAIN * old.score
    return new.score < old.score


class SignSpotter:
    """Feeds 30 fps feature frames through the bank and reports finished signs."""

    def __init__(self, bank: SignBank, settle_s: float = 0.25, history_s: float = 60.0, space_y: float = SPACE_Y):
        self.bank = bank
        self.space_y = space_y
        self.settle = int(round(settle_s * SIGN_FPS))
        self.repeat_gap = int(round(REPEAT_GAP_S * SIGN_FPS))
        self.hist = int(history_s * SIGN_FPS)
        self.i = 0
        self.times: deque = deque(maxlen=self.hist)
        self.pos: deque = deque(maxlen=self.hist)
        self.emitted: deque = deque(maxlen=16)
        self.reset()

    def reset(self) -> None:
        """Forget partial matches (e.g. after tracking was lost)."""
        self.dp = _DP(self.bank)
        self.best: Detection | None = None
        self.later: list[Detection] = []

    @property
    def busy(self) -> bool:
        return self.best is not None

    def _base(self) -> int:
        return self.i - len(self.times)

    def _time(self, i: int) -> float:
        return self.times[int(np.clip(i - self._base(), 0, len(self.times) - 1))]

    def _positions(self, i0: int, i1: int) -> np.ndarray:
        base = self._base()
        a = max(0, i0 - base)
        b = min(len(self.pos) - 1, i1 - base)
        return np.array(list(islice(self.pos, a, b + 1))) if b > a else np.zeros((0, 3))

    def _movement_ok(self, k: int, i0: int, i1: int) -> bool:
        """Did the dominant hand move about as much (and as steadily) as in the recording?"""
        tp = self.bank.d_path[k]
        if tp < MOVE_MIN_PATH:
            return True
        pos = self._positions(i0, i1)
        travel = path_length(pos) if len(pos) > 1 else 0.0
        if not (MOVE_RATIO[0] * tp <= travel <= MOVE_RATIO[1] * tp + 0.04):
            return False
        return moving_fraction(pos) >= MOVE_FRAC_RATIO * self.bank.move_frac[k]

    def push(self, x: np.ndarray, t: float) -> list[Detection]:
        bank = self.bank
        self.times.append(t)
        self.pos.append(np.asarray(x[0:3], np.float64))
        i = self.i
        self.i += 1
        avg, start, end_skip = self.dp.step(bank.local_costs(x), i)
        n = i - start + 1
        ok = (avg <= bank.thr) & (n <= MAX_SLOWDOWN * bank.lengths + 3) & (start >= self._base())
        for k in np.flatnonzero(ok):
            s = int(start[k])
            if not self._movement_ok(int(k), s, i):
                continue
            self._offer(Detection(bank.templates[k].label, self._time(s), t, float(avg[k] / bank.thr[k]),
                                  float(avg[k]), float(bank.thr[k]), int(k), s, i, int(end_skip[k])))
        out = []
        if self._falling():
            return out            # maybe the hands are being lowered: wait and see (flush decides)
        while self.best is not None and i - self.best.i1 >= self.settle + self.best.end_skip:
            out.append(self._emit())
        return out

    def _falling(self) -> bool:
        """The dominant hand is going down fast, or is already below the signing space."""
        if not self.pos:
            return False
        if self.pos[-1][1] < self.space_y:
            return True
        if len(self.pos) < VEL_LAG + 1:
            return False
        return (self.pos[-1][1] - self.pos[-1 - VEL_LAG][1]) / (VEL_LAG * SIGN_DT) < -DROP_SPEED

    def flush(self, dropped: bool = False) -> list[Detection]:
        """Report whatever is pending now and start over. dropped=True: the hands were
        lowered; a so-so match that only fits by using that downward movement is dropped."""
        cut = self._drop_start() if dropped else None
        out = []
        while self.best is not None:
            det = self._emit()
            if cut is not None and det.i1 > cut + 2 and det.score > DROP_KEEP_SCORE:
                self.emitted.pop()
                continue
            out.append(det)
        self.reset()
        return out

    def _drop_start(self) -> int | None:
        """Stream index where the hands started going down (None if they didn't)."""
        n = min(len(self.pos), int(3 * SIGN_FPS))
        if n < 6:
            return None
        P = np.array(list(islice(self.pos, len(self.pos) - n, len(self.pos))))
        vy = (P[3:, 1] - P[:-3, 1]) / (3 * SIGN_DT)
        j = len(vy) - 1
        rest = 0
        while j >= 0 and vy[j] > -DROP_SPEED and rest < int(0.6 * SIGN_FPS):   # already down and still
            j -= 1
            rest += 1
        if j < 0 or vy[j] > -DROP_SPEED:
            return None
        while j > 0 and vy[j - 1] <= -DROP_SPEED:
            j -= 1
        return self.i - n + j

    def _offer(self, det: Detection) -> None:
        for e in self.emitted:
            if _overlap(e, det) and not _covers(det, e):
                return            # already typed (or a weaker reading of the same movement)
            if e.label == det.label and det.i0 - e.i1 <= self.repeat_gap:
                return            # the second half of a sign that was already typed
        if self.best is None:
            self.best = det
            return
        if _overlap(self.best, det):
            if _prefer(det, self.best):
                self.best = det
            return
        for j, other in enumerate(self.later):
            if _overlap(other, det):
                if _prefer(det, other):
                    self.later[j] = det
                return
        self.later.append(det)

    def _emit(self) -> Detection:
        det = self.best
        lens = self.bank.lengths
        self.dp.forget_before(np.flatnonzero(lens <= 1.4 * lens[det.k]), det.i1)
        self.emitted.append(det)
        self.later = [c for c in self.later if not _overlap(c, det)]
        self.best = None
        if self.later:
            self.later.sort(key=lambda c: c.i1)
            self.best = self.later.pop(0)
        return det


# ---------------------------------------------------------------- accuracy check
def _summary(t: SignTemplate) -> np.ndarray:
    Z = t.F / SCALE
    return np.concatenate([p.mean(axis=0) for p in np.array_split(Z, 3)])


@dataclass
class SignReport:
    overall: float
    per_label: dict = field(default_factory=dict)
    untestable: list = field(default_factory=list)

    def worst(self, n: int = 6):
        rows = []
        for label, info in self.per_label.items():
            conf = info["confused_with"][0][0] if info["confused_with"] else None
            rows.append((label, info["acc"], conf))
        rows.sort(key=lambda r: r[1])
        return [r for r in rows[:n] if r[1] < 0.999]


def evaluate_bank(bank: SignBank, shortlist: int = 6, should_stop=None) -> SignReport:
    """Every recording is tested against the other recordings (of all signs)."""
    T = bank.templates
    summaries = np.stack([_summary(t) for t in T])
    counts: dict[str, int] = {}
    for t in T:
        counts[t.label] = counts.get(t.label, 0) + 1
    per: dict[str, dict] = {}
    for a, ta in enumerate(T):
        if should_stop is not None and should_stop():
            break
        if counts[ta.label] < 2 or ta.source:
            continue
        d = np.linalg.norm(summaries - summaries[a], axis=1)
        d[a] = np.inf
        labels: list[str] = []
        for k in np.argsort(d):
            if T[k].label not in labels:
                labels.append(T[k].label)
            if len(labels) >= shortlist:
                break
        if ta.label not in labels:
            labels.append(ta.label)
        keep = [k for k in range(len(T)) if k != a and T[k].label in labels]
        sub = bank.subset(keep)
        cost, s0, s1 = scan(sub, ta.F)
        score = cost / sub.thr
        best = None              # same choice as live: a longer sign/phrase wins over the signs it contains
        for k in sorted(np.flatnonzero(score <= 1.0), key=lambda k: score[k]):
            c = Detection(sub.templates[k].label, 0.0, 0.0, float(score[k]), float(cost[k]), float(sub.thr[k]),
                          int(k), int(s0[k]), int(s1[k]))
            if best is None or _prefer(c, best):
                best = c
        pred = best.label if best is not None else None
        info = per.setdefault(ta.label, {"n": 0, "ok": 0, "unknown": 0, "conf": {}})
        info["n"] += 1
        if pred == ta.label:
            info["ok"] += 1
        elif pred is None:
            info["unknown"] += 1
        else:
            info["conf"][pred] = info["conf"].get(pred, 0) + 1
    out, tot, ok = {}, 0, 0
    for label, info in per.items():
        n = max(info["n"], 1)
        conf = sorted(info["conf"].items(), key=lambda kv: -kv[1])
        out[label] = {"acc": info["ok"] / n, "unknown": info["unknown"] / n, "n": info["n"],
                      "confused_with": [(c, v / n) for c, v in conf]}
        tot += info["n"]
        ok += info["ok"]
    untestable = sorted(lab for lab, c in counts.items() if c < 2)
    return SignReport(ok / tot if tot else 0.0, out, untestable)
