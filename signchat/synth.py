"""A simple synthetic hand, used by the built-in simulator and the tests.

It is *not* an accurate model of ASL handshapes; it only needs to produce 26
distinct, plausible-looking shapes (including some deliberately similar ones,
like A/S/T/M/N) so the whole pipeline can be exercised without VR hardware.

Hand-local frame of a right hand: +x towards the thumb side, +y towards the
fingertips, +z out of the palm.  Distances in metres.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

from .hand import HandFrame

FINGER_NAMES = ("index", "middle", "ring", "pinky")
FINGER_BASE = {"index": (0.022, 0.095, 0.0), "middle": (0.0, 0.100, 0.0),
               "ring": (-0.018, 0.093, 0.0), "pinky": (-0.034, 0.083, 0.0)}
FINGER_LEN = {"index": (0.042, 0.025, 0.022), "middle": (0.046, 0.029, 0.023),
              "ring": (0.043, 0.027, 0.023), "pinky": (0.033, 0.019, 0.020)}
FINGER_FAN = {"index": 0.08, "middle": 0.0, "ring": -0.07, "pinky": -0.15}
THUMB_CMC = np.array([0.024, 0.028, 0.012])
THUMB_LEN = (0.045, 0.032, 0.027)
OPPOSE_DIR = np.array([-1.0, 0.35, 0.9]) / np.linalg.norm([-1.0, 0.35, 0.9])

UP, FORWARD, BACK, LEFT, DOWN = (0, 1, 0), (0, 0, -1), (0, 0, 1), (-1, 0, 0), (0, -1, 0)


@dataclass(frozen=True)
class Pose:
    fingers: tuple  # 4 x (mcp, pip, dip, splay)  index, middle, ring, pinky
    thumb: tuple    # (azimuth, elevation, mcp_flex, ip_flex)
    finger_dir: tuple = UP
    palm_dir: tuple = FORWARD

    def vector(self) -> np.ndarray:
        return np.concatenate([np.ravel(self.fingers), self.thumb, self.finger_dir, self.palm_dir]).astype(float)

    @staticmethod
    def from_vector(v: np.ndarray) -> "Pose":
        v = np.asarray(v, float)
        return Pose(tuple(map(tuple, v[:16].reshape(4, 4))), tuple(v[16:20]), tuple(v[20:23]), tuple(v[23:26]))


EXT = (0.0, 0.0, 0.0, 0.0)
CURL = (1.45, 1.6, 0.9, 0.0)
HOOK = (0.15, 1.5, 1.1, 0.0)
HALF = (0.6, 0.8, 0.5, 0.0)
OCURL = (0.9, 1.0, 0.6, 0.0)
CLAW = (0.5, 1.6, 1.2, 0.0)
OVER = (1.5, 1.2, 0.6, 0.0)

T_SIDE = (0.15, 0.05, 0.1, 0.1)
T_ACROSS = (-0.5, 0.9, 0.6, 0.4)
T_FRONT = (-0.3, 1.1, 0.9, 0.6)
T_OUT = (1.2, 0.2, 0.0, 0.0)
T_OPPOSE = (0.2, 1.0, 0.5, 0.4)
T_UNDER = (-0.2, 1.3, 1.0, 0.8)
T_BETWEEN = (0.05, 0.9, 0.5, 0.2)
T_UNDER3 = (-0.35, 1.2, 0.7, 0.5)
T_UNDER2 = (-0.15, 1.1, 0.6, 0.4)
T_UP = (0.35, 0.3, 0.0, 0.0)
T_PARALLEL = (0.6, 0.1, 0.0, 0.0)


def _f(*fs):
    return tuple(fs)


def _spl(base, s):
    return (base[0], base[1], base[2], s)


LETTER_POSES: dict[str, Pose] = {
    "A": Pose(_f(CURL, CURL, CURL, CURL), T_SIDE),
    "B": Pose(_f(_spl(EXT, -0.06), EXT, _spl(EXT, 0.05), _spl(EXT, 0.12)), T_ACROSS),
    "C": Pose(_f(HALF, HALF, HALF, HALF), (0.5, 0.9, 0.3, 0.2), UP, LEFT),
    "D": Pose(_f(EXT, OCURL, OCURL, OCURL), T_OPPOSE),
    "E": Pose(_f(CLAW, CLAW, CLAW, CLAW), T_UNDER),
    "F": Pose(_f(OCURL, _spl(EXT, 0.05), _spl(EXT, 0.05), _spl(EXT, 0.05)), T_OPPOSE),
    "G": Pose(_f(EXT, CURL, CURL, CURL), T_PARALLEL, LEFT, BACK),
    "H": Pose(_f(_spl(EXT, -0.06), EXT, CURL, CURL), T_ACROSS, LEFT, BACK),
    "I": Pose(_f(CURL, CURL, CURL, EXT), T_FRONT),
    "J": Pose(_f(CURL, CURL, CURL, EXT), T_FRONT),
    "K": Pose(_f(EXT, (0.5, 0.0, 0.0, -0.1), CURL, CURL), T_UP),
    "L": Pose(_f(EXT, CURL, CURL, CURL), T_OUT),
    "M": Pose(_f(OVER, OVER, OVER, CURL), T_UNDER3),
    "N": Pose(_f(OVER, OVER, CURL, CURL), T_UNDER2),
    "O": Pose(_f(OCURL, OCURL, OCURL, OCURL), T_OPPOSE),
    "P": Pose(_f(EXT, (0.5, 0.0, 0.0, -0.1), CURL, CURL), T_UP, (0.0, -0.7, -0.7), LEFT),
    "Q": Pose(_f(EXT, CURL, CURL, CURL), T_PARALLEL, DOWN, LEFT),
    "R": Pose(_f(_spl(EXT, -0.14), _spl(EXT, 0.12), CURL, CURL), T_ACROSS),
    "S": Pose(_f(CURL, CURL, CURL, CURL), T_FRONT),
    "T": Pose(_f(CURL, CURL, CURL, CURL), T_BETWEEN),
    "U": Pose(_f(_spl(EXT, -0.06), EXT, CURL, CURL), T_ACROSS),
    "V": Pose(_f(_spl(EXT, 0.18), _spl(EXT, -0.18), CURL, CURL), T_ACROSS),
    "W": Pose(_f(_spl(EXT, 0.18), EXT, _spl(EXT, -0.12), CURL), T_ACROSS),
    "X": Pose(_f(HOOK, CURL, CURL, CURL), T_ACROSS),
    "Y": Pose(_f(CURL, CURL, CURL, EXT), T_OUT),
    "Z": Pose(_f(EXT, CURL, CURL, CURL), T_ACROSS, (0.0, 0.5, -0.85), LEFT),
    "_rest": Pose(_f((0.3, 0.4, 0.2, 0.0),) * 4, (0.3, 0.4, 0.2, 0.1), (0.0, 0.3, -0.95), LEFT),
}

SIGN_POSITION = np.array([0.16, -0.28, -0.38])   # in front of the right shoulder, below the eyes
REST_POSITION = np.array([0.22, -0.80, -0.15])   # hand lowered
MIRROR = np.array([-1.0, 1.0, 1.0])


def _rotate_towards(v: np.ndarray, target: np.ndarray, angle: float) -> np.ndarray:
    axis = np.cross(v, target)
    n = np.linalg.norm(axis)
    if n < 1e-9 or angle == 0:
        return v
    k = axis / n
    return v * np.cos(angle) + np.cross(k, v) * np.sin(angle) + k * np.dot(k, v) * (1 - np.cos(angle))


def local_points(pose: Pose, scale: float = 1.0) -> np.ndarray:
    pts = np.zeros((21, 3))
    # thumb
    az, el, mcp, ip = pose.thumb
    m = np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])
    pts[1] = THUMB_CMC
    pts[2] = pts[1] + THUMB_LEN[0] * m
    d1 = _rotate_towards(m, OPPOSE_DIR, mcp)
    pts[3] = pts[2] + THUMB_LEN[1] * d1
    d2 = _rotate_towards(d1, OPPOSE_DIR, ip)
    pts[4] = pts[3] + THUMB_LEN[2] * d2
    # fingers
    zhat = np.array([0.0, 0.0, 1.0])
    for fi, name in enumerate(FINGER_NAMES):
        f_mcp, f_pip, f_dip, splay = pose.fingers[fi]
        a = FINGER_FAN[name] + splay
        d0 = np.array([np.sin(a), np.cos(a), 0.0])
        base = 5 + 4 * fi
        pts[base] = FINGER_BASE[name]
        th = 0.0
        for j, (flex, length) in enumerate(zip((f_mcp, f_pip, f_dip), FINGER_LEN[name])):
            th += flex
            pts[base + j + 1] = pts[base + j] + length * (np.cos(th) * d0 + np.sin(th) * zhat)
    return pts * scale


def orientation_matrix(finger_dir, palm_dir) -> np.ndarray:
    y = np.asarray(finger_dir, float)
    y = y / np.linalg.norm(y)
    z = np.asarray(palm_dir, float)
    z = z - y * np.dot(z, y)
    z = z / np.linalg.norm(z)
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def render(pose: Pose, wrist: np.ndarray, hand: str = "right", scale: float = 1.0,
           roll: float = 0.0) -> np.ndarray:
    """21 viewer-space points for a pose placed at `wrist`."""
    R = orientation_matrix(pose.finger_dir, pose.palm_dir)
    if roll:
        c, s = np.cos(roll), np.sin(roll)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ R
    pts = local_points(pose, scale) @ R.T + wrist
    if hand == "left":
        pts = pts * np.array([-1.0, 1.0, 1.0])
    return pts


@dataclass
class SimUser:
    """A simulated person: hand size and personal habits for every sign."""
    seed: int = 0
    hand: str = "right"
    habit: float = 0.05      # how much their signs differ from the textbook poses (rad)
    take_var: float = 0.03   # how much they vary between repetitions
    jitter_m: float = 0.0015  # tracking noise per landmark

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.scale = float(rng.uniform(0.92, 1.08))
        self.personal = {k: self._perturb(p, rng, self.habit) for k, p in LETTER_POSES.items()}
        self._sign_habits: dict = {}

    @property
    def other_hand(self) -> str:
        return "left" if self.hand == "right" else "right"

    def sign_habit(self, gloss: str):
        """How this person signs a word compared with everyone else: (location offset, size, speed)."""
        h = self._sign_habits.get(gloss)
        if h is None:
            rng = np.random.default_rng([self.seed, zlib.crc32(gloss.encode())])
            h = (rng.normal(0, self.habit * 0.4, 3), float(rng.uniform(0.88, 1.12)), float(rng.uniform(0.85, 1.15)))
            self._sign_habits[gloss] = h
        return h

    @staticmethod
    def _perturb(pose: Pose, rng, amount: float) -> Pose:
        v = pose.vector()
        v[:20] += rng.normal(0, amount, 20)
        v[20:26] += rng.normal(0, amount * 0.8, 6)
        return Pose.from_vector(v)

    def take_pose(self, label: str, rng) -> Pose:
        return self._perturb(self.personal[label], rng, self.take_var)


def _lerp_pose(a: Pose, b: Pose, u: float) -> Pose:
    return Pose.from_vector((1 - u) * a.vector() + u * b.vector())


def _smooth(u: float) -> float:
    return u * u * (3 - 2 * u)


J_PATH = np.array([[0.0, 0.0], [0.0, -0.05], [-0.01, -0.085], [-0.04, -0.095], [-0.06, -0.07]])
Z_PATH = np.array([[0.0, 0.0], [0.065, 0.0], [0.0, -0.065], [0.065, -0.065]])
MOTION_PATHS = {"J": (J_PATH, 0.6), "Z": (Z_PATH, 0.8)}


def _path_point(path: np.ndarray, u: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = u * cum[-1]
    return np.array([np.interp(s, cum, path[:, 0]), np.interp(s, cum, path[:, 1])])


class Performer:
    """Turns a script of signs into a stream of frames.

    Script items: a letter ("A"), " " (pause = space), "|" (lower the hand),
    or ("rest", seconds).  Double letters get a small bounce automatically.
    """

    def __init__(self, user: SimUser, fps: float = 60.0, seed: int = 1, hold_s: float = 0.55,
                 trans_s: float = 0.28, pause_s: float = 1.6, drop_s: float = 1.6, t0: float = 0.0):
        self.user, self.fps, self.hold_s, self.trans_s = user, fps, hold_s, trans_s
        self.pause_s, self.drop_s = pause_s, drop_s
        self.rng = np.random.default_rng(seed)
        self.rng_n = np.random.default_rng(seed + 7919)   # the other hand has its own noise
        self.t = t0
        self.pose = user.take_pose("_rest", self.rng)
        self.pos = REST_POSITION.copy()
        self.roll = 0.0
        # the other (non-dominant) hand, in mirrored "right-handed signer" coordinates (x < 0)
        self.npose = user.take_pose("_rest", self.rng_n)
        self.npos = REST_POSITION * MIRROR
        self.nroll = 0.0
        self.other_visible = True
        self.frames: list[HandFrame] = []          # the dominant hand
        self.frames_n: list[HandFrame | None] = []  # the other hand at the same moments

    def _emit(self, pose: Pose, pos: np.ndarray, roll: float | None = None):
        roll = self.roll if roll is None else roll
        pts = render(pose, pos, self.user.hand, self.user.scale, roll)
        pts = pts + self.rng.normal(0, self.user.jitter_m, pts.shape)
        self.frames.append(HandFrame(self.t, self.user.hand, pts, head_relative=True, quality="sim"))
        if self.other_visible:
            other = self.user.other_hand
            npts = render(self.npose, self.npos * MIRROR, other, self.user.scale, self.nroll)
            npts = npts + self.rng_n.normal(0, self.user.jitter_m, npts.shape)
            self.frames_n.append(HandFrame(self.t, other, npts, head_relative=True, quality="sim"))
        else:
            self.frames_n.append(None)
        self.t += 1.0 / self.fps

    def batches(self, start: int = 0) -> list[list[HandFrame]]:
        """Tracking updates as an input source would deliver them (both hands)."""
        return [[f] + ([n] if n is not None else []) for f, n in zip(self.frames[start:], self.frames_n[start:])]

    def _move(self, pose: Pose, pos: np.ndarray, dur: float):
        a_pose, a_pos, a_roll = self.pose, self.pos, self.roll
        n = max(1, int(round(dur * self.fps)))
        for i in range(1, n + 1):
            u = _smooth(i / n)
            self._emit(_lerp_pose(a_pose, pose, u), (1 - u) * a_pos + u * pos, (1 - u) * a_roll)
        self.pose, self.pos, self.roll = pose, pos.copy(), 0.0

    def _move_both(self, pose: Pose, pos: np.ndarray, roll: float, npose: Pose, npos: np.ndarray,
                   nroll: float, dur: float):
        a_pose, a_pos, a_roll = self.pose, self.pos, self.roll
        b_pose, b_pos, b_roll = self.npose, self.npos, self.nroll
        n = max(1, int(round(dur * self.fps)))
        for i in range(1, n + 1):
            u = _smooth(i / n)
            self.npose = _lerp_pose(b_pose, npose, u)
            self.npos = (1 - u) * b_pos + u * npos
            self.nroll = (1 - u) * b_roll + u * nroll
            self._emit(_lerp_pose(a_pose, pose, u), (1 - u) * a_pos + u * pos, (1 - u) * a_roll + u * roll)
        self.pose, self.pos, self.roll = pose, np.array(pos, float), roll
        self.npose, self.npos, self.nroll = npose, np.array(npos, float), nroll

    def lower_hands(self, dur: float = 0.4) -> None:
        self._move_both(self.user.take_pose("_rest", self.rng), REST_POSITION, 0.0,
                        self.user.take_pose("_rest", self.rng_n), REST_POSITION * MIRROR, 0.0, dur)

    def _hold(self, dur: float, drift: float = 0.002):
        n = max(1, int(round(dur * self.fps)))
        for _ in range(n):
            self._emit(self.pose, self.pos + self.rng.normal(0, drift / 3, 3))

    def motion(self, label: str, scale: float = 1.0, dur: float | None = None):
        path, base_dur = MOTION_PATHS[label]
        dur = (dur or base_dur) * float(self.rng.uniform(0.85, 1.15))
        path = path * scale * float(self.rng.uniform(0.85, 1.15))
        start = self.pos.copy()
        n = max(4, int(round(dur * self.fps)))
        for i in range(1, n + 1):
            u = _smooth(i / n)
            off = _path_point(path, u)
            p = start + np.array([off[0], off[1], 0.0])
            self._emit(self.pose, p)
        self.pos = start + np.array([*_path_point(path, 1.0), 0.0])

    def play(self, script) -> list[HandFrame]:
        prev = None
        sign_pos = SIGN_POSITION.copy()
        for item in script:
            if isinstance(item, tuple) and item[0] == "rest":
                self.lower_hands(0.4)
                self._hold(item[1])
                prev = None
                continue
            if isinstance(item, tuple) and item[0] == "sign":
                self.sign(item[1], *item[2:])
                prev = None
                continue
            if item == "|":
                self.lower_hands(0.4)
                self._hold(self.drop_s)
                prev = None
                continue
            if item == " ":
                self._hold(self.pause_s)
                prev = None
                continue
            label = item.upper()
            pose = self.user.take_pose(label, self.rng)
            if label == prev and label not in MOTION_PATHS:
                # double letter: small bounce sideways and back
                bounce = sign_pos + np.array([0.045, 0.0, 0.0])
                self._move(self.pose, bounce, 0.14)
                self._move(self.pose, sign_pos, 0.14)
                self._hold(self.hold_s)
            else:
                self._move(pose, sign_pos, self.trans_s)
                if label in MOTION_PATHS:
                    self._hold(0.12)
                    self.motion(label)
                    self._hold(0.35)
                    self._move(self.pose, sign_pos, 0.2)
                else:
                    self._hold(self.hold_s)
            prev = label
        return self.frames

    def sign(self, gloss: str, kind: str | None = None, speed: float = 1.0, trans_s: float | None = None):
        """Sign a word or phrase, starting from wherever the hands are."""
        parts = phrase_parts(gloss, kind)
        for n, part in enumerate(parts):
            self._sign_one(sign_spec(part), speed, trans_s if trans_s is not None else (0.2 if n else None))
        return self.frames


    def _sign_one(self, spec: SignSpec, speed: float = 1.0, trans_s: float | None = None) -> None:
        user, rng = self.user, self.rng
        loc_off, h_scale, h_speed = user.sign_habit(spec.gloss)
        loc = np.array(spec.loc) + loc_off + rng.normal(0, 0.008, 3)
        size = spec.size * h_scale * float(rng.uniform(0.93, 1.07))
        dur = spec.dur / (h_speed * speed) * float(rng.uniform(0.9, 1.1))
        d0 = user.take_pose(spec.d_start, rng)
        d1 = d0 if spec.d_end == spec.d_start else user.take_pose(spec.d_end, rng)

        def other_at(u):
            if spec.two == "mirror":
                off, roll = sign_offset(spec, u, size)
                return (loc + off) * MIRROR, roll
            if spec.two == "alternate":
                off, roll = sign_offset(spec, (u + 0.5) % 1.0, size)
                return (loc + off) * MIRROR, roll
            if spec.two == "base":
                return np.array(spec.n_loc) + loc_off, 0.0
            return None, 0.0

        if spec.two:
            n0 = user.take_pose(spec.n_pose, self.rng_n)
            npos0, nroll0 = other_at(0.0)
        else:
            n0 = self.npose if np.allclose(self.npos, REST_POSITION * MIRROR, atol=0.05) else \
                user.take_pose("_rest", self.rng_n)
            npos0, nroll0 = REST_POSITION * MIRROR, 0.0
        trans = self.trans_s if trans_s is None else trans_s
        self._move_both(d0, loc, 0.0, n0, npos0, nroll0, trans)
        n = max(4, int(round(dur * self.fps)))
        for i in range(1, n + 1):
            u = i / n
            off, roll = sign_offset(spec, u, size)
            pose = d0 if d1 is d0 else _lerp_pose(d0, d1, _smooth(float(np.clip((u - 0.25) / 0.5, 0.0, 1.0))))
            if spec.two:
                self.npos, self.nroll = other_at(u)
            self.pose, self.pos, self.roll = pose, loc + off, roll
            self._emit(pose, loc + off, roll)
        self._hold(0.08)

    def hold_sign(self, label: str, dur: float) -> list[HandFrame]:
        """Used for recording takes: form the sign and hold it."""
        self._move(self.user.take_pose(label, self.rng), SIGN_POSITION, self.trans_s)
        self._hold(dur)
        return self.frames


# ---------------------------------------------------------------- whole signs
# Made-up signs, generated from the gloss so every gloss always looks the same.
# They are NOT real ASL; they only need to be distinct, plausible hand
# movements (one- and two-handed, with handshape changes) to test the
# recogniser and to let people try the app without VR.

SIGN_LOCATIONS = {          # wrist positions relative to the eyes (m), right-handed signer
    "forehead": (0.04, 0.02, -0.16), "eyes": (0.07, -0.04, -0.20), "chin": (0.03, -0.17, -0.18),
    "cheek": (0.11, -0.10, -0.16), "chest": (0.05, -0.40, -0.24), "shoulder": (0.17, -0.27, -0.26),
    "neutral": (0.13, -0.32, -0.40), "low": (0.11, -0.47, -0.36), "side": (0.27, -0.22, -0.33),
}
SIGN_SHAPES = tuple("ABCDEFGHIKLMNOPRSTUVWXY")
_DIRS = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, -1), (0, 0, 1)], float)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) or 1.0)


@dataclass(frozen=True)
class SignSpec:
    gloss: str
    d_start: str            # dominant handshape (a letter pose)
    d_end: str              # handshape at the end
    loc: tuple
    move: str               # line, arc, circle, tap, twist, shake
    vec: tuple
    vec2: tuple
    size: float
    dur: float
    two: str = ""           # "", "mirror", "base" or "alternate"
    n_pose: str = "B"
    n_loc: tuple = (0.0, 0.0, 0.0)


def sign_spec(gloss: str) -> SignSpec:
    rng = np.random.default_rng(zlib.crc32(gloss.encode()))
    d_start = str(rng.choice(SIGN_SHAPES))
    d_end = d_start if rng.random() < 0.65 else str(rng.choice(SIGN_SHAPES))
    loc = np.array(SIGN_LOCATIONS[str(rng.choice(list(SIGN_LOCATIONS)))]) + rng.normal(0, 0.015, 3)
    move = str(rng.choice(["line", "line", "arc", "arc", "circle", "tap", "twist", "shake"]))
    vec = _unit(_DIRS[rng.integers(len(_DIRS))] + rng.normal(0, 0.35, 3))
    vec2 = _unit(np.cross(vec, _unit(rng.normal(0, 1, 3))))
    size = {"line": rng.uniform(0.10, 0.22), "arc": rng.uniform(0.10, 0.20), "circle": rng.uniform(0.04, 0.07),
            "tap": rng.uniform(0.03, 0.05), "twist": 0.02, "shake": rng.uniform(0.04, 0.06)}[move]
    dur = float(rng.uniform(0.5, 0.9))
    two, n_pose, n_loc = "", "B", REST_POSITION * MIRROR
    if rng.random() < 0.4:
        two = str(rng.choice(["mirror", "base", "alternate"]))
        if two == "alternate" and move not in ("circle", "tap", "shake"):
            two = "mirror"
        if two == "base":
            n_pose = str(rng.choice(["B", "A", "S", "C", "O"]))
            n_loc = np.array([-0.05, -0.42, -0.32]) + rng.normal(0, 0.01, 3)
            loc = np.array([0.03, -0.32, -0.32]) + rng.normal(0, 0.01, 3)
        else:
            n_pose = d_start
            loc = loc.copy()
            loc[0] = max(loc[0], 0.10)
            n_loc = loc * MIRROR
    return SignSpec(gloss, d_start, d_end, tuple(loc), move, tuple(vec), tuple(vec2), float(size), dur, two,
                    n_pose, tuple(n_loc))


def sign_offset(spec: SignSpec, u: float, size: float) -> tuple[np.ndarray, float]:
    """Where the dominant hand is (relative to the start) at u in [0, 1], and its roll."""
    vec, vec2 = np.array(spec.vec), np.array(spec.vec2)
    su = _smooth(float(np.clip(u, 0.0, 1.0)))
    if spec.move == "line":
        return vec * size * su, 0.0
    if spec.move == "arc":
        return vec * size * su + vec2 * 0.35 * size * np.sin(np.pi * su), 0.0
    if spec.move == "circle":
        th = 2 * np.pi * u
        return size * ((np.cos(th) - 1) * vec + np.sin(th) * vec2), 0.0
    if spec.move == "tap":
        return vec * size * (1 - np.cos(4 * np.pi * u)) / 2, 0.0
    if spec.move == "shake":
        return vec * size * np.sin(4 * np.pi * u), 0.0
    return vec * size * su, 1.2 * su        # twist


def phrase_parts(gloss: str, kind: str | None = None) -> list[str]:
    """The separate signs a phrase is made of (for the simulated signer)."""
    from .vocab import STARTER_BY_GLOSS
    entry = STARTER_BY_GLOSS.get(gloss)
    if kind is None:
        kind = entry.kind if entry is not None else "word"
    words = gloss.split("-")
    if kind != "phrase" or len(words) < 2:
        return [gloss]
    parts, i = [], 0
    while i < len(words):
        for j in range(len(words), i, -1):
            cand = "-".join(words[i:j])
            e = STARTER_BY_GLOSS.get(cand)
            if j == i + 1 or (e is not None and e.kind == "word"):
                parts.append(cand)
                i = j
                break
    return parts
