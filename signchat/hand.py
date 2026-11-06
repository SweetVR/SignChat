"""The hand data every input source produces.

All sources convert their tracking data into the same 21 landmarks that
MediaPipe uses, expressed in a *viewer* frame seen from the signer's eyes:

    x = to the signer's right, y = up, z = backwards (towards the signer)

For SteamVR the viewer is the headset and units are metres.  For a webcam the
camera image is rotated into the same convention (units are approximately
metres as well).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

NUM_LANDMARKS = 21
WRIST = 0
THUMB = (1, 2, 3, 4)
INDEX = (5, 6, 7, 8)
MIDDLE = (9, 10, 11, 12)
RING = (13, 14, 15, 16)
PINKY = (17, 18, 19, 20)
TIPS = (4, 8, 12, 16, 20)
MOTION_POINTS = (0, 4, 8, 12, 16, 20)  # wrist + fingertips, used for speed

# SteamVR hand skeleton (31 bones) -> MediaPipe landmark order.
# Bone 1 is the wrist; each finger's joints start at its proximal bone.
STEAMVR_BONE_FOR_LANDMARK = (
    1,              # wrist
    2, 3, 4, 5,     # thumb metacarpal, proximal, distal, tip
    7, 8, 9, 10,    # index  MCP, PIP, DIP, tip
    12, 13, 14, 15,  # middle
    17, 18, 19, 20,  # ring
    22, 23, 24, 25,  # pinky
)


@dataclass
class HandFrame:
    t: float               # seconds (monotonic clock of the source)
    hand: str              # "left" or "right"
    points: np.ndarray     # (21, 3) in the viewer frame described above
    head_relative: bool = False  # True when the viewer is the headset (enables "hand lowered" checks)
    quality: str = ""      # "full", "partial", "estimated", "camera", "sim"
    # 3x3 rotation from the viewer frame to a *level* frame: same origin and
    # facing direction as the head, but not tilted when you look up/down or
    # tilt your head. None = the viewer frame is already level.
    level: np.ndarray | None = None

    def palm_size(self) -> float:
        return float(np.linalg.norm(self.points[9] - self.points[0]))

    def level_points(self) -> np.ndarray:
        pts = np.asarray(self.points, dtype=np.float64)
        return pts if self.level is None else pts @ np.asarray(self.level, np.float64).T


def level_rotation(head_rot: np.ndarray) -> np.ndarray:
    """Rotation from the head frame to a level frame that only follows the
    head's left/right turning (yaw), for a head rotation matrix whose columns
    are the head's x, y, z axes in a y-up world."""
    head_rot = np.asarray(head_rot, np.float64)
    fwd = -head_rot[:, 2]
    flat = np.array([fwd[0], 0.0, fwd[2]])
    n = float(np.linalg.norm(flat))
    if n < 1e-3:   # looking straight up or down: use the head's up axis instead
        up = head_rot[:, 1] * (1.0 if fwd[1] < 0 else -1.0)
        flat = np.array([up[0], 0.0, up[2]])
        n = float(np.linalg.norm(flat)) or 1.0
    z = -flat / n
    y = np.array([0.0, 1.0, 0.0])
    x = np.cross(y, z)
    yaw_rot = np.column_stack([x, y, z])
    return yaw_rot.T @ head_rot


def canonical_points(frame: HandFrame) -> np.ndarray:
    """Points with a left hand mirrored into right-hand space.

    Everything downstream (features, motion paths) works on right hands, so a
    profile recorded with one hand also works with the other.
    """
    pts = np.asarray(frame.points, dtype=np.float64)
    if frame.hand == "left":
        pts = pts * np.array([-1.0, 1.0, 1.0])
    return pts


class SpeedMeter:
    """How fast the hand moves (wrist + fingertips), in palm lengths per second.

    Measured over a ~0.1 s window rather than frame to frame, so normal
    tracking jitter (worse at 90-120 Hz) doesn't look like movement.
    """

    def __init__(self, window: float = 0.1, max_gap: float = 0.25):
        self.window = window
        self.max_gap = max_gap
        self.buf: deque = deque()

    def reset(self) -> None:
        self.buf.clear()

    def update(self, t: float, pts: np.ndarray) -> float:
        p = np.asarray(pts, np.float64)[list(MOTION_POINTS)]
        if self.buf and (t - self.buf[-1][0] > self.max_gap or t <= self.buf[-1][0]):
            self.buf.clear()
        self.buf.append((t, p))
        while len(self.buf) > 1 and self.buf[1][0] <= t - self.window:
            self.buf.popleft()
        t0, p0 = self.buf[0]
        dt = t - t0
        if dt < 0.02:
            return 0.0
        palm = float(np.linalg.norm(pts[9] - pts[0])) or 1e-6
        return float(np.linalg.norm(p - p0, axis=1).mean() / dt / palm)


def speeds_for_sequence(times: np.ndarray, pts_seq: np.ndarray) -> np.ndarray:
    """Per-frame speeds (palm lengths/s) for a recorded sequence."""
    meter = SpeedMeter()
    return np.array([meter.update(float(t), p) for t, p in zip(times, pts_seq)])
