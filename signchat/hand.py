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

    def palm_size(self) -> float:
        return float(np.linalg.norm(self.points[9] - self.points[0]))


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
