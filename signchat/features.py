"""Turn 21 hand landmarks into a feature vector for the classifier.

Feature groups
--------------
shape        landmark positions in a hand-local frame, scaled by palm size
             (ignores where the hand is and how it is rotated)
contacts     thumb-to-finger and fingertip-to-fingertip distances
angles       how much every joint is bent, plus the spread between fingers
orientation  which way the fingers point and the palm faces, as seen from the
             signer's eyes (separates G/H from U, K from P, and so on)

Expects canonical (right-hand) points, see hand.canonical_points.
"""

from __future__ import annotations

import numpy as np

COORD_IDX = np.array([i for i in range(1, 21) if i != 9])  # wrist is the origin, middle MCP is the y axis
CONTACT_PAIRS = np.array([
    (4, 8), (4, 12), (4, 16), (4, 20),   # thumb tip to each fingertip
    (4, 6), (4, 10), (4, 14), (4, 5),    # thumb tip to index/middle/ring middle joints, index knuckle
    (8, 12), (12, 16), (16, 20), (8, 16),  # fingertip spread
])
ANGLE_TRIPLES = np.array([
    (0, 1, 2), (1, 2, 3), (2, 3, 4),
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    (0, 9, 10), (9, 10, 11), (10, 11, 12),
    (0, 13, 14), (13, 14, 15), (14, 15, 16),
    (0, 17, 18), (17, 18, 19), (18, 19, 20),
])
SPREAD_SEGMENTS = np.array([
    ((2, 3), (5, 6)),
    ((5, 6), (9, 10)),
    ((9, 10), (13, 14)),
    ((13, 14), (17, 18)),
])

GROUP_ORDER = ("shape", "contacts", "angles", "orientation")
GROUP_SIZES = {
    "shape": len(COORD_IDX) * 3,
    "contacts": len(CONTACT_PAIRS),
    "angles": len(ANGLE_TRIPLES) + len(SPREAD_SEGMENTS),
    "orientation": 6,
}
FEATURE_DIM = sum(GROUP_SIZES.values())


def group_slices() -> dict[str, slice]:
    out, start = {}, 0
    for g in GROUP_ORDER:
        out[g] = slice(start, start + GROUP_SIZES[g])
        start += GROUP_SIZES[g]
    return out


GROUP_SLICES = group_slices()
SHAPE_MASK = np.zeros(FEATURE_DIM, dtype=bool)
for _g in ("shape", "contacts", "angles"):
    SHAPE_MASK[GROUP_SLICES[_g]] = True


def hand_basis(pts: np.ndarray):
    """Origin (wrist), 3x3 basis (rows: x towards the index side, y towards the
    fingers, z out of the palm) and palm length."""
    origin = pts[0]
    y = pts[9] - origin
    palm = float(np.linalg.norm(y))
    if palm < 1e-6:
        raise ValueError("degenerate hand: wrist and middle knuckle coincide")
    y = y / palm
    across = pts[5] - pts[17]
    x = across - y * float(np.dot(across, y))
    nx = float(np.linalg.norm(x))
    if nx < 1e-9:
        # fall back to any vector perpendicular to y
        helper = np.array([1.0, 0.0, 0.0]) if abs(y[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
        x = np.cross(helper, y)
        nx = float(np.linalg.norm(x))
    x = x / nx
    z = np.cross(x, y)
    return origin, np.stack([x, y, z]), palm


def _angles(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    nu = np.linalg.norm(u, axis=-1)
    nv = np.linalg.norm(v, axis=-1)
    cos = np.einsum("ij,ij->i", u, v) / np.maximum(nu * nv, 1e-12)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def extract_features(pts: np.ndarray) -> np.ndarray:
    """Feature vector (float32, length FEATURE_DIM) for canonical points."""
    pts = np.asarray(pts, dtype=np.float64)
    origin, basis, palm = hand_basis(pts)
    local = (pts - origin) @ basis.T / palm
    shape = local[COORD_IDX].ravel()

    contacts = np.linalg.norm(pts[CONTACT_PAIRS[:, 0]] - pts[CONTACT_PAIRS[:, 1]], axis=1) / palm

    p0 = pts[ANGLE_TRIPLES[:, 0]]
    p1 = pts[ANGLE_TRIPLES[:, 1]]
    p2 = pts[ANGLE_TRIPLES[:, 2]]
    bends = _angles(p1 - p0, p2 - p1)
    seg_a = pts[SPREAD_SEGMENTS[:, 0, 1]] - pts[SPREAD_SEGMENTS[:, 0, 0]]
    seg_b = pts[SPREAD_SEGMENTS[:, 1, 1]] - pts[SPREAD_SEGMENTS[:, 1, 0]]
    spreads = _angles(seg_a, seg_b)

    orientation = np.concatenate([basis[1], basis[2]])
    return np.concatenate([shape, contacts, bends, spreads, orientation]).astype(np.float32)


def extract_many(pts_seq: np.ndarray) -> np.ndarray:
    return np.stack([extract_features(p) for p in pts_seq]) if len(pts_seq) else np.zeros((0, FEATURE_DIM), np.float32)
