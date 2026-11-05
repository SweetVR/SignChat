import numpy as np

from signchat.classifier import SignModel
from signchat.features import extract_many
from signchat.hand import speeds_for_sequence
from signchat.motion import MotionMatcher, build_template, dtw_distance, find_motion_segment, normalize_path
from signchat.synth import LETTER_POSES, SIGN_POSITION, Performer, SimUser


def _seq(frames):
    t = np.array([f.t for f in frames])
    pts = np.stack([f.points for f in frames])
    return t, pts, extract_many(pts), speeds_for_sequence(t, pts)


def _at_sign(user, label, seed):
    perf = Performer(user, seed=seed)
    perf.pose, perf.pos = user.take_pose(label, perf.rng), SIGN_POSITION.copy()
    perf._hold(0.5)
    return perf


def _motion_take(user, label, seed):
    perf = _at_sign(user, label, seed)
    perf.motion(label)
    perf._hold(0.4)
    return _seq(perf.frames)


def _static_model(user):
    data = []
    for i, lab in enumerate(l for l in LETTER_POSES if l not in ("J", "Z")):
        frames = Performer(user, seed=500 + i).hold_sign(lab, 1.5)[36:]
        t, pts, feats, _ = _seq(frames)
        data.append((lab, lab, feats))
    return SignModel().fit(data)


def _matcher(user, model):
    templates = []
    for label in ("J", "Z"):
        for k in range(3):
            t, pts, feats, sp = _motion_take(user, label, 40 + k + (0 if label == "J" else 10))
            tm = build_template(label, f"{label}{k}", t, pts, feats, sp)
            assert tm is not None
            templates.append(tm)
    return MotionMatcher(templates, model)


def _segment(seq):
    t, pts, feats, sp = seq
    a, b = find_motion_segment(t, sp, pts_seq=pts)
    return t[a:b + 1], pts[a:b + 1], feats[a:b + 1]


def test_dtw_prefers_the_same_shape():
    z = normalize_path(np.array([[0, 0], [1, 0], [0, -1], [1, -1]], float))
    z2 = normalize_path(np.array([[0, 0], [1.2, 0.05], [0.1, -0.9], [1.1, -1.0]], float))
    line = normalize_path(np.array([[0, 0], [1, 0]], float))
    assert dtw_distance(z, z2) < 0.1 < dtw_distance(z, line)


def test_j_and_z_are_recognised():
    user = SimUser(seed=5)
    matcher = _matcher(user, _static_model(user))
    for label in ("J", "Z"):
        for k in range(3):
            m = matcher.match(*_segment(_motion_take(user, label, 900 + k + (0 if label == "J" else 10))))
            assert m is not None and m.label == label, (label, k, m)


def test_ordinary_moves_are_not_j_or_z():
    user = SimUser(seed=5)
    matcher = _matcher(user, _static_model(user))
    # a double-letter bounce on L, and a plain sideways move with an I handshape
    perf = _at_sign(user, "L", 77)
    perf._move(perf.pose, SIGN_POSITION + np.array([0.045, 0, 0]), 0.14)
    perf._move(perf.pose, SIGN_POSITION, 0.14)
    perf._hold(0.4)
    assert matcher.match(*_segment(_seq(perf.frames))) is None
    perf = _at_sign(user, "I", 78)
    perf._move(perf.pose, SIGN_POSITION + np.array([0.09, 0.0, 0]), 0.4)
    perf._hold(0.4)
    assert matcher.match(*_segment(_seq(perf.frames))) is None
