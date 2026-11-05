import numpy as np

from signchat.classifier import SignModel
from signchat.features import FEATURE_DIM, GROUP_SLICES, extract_features
from signchat.hand import HandFrame, canonical_points
from signchat.synth import LETTER_POSES, SIGN_POSITION, Performer, Pose, SimUser, render

STATIC = [l for l in LETTER_POSES if l not in ("J", "Z")]


def rot(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def test_shape_features_ignore_position_rotation_and_size():
    pts = render(LETTER_POSES["K"], SIGN_POSITION)
    f0 = extract_features(pts)
    moved = (pts - pts[0]) @ rot([0.3, 1, 0.2], 0.9).T * 1.3 + np.array([1.0, -2.0, 0.5])
    f1 = extract_features(moved)
    shape = slice(0, GROUP_SLICES["orientation"].start)
    assert f0.shape == (FEATURE_DIM,)
    assert np.allclose(f0[shape], f1[shape], atol=1e-4)
    # orientation features do change when the hand turns
    assert not np.allclose(f0[GROUP_SLICES["orientation"]], f1[GROUP_SLICES["orientation"]], atol=1e-2)


def test_left_hand_maps_onto_right_hand():
    pose = LETTER_POSES["G"]
    right = HandFrame(0.0, "right", render(pose, SIGN_POSITION, "right"))
    left = HandFrame(0.0, "left", render(pose, SIGN_POSITION, "left"))
    assert np.allclose(extract_features(canonical_points(right)), extract_features(canonical_points(left)), atol=1e-5)


def _take(user, label, seed):
    frames = Performer(user, seed=seed).hold_sign(label, 2.3)[36:]
    return np.stack([extract_features(canonical_points(f)) for f in frames])


def _fit(user):
    data = [(lab, f"{lab}{k}", _take(user, lab, 100 * i + k)) for i, lab in enumerate(STATIC) for k in range(2)]
    return SignModel().fit(data)


def test_classifier_recognises_new_repetitions():
    user = SimUser(seed=3)
    model = _fit(user)
    ok = n = 0
    for i, lab in enumerate(STATIC):
        for x in _take(user, lab, 9000 + i)[::6]:
            n += 1
            ok += model.predict(x).label == lab
    assert ok / n > 0.95


def test_accuracy_check_reports_confusable_letters():
    model = _fit(SimUser(seed=3))
    report = model.evaluate()
    assert report.overall > 0.9
    worst = {lab for lab, acc, _ in report.worst(4)}
    assert worst & {"M", "N", "S", "R", "U"}  # the deliberately similar synthetic shapes


def test_random_hand_shapes_are_mostly_rejected():
    model = _fit(SimUser(seed=3))
    rng = np.random.default_rng(0)
    rejected = 0
    for _ in range(200):
        v = LETTER_POSES["B"].vector()
        v[:16] = rng.uniform(0, 1.5, 16)
        v[16:20] = rng.uniform(-0.5, 1.2, 4)
        rejected += model.predict(extract_features(render(Pose.from_vector(v), SIGN_POSITION))).label is None
    assert rejected / 200 > 0.6


def test_empty_model_predicts_nothing():
    assert SignModel().predict(np.zeros(FEATURE_DIM)).label is None
