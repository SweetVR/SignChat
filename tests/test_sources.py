import json
import time
from types import SimpleNamespace as NS

import numpy as np

from signchat.config import Settings
from signchat.features import extract_features
from signchat.hand import HandFrame, canonical_points
from signchat.osc_in import OscToggleListener
from signchat.paths import steamvr_manifest
from signchat.sources.steamvr import bones_to_view
from signchat.sources.webcam import convert_result
from signchat.synth import LETTER_POSES, SIGN_POSITION, render


def test_steamvr_manifest_and_bindings_are_consistent():
    man = json.loads(steamvr_manifest().read_text())
    names = {a["name"] for a in man["actions"]}
    for b in man["default_bindings"]:
        binding = json.loads((steamvr_manifest().parent / b["binding_url"]).read_text())
        assert binding["controller_type"] == b["controller_type"]
        outs = [x["output"] for group in binding["bindings"]["/actions/signchat"].values() for x in group]
        assert set(outs) == names


def test_headset_space_conversion_matches_the_viewer_convention():
    # hand 40 cm in front of a headset that is turned 90 degrees to the left
    yaw = np.pi / 2
    head = np.array([[np.cos(yaw), 0, np.sin(yaw), 0.0], [0, 1, 0, 1.6], [-np.sin(yaw), 0, np.cos(yaw), 0.0]])
    ahead_world = -head[:, 2] * 0.4 + head[:, 3]
    raw = np.hstack([np.eye(3), ahead_world[:, None]])
    pts = bones_to_view(np.zeros((21, 3)), raw, head)
    assert np.allclose(pts[0], [0, 0, -0.4], atol=1e-9)   # straight ahead is -z


def _fake_mediapipe(pts_view, width=1280, height=720, label="Left"):
    """Build a MediaPipe-like result from viewer-space points (inverse of convert_result)."""
    cam = pts_view * np.array([-1, -1, 1])                       # signer -> camera axes
    world = cam - cam.mean(axis=0)
    px = cam[:, :2] / 0.0006 + np.array([width / 2, height / 2])  # ~0.6 mm per pixel
    lm = lambda a, b, c: NS(x=a, y=b, z=c)
    return NS(hand_landmarks=[[lm(x / width, y / height, 0.0) for x, y in px]],
              hand_world_landmarks=[[lm(*p) for p in world]],
              handedness=[[NS(category_name=label, score=0.99)]])


def test_webcam_conversion_keeps_the_hand_shape_and_side():
    pts = render(LETTER_POSES["L"], SIGN_POSITION)
    frames = convert_result(_fake_mediapipe(pts, label="Left"), 1280, 720, 1.0)
    assert len(frames) == 1 and frames[0].hand == "right"       # MediaPipe labels are mirrored
    f_ref = extract_features(canonical_points(HandFrame(0, "right", pts)))
    f_cam = extract_features(canonical_points(frames[0]))
    assert np.allclose(f_ref, f_cam, atol=1e-4)
    assert np.allclose(frames[0].points - frames[0].points[0], pts - pts[0], atol=1e-6)


def test_simulated_source_drives_the_engine():
    from signchat.sources.simulator import SimulatorSource
    got = []
    src = SimulatorSource(Settings(source="sim"), lambda frames, now: got.extend(frames), lambda s, m: None)
    src.start()
    try:
        src.press("A")
        time.sleep(0.6)
    finally:
        src.stop()
    assert len(got) > 15 and all(f.hand == "right" for f in got)


def test_avatar_toggle_listener():
    from pythonosc.udp_client import SimpleUDPClient
    seen = []
    lst = OscToggleListener(0, "SignChat", seen.append)
    lst.start()
    try:
        port = lst.server.server_address[1]
        SimpleUDPClient("127.0.0.1", port).send_message("/avatar/parameters/SignChat", True)
        deadline = time.time() + 2
        while not seen and time.time() < deadline:
            time.sleep(0.02)
    finally:
        lst.stop()
    assert seen == [True]
