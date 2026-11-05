"""Runs the SteamVR source against a fake OpenVR runtime.

The fake uses pyopenvr's real ctypes structures and constants and only
replaces the calls into SteamVR, so it checks how SignChat calls the API
(argument types, arrays, structs) without needing a headset.
"""

import ctypes
import sys
import time
import types
from pathlib import Path

import numpy as np
import openvr as real

from signchat.config import Settings
from signchat.hand import STEAMVR_BONE_FOR_LANDMARK
from signchat.synth import LETTER_POSES, local_points

HEAD_POS = np.array([0.0, 1.6, 0.0])
HAND_POS = np.array([0.15, 1.35, -0.35])


def _pose_matrix(m, pos):
    for i in range(3):
        for j in range(4):
            m.m[i][j] = (1.0 if i == j else 0.0) if j < 3 else float(pos[i])


def make_fake(log):
    fake = types.ModuleType("openvr")
    for name in dir(real):
        if not name.startswith("_"):
            setattr(fake, name, getattr(real, name))
    fake.isRuntimeInstalled = lambda: True
    fake.shutdown = lambda: log.setdefault("shutdowns", []).append(1)
    model_pts = local_points(LETTER_POSES["A"])

    class FakeSystem:
        def pollNextEvent(self, event):
            return False

        def getDeviceToAbsoluteTrackingPose(self, origin, t, poses):
            assert isinstance(poses, ctypes.Array) and origin == real.TrackingUniverseStanding
            _pose_matrix(poses[0].mDeviceToAbsoluteTracking, HEAD_POS)
            poses[0].bPoseIsValid = 1
            return poses

    class FakeInput:
        def setActionManifestPath(self, path):
            assert Path(path).is_absolute() and Path(path).exists()
            log["manifest"] = path

        def getActionSetHandle(self, name):
            assert name == "/actions/signchat"
            return 11

        def getActionHandle(self, name):
            log.setdefault("actions", []).append(name)
            return 100 + len(log["actions"])

        def updateActionState(self, sets):
            assert isinstance(sets, ctypes.Array) and sets[0].ulActionSet == 11
            return sets

        def getSkeletalActionData(self, action):
            d = real.InputSkeletalActionData_t()
            d.bActive = 1
            return d

        def getPoseActionDataRelativeToNow(self, action, origin, t, device):
            assert device == real.k_ulInvalidInputValueHandle
            d = real.InputPoseActionData_t()
            d.bActive = 1
            d.pose.bPoseIsValid = 1
            _pose_matrix(d.pose.mDeviceToAbsoluteTracking, HAND_POS)
            return d

        def getSkeletalBoneData(self, action, space, motion_range, bones):
            assert isinstance(bones, ctypes.Array) and len(bones) == 31
            assert space == real.VRSkeletalTransformSpace_Model
            assert motion_range == real.VRSkeletalMotionRange_WithoutController
            for lm, bone in enumerate(STEAMVR_BONE_FOR_LANDMARK):
                for k in range(3):
                    bones[bone].position.v[k] = model_pts[lm][k]
            return bones

        def getSkeletalTrackingLevel(self, action):
            return real.EVRSkeletalTrackingLevel(real.VRSkeletalTracking_Full)

        def triggerHapticVibrationAction(self, action, start, dur, freq, amp, device):
            log.setdefault("haptics", []).append((start, dur, freq, amp))

    class FakeOverlay:
        def createOverlay(self, key, name):
            return 7

        def setOverlayWidthInMeters(self, handle, w):
            assert 0.1 < w < 1.0

        def setOverlayTransformTrackedDeviceRelative(self, handle, device, m):
            assert isinstance(m, real.HmdMatrix34_t) and device == real.k_unTrackedDeviceIndex_Hmd

        def setOverlayRaw(self, handle, buf, w, h, bpp):
            assert isinstance(buf, ctypes.Array) and len(buf) == w * h * bpp
            log.setdefault("hud", []).append((w, h))

        def showOverlay(self, handle):
            log["shown"] = True

        def hideOverlay(self, handle):
            log["shown"] = False

        def destroyOverlay(self, handle):
            log["destroyed"] = True

    def init(app_type, info=None):
        log.setdefault("inits", []).append(app_type)
        return FakeSystem()

    fake.init = init
    fake.VRInput = FakeInput
    fake.VRSystem = FakeSystem
    fake.VROverlay = FakeOverlay
    return fake


def test_steamvr_source_with_fake_runtime(monkeypatch):
    log = {}
    monkeypatch.setitem(sys.modules, "openvr", make_fake(log))
    from signchat.sources.steamvr import SteamVRSource

    frames, statuses = [], []
    src = SteamVRSource(Settings(), lambda fr, now: frames.extend(fr), lambda st, msg: statuses.append((st, msg)))
    src.hud_provider = lambda: {"enabled": True, "label": "A", "hold": 0.4, "text": "Hi", "rec": None}
    src.start()
    src.haptic("done")
    time.sleep(0.6)
    src.stop()

    assert log["inits"][:2] == [real.VRApplication_Background, real.VRApplication_Overlay]
    assert {"/actions/signchat/in/skeleton_left", "/actions/signchat/in/pose_right"} <= set(log["actions"])
    assert any(st == "ok" and "hand tracking" in msg for st, msg in statuses)
    assert {f.hand for f in frames} == {"left", "right"}
    f = frames[-1]
    assert f.head_relative and f.quality == "full"
    expected = local_points(LETTER_POSES["A"]) + HAND_POS - HEAD_POS
    assert np.allclose(f.points, expected, atol=1e-5)
    assert log.get("hud") and log.get("shown") is True and log.get("destroyed")
    assert len(log.get("haptics", [])) == 4      # "done" = 2 pulses x 2 hands
