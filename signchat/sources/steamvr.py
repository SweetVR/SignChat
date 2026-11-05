"""SteamVR hand skeleton input (Index Knuckles, Quest hand tracking through
Steam Link / Virtual Desktop, and any other device with SteamVR skeletal input).

SignChat runs as an OpenVR *overlay* application next to VRChat, the same way
tools like OVR Advanced Settings or ThumbParams do, and reads the hand
skeletons through its own action manifest (signchat/steamvr/actions.json).
It also draws a small status panel in VR and buzzes the controllers during
recording.
"""

from __future__ import annotations

import ctypes
import logging
import math
import time

import numpy as np

from ..hand import STEAMVR_BONE_FOR_LANDMARK, HandFrame
from ..paths import steamvr_manifest
from .base import HandSource

log = logging.getLogger(__name__)

ACTION_SET = "/actions/signchat"
HANDS = ("left", "right")
LEVEL_NAMES = {0: "estimated", 1: "partial", 2: "full"}
HAPTIC_PATTERNS = {  # list of (start_s, duration_s, frequency_hz, amplitude)
    "tick": [(0.0, 0.025, 160.0, 0.35)],
    "go": [(0.0, 0.18, 120.0, 0.9)],
    "done": [(0.0, 0.05, 160.0, 0.6), (0.12, 0.05, 160.0, 0.6)],
    "error": [(0.0, 0.25, 60.0, 1.0)],
}
LOOP_HZ = 90.0


def mat34_to_np(m) -> np.ndarray:
    """openvr.HmdMatrix34_t -> (3, 4) numpy array [R | t]."""
    return np.array([[m.m[i][j] for j in range(4)] for i in range(3)], dtype=np.float64)


def bones_to_view(model_pts: np.ndarray, raw_pose: np.ndarray, head_pose: np.ndarray) -> np.ndarray:
    """Hand landmarks in skeleton model space -> headset (viewer) space.

    model_pts: (21, 3) bone positions relative to the hand's /pose/raw
    raw_pose, head_pose: (3, 4) device-to-tracking-space matrices
    """
    world = model_pts @ raw_pose[:, :3].T + raw_pose[:, 3]
    return (world - head_pose[:, 3]) @ head_pose[:, :3]


def bone_positions(bones) -> np.ndarray:
    """(31,) VRBoneTransform_t array -> (21, 3) landmark positions."""
    raw = np.frombuffer(bones, dtype=np.float32).reshape(len(bones), 8)  # position xyzw + quaternion wxyz
    return raw[list(STEAMVR_BONE_FOR_LANDMARK), :3].astype(np.float64)


class VRHud:
    """A small panel floating below your view, drawn with Pillow."""

    def __init__(self, openvr, settings):
        from ..hud_render import OUT_H, OUT_W
        self.openvr = openvr
        self.W, self.H = OUT_W, OUT_H
        self.ov = openvr.VROverlay()
        self.handle = self.ov.createOverlay("signchat.hud", "SignChat")
        self.ov.setOverlayWidthInMeters(self.handle, 0.26)
        self.visible = False
        self.place(settings.hud_distance_m, settings.hud_drop_m)

    def place(self, distance: float, drop: float) -> None:
        o = self.openvr
        th = math.atan2(drop, distance)          # tilt the panel so it faces your eyes
        c, s = math.cos(th), math.sin(th)
        m = o.HmdMatrix34_t()
        rows = ((1, 0, 0, 0), (0, c, s, -drop), (0, -s, c, -distance))
        for i, row in enumerate(rows):
            for j, v in enumerate(row):
                m.m[i][j] = v
        self.ov.setOverlayTransformTrackedDeviceRelative(self.handle, o.k_unTrackedDeviceIndex_Hmd, m)

    def update(self, state: dict) -> None:
        from ..hud_render import render_hud_bytes
        show = bool(state.get("enabled") or state.get("rec"))
        if show:
            data = render_hud_bytes(state)
            buf = (ctypes.c_char * len(data)).from_buffer_copy(data)
            self.ov.setOverlayRaw(self.handle, buf, self.W, self.H, 4)
            if not self.visible:
                self.ov.showOverlay(self.handle)
                self.visible = True
        elif self.visible:
            self.ov.hideOverlay(self.handle)
            self.visible = False

    def close(self) -> None:
        try:
            self.ov.destroyOverlay(self.handle)
        except Exception:
            pass


class SteamVRSource(HandSource):
    kind = "steamvr"
    name = "SteamVR"
    has_hud = True
    has_haptics = True

    def run(self) -> None:
        import openvr

        if not openvr.isRuntimeInstalled():
            self.status("error", "SteamVR isn't installed on this PC.")
            return
        while not self.stopping:
            if not self._steamvr_running(openvr):
                self.status("waiting", "Waiting for SteamVR to start…")
                self._stop.wait(3.0)
                continue
            try:
                self._session(openvr)
            except openvr.error_code.OpenVRError as exc:
                log.warning("SteamVR session ended: %r", exc)
                self.status("error", f"SteamVR error: {type(exc).__name__}. Retrying…")
                self._stop.wait(3.0)
            finally:
                try:
                    openvr.shutdown()
                except Exception:
                    pass
        self.status("stopped", "Stopped")

    @staticmethod
    def _steamvr_running(openvr) -> bool:
        # A "background" app can't start SteamVR, so this only succeeds if it's already running.
        try:
            openvr.init(openvr.VRApplication_Background)
        except openvr.error_code.OpenVRError:
            return False
        openvr.shutdown()
        return True

    def _session(self, openvr) -> None:
        vrsys = openvr.init(openvr.VRApplication_Overlay)
        vrinput = openvr.VRInput()
        vrinput.setActionManifestPath(str(steamvr_manifest()))
        action_set = vrinput.getActionSetHandle(ACTION_SET)
        acts = {h: {"skel": vrinput.getActionHandle(f"{ACTION_SET}/in/skeleton_{h}"),
                    "pose": vrinput.getActionHandle(f"{ACTION_SET}/in/pose_{h}"),
                    "haptic": vrinput.getActionHandle(f"{ACTION_SET}/out/haptic_{h}")} for h in HANDS}
        sets = (openvr.VRActiveActionSet_t * 1)()
        sets[0].ulActionSet = action_set
        bones = (openvr.VRBoneTransform_t * 31)()
        poses = (openvr.TrackedDevicePose_t * 1)()   # index 0 is always the headset
        event = openvr.VREvent_t()

        hud = None
        if self.settings.hud_enabled:
            try:
                hud = VRHud(openvr, self.settings)
            except openvr.error_code.OpenVRError as exc:
                log.warning("no VR status panel: %r", exc)

        self.status("ok", "Connected to SteamVR")
        last_hud = 0.0
        seen_hand = False
        started = time.monotonic()
        warned = False
        try:
            while not self.stopping:
                tick = time.monotonic()
                while vrsys.pollNextEvent(event):
                    if event.eventType == openvr.VREvent_Quit:
                        self.status("waiting", "SteamVR closed. Waiting for it to start again…")
                        return
                vrinput.updateActionState(sets)
                vrsys.getDeviceToAbsoluteTrackingPose(openvr.TrackingUniverseStanding, 0.0, poses)
                frames = []
                if poses[0].bPoseIsValid:
                    head = mat34_to_np(poses[0].mDeviceToAbsoluteTracking)
                    for h in HANDS:
                        f = self._read_hand(openvr, vrinput, acts[h], h, head, bones, tick)
                        if f is not None:
                            frames.append(f)
                if frames and not seen_hand:
                    seen_hand = True
                    q = ", ".join(sorted({f"{f.hand}: {f.quality}" for f in frames}))
                    self.status("ok", f"Connected to SteamVR · hand tracking ({q})")
                if not seen_hand and not warned and tick - started > 8.0:
                    warned = True
                    self.status("waiting", "Connected, but SteamVR isn't sending hand data to SignChat yet. "
                                           "Check that your controllers/hands are tracked, then see "
                                           "'Hands not detected' in the README.")
                self.on_update(frames, tick)

                while not self._haptics.empty():
                    self._play_haptic(openvr, vrinput, acts, self._haptics.get_nowait())
                if hud is not None and self.hud_provider is not None and tick - last_hud >= 0.1:
                    last_hud = tick
                    try:
                        hud.update(self.hud_provider())
                    except Exception as exc:  # the panel is optional; keep tracking alive
                        log.warning("VR status panel failed, turning it off: %r", exc)
                        hud.close()
                        hud = None
                spare = 1.0 / LOOP_HZ - (time.monotonic() - tick)
                if spare > 0:
                    self._stop.wait(spare)
        finally:
            if hud is not None:
                hud.close()

    def _read_hand(self, openvr, vrinput, act, hand, head, bones, now):
        try:
            if not vrinput.getSkeletalActionData(act["skel"]).bActive:
                return None
            pose = vrinput.getPoseActionDataRelativeToNow(act["pose"], openvr.TrackingUniverseStanding, 0.0,
                                                          openvr.k_ulInvalidInputValueHandle)
            if not (pose.bActive and pose.pose.bPoseIsValid):
                return None
            vrinput.getSkeletalBoneData(act["skel"], openvr.VRSkeletalTransformSpace_Model,
                                        openvr.VRSkeletalMotionRange_WithoutController, bones)
            level = vrinput.getSkeletalTrackingLevel(act["skel"])
        except openvr.error_code.InputError:
            return None
        level = int(getattr(level, "value", level))
        pts = bones_to_view(bone_positions(bones), mat34_to_np(pose.pose.mDeviceToAbsoluteTracking), head)
        if not np.all(np.isfinite(pts)):
            return None
        return HandFrame(now, hand, pts, head_relative=True, quality=LEVEL_NAMES.get(level, "unknown"))

    def _play_haptic(self, openvr, vrinput, acts, pattern: str) -> None:
        for start, dur, freq, amp in HAPTIC_PATTERNS.get(pattern, []):
            for h in HANDS:
                try:
                    vrinput.triggerHapticVibrationAction(acts[h]["haptic"], start, dur, freq, amp,
                                                         openvr.k_ulInvalidInputValueHandle)
                except openvr.error_code.InputError:
                    pass
