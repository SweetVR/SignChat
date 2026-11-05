"""Webcam hand tracking with Google MediaPipe (optional add-on).

Useful for VRChat desktop mode, or to try SignChat at your desk.  Needs
`pip install -r requirements-webcam.txt` (mediapipe + opencv).  The hand model
(about 8 MB) is downloaded once on first use.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

from ..hand import HandFrame
from ..paths import app_dir
from .base import HandSource

log = logging.getLogger(__name__)

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")


def model_path() -> Path:
    return app_dir() / "models" / "hand_landmarker.task"


def ensure_model(status) -> Path:
    path = model_path()
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    status("waiting", "Downloading the hand model (about 8 MB, first time only)…")
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    os.replace(tmp, path)
    return path


def convert_result(result, width: int, height: int, t: float) -> list[HandFrame]:
    """MediaPipe HandLandmarkerResult (from a non-mirrored camera image) -> HandFrames
    in the signer's own frame: x = their right, y = up, z = away from the camera."""
    frames = []
    for i, image_lms in enumerate(result.hand_landmarks):
        if i >= len(result.hand_world_landmarks) or i >= len(result.handedness):
            break
        # MediaPipe assumes a mirrored (selfie) image; ours isn't, so its labels are swapped.
        label = result.handedness[i][0].category_name.lower()
        hand = "right" if label == "left" else "left"
        world = np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[i]], dtype=np.float64)
        img = np.array([[p.x * width, p.y * height] for p in image_lms], dtype=np.float64)
        # camera axes (x right, y down, z away) -> signer axes: rotate 180 degrees about z
        view = world * np.array([-1.0, -1.0, 1.0])
        palm_m = float(np.linalg.norm(world[9] - world[0]))
        palm_px = float(np.linalg.norm(img[9] - img[0]))
        if palm_m < 1e-4 or palm_px < 1.0:
            continue
        scale = palm_m / palm_px
        wrist = np.array([-(img[0, 0] - width / 2) * scale, -(img[0, 1] - height / 2) * scale, 0.0])
        pts = view - view[0] + wrist
        frames.append(HandFrame(t, hand, pts, head_relative=False, quality="camera"))
    return frames


class WebcamSource(HandSource):
    kind = "webcam"
    name = "Webcam"

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.preview_cb = None   # callable(rgb_uint8_array) for the UI

    def run(self) -> None:
        try:
            import cv2
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision
        except ImportError:
            self.status("error", "The webcam add-on isn't installed. Run: pip install -r requirements-webcam.txt")
            return
        try:
            model = ensure_model(self.status)
        except OSError as exc:
            self.status("error", f"Couldn't download the hand model: {exc}")
            return
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.settings.camera_index, backend)
        if not cap.isOpened():
            self.status("error", f"Couldn't open camera {self.settings.camera_index}.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5)
        landmarker = vision.HandLandmarker.create_from_options(opts)
        self.status("ok", f"Camera {self.settings.camera_index} running")
        last_ts, last_preview = -1, 0.0
        try:
            while not self.stopping:
                ok, frame = cap.read()
                if not ok:
                    self.status("error", "The camera stopped sending pictures.")
                    self._stop.wait(1.0)
                    continue
                now = time.monotonic()
                ts = max(int(now * 1000), last_ts + 1)
                last_ts = ts
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
                self.on_update(convert_result(result, w, h, now), now)
                if self.preview_cb is not None and now - last_preview > 1 / 15:
                    last_preview = now
                    small = cv2.resize(cv2.flip(rgb, 1), (320, int(320 * h / w)))
                    for lms in result.hand_landmarks:
                        for p in lms:
                            cv2.circle(small, (int((1 - p.x) * small.shape[1]), int(p.y * small.shape[0])), 3,
                                       (72, 214, 140), -1)
                    self.preview_cb(small)
        finally:
            landmarker.close()
            cap.release()
        self.status("stopped", "Stopped")
