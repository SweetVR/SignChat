"""Hand-tracking input sources."""

from __future__ import annotations

SOURCES = {
    "steamvr": "SteamVR (Index Knuckles, Quest hand tracking, other VR hands)",
    "webcam": "Webcam (desktop mode, needs the webcam add-on)",
    "sim": "Simulated hand (try SignChat without VR)",
}


def create_source(kind: str, settings, on_update, on_status):
    if kind == "steamvr":
        from .steamvr import SteamVRSource
        return SteamVRSource(settings, on_update, on_status)
    if kind == "webcam":
        from .webcam import WebcamSource
        return WebcamSource(settings, on_update, on_status)
    if kind == "sim":
        from .simulator import SimulatorSource
        return SimulatorSource(settings, on_update, on_status)
    raise ValueError(f"unknown input source {kind!r}")
