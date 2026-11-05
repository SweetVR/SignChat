"""User settings, saved as JSON next to the profiles."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, fields

from .paths import app_dir

log = logging.getLogger(__name__)


@dataclass
class Settings:
    # input
    source: str = "steamvr"          # steamvr | webcam | sim
    hand: str = "right"              # the hand you fingerspell with
    camera_index: int = 0
    # VRChat
    osc_host: str = "127.0.0.1"
    osc_port: int = 9000
    live_captions: bool = True       # show letters in the chatbox while you spell
    live_interval_s: float = 2.0     # VRChat times out chatbox spam; keep this >= 1.5
    sound_on_send: bool = False
    osc_toggle_enabled: bool = False  # let an avatar parameter turn signing on/off
    osc_listen_port: int = 9001
    osc_toggle_param: str = "SignChat"
    # typing
    case_mode: str = "sentence"      # sentence | upper | lower
    hold_s: float = 0.30
    word_gap_s: float = 1.2
    drop_send_s: float = 1.0
    idle_send_s: float = 0.0
    drop_height_m: float = 0.55      # hand more than this far below your eyes = lowered
    min_conf: float = 0.55
    # recognition
    knn_k: int = 7
    reject_ratio: float = 3.0
    motion_tolerance: float = 0.30
    # recording
    countdown_s: float = 3.0
    static_record_s: float = 2.0
    motion_record_s: float = 2.5
    rest_record_s: float = 5.0
    takes_static: int = 2
    takes_motion: int = 3
    # VR extras
    hud_enabled: bool = True
    hud_distance_m: float = 0.55
    hud_drop_m: float = 0.20
    haptics: bool = True
    profile_name: str = "default"

    @classmethod
    def load(cls) -> "Settings":
        path = app_dir() / "settings.json"
        s = cls()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                known = {f.name: f.type for f in fields(cls)}
                for k, v in data.items():
                    if k in known:
                        setattr(s, k, type(getattr(s, k))(v))
            except Exception as exc:
                log.warning("could not read settings (%s); using defaults", exc)
        return s

    def save(self) -> None:
        path = app_dir() / "settings.json"
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
