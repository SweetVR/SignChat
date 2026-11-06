"""Decides *when* a recognised handshape becomes a typed letter.

Rules (all timings are adjustable in Settings):
  * hold a handshape still for `hold_s` to type it
  * the same letter types again only after the hand shows a different shape,
    or after a small bounce/slide (how double letters like the LL in HELLO
    are signed)
  * keeping the hand still for `word_gap_s` after a letter adds a space
  * lowering the hand (or losing tracking) for `drop_send_s` sends the message
  * optional signs for space / delete / send / clear need a slightly longer hold
"""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass
from typing import NamedTuple

from .classifier import REST_LABEL

ACTION_LABELS = {"_space": "space", "_backspace": "backspace", "_send": "send", "_clear": "clear"}


@dataclass
class TimingConfig:
    hold_s: float = 0.30
    action_hold_s: float = 0.55
    still_speed: float = 1.6      # palm lengths per second
    min_conf: float = 0.55
    smooth_frames: int = 5
    release_s: float = 0.18
    vote_share: float = 0.6       # share of frames during the hold that must agree
    repeat_path: float = 0.35     # palm lengths of movement that counts as a bounce
    word_gap_s: float = 1.2       # 0 disables automatic spaces
    drop_send_s: float = 1.0      # 0 disables sending when the hand is lowered
    idle_send_s: float = 0.0      # 0 disables sending after a long pause


class Event(NamedTuple):
    kind: str            # "letter", "space", "backspace", "send", "clear"
    value: str | None
    t: float


class LetterSegmenter:
    def __init__(self, cfg: TimingConfig | None = None):
        self.cfg = cfg or TimingConfig()
        self.reset()

    def reset(self) -> None:
        self.hist: deque = deque(maxlen=max(1, self.cfg.smooth_frames))
        self.win: deque = deque()
        self.prev_t: float | None = None
        self.cand: str | None = None
        self.last_label: str | None = None
        self.last_commit_t: float | None = None
        self.released = True
        self.diff_since: float | None = None
        self.move_path = 0.0
        self.inactive_since: float | None = None
        self.has_text = False
        self.space_pending = False
        self.smoothed: str | None = None
        self.hold_progress = 0.0
        self.still_time = 0.0   # how long the hand has been still since the last typed letter
        self.bounce_ok = True   # may a small bounce repeat the last letter?

    # ------------------------------------------------------------------
    def mark_commit(self, t: float, shape_label: str | None) -> None:
        """Called when a moving letter (J/Z) was typed; `shape_label` is the
        static shape the hand ends in, which must not be typed right after."""
        self.last_label = shape_label
        self.last_commit_t = t
        self.released = False
        self.move_path = 0.0
        self.diff_since = None
        self.win.clear()
        self.has_text = True
        self.space_pending = True
        self.still_time = 0.0
        self.bounce_ok = False   # moving back after J/Z must not repeat the shape it ended in

    def after_sign(self, t: float, end_shape: str | None) -> None:
        """A whole sign was recognised. If the hand is still in the handshape the
        sign ended with, don't type that handshape as a letter."""
        self.has_text = True
        self.last_commit_t = t
        self.still_time = 0.0
        if end_shape is not None and end_shape in (self.cand, self.smoothed, self.last_label):
            self.last_label = end_shape
            self.released = False
            self.bounce_ok = False
            self.diff_since = None
            self.win.clear()
            self.hold_progress = 0.0

    def _smooth(self, label: str | None) -> str | None:
        self.hist.append(label)
        counts = Counter(x for x in self.hist if x is not None)
        if not counts:
            return None
        lab, n = counts.most_common(1)[0]
        return lab if n >= math.ceil(0.6 * len(self.hist)) else None

    def _commit_event(self, sign: str, t: float) -> Event | None:
        if sign.startswith("_"):
            kind = ACTION_LABELS.get(sign)
            return Event(kind, None, t) if kind else None
        return Event("letter", sign, t)

    # ------------------------------------------------------------------
    def update(self, t: float, label: str | None, conf: float, speed: float, active: bool) -> list[Event]:
        cfg = self.cfg
        events: list[Event] = []
        dt = 0.0 if self.prev_t is None else max(0.0, t - self.prev_t)
        self.prev_t = t

        if not active:
            if self.inactive_since is None:
                self.inactive_since = t
            self.hist.clear()
            self.win.clear()
            self.cand = None
            self.smoothed = None
            self.hold_progress = 0.0
            self.released = True
            if self.has_text and cfg.drop_send_s > 0 and t - self.inactive_since >= cfg.drop_send_s:
                events.append(Event("send", None, t))
                self.has_text = False
                self.space_pending = False
            return events
        self.inactive_since = None

        sign = self._smooth(label if conf >= cfg.min_conf else None)
        if sign == REST_LABEL:
            sign = None
        self.smoothed = sign

        # Has the last typed letter been "let go" so it may be typed again?
        if sign != self.last_label:
            if self.diff_since is None:
                self.diff_since = t
            if t - self.diff_since >= cfg.release_s:
                self.released = True
        else:
            self.diff_since = None
            if not self.released and self.bounce_ok and speed > cfg.still_speed:
                self.move_path += speed * dt
                if self.move_path >= cfg.repeat_path:
                    self.released = True

        # Pause timer for automatic spaces: only counts while the hand is still,
        # so slow letters (J, Z, awkward transitions) don't insert spaces.
        if speed > cfg.still_speed:
            self.still_time = 0.0
        else:
            self.still_time += dt

        # Hold: while the hand is still, collect one vote per frame. A sign is
        # typed when it wins most votes over the whole hold time. Voting (instead
        # of needing every frame to agree) copes with similar-looking letters
        # flickering for a frame or two.
        frame_sign = label if (label is not None and conf >= cfg.min_conf and label != REST_LABEL) else None
        if speed > cfg.still_speed:
            self.win.clear()
        else:
            self.win.append((t, frame_sign))
        horizon = max(cfg.hold_s, cfg.action_hold_s)
        while self.win and self.win[0][0] < t - horizon - 1e-9:
            self.win.popleft()
        self.hold_progress = 0.0
        votes = Counter(l for _, l in self.win if l is not None)
        if votes:
            top = votes.most_common(1)[0][0]
            need = cfg.action_hold_s if top.startswith("_") else cfg.hold_s
            recent = [l for tt, l in self.win if tt >= t - need - 1e-9]
            share = recent.count(top) / len(recent)
            still_for = t - self.win[0][0]
            self.cand = top
            if (top != self.last_label or self.released) and share >= cfg.vote_share:
                self.hold_progress = min(1.0, still_for / need) if need > 0 else 1.0
                if still_for >= need - 1e-9:
                    ev = self._commit_event(top, t)
                    self.last_label = top
                    self.last_commit_t = t
                    self.released = False
                    self.move_path = 0.0
                    self.diff_since = None
                    self.hold_progress = 0.0
                    self.still_time = 0.0
                    self.bounce_ok = True
                    self.win.clear()
                    if ev is not None:
                        events.append(ev)
                        if ev.kind == "letter":
                            self.has_text = True
                            self.space_pending = True
                        elif ev.kind == "space":
                            self.space_pending = False
        else:
            self.cand = None

        # Pauses.
        if self.has_text and self.last_commit_t is not None:
            quiet = t - self.last_commit_t
            if cfg.word_gap_s > 0 and self.space_pending and self.still_time >= cfg.word_gap_s:
                events.append(Event("space", None, t))
                self.space_pending = False
            if cfg.idle_send_s > 0 and quiet >= cfg.idle_send_s:
                events.append(Event("send", None, t))
                self.has_text = False
                self.space_pending = False
        return events
