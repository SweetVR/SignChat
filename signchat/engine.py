"""Ties everything together: hand frames in, chatbox messages out.

Thread model: an input source calls `on_update(frames, now)` from its own
thread for every tracking update; the UI calls the public methods from the
UI thread.  Everything shares one lock.  Results are reported through
`emit(dict)`, which the UI drains on its own timer.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .classifier import REST_LABEL, Prediction, SignModel
from .config import Settings
from .features import extract_features, extract_many
from .hand import HandFrame, SpeedMeter, canonical_points, speeds_for_sequence
from .motion import MotionMatcher, build_template, find_motion_segment
from .osc_out import ChatboxSender
from .profile import Profile, Take
from .segmenter import Event, LetterSegmenter, TimingConfig
from .text_builder import TextBuilder

log = logging.getLogger(__name__)

LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
MOTION_LETTERS = ("J", "Z")
SPECIAL_SIGNS = {
    "_rest": "Resting hand",
    "_space": "Space",
    "_backspace": "Delete",
    "_send": "Send",
    "_clear": "Clear",
}
GUIDED_ORDER = ("_rest",) + LETTERS
PROMPT_S = 1.2
MOTION_ON, MOTION_OFF, MOTION_QUIET_S, MOTION_MAX_S = 2.2, 1.2, 0.08, 2.5


def label_kind(label: str) -> str:
    return "motion" if label in MOTION_LETTERS else "static"


def display_name(label: str) -> str:
    return SPECIAL_SIGNS.get(label, label)


@dataclass
class RecordJob:
    label: str
    kind: str
    capture_s: float
    take_no: int = 1
    takes_total: int = 1
    retried: bool = False


def _canon_seq(points: np.ndarray, hand: str) -> np.ndarray:
    pts = np.asarray(points, np.float64)
    return pts * np.array([-1.0, 1.0, 1.0]) if hand == "left" else pts


class Engine:
    def __init__(self, settings: Settings, profile: Profile, sender: ChatboxSender,
                 emit: Callable[[dict], None] | None = None):
        self.s = settings
        self.profile = profile
        self.sender = sender
        self.emit = emit or (lambda ev: None)
        self.lock = threading.RLock()
        self.source_kind = settings.source
        self.enabled = False
        self.model: SignModel | None = None
        self.matcher: MotionMatcher | None = None
        self.text = TextBuilder(settings.case_mode)
        self.seg = LetterSegmenter(self._timing())
        self.sender.on_sent = self._on_sent
        self._feat_cache: dict[str, np.ndarray] = {}
        # live tracking state
        self.meter = SpeedMeter()
        self.speed = 0.0
        self.active = False
        self.last_pred: Prediction | None = None
        self.history: deque = deque(maxlen=600)
        self._moving = False
        self._move_start = 0.0
        self._quiet_since: float | None = None
        self._pred_emit_t = -1.0
        self._last_frame_t: float | None = None
        # recording state
        self.jobs: deque[RecordJob] = deque()
        self.job: RecordJob | None = None
        self.phase: str | None = None
        self.phase_t0 = 0.0
        self.captured: list[HandFrame] = []
        self._next_job_at = 0.0
        self._count_shown: int | None = None
        self._rec_emit_t = -1.0
        self.now = 0.0

    # ---------------------------------------------------------------- setup
    def _timing(self) -> TimingConfig:
        s = self.s
        return TimingConfig(hold_s=s.hold_s, word_gap_s=s.word_gap_s, drop_send_s=s.drop_send_s,
                            idle_send_s=s.idle_send_s, min_conf=s.min_conf)

    def apply_settings(self, settings: Settings) -> None:
        with self.lock:
            retrain = (settings.knn_k != self.s.knn_k or settings.reject_ratio != self.s.reject_ratio
                       or settings.motion_tolerance != self.s.motion_tolerance)
            self.s = settings
            self.seg.cfg = self._timing()
            self.text.case_mode = settings.case_mode
            self.sender.min_interval = settings.live_interval_s
            self.sender.sound_on_send = settings.sound_on_send
            if retrain:
                self.retrain()

    def set_sender(self, sender: ChatboxSender) -> None:
        with self.lock:
            self.sender = sender
            sender.on_sent = self._on_sent

    def set_source_kind(self, kind: str) -> None:
        with self.lock:
            self.source_kind = kind
            self.meter.reset()
            self.retrain()

    def set_enabled(self, on: bool) -> None:
        with self.lock:
            self.enabled = on
            self.seg.reset()
            self.seg.has_text = not self.text.is_empty()
            self._moving = False
            self.emit({"type": "enabled", "on": on})

    # ------------------------------------------------------------- training
    def _features_for(self, take: Take) -> np.ndarray:
        f = self._feat_cache.get(take.id)
        if f is None:
            f = extract_many(_canon_seq(take.points, take.hand))
            self._feat_cache[take.id] = f
        return f

    def retrain(self) -> None:
        with self.lock:
            takes = self.profile.takes_for(self.source_kind)
            static = [(t.label, t.id, self._features_for(t)) for t in takes if t.kind == "static"]
            self.model = SignModel(k=self.s.knn_k, reject_ratio=self.s.reject_ratio).fit(static) if static else None
            templates = []
            for t in takes:
                if t.kind != "motion":
                    continue
                canon = _canon_seq(t.points, t.hand)
                tmpl = build_template(t.label, t.id, t.t, canon, self._features_for(t),
                                      speeds_for_sequence(t.t, canon))
                if tmpl is not None:
                    templates.append(tmpl)
            self.matcher = MotionMatcher(templates, self.model, dtw_tol=self.s.motion_tolerance)
            self.emit({"type": "model", "counts": self.profile.counts(self.source_kind),
                       "labels": self.model.labels if self.model else [],
                       "motion": self.matcher.labels})

    def delete_label(self, label: str) -> int:
        with self.lock:
            n = self.profile.remove_label(label, self.source_kind)
            self._save_and_retrain()
            return n

    def delete_all(self) -> None:
        with self.lock:
            self.profile.takes = [t for t in self.profile.takes if t.source != self.source_kind]
            self._save_and_retrain()

    def import_takes(self, takes) -> int:
        with self.lock:
            known = {t.id for t in self.profile.takes}
            added = [t for t in takes if t.id not in known]
            self.profile.takes.extend(added)
            self._save_and_retrain()
            return len(added)

    def _save_and_retrain(self) -> None:
        try:
            self.profile.save()
        except OSError as exc:
            log.error("could not save profile: %s", exc)
        self.retrain()

    def counts(self) -> dict[str, int]:
        with self.lock:
            return self.profile.counts(self.source_kind)

    def evaluate(self) -> dict:
        """Accuracy check: every recording is tested against the others."""
        with self.lock:
            report = {"static": None, "motion": {}}
            if self.model is not None:
                report["static"] = self.model.evaluate()
            motion_takes = [t for t in self.profile.takes_for(self.source_kind) if t.kind == "motion"]
            for t in motion_takes:
                others = [m for m in (self.matcher.templates if self.matcher else []) if m.take_id != t.id]
                if not any(o.label == t.label for o in others):
                    continue
                canon = _canon_seq(t.points, t.hand)
                feats = self._features_for(t)
                speeds = speeds_for_sequence(t.t, canon)
                seg = find_motion_segment(t.t, speeds, pts_seq=canon)
                ok = False
                if seg is not None:
                    a, b = seg
                    m = MotionMatcher(others, self.model, dtw_tol=self.s.motion_tolerance).match(
                        t.t[a:b + 1], canon[a:b + 1], feats[a:b + 1])
                    ok = m is not None and m.label == t.label
                hit, n = report["motion"].get(t.label, (0, 0))
                report["motion"][t.label] = (hit + int(ok), n + 1)
            return report

    # ------------------------------------------------------------ recording
    def record(self, label: str, takes: int | None = None) -> None:
        with self.lock:
            kind = label_kind(label)
            if takes is None:
                takes = self.s.takes_motion if kind == "motion" else (1 if label == REST_LABEL else self.s.takes_static)
            capture = {"motion": self.s.motion_record_s}.get(kind, self.s.static_record_s)
            if label == REST_LABEL:
                capture = self.s.rest_record_s
            for i in range(takes):
                self.jobs.append(RecordJob(label, kind, capture, i + 1, takes))
            self.emit({"type": "queue", "remaining": len(self.jobs)})

    def guided(self, labels=GUIDED_ORDER) -> int:
        """Queue every sign that still needs recordings. Returns the number of takes queued."""
        with self.lock:
            counts = self.profile.counts(self.source_kind)
            queued = 0
            for label in labels:
                want = (self.s.takes_motion if label_kind(label) == "motion"
                        else 1 if label == REST_LABEL else self.s.takes_static)
                missing = want - counts.get(label, 0)
                if missing > 0:
                    self.record(label, missing)
                    queued += missing
            return queued

    def cancel_recording(self) -> None:
        with self.lock:
            self.jobs.clear()
            self.job = None
            self.phase = None
            self.captured = []
            self.emit({"type": "rec", "phase": None})

    @property
    def recording(self) -> bool:
        return self.job is not None or bool(self.jobs)

    def _rec_event(self, now: float, force: bool = False) -> None:
        if not force and now - self._rec_emit_t < 0.1:
            return
        self._rec_emit_t = now
        if self.job is None:
            self.emit({"type": "rec", "phase": None, "queue": len(self.jobs),
                       "waiting": bool(self.jobs) and not self._hand_seen(now)})
            return
        j = self.job
        elapsed = now - self.phase_t0
        total = {"prompt": PROMPT_S, "countdown": self.s.countdown_s, "capture": j.capture_s}.get(self.phase, 1.0)
        self.emit({"type": "rec", "phase": self.phase, "label": j.label, "kind": j.kind,
                   "remaining": max(0.0, total - elapsed), "progress": min(1.0, elapsed / total if total else 1.0),
                   "take_no": j.take_no, "takes_total": j.takes_total, "queue": len(self.jobs)})

    def _haptic(self, pattern: str) -> None:
        if self.s.haptics:
            self.emit({"type": "haptic", "pattern": pattern})

    def _hand_seen(self, now: float) -> bool:
        return self._last_frame_t is not None and now - self._last_frame_t < 2.0

    def _tick_recording(self, now: float) -> None:
        if self.job is None:
            if not self.jobs or now < self._next_job_at:
                return
            if not self._hand_seen(now):
                self._rec_event(now)   # wait until hand tracking works before prompting
                return
            self.job = self.jobs.popleft()
            self.phase, self.phase_t0 = "prompt", now
            self._rec_event(now, force=True)
        elapsed = now - self.phase_t0
        if self.phase == "prompt" and elapsed >= PROMPT_S:
            self.phase, self.phase_t0, self._count_shown = "countdown", now, None
            elapsed = 0.0
        if self.phase == "countdown":
            remaining = self.s.countdown_s - elapsed
            if remaining <= 0:
                self.phase, self.phase_t0, self.captured = "capture", now, []
                self._haptic("go")
                self._rec_event(now, force=True)
                return
            sec = math.ceil(remaining)
            if sec != self._count_shown:
                self._count_shown = sec
                self._haptic("tick")
        if self.phase == "capture" and elapsed >= self.job.capture_s:
            self._finish_capture(now)
            return
        self._rec_event(now)

    def _finish_capture(self, now: float) -> None:
        job, frames = self.job, self.captured
        ok, message, take = self._make_take(job, frames)
        if ok:
            self.profile.add(take)
            try:
                self.profile.save()
            except OSError as exc:
                log.error("could not save profile: %s", exc)
            self.retrain()
            self._haptic("done")
        else:
            self._haptic("error")
            if not job.retried:
                job.retried = True
                self.jobs.appendleft(job)
        self.emit({"type": "take", "label": job.label, "ok": ok, "message": message,
                   "counts": self.profile.counts(self.source_kind)})
        self.job, self.phase, self.captured = None, None, []
        self._next_job_at = now + 0.8
        self._rec_event(now, force=True)

    def _make_take(self, job: RecordJob, frames: list[HandFrame]):
        hand_word = "right" if self.s.hand == "right" else "left"
        min_frames = max(8, int(job.capture_s * 12))
        if len(frames) < min_frames:
            return False, f"Couldn't see your {hand_word} hand clearly. Check tracking and try again.", None
        t = np.array([f.t for f in frames], np.float64)
        pts = np.stack([np.asarray(f.points, np.float32) for f in frames])
        hand = frames[0].hand
        canon = _canon_seq(pts, hand)
        message = "Saved"
        if job.kind == "motion":
            feats = extract_many(canon)
            tmpl = build_template(job.label, "check", t, canon, feats, speeds_for_sequence(t, canon))
            if tmpl is None:
                return False, f"No movement seen. Start the {job.label} motion right after GO.", None
        else:
            keep = t >= t[0] + 0.25
            t, pts, canon = t[keep], pts[keep], canon[keep]
            if job.label != REST_LABEL:
                speeds = speeds_for_sequence(t, canon)
                if float(np.median(speeds)) > self.seg.cfg.still_speed * 1.2:
                    message = "Saved, but your hand was moving. Holding still gives better results."
            if len(t) > 90:
                sel = np.unique(np.linspace(0, len(t) - 1, 90).round().astype(int))
                t, pts = t[sel], pts[sel]
        take = Take(label=job.label, kind=job.kind, source=self.source_kind, hand=hand, t=t, points=pts)
        return True, message, take

    # ------------------------------------------------------------- live path
    def on_update(self, frames: list[HandFrame], now: float) -> None:
        """Called by the input source for every tracking update (may be empty)."""
        with self.lock:
            self.now = now
            frame = next((f for f in frames if f.hand == self.s.hand), None)
            if frame is not None:
                self._process_frame(frame)
            else:
                self._process_missing(now, other_hand=bool(frames))
            self._tick_recording(now)
            self.sender.tick(now)

    def _is_active(self, frame: HandFrame) -> bool:
        if frame.head_relative:
            return float(frame.points[0][1]) >= -self.s.drop_height_m
        return True

    def _process_missing(self, now: float, other_hand: bool = False) -> None:
        if self._last_frame_t is not None and now - self._last_frame_t < 0.25:
            return  # brief tracking hiccup
        self.active = False
        self.meter.reset()
        self.speed = 0.0
        if self.enabled and not self.recording:
            for ev in self.seg.update(now, None, 0.0, 0.0, False):
                self._apply_event(ev)
        if now - self._pred_emit_t >= 0.1:
            self._pred_emit_t = now
            self.emit({"type": "pred", "label": None, "raw": None, "conf": 0.0, "hold": 0.0,
                       "active": False, "tracked": False, "other_hand": other_hand})

    def _process_frame(self, frame: HandFrame) -> None:
        t = frame.t
        self._last_frame_t = t
        pts = canonical_points(frame)
        try:
            feats = extract_features(pts)
        except ValueError:
            return
        self.speed = self.meter.update(t, pts)
        self.active = self._is_active(frame)
        pred = self.model.predict(feats) if self.model is not None else None
        self.last_pred = pred
        self.history.append((t, pts, feats))

        if self.phase == "capture":
            self.captured.append(frame)
        elif self.enabled and not self.recording:
            if self.active:
                self._motion_step(t)
            else:
                self._moving = False
            for ev in self.seg.update(t, pred.label if pred else None, pred.conf if pred else 0.0,
                                      self.speed, self.active):
                self._apply_event(ev)

        if t - self._pred_emit_t >= 1 / 15:
            self._pred_emit_t = t
            self.emit({"type": "pred", "label": pred.label if pred else None,
                       "raw": pred.raw_label if pred else None, "conf": pred.conf if pred else 0.0,
                       "ratio": pred.ratio if pred else None, "hold": self.seg.hold_progress,
                       "active": self.active, "tracked": True, "quality": frame.quality})

    # ------------------------------------------------------------- motion
    def _motion_step(self, t: float) -> None:
        if self.matcher is None or not self.matcher.templates:
            return
        sp = self.speed
        if not self._moving:
            if sp > MOTION_ON:
                self._moving, self._move_start, self._quiet_since = True, t, None
            return
        if t - self._move_start > MOTION_MAX_S:
            self._moving = False
            return
        if sp < MOTION_OFF:
            if self._quiet_since is None:
                self._quiet_since = t
            elif t - self._quiet_since >= MOTION_QUIET_S:
                self._moving = False
                start, end = self._move_start - 0.05, self._quiet_since + 0.03
                seg = [h for h in self.history if start <= h[0] <= end]
                if len(seg) >= 4:
                    self._try_motion(seg, t)
        else:
            self._quiet_since = None

    def _try_motion(self, seg: list, t_now: float) -> None:
        times = np.array([h[0] for h in seg])
        pts = np.stack([h[1] for h in seg])
        feats = np.stack([h[2] for h in seg])
        m = self.matcher.match(times, pts, feats)
        if m is None:
            return
        start_label = m.template.start_label if m.template is not None else None
        if (start_label is not None and self.seg.last_commit_t is not None
                and m.start_t - self.seg.last_commit_t <= 1.0
                and self.seg.last_label == start_label and self.text.last_char() == start_label):
            self.text.backspace()  # the hold before the motion typed e.g. "I"; it was really "J"
        self.text.add_letter(m.label)
        end_shape = self.model.predict(feats[-1]).raw_label if self.model else None
        self.seg.mark_commit(t_now, end_shape)
        self.emit({"type": "commit", "label": m.label, "motion": True})
        self._text_changed()

    # --------------------------------------------------------------- text
    def _apply_event(self, ev: Event) -> None:
        if ev.kind == "letter":
            self.text.add_letter(ev.value)
            self.emit({"type": "commit", "label": ev.value, "motion": False})
        elif ev.kind == "space":
            if not self.text.add_space():
                return
        elif ev.kind == "backspace":
            self.text.backspace()
        elif ev.kind == "clear":
            self.text.clear()
        elif ev.kind == "send":
            self._send()
            return
        self._text_changed()

    def _text_changed(self) -> None:
        while True:
            part = self.text.pop_overflow()
            if not part:
                break
            self.sender.send_final(part)
        self.seg.has_text = not self.text.is_empty()
        shown = self.text.display()
        if self.s.live_captions and shown.strip():
            self.sender.show_live(shown.strip())
        self.sender.set_typing(not self.text.is_empty())
        self.emit({"type": "text", "text": shown})

    def _send(self) -> None:
        msg = self.text.finalize()
        self.seg.has_text = False
        self.seg.space_pending = False
        if msg:
            self.sender.send_final(msg)
        else:
            self.sender.set_typing(False)
        self.emit({"type": "text", "text": ""})

    def _on_sent(self, text: str, final: bool) -> None:
        self.emit({"type": "sent", "text": text, "final": final})

    # UI buttons
    def send_now(self) -> None:
        with self.lock:
            self._send()
            self.sender.tick(self.now)

    def backspace(self) -> None:
        with self.lock:
            self.text.backspace()
            self._text_changed()

    def clear(self) -> None:
        with self.lock:
            self.text.clear()
            self._text_changed()

    def hud_state(self) -> dict:
        """Snapshot for the in-VR display."""
        with self.lock:
            p = self.last_pred
            state = {"enabled": self.enabled, "text": self.text.display(), "label": p.label if p else None,
                     "hold": self.seg.hold_progress, "active": self.active, "rec": None}
            if self.job is not None:
                elapsed = self.now - self.phase_t0
                total = {"prompt": PROMPT_S, "countdown": self.s.countdown_s,
                         "capture": self.job.capture_s}.get(self.phase, 1.0)
                state["rec"] = {"label": self.job.label, "phase": self.phase,
                                "remaining": max(0.0, total - elapsed),
                                "progress": min(1.0, elapsed / total if total else 1.0),
                                "kind": self.job.kind, "take_no": self.job.take_no,
                                "takes_total": self.job.takes_total}
            return state
