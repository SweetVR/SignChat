"""Ties everything together: hand frames in, chatbox messages out.

Thread model: an input source calls `on_update(frames, now)` from its own
thread for every tracking update; the UI calls the public methods from the
UI thread.  Everything shares one lock.  Results are reported through
`emit(dict)`, which the UI drains on its own timer.

Two recognisers run side by side on the same hands:
  * letters (fingerspelling): the dominant hand, typed when you hold a
    handshape still (J and Z by their motion)
  * words and phrases: both hands, recognised from their movement
A word or phrase takes back the letters that were typed while it was being
signed, and a phrase replaces the separate signs it is made of.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import Counter, deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import signs as sg
from .classifier import REST_LABEL, Prediction, SignModel
from .config import Settings
from .features import extract_features, extract_many
from .hand import HandFrame, SpeedMeter, canonical_points, speeds_for_sequence
from .motion import MotionMatcher, build_template, find_motion_segment
from .osc_out import ChatboxSender
from .profile import Profile, Take
from .segmenter import Event, LetterSegmenter, TimingConfig
from .text_builder import TextBuilder
from .vocab import MAX_TEXT, VocabEntry, default_text, entry_for, normalize_gloss, validate_gloss

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
SIGN_LOG_S = 120.0            # how much recent signing is kept for "Fix a mistake"
SIGN_MAX_TEMPLATES = 8        # recordings per sign used for live matching (the newest ones)
FINGERSPELL_GUARD = 3         # a sign may not take back this many fingerspelled letters
SIGN_IDLE_S = 0.35            # hands down this long = no sign in progress
SIGN_TAKE_HZ = 45.0           # sign recordings are stored at up to this rate


def is_sign_label(label: str) -> bool:
    return len(label) >= 2 and not label.startswith("_")


def label_kind(label: str) -> str:
    if label in MOTION_LETTERS:
        return "motion"
    if is_sign_label(label):
        return "sign"
    return "static"


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


def _hands(frames: list[HandFrame]) -> tuple[HandFrame | None, HandFrame | None]:
    right = next((f for f in frames if f.hand == "right"), None)
    left = next((f for f in frames if f.hand == "left"), None)
    return right, left


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
        self.bank: sg.SignBank | None = None
        self.spotter: sg.SignSpotter | None = None
        self.packs: list = []                     # packs.Pack objects in use
        self.vocab: dict[str, VocabEntry] = self._load_vocab()
        self.text = TextBuilder(settings.case_mode)
        self.seg = LetterSegmenter(self._timing())
        self.sender.on_sent = self._on_sent
        self._feat_cache: dict[str, np.ndarray] = {}
        self._tmpl_cache: dict[tuple, sg.SignTemplate | None] = {}
        self._intra_cache: dict[tuple, float | None] = {}
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
        # words & phrases
        self.live_enc = sg.LiveEncoder(settings.hand)
        self._sign_idle = True
        self._sign_up_t: float | None = None
        self.frame_log: deque = deque(maxlen=int(SIGN_LOG_S * sg.SIGN_FPS))
        self._log_t = -1e9
        self.token_hist: deque = deque(maxlen=40)   # what was typed recently, for "Fix a mistake"
        # recording state
        self.jobs: deque[RecordJob] = deque()
        self.job: RecordJob | None = None
        self.phase: str | None = None
        self.phase_t0 = 0.0
        self.captured: list[HandFrame] = []
        self.captured_pairs: list = []
        self._next_job_at = 0.0
        self._count_shown: int | None = None
        self._rec_emit_t = -1.0
        self.now = 0.0

    # ---------------------------------------------------------------- setup
    def _timing(self) -> TimingConfig:
        s = self.s
        return TimingConfig(hold_s=s.hold_s, word_gap_s=s.word_gap_s, drop_send_s=s.drop_send_s,
                            idle_send_s=s.idle_send_s, min_conf=s.min_conf)

    def _space_y(self) -> float:
        """Palm height (m, relative to the eyes) below which the hands count as lowered."""
        return -(self.s.drop_height_m + 0.05)

    def apply_settings(self, settings: Settings) -> None:
        with self.lock:
            old = self.s
            retrain = any(getattr(settings, k) != getattr(old, k) for k in (
                "knn_k", "reject_ratio", "motion_tolerance", "sign_tolerance", "drop_height_m", "signs_enabled"))
            self.s = settings
            self.seg.cfg = self._timing()
            self.text.case_mode = settings.case_mode
            self.sender.min_interval = settings.live_interval_s
            self.sender.sound_on_send = settings.sound_on_send
            if settings.hand != old.hand:
                self.live_enc = sg.LiveEncoder(settings.hand)
                self._reset_signs()
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

    def set_packs(self, packs: list) -> None:
        """Template packs to use (packs.Pack objects)."""
        with self.lock:
            self.packs = list(packs)
            self.retrain()

    def set_enabled(self, on: bool) -> None:
        with self.lock:
            self.enabled = on
            self.seg.reset()
            self.seg.has_text = not self.text.is_empty()
            self._moving = False
            self._reset_signs()
            self.emit({"type": "enabled", "on": on})

    def _reset_signs(self) -> None:
        if self.spotter is not None:
            self.spotter.reset()
        self.live_enc.reset()
        self._sign_idle = True
        self._sign_up_t = None

    # ------------------------------------------------------------- vocabulary
    def _load_vocab(self) -> dict[str, VocabEntry]:
        out = {}
        for gloss, d in (self.profile.meta.get("signs") or {}).items():
            if isinstance(d, dict) and not validate_gloss(gloss):
                out[gloss] = VocabEntry.from_dict(gloss, d)
        return out

    def _store_vocab(self) -> None:
        self.profile.meta["signs"] = {g: e.to_dict() for g, e in self.vocab.items()}

    def entry(self, gloss: str) -> VocabEntry:
        """What a sign types (your own list first, then template packs, then the built-in list)."""
        if gloss in self.vocab:
            return self.vocab[gloss]
        for pack in self.packs:
            if gloss in pack.vocab:
                return pack.vocab[gloss]
        return entry_for(gloss)

    def add_sign(self, gloss: str, text: str | None = None, kind: str | None = None,
                 hands: str = "auto") -> tuple[str | None, str]:
        """Add a word or phrase to your list. Returns (error message or None, gloss)."""
        g = normalize_gloss(gloss)
        err = validate_gloss(g)
        if err:
            return err, g
        with self.lock:
            base = self.entry(g)
            text = (text if text is not None else base.text).strip()[:MAX_TEXT] or default_text(g)
            self.vocab[g] = VocabEntry(g, text, kind or base.kind, hands if hands in ("auto", "one", "two")
                                       else "auto", base.category)
            self._store_vocab()
            self._save_and_retrain()
        return None, g

    def add_common_signs(self, glosses: list[str]) -> int:
        with self.lock:
            n = 0
            for g in glosses:
                if g not in self.vocab and not validate_gloss(g):
                    self.vocab[g] = self.entry(g)
                    n += 1
            self._store_vocab()
            self._save_and_retrain()
            return n

    def update_sign(self, gloss: str, text: str | None = None, kind: str | None = None,
                    hands: str | None = None, sensitivity: float | None = None) -> None:
        with self.lock:
            e = self.vocab.get(gloss) or self.entry(gloss)
            if text is not None:
                e.text = text.strip()[:MAX_TEXT] or default_text(gloss)
            if kind in ("word", "phrase"):
                e.kind = kind
            if hands in ("auto", "one", "two"):
                e.hands = hands
            self.vocab[gloss] = e
            self._store_vocab()
            if sensitivity is not None:
                sens = self.profile.meta.setdefault("sign_sensitivity", {})
                if abs(sensitivity - 1.0) < 1e-3:
                    sens.pop(gloss, None)
                else:
                    sens[gloss] = float(np.clip(sensitivity, 0.4, 1.6))
            self._save_and_retrain()

    def remove_sign(self, gloss: str) -> int:
        """Remove a word/phrase and your recordings of it."""
        with self.lock:
            self.vocab.pop(gloss, None)
            self._store_vocab()
            (self.profile.meta.get("sign_sensitivity") or {}).pop(gloss, None)
            n = self.profile.remove_label(gloss, self.source_kind)
            self._save_and_retrain()
            return n

    def sign_sensitivity(self, gloss: str) -> float:
        return float((self.profile.meta.get("sign_sensitivity") or {}).get(gloss, 1.0))

    def sign_rows(self) -> list[dict]:
        """Everything in the Words & phrases list: your own list plus signs from template packs."""
        with self.lock:
            own = self.profile.counts(self.source_kind)
            packc = self.pack_counts()
            glosses = list(self.vocab)
            for g in packc:
                if is_sign_label(g) and g not in self.vocab:
                    glosses.append(g)
            rows = []
            for g in glosses:
                e = self.entry(g)
                two = None
                if self.bank is not None:
                    two = next((t.two_handed for t in self.bank.templates if t.label == g), None)
                rows.append({"gloss": g, "text": e.text, "kind": e.kind, "hands": e.hands, "two_handed": two,
                             "own": own.get(g, 0), "pack": packc.get(g, 0), "mine": g in self.vocab,
                             "ready": self.bank is not None and g in self.bank.labels,
                             "sensitivity": self.sign_sensitivity(g)})
            return rows

    # ------------------------------------------------------------- training
    def _features_for(self, take: Take) -> np.ndarray:
        f = self._feat_cache.get(take.id)
        if f is None:
            f = extract_many(_canon_seq(take.points, take.hand))
            self._feat_cache[take.id] = f
        return f

    def _all_takes(self) -> tuple[list[Take], dict[str, str]]:
        """Your recordings, plus template-pack recordings of anything you haven't recorded yourself.
        Also returns {take id: pack name} for the pack ones."""
        own = self.profile.takes_for(self.source_kind)
        mine = {t.label for t in own}
        takes, src = list(own), {}
        for pack in self.packs:
            for t in pack.takes:
                if t.source == self.source_kind and t.label not in mine:
                    takes.append(t)
                    src[t.id] = pack.info.name
        return takes, src

    def pack_counts(self) -> dict[str, int]:
        with self.lock:
            takes, src = self._all_takes()
            return dict(Counter(t.label for t in takes if t.id in src))

    def retrain(self) -> None:
        with self.lock:
            takes, src = self._all_takes()
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
            self._retrain_signs([t for t in takes if t.kind == "sign"], src)
            self.emit({"type": "model", "counts": self.profile.counts(self.source_kind),
                       "pack_counts": dict(Counter(t.label for t in takes if t.id in src)),
                       "labels": self.model.labels if self.model else [],
                       "motion": self.matcher.labels,
                       "signs": list(self.bank.labels) if self.bank else []})

    def _sign_template(self, t: Take, space_y: float, source: str) -> sg.SignTemplate | None:
        key = (t.id, round(space_y, 3))
        if key not in self._tmpl_cache:
            tm = None
            pts = np.asarray(t.points)
            if pts.ndim == 4 and pts.shape[1:] == (2, 21, 3) and len(pts) == len(t.t) >= 2:
                present = t.present if t.present is not None else np.ones((len(t.t), 2), bool)
                try:
                    tm = sg.build_template(t.label, t.id, t.t, pts, present, t.hand, "auto", space_y,
                                           head_relative=t.source != "webcam", source=source, trim=not t.clip)
                except Exception:
                    log.exception("could not use recording %s of %s", t.id, t.label)
            self._tmpl_cache[key] = tm
        return self._tmpl_cache[key]

    def _retrain_signs(self, takes: list[Take], src: dict[str, str]) -> None:
        space_y = self._space_y()
        by_label: dict[str, list] = {}
        for t in takes:
            tm = self._sign_template(t, space_y, src.get(t.id, ""))
            if tm is not None:
                by_label.setdefault(t.label, []).append((t.created, tm))
        templates = []
        for items in by_label.values():
            items.sort(key=lambda x: -x[0])
            templates += [tm for _, tm in items[:SIGN_MAX_TEMPLATES]]
        if not templates:
            self.bank = self.spotter = None
            return
        sg.harmonize_hands(templates, {g: self.entry(g).hands for g in by_label})
        groups: dict[str, list] = {}
        for tm in templates:
            groups.setdefault(tm.label, []).append(tm)
        intra = {}
        for label, ts in groups.items():
            key = (label, tuple(sorted(t.take_id for t in ts)), ts[0].two_handed, round(space_y, 3))
            if key not in self._intra_cache:
                self._intra_cache[key] = sg.intra_cost(ts)
            intra[label] = self._intra_cache[key]
        sens = {g: float(v) for g, v in (self.profile.meta.get("sign_sensitivity") or {}).items()}
        self.bank = sg.SignBank(templates, tolerance=self.s.sign_tolerance, sensitivity=sens, intra=intra)
        self.spotter = sg.SignSpotter(self.bank, space_y=space_y if self.source_kind != "webcam" else -1e9)
        self.live_enc.reset()
        self._sign_idle = True

    def delete_label(self, label: str) -> int:
        with self.lock:
            n = self.profile.remove_label(label, self.source_kind)
            self._save_and_retrain()
            return n

    def delete_all(self) -> None:
        with self.lock:
            self.profile.takes = [t for t in self.profile.takes if t.source != self.source_kind]
            self._save_and_retrain()

    def import_takes(self, takes, meta: dict | None = None) -> int:
        with self.lock:
            known = {t.id for t in self.profile.takes}
            added = [t for t in takes if t.id not in known]
            self.profile.takes.extend(added)
            for gloss, d in ((meta or {}).get("signs") or {}).items():
                if gloss not in self.vocab and isinstance(d, dict) and not validate_gloss(gloss):
                    self.vocab[gloss] = VocabEntry.from_dict(gloss, d)
            self._store_vocab()
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

    def evaluate(self, should_stop=None) -> dict:
        """Accuracy check: every recording is tested against the others.
        Runs without holding the lock for long, so signing keeps working."""
        with self.lock:
            model, matcher, bank = self.model, self.matcher, self.bank
            motion_takes = [t for t in self.profile.takes_for(self.source_kind) if t.kind == "motion"]
            tol = self.s.motion_tolerance
        report = {"static": None, "motion": {}, "signs": None}
        if model is not None:
            report["static"] = model.evaluate()
        for t in motion_takes:
            others = [m for m in (matcher.templates if matcher else []) if m.take_id != t.id]
            if not any(o.label == t.label for o in others):
                continue
            canon = _canon_seq(t.points, t.hand)
            feats = self._features_for(t)
            speeds = speeds_for_sequence(t.t, canon)
            seg = find_motion_segment(t.t, speeds, pts_seq=canon)
            ok = False
            if seg is not None:
                a, b = seg
                m = MotionMatcher(others, model, dtw_tol=tol).match(t.t[a:b + 1], canon[a:b + 1], feats[a:b + 1])
                ok = m is not None and m.label == t.label
            hit, n = report["motion"].get(t.label, (0, 0))
            report["motion"][t.label] = (hit + int(ok), n + 1)
        if bank is not None:
            report["signs"] = sg.evaluate_bank(bank, should_stop=should_stop)
        return report

    # ------------------------------------------------------------ recording
    def record(self, label: str, takes: int | None = None) -> None:
        with self.lock:
            kind = label_kind(label)
            if kind == "sign":
                entry = self.entry(label)
                if label not in self.vocab:
                    self.vocab[label] = entry
                    self._store_vocab()
                if takes is None:
                    takes = self.s.takes_sign
                capture = self.s.phrase_record_s if entry.kind == "phrase" else self.s.sign_record_s
            else:
                if takes is None:
                    takes = self.s.takes_motion if kind == "motion" else (1 if label == REST_LABEL else self.s.takes_static)
                capture = {"motion": self.s.motion_record_s}.get(kind, self.s.static_record_s)
                if label == REST_LABEL:
                    capture = self.s.rest_record_s
            for i in range(takes):
                self.jobs.append(RecordJob(label, kind, capture, i + 1, takes))
            self.emit({"type": "queue", "remaining": len(self.jobs)})

    def guided(self, labels=GUIDED_ORDER) -> int:
        """Queue every letter that still needs recordings. Returns the number of takes queued."""
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

    def guided_signs(self, glosses: list[str] | None = None) -> int:
        """Queue every word/phrase in your list that has fewer than `takes_sign` recordings."""
        with self.lock:
            counts = self.profile.counts(self.source_kind)
            queued = 0
            for g in (glosses if glosses is not None else list(self.vocab)):
                missing = self.s.takes_sign - counts.get(g, 0)
                if missing > 0:
                    self.record(g, missing)
                    queued += missing
            return queued

    def cancel_recording(self) -> None:
        with self.lock:
            self.jobs.clear()
            self.job = None
            self.phase = None
            self.captured = []
            self.captured_pairs = []
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
                self.phase, self.phase_t0, self.captured, self.captured_pairs = "capture", now, [], []
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
        job = self.job
        frames = self.captured_pairs if job.kind == "sign" else self.captured
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
        self.job, self.phase, self.captured, self.captured_pairs = None, None, [], []
        self._next_job_at = now + 0.8
        self._rec_event(now, force=True)

    def _make_take(self, job: RecordJob, frames: list):
        if job.kind == "sign":
            return self._make_sign_take(job, frames)
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
        take = Take(label=job.label, kind=job.kind, source=self.source_kind, hand=hand, t=t, points=pts,
                    quality=frames[0].quality)
        return True, message, take

    def _pairs_to_arrays(self, pairs: list, max_hz: float = SIGN_TAKE_HZ):
        """[(t, right frame, left frame)] -> times, (n, 2, 21, 3) level points, (n, 2) presence."""
        kept, last = [], -1e9
        for p in pairs:
            if p[0] >= last + 1.0 / max_hz - 1e-6:
                kept.append(p)
                last = p[0]
        t = np.array([p[0] for p in kept], np.float64)
        P = np.zeros((len(kept), 2, 21, 3), np.float32)
        M = np.zeros((len(kept), 2), bool)
        for i, (_, r, l) in enumerate(kept):
            if r is not None:
                P[i, 0], M[i, 0] = r.level_points(), True
            if l is not None:
                P[i, 1], M[i, 1] = l.level_points(), True
        return t, P, M

    def _dominant_quality(self, pairs: list) -> str:
        idx = 0 if self.s.hand == "right" else 1
        q = Counter(p[1 + idx].quality for p in pairs if p[1 + idx] is not None)
        return q.most_common(1)[0][0] if q else ""

    def _make_sign_take(self, job: RecordJob, pairs: list):
        idx = 0 if self.s.hand == "right" else 1
        seen = sum(1 for p in pairs if p[1 + idx] is not None)
        if seen < max(8, int(job.capture_s * 12)):
            return False, f"Couldn't see your {self.s.hand} hand clearly. Check tracking and try again.", None
        t, P, M = self._pairs_to_arrays(pairs)
        entry = self.entry(job.label)
        tm = sg.build_template(job.label, "check", t, P, M, self.s.hand, entry.hands, self._space_y(),
                               head_relative=self.source_kind != "webcam")
        if tm is None:
            return False, f"No sign seen. Raise your hands and sign {job.label} right after GO.", None
        message = "Saved"
        if tm.cut_off:
            message = (f"Saved, but you were still signing when the recording stopped. If {job.label} got cut off, "
                       "record it again or make the recording time longer in Settings.")
        take = Take(label=job.label, kind="sign", source=self.source_kind, hand=self.s.hand, t=t, points=P,
                    present=M, quality=self._dominant_quality(pairs))
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
            self._process_signs(frames, frame, now)
            self._tick_recording(now)
            self.sender.tick(now)

    def _is_active(self, frame: HandFrame) -> bool:
        if frame.head_relative:
            return float(frame.level_points()[0][1]) >= -self.s.drop_height_m
        return True

    def _in_sign_space(self, frame: HandFrame) -> bool:
        if not frame.head_relative:
            return True
        return float(frame.level_points()[sg.PALM_IDX, 1].mean()) >= self._space_y()

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

    # ------------------------------------------------------------- words & phrases
    def _process_signs(self, frames: list[HandFrame], d_frame: HandFrame | None, now: float) -> None:
        right, left = _hands(frames)
        if self.phase == "capture" and self.job is not None and self.job.kind == "sign":
            if right is not None or left is not None:
                t = d_frame.t if d_frame is not None else (right or left).t
                self.captured_pairs.append((t, right, left))
            return
        if not self.enabled or self.recording:
            return
        t = d_frame.t if d_frame is not None else (frames[0].t if frames else now)
        if frames and t - self._log_t >= 1.0 / sg.SIGN_FPS - 1e-6:
            self._log_t = t
            self.frame_log.append((t, right, left))
        if self.spotter is None or not self.s.signs_enabled:
            return
        if d_frame is not None and self._in_sign_space(d_frame):
            self._sign_up_t = t
        if self._sign_up_t is None or t - self._sign_up_t > SIGN_IDLE_S:
            if not self._sign_idle:
                self._sign_idle = True
                self.live_enc.reset()
                for det in self.spotter.flush(dropped=d_frame is not None):
                    self._apply_sign(det)
            return
        self._sign_idle = False
        r = right.level_points() if right is not None else None
        l = left.level_points() if left is not None else None
        for g, x in self.live_enc.update(t, r, l):
            for det in self.spotter.push(x, g):
                self._apply_sign(det)

    def _remember(self, tok, **info) -> None:
        self.token_hist.append({"id": tok.id, "kind": tok.kind, "text": tok.text, "label": tok.label,
                                "t0": tok.t0, "t1": tok.t1, "fixed": None, **info})

    def _shape_at(self, t: float) -> str | None:
        """The handshape the letter classifier saw at time t."""
        if self.model is None or not self.history:
            return None
        h = min(self.history, key=lambda h: abs(h[0] - t))
        return self.model.predict(h[2]).raw_label if abs(h[0] - t) < 0.2 else None

    def _hand_still_since(self, t0: float, max_move: float = 0.04) -> bool:
        wrists = [h[1][0] for h in self.history if h[0] >= t0 - 0.02]
        if len(wrists) < 2:
            return True
        w = np.asarray(wrists)
        return float(np.linalg.norm(w - w[0], axis=1).max()) < max_move

    def _apply_sign(self, det: sg.Detection) -> None:
        if self.text.letters_since(det.start_t, det.end_t) >= FINGERSPELL_GUARD:
            log.info("ignored %s (%.2f): it overlaps fingerspelling", det.label, det.score)
            return
        entry = self.entry(det.label)
        tok, removed = self.text.add_word(entry.text, det.label, det.start_t, det.end_t)
        self._remember(tok, score=det.score, removed=removed)
        # still holding the handshape the sign ended with: don't type it as a letter
        end_shape = self._shape_at(det.end_t) if self._hand_still_since(det.end_t) else None
        self.seg.after_sign(self.now, end_shape)
        self.emit({"type": "commit", "label": det.label, "sign": True, "text": entry.text, "token": tok.id,
                   "replaced": [t.text for t in removed if t.kind != "space"]})
        self._text_changed()

    # ------------------------------------------------------------- motion letters
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
        tok = self.text.add_letter(m.label, m.start_t, t_now)
        self._remember(tok)
        end_shape = self.model.predict(feats[-1]).raw_label if self.model else None
        self.seg.mark_commit(t_now, end_shape)
        self.emit({"type": "commit", "label": m.label, "motion": True, "token": tok.id})
        self._text_changed()

    # --------------------------------------------------------------- text
    def _apply_event(self, ev: Event) -> None:
        if ev.kind == "letter":
            tok = self.text.add_letter(ev.value, ev.t - self.s.hold_s, ev.t)
            self._remember(tok)
            self.emit({"type": "commit", "label": ev.value, "motion": False, "token": tok.id})
        elif ev.kind == "space":
            if not self.text.add_space(ev.t):
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

    # ------------------------------------------------------------- fixing mistakes
    def recent_tokens(self, n: int = 12) -> list[dict]:
        """What was typed recently (newest last), for the "Fix a mistake" list."""
        with self.lock:
            out = []
            for info in list(self.token_hist)[-n:]:
                d = {k: v for k, v in info.items() if k != "removed"}
                d["in_message"] = self.text.find(info["id"]) is not None
                out.append(d)
            return out

    def _logged(self, t0: float, t1: float) -> list:
        return [e for e in self.frame_log if t0 <= e[0] <= t1]

    def _learn_letter(self, label: str, frames: list) -> bool:
        idx = 1 if self.s.hand == "right" else 2
        dom = [e[idx] for e in frames if e[idx] is not None]
        if len(dom) < 4:
            return False
        t = np.array([f.t for f in dom], np.float64)
        pts = np.stack([np.asarray(f.points, np.float32) for f in dom])
        self.profile.add(Take(label=label, kind="static", source=self.source_kind, hand=dom[0].hand, t=t,
                              points=pts, quality=dom[0].quality))
        return True

    def _learn_sign(self, label: str, frames: list) -> bool:
        if len(frames) < 8:
            return False
        t, P, M = self._pairs_to_arrays(frames)
        if label not in self.vocab:
            self.vocab[label] = self.entry(label)
            self._store_vocab()
        self.profile.add(Take(label=label, kind="sign", source=self.source_kind, hand=self.s.hand, t=t,
                              points=P, present=M, quality=self._dominant_quality(frames), clip=True))
        return True

    def _tighten(self, gloss: str) -> None:
        sens = self.profile.meta.setdefault("sign_sensitivity", {})
        sens[gloss] = round(max(0.5, float(sens.get(gloss, 1.0)) * 0.85), 3)

    def fix_token(self, token_id: int, new_label: str | None) -> str:
        """Correct something that was typed, and learn from it.

        new_label: the letter or sign it really was, or None for "nothing"
        (I wasn't signing that).  Returns a short message for the user."""
        with self.lock:
            info = next((i for i in self.token_hist if i["id"] == token_id), None)
            if info is None:
                return "That's too long ago to fix."
            in_msg = self.text.find(token_id) is not None
            old = info["label"]
            learned = ""
            if info["kind"] == "letter":
                frames = self._logged(info["t0"] - 0.05, info["t1"] + 0.02)
                if new_label is None:
                    if self._learn_letter(REST_LABEL, frames):
                        learned = f"SignChat will be less likely to type {old} for that handshape."
                    if in_msg:
                        self.text.remove(token_id)
                elif label_kind(new_label) == "sign":
                    if in_msg:
                        self.text.replace(token_id, "word", self.entry(new_label).text, new_label)
                else:
                    if label_kind(new_label) == "static" and self._learn_letter(new_label, frames):
                        learned = f"Saved that handshape as another recording of {new_label}."
                    if in_msg:
                        self.text.replace(token_id, "letter", new_label, new_label)
            else:
                if new_label is None or label_kind(new_label) != "sign":
                    self._tighten(old)
                    learned = f"{old} now needs to be signed more exactly before it's typed."
                    if in_msg:
                        if new_label is None:
                            self.text.replace_with(token_id, info.get("removed") or [])
                        else:
                            self.text.replace(token_id, "letter", new_label, new_label)
                else:
                    frames = self._logged(info["t0"] - 0.1, info["t1"] + 0.1)
                    if self._learn_sign(new_label, frames):
                        learned = f"Saved that as another recording of {new_label}."
                    if in_msg:
                        self.text.replace(token_id, "word", self.entry(new_label).text, new_label)
            info["fixed"] = new_label or ""
            if in_msg:
                self._text_changed()
            if learned:
                self._save_and_retrain()
            return learned or "Fixed."

    # ------------------------------------------------------------- VR panel
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
                if self.job.kind == "sign":
                    state["rec"]["sign_kind"] = self.entry(self.job.label).kind
            return state
