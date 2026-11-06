"""A simulated signer so you can try SignChat (and its whole training flow)
without VR: type letters in the app window, or pick a word to sign, and the
virtual hands do it; during recording they follow the prompts by themselves.

The simulated words are made-up movements, not real ASL.
"""

from __future__ import annotations

import queue
import time

import numpy as np

from ..synth import REST_POSITION, SIGN_POSITION, Performer, SimUser
from .base import HandSource


class SimulatorSource(HandSource):
    kind = "sim"
    name = "Simulated hands"

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.keys: queue.Queue = queue.Queue()
        self.auto_follow = True     # follow recording prompts automatically
        self.user = SimUser(seed=42, hand=self.settings.hand)

    def press(self, key: str) -> None:
        """A letter, " " (pause), "\\n" (lower the hands)."""
        self.keys.put(key)

    def perform_sign(self, gloss: str, kind: str | None = None) -> None:
        """Sign a word or phrase (made-up simulated movement)."""
        self.keys.put(("sign", gloss, kind))

    def _follow_prompt(self, perf: Performer, rec: dict, state: dict) -> bool:
        label, phase = rec["label"], rec["phase"]
        key = (label, rec.get("take_no"))
        if rec.get("kind") == "sign":
            if phase in ("prompt", "countdown") and state.get("shape") != key:
                state["shape"] = key
                perf.lower_hands(0.3)
                return True
            if phase == "capture" and state.get("moved") != key:
                state["moved"] = key
                perf._hold(0.3)
                perf.sign(label, rec.get("sign_kind"))
                perf._hold(0.25)
                perf.lower_hands(0.4)
                return True
            return False
        pos = REST_POSITION if label == "_rest" else SIGN_POSITION
        if phase in ("prompt", "countdown"):
            if state.get("shape") != key:
                state["shape"] = key
                perf._move(self.user.take_pose(label, perf.rng), pos, 0.3)
                return True
        elif phase == "capture":
            if rec.get("kind") == "motion" and state.get("moved") != key:
                state["moved"] = key
                perf._hold(0.1)
                perf.motion(label)
                perf._hold(0.3)
                return True
            if label == "_rest":
                perf._move(self.user.take_pose("_rest", perf.rng), REST_POSITION + perf.rng.normal(0, 0.03, 3), 0.25)
                return True
        return False

    def _do_key(self, perf: Performer, key, last_key: str | None) -> str | None:
        if isinstance(key, tuple):
            perf.sign(key[1], key[2])
            return None
        if key.upper() == last_key and key.upper() not in ("J", "Z") and key.strip():
            # same letter again: a small bounce, the way double letters are signed
            perf._move(perf.pose, SIGN_POSITION + np.array([0.045, 0.0, 0.0]), 0.14)
            perf._move(perf.pose, SIGN_POSITION, 0.14)
            perf._hold(0.55)
        else:
            perf.play(["|" if key == "\n" else key])
        return key.upper() if key.strip() else None

    def run(self) -> None:
        self.status("ok", "Simulated hands ready. Type letters or pick a word to sign it.")
        perf = Performer(self.user, seed=int(time.time()) % 10000, t0=time.monotonic())
        follow_state: dict = {}
        sent = 0
        last_key = None
        while not self.stopping:
            now = time.monotonic()
            if sent >= len(perf.frames):
                perf.frames.clear()
                perf.frames_n.clear()
                sent = 0
                perf.t = max(perf.t, now)
                rec = None
                if self.auto_follow and self.hud_provider is not None:
                    rec = self.hud_provider().get("rec")
                if rec and self._follow_prompt(perf, rec, follow_state):
                    pass
                elif not rec and not self.keys.empty():
                    last_key = self._do_key(perf, self.keys.get_nowait(), last_key)
                else:
                    perf._hold(0.05)
            latest = None
            while sent < len(perf.frames) and perf.frames[sent].t <= now:
                latest = sent
                sent += 1
            if latest is None:
                self.on_update([], now)
            else:
                batch = [perf.frames[latest]]
                if perf.frames_n[latest] is not None:
                    batch.append(perf.frames_n[latest])
                self.on_update(batch, now)
            self._stop.wait(1 / 60)
        self.status("stopped", "Stopped")
