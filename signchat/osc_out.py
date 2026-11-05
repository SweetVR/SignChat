"""Talking to VRChat over OSC.

VRChat listens on UDP 127.0.0.1:9000.  /chatbox/input takes
(text, send_immediately, play_notification_sound) and /chatbox/typing
toggles the "..." typing bubble.  VRChat times out chatbox spam, so live
updates are coalesced and rate limited.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Callable

log = logging.getLogger(__name__)

SendFn = Callable[[str, list], None]


def udp_sender(host: str, port: int) -> SendFn:
    from pythonosc.udp_client import SimpleUDPClient

    client = SimpleUDPClient(host, port)

    def send(address: str, args: list) -> None:
        client.send_message(address, args)

    return send


class ChatboxSender:
    def __init__(self, host: str = "127.0.0.1", port: int = 9000, min_interval: float = 2.0,
                 sound_on_send: bool = False, send_fn: SendFn | None = None):
        self.min_interval = min_interval
        self.sound_on_send = sound_on_send
        self._send = send_fn or udp_sender(host, port)
        self._finals: deque[str] = deque()
        self._live: str | None = None
        self._last_sent_t: float | None = None
        self._last_text: str | None = None
        self._typing: bool | None = None
        self._lock = threading.Lock()
        self.on_sent: Callable[[str, bool], None] | None = None

    def _emit(self, address: str, args: list) -> None:
        try:
            self._send(address, args)
        except OSError as exc:  # VRChat not running is fine for UDP, but report real errors
            log.warning("OSC send failed: %s", exc)

    def set_typing(self, on: bool) -> None:
        with self._lock:
            if on != self._typing:
                self._typing = on
                self._emit("/chatbox/typing", [bool(on)])

    def show_live(self, text: str) -> None:
        """Replace the message shown while the person is still spelling."""
        with self._lock:
            self._live = text

    def send_final(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        with self._lock:
            self._finals.append(text)
            self._live = None

    def pending(self) -> bool:
        return bool(self._finals) or self._live is not None

    def tick(self, now: float) -> None:
        with self._lock:
            if self._last_sent_t is not None and now - self._last_sent_t < self.min_interval:
                return
            if self._finals:
                text, final = self._finals.popleft(), True
            elif self._live is not None:
                text, final, self._live = self._live, False, None
                if not text.strip() or text == self._last_text:
                    return
            else:
                return
            self._emit("/chatbox/input", [text[:144], True, bool(final and self.sound_on_send)])
            self._last_sent_t = now
            self._last_text = text
            if final:
                self._emit("/chatbox/typing", [False])
                self._typing = False
        if self.on_sent:
            self.on_sent(text, final)
