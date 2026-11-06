"""Talking to VRChat (and TTS Voice Wizard) over OSC.

VRChat listens on UDP 127.0.0.1:9000.  /chatbox/input takes
(text, send_immediately, play_notification_sound) and /chatbox/typing
toggles the "..." typing bubble.  VRChat times out chatbox spam, so live
updates are coalesced and rate limited.

TTS Voice Wizard (https://github.com/VRCWizard/TTS-Voice-Wizard) has an OSC
listener (port 4026 by default; turn it on in its OSC Listener settings).
When it's used, finished messages go to TTS Voice Wizard instead of straight
to the chatbox, and it shows, speaks and/or translates them with its own
settings, so the two apps don't overwrite each other.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Callable

log = logging.getLogger(__name__)

SendFn = Callable[[str, list], None]
TTSVW_SPEAK = "/TTSVoiceWizard/TextToSpeech"
TTSVW_TEXT = "/TTSVoiceWizard/TextToText"


def udp_sender(host: str, port: int) -> SendFn:
    from pythonosc.udp_client import SimpleUDPClient

    client = SimpleUDPClient(host, port)

    def send(address: str, args: list) -> None:
        client.send_message(address, args)

    return send


def ttsvw_sink(host: str = "127.0.0.1", port: int = 4026, speak: bool = True,
               send_fn: SendFn | None = None) -> Callable[[str], None]:
    """Finished messages -> TTS Voice Wizard's OSC listener (it uses its own voice/chatbox settings)."""
    send = send_fn or udp_sender(host, port)
    address = TTSVW_SPEAK if speak else TTSVW_TEXT

    def sink(text: str) -> None:
        send(address, [text])

    return sink


class ChatboxSender:
    def __init__(self, host: str = "127.0.0.1", port: int = 9000, min_interval: float = 2.0,
                 sound_on_send: bool = False, send_fn: SendFn | None = None,
                 final_sink: Callable[[str], None] | None = None):
        self.min_interval = min_interval
        self.sound_on_send = sound_on_send
        self.final_sink = final_sink     # if set, finished messages go here instead of the chatbox
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
            if final and self.final_sink is not None:
                try:
                    self.final_sink(text)
                except OSError as exc:
                    log.warning("sending to TTS Voice Wizard failed: %s", exc)
            else:
                self._emit("/chatbox/input", [text[:144], True, bool(final and self.sound_on_send)])
            self._last_sent_t = now
            self._last_text = text
            if final:
                self._emit("/chatbox/typing", [False])
                self._typing = False
        if self.on_sent:
            self.on_sent(text, final)
