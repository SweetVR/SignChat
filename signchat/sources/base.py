"""Common plumbing for hand-tracking sources (each runs in its own thread)."""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable

from ..config import Settings
from ..hand import HandFrame

log = logging.getLogger(__name__)

UpdateFn = Callable[[list[HandFrame], float], None]
StatusFn = Callable[[str, str], None]   # (state, message); state: ok | waiting | error | stopped


class HandSource:
    kind = "base"
    name = "Base"
    has_hud = False
    has_haptics = False

    def __init__(self, settings: Settings, on_update: UpdateFn, on_status: StatusFn):
        self.settings = settings
        self.on_update = on_update
        self.on_status = on_status
        self.hud_provider: Callable[[], dict] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._haptics: queue.Queue = queue.Queue()

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_safe, name=f"source-{self.kind}", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def haptic(self, pattern: str) -> None:
        self._haptics.put(pattern)

    def status(self, state: str, message: str) -> None:
        try:
            self.on_status(state, message)
        except Exception:  # never let the UI callback kill tracking
            log.exception("status callback failed")

    def _run_safe(self) -> None:
        try:
            self.run()
        except Exception as exc:
            log.exception("input source crashed")
            self.status("error", f"{self.name} stopped: {exc}")

    def run(self) -> None:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError
