"""Optional: let an avatar parameter switch signing on and off.

Add a Bool parameter (default name "SignChat") to your avatar's expression
parameters and a toggle in your expressions menu.  VRChat sends it to port
9001, where SignChat listens.  Many OSC apps want port 9001, so this is off by
default; use an OSC router if another app already uses it.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

log = logging.getLogger(__name__)


class OscToggleListener:
    def __init__(self, port: int, param: str, on_change: Callable[[bool], None]):
        self.port = port
        self.param = param
        self.on_change = on_change
        self.server = None

    def start(self) -> None:
        from pythonosc.dispatcher import Dispatcher
        from pythonosc.osc_server import ThreadingOSCUDPServer

        disp = Dispatcher()
        disp.map(f"/avatar/parameters/{self.param}", self._handle)
        self.server = ThreadingOSCUDPServer(("127.0.0.1", self.port), disp)  # OSError if the port is taken
        threading.Thread(target=self.server.serve_forever, name="osc-in", daemon=True).start()

    def _handle(self, address: str, *args) -> None:
        if args:
            try:
                self.on_change(bool(args[0]))
            except Exception:
                log.exception("toggle handler failed")

    def stop(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
