"""SignChat desktop window (PySide6)."""

from __future__ import annotations

import dataclasses
import logging
import logging.handlers
import os
import queue
import sys
import threading
from pathlib import Path

from PySide6.QtCore import QRectF, QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QColor, QDesktopServices, QFont, QIcon, QImage, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QMainWindow, QMenu,
    QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QToolButton, QVBoxLayout, QWidget,
)

from . import APP_NAME, __version__
from .config import Settings
from .engine import LETTERS, MOTION_LETTERS, Engine, display_name
from .osc_in import OscToggleListener
from .osc_out import ChatboxSender
from .paths import app_dir, logs_dir
from .profile import Profile
from .sources import SOURCES, create_source

log = logging.getLogger(__name__)

BG = "#14161c"
PANEL = "#1c1f27"
PANEL_2 = "#252935"
BORDER = "#323746"
TEXT = "#eef0f4"
MUTED = "#9aa1b1"
GREEN = "#48d68c"
AMBER = "#ffba49"
RED = "#ff6363"
BLUE = "#6aa8ff"

STYLE = f"""
QMainWindow, QDialog {{ background: {BG}; }}
QWidget {{ color: {TEXT}; font-size: 10.5pt; }}
QFrame#panel {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 14px; }}
QLabel {{ background: transparent; }}
QLabel#h1 {{ font-size: 15pt; font-weight: 600; }}
QLabel#h2 {{ font-size: 12pt; font-weight: 600; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#message {{ font-size: 20pt; background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 12px 14px; }}
QLabel#chip {{ background: {PANEL_2}; border-radius: 12px; padding: 5px 12px; }}
QPushButton, QToolButton {{ background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 9px;
    padding: 7px 14px; }}
QPushButton:hover, QToolButton:hover {{ border-color: #4a5266; }}
QPushButton:disabled {{ color: #5b6172; }}
QPushButton#primary {{ background: #2f6f4f; border-color: #3d8c64; font-weight: 600; }}
QPushButton#primary:hover {{ background: #367d5a; }}
QPushButton#start {{ font-size: 14pt; font-weight: 600; padding: 12px; background: #2f6f4f; border-color: #3d8c64; }}
QPushButton#start[on="true"] {{ background: #6b2f35; border-color: #8c3d45; }}
QPushButton#tile {{ font-size: 15pt; font-weight: 600; min-width: 54px; min-height: 54px; padding: 2px; }}
QPushButton#tile[state="empty"] {{ color: {MUTED}; }}
QPushButton#tile[state="partial"] {{ border: 2px solid {AMBER}; }}
QPushButton#tile[state="done"] {{ border: 2px solid {GREEN}; }}
QPushButton#tile:checked {{ background: #33405a; }}
QListWidget {{ background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 10px; padding: 6px; }}
QListWidget::item {{ padding: 4px 2px; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{ background: {PANEL_2}; border: 1px solid {BORDER};
    border-radius: 8px; padding: 5px 8px; }}
QGroupBox {{ border: 1px solid {BORDER}; border-radius: 10px; margin-top: 14px; padding: 10px 8px 8px 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 4px; color: {MUTED}; }}
QScrollArea {{ border: none; background: {BG}; }}
QScrollArea > QWidget > QWidget {{ background: {BG}; }}
QMenu {{ background: {PANEL}; border: 1px solid {BORDER}; }}
QMenu::item:selected {{ background: #33405a; }}
"""


def setup_logging() -> Path:
    path = logs_dir() / "signchat.log"
    handler = logging.handlers.RotatingFileHandler(path, maxBytes=1_000_000, backupCount=2, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    return path


def make_icon() -> QIcon:
    pm = QPixmap(64, 64)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor("#2f6f4f"))
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(4, 4, 56, 56, 14, 14)
    p.setPen(QColor("white"))
    f = QFont()
    f.setPixelSize(30)
    f.setBold(True)
    p.setFont(f)
    p.drawText(QRectF(4, 4, 56, 56), Qt.AlignCenter, "Aa")
    p.end()
    return QIcon(pm)


class LetterRing(QWidget):
    """Big current-letter display with a "hold to type" progress ring."""

    def __init__(self):
        super().__init__()
        self.text = "–"
        self.progress = 0.0
        self.color = QColor(MUTED)
        self.flash = 0
        self.setMinimumSize(170, 170)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def sizeHint(self) -> QSize:
        return QSize(170, 170)

    def set_state(self, text: str, progress: float, color: str) -> None:
        self.text, self.progress, self.color = text, progress, QColor(color)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(12, 12, self.width() - 24, self.height() - 24)
        p.setPen(QPen(QColor(BORDER), 10))
        p.drawEllipse(r)
        if self.progress > 0 or self.flash:
            p.setPen(QPen(QColor(GREEN) if self.flash else self.color, 10, Qt.SolidLine, Qt.RoundCap))
            span = 360 if self.flash else 360 * min(1.0, self.progress)
            p.drawArc(r, 90 * 16, int(-span * 16))
        f = QFont()
        f.setPixelSize(74 if len(self.text) <= 2 else 30)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(TEXT))
        p.drawText(r, Qt.AlignCenter, self.text)
        p.end()


def tile_text(label: str) -> str:
    return {"_rest": "Rest", "_space": "Space", "_backspace": "Del", "_send": "Send", "_clear": "Clear"}.get(label, label)


class SettingsDialog(QDialog):
    def __init__(self, s: Settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SignChat settings")
        self.s = dataclasses.replace(s)
        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        lay = QVBoxLayout(body)
        scroll.setWidget(body)
        outer.addWidget(scroll)
        self.resize(640, 700)
        self.fields = {}

        def group(title):
            g = QGroupBox(title)
            f = QFormLayout(g)
            f.setRowWrapPolicy(QFormLayout.WrapLongRows)
            f.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            lay.addWidget(g)
            return f

        def combo(form, key, label, options):
            c = QComboBox()
            c.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            c.setMinimumContentsLength(18)
            for value, text in options:
                c.addItem(text, value)
            c.setCurrentIndex(max(0, [v for v, _ in options].index(getattr(self.s, key))
                                  if getattr(self.s, key) in [v for v, _ in options] else 0))
            form.addRow(label, c)
            self.fields[key] = c

        def dspin(form, key, label, lo, hi, step, suffix=""):
            w = QDoubleSpinBox()
            w.setRange(lo, hi)
            w.setSingleStep(step)
            w.setDecimals(2)
            w.setSuffix(suffix)
            w.setValue(getattr(self.s, key))
            form.addRow(label, w)
            self.fields[key] = w

        def ispin(form, key, label, lo, hi):
            w = QSpinBox()
            w.setRange(lo, hi)
            w.setValue(getattr(self.s, key))
            form.addRow(label, w)
            self.fields[key] = w

        def check(form, key, label):
            w = QCheckBox(label)
            w.setChecked(getattr(self.s, key))
            form.addRow(w)
            self.fields[key] = w

        def text(form, key, label):
            w = QLineEdit(str(getattr(self.s, key)))
            form.addRow(label, w)
            self.fields[key] = w

        f = group("Hand tracking")
        combo(f, "source", "Input", list(SOURCES.items()))
        combo(f, "hand", "I fingerspell with my", [("right", "Right hand"), ("left", "Left hand")])
        ispin(f, "camera_index", "Webcam number", 0, 9)

        f = group("VRChat")
        text(f, "osc_host", "VRChat address")
        ispin(f, "osc_port", "VRChat OSC port", 1, 65535)
        check(f, "live_captions", "Show letters in the chatbox while I spell")
        dspin(f, "live_interval_s", "Update the chatbox at most every", 1.0, 5.0, 0.1, " s")
        check(f, "sound_on_send", "Play the chatbox sound when a message is sent")
        check(f, "osc_toggle_enabled", "Let an avatar toggle turn signing on/off")
        text(f, "osc_toggle_param", "Avatar parameter name (Bool)")
        ispin(f, "osc_listen_port", "Listen on port", 1, 65535)

        f = group("Typing")
        dspin(f, "hold_s", "Hold a sign still for", 0.1, 1.5, 0.05, " s")
        dspin(f, "word_gap_s", "Add a space after keeping still for (0 = never)", 0.0, 5.0, 0.1, " s")
        dspin(f, "drop_send_s", "Send when my hand is lowered for (0 = never)", 0.0, 5.0, 0.1, " s")
        dspin(f, "idle_send_s", "Send after no new letter for (0 = never)", 0.0, 30.0, 0.5, " s")
        dspin(f, "drop_height_m", "Hand counts as lowered this far below my eyes", 0.2, 1.2, 0.05, " m")
        combo(f, "case_mode", "Capital letters", [("sentence", "Sentence case"), ("upper", "ALL CAPS"),
                                                  ("lower", "all lowercase")])

        f = group("Recognition")
        dspin(f, "reject_ratio", "Tolerance (lower = stricter, fewer wrong letters)", 1.2, 6.0, 0.1)
        dspin(f, "min_conf", "Minimum agreement", 0.3, 0.95, 0.05)
        dspin(f, "motion_tolerance", "J/Z tolerance (lower = stricter)", 0.1, 0.6, 0.02)
        ispin(f, "knn_k", "Examples compared per frame", 1, 25)

        f = group("Recording")
        dspin(f, "countdown_s", "Countdown", 1.0, 6.0, 0.5, " s")
        dspin(f, "static_record_s", "Record a still sign for", 1.0, 5.0, 0.5, " s")
        dspin(f, "motion_record_s", "Record a moving sign for", 1.5, 5.0, 0.5, " s")
        ispin(f, "takes_static", "Recordings per letter", 1, 6)
        ispin(f, "takes_motion", "Recordings per J / Z", 1, 8)

        f = group("In VR (SteamVR only)")
        check(f, "hud_enabled", "Show a status panel in VR")
        dspin(f, "hud_distance_m", "Panel distance", 0.3, 1.5, 0.05, " m")
        dspin(f, "hud_drop_m", "Panel below eye level", 0.0, 0.6, 0.02, " m")
        check(f, "haptics", "Buzz the controllers during recording")

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def result_settings(self) -> Settings:
        s = dataclasses.replace(self.s)
        for key, w in self.fields.items():
            if isinstance(w, QComboBox):
                setattr(s, key, w.currentData())
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                setattr(s, key, type(getattr(s, key))(w.value()))
            elif isinstance(w, QCheckBox):
                setattr(s, key, w.isChecked())
            elif isinstance(w, QLineEdit):
                setattr(s, key, w.text().strip() or getattr(self.s, key))
        return s


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings | None = None):
        super().__init__()
        self.settings = settings or Settings.load()
        self.events: queue.Queue = queue.Queue()
        self.profile = Profile.load_or_create(self.settings.profile_name)
        self.sender = self._make_sender()
        self.engine = Engine(self.settings, self.profile, self.sender, self.events.put)
        self.source = None
        self.listener: OscToggleListener | None = None
        self.preview_image: QImage | None = None
        self.selected = "A"
        self.counts: dict[str, int] = {}
        self.model_labels: list[str] = []

        self.setWindowTitle(f"{APP_NAME} – ASL fingerspelling to the VRChat chatbox")
        self.setWindowIcon(make_icon())
        self.resize(1180, 760)
        self._build()
        self.engine.set_source_kind(self.settings.source)
        self._start_source()
        self._start_listener()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._drain)
        self.timer.start(33)

    # ------------------------------------------------------------ plumbing
    def _make_sender(self) -> ChatboxSender:
        s = self.settings
        return ChatboxSender(s.osc_host, s.osc_port, s.live_interval_s, s.sound_on_send)

    def _start_source(self) -> None:
        self.source = create_source(self.settings.source, self.settings, self.engine.on_update, self._on_status)
        self.source.hud_provider = self.engine.hud_state
        if hasattr(self.source, "preview_cb"):
            self.source.preview_cb = self._on_preview
        self.source.start()
        self.sim_box.setVisible(self.settings.source == "sim")
        self.preview.setVisible(self.settings.source == "webcam")

    def _stop_source(self) -> None:
        if self.source is not None:
            self.source.stop()
            self.source = None

    def _start_listener(self) -> None:
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
        if not self.settings.osc_toggle_enabled:
            return
        self.listener = OscToggleListener(self.settings.osc_listen_port, self.settings.osc_toggle_param,
                                          lambda on: self.events.put({"type": "toggle", "on": on}))
        try:
            self.listener.start()
        except OSError as exc:
            self.listener = None
            self.events.put({"type": "notice", "text": f"Couldn't listen on port {self.settings.osc_listen_port} "
                                                       f"({exc.strerror or exc}). Another OSC app may be using it."})

    def _on_status(self, state: str, message: str) -> None:
        log.info("input %s: %s", state, message)
        self.events.put({"type": "status", "state": state, "message": message})

    def _on_preview(self, rgb) -> None:
        h, w = rgb.shape[:2]
        self.preview_image = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format_RGB888).copy()

    # ------------------------------------------------------------- layout
    def _panel(self) -> tuple[QFrame, QVBoxLayout]:
        fr = QFrame()
        fr.setObjectName("panel")
        lay = QVBoxLayout(fr)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(10)
        return fr, lay

    def _build(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(16, 12, 16, 16)
        outer.setSpacing(12)

        top = QHBoxLayout()
        title = QLabel(APP_NAME)
        title.setObjectName("h1")
        top.addWidget(title)
        self.status_chip = QLabel("Starting…")
        self.status_chip.setObjectName("chip")
        top.addWidget(self.status_chip)
        top.addStretch(1)
        self.vrc_chip = QLabel()
        self.vrc_chip.setObjectName("chip")
        top.addWidget(self.vrc_chip)
        more = QToolButton()
        more.setText("Profile ▾")
        more.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(more)
        for text, fn in (("Export recordings…", self._export), ("Import recordings…", self._import),
                         ("Delete all recordings for this input", self._reset), (None, None),
                         ("Open SignChat data folder", self._open_folder)):
            if text is None:
                menu.addSeparator()
                continue
            act = QAction(text, menu)
            act.triggered.connect(fn)
            menu.addAction(act)
        more.setMenu(menu)
        top.addWidget(more)
        gear = QPushButton("Settings")
        gear.clicked.connect(self._open_settings)
        top.addWidget(gear)
        outer.addLayout(top)

        cols = QHBoxLayout()
        cols.setSpacing(14)
        outer.addLayout(cols, 1)

        # ---- live panel
        live, lay = self._panel()
        cols.addWidget(live, 5)
        h = QLabel("Live")
        h.setObjectName("h2")
        lay.addWidget(h)
        self.start_btn = QPushButton("Start signing")
        self.start_btn.setObjectName("start")
        self.start_btn.clicked.connect(self._toggle_signing)
        lay.addWidget(self.start_btn)
        row = QHBoxLayout()
        self.ring = LetterRing()
        row.addWidget(self.ring)
        info = QVBoxLayout()
        self.pred_label = QLabel("Hold a handshape still to type it.")
        self.pred_label.setWordWrap(True)
        info.addWidget(self.pred_label)
        self.hand_label = QLabel("")
        self.hand_label.setObjectName("muted")
        self.hand_label.setWordWrap(True)
        info.addWidget(self.hand_label)
        info.addStretch(1)
        row.addLayout(info, 1)
        lay.addLayout(row)
        self.message = QLabel("")
        self.message.setObjectName("message")
        self.message.setWordWrap(True)
        self.message.setMinimumHeight(120)
        self.message.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        lay.addWidget(self.message)
        self.count_label = QLabel("0 / 144")
        self.count_label.setObjectName("muted")
        self.count_label.setAlignment(Qt.AlignRight)
        lay.addWidget(self.count_label)
        btns = QHBoxLayout()
        for text, fn in (("Send now", self.engine.send_now), ("Delete letter", self.engine.backspace),
                         ("Clear", self.engine.clear)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            btns.addWidget(b)
        lay.addLayout(btns)
        self.sim_box = QLineEdit()
        self.sim_box.setPlaceholderText("Simulated hand: type letters here (space = pause, Enter = lower hand)")
        self.sim_box.textEdited.connect(self._sim_typed)
        self.sim_box.returnPressed.connect(lambda: self._sim_press("\n"))
        lay.addWidget(self.sim_box)
        self.preview = QLabel()
        self.preview.setFixedHeight(180)
        self.preview.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.preview)
        sent_h = QLabel("Sent to VRChat")
        sent_h.setObjectName("muted")
        lay.addWidget(sent_h)
        self.sent_list = QListWidget()
        lay.addWidget(self.sent_list, 1)

        # ---- teach panel
        teach, lay = self._panel()
        cols.addWidget(teach, 6)
        h = QLabel("Teach SignChat your handshapes")
        h.setObjectName("h2")
        lay.addWidget(h)
        sub = QLabel("Record each letter twice (J and Z three times, with their movement). It takes a few "
                     "minutes and adapts to your hand, your hardware and your signing style. Use the "
                     "handshapes you normally use in VR.")
        sub.setObjectName("muted")
        sub.setWordWrap(True)
        lay.addWidget(sub)
        row = QHBoxLayout()
        self.guided_btn = QPushButton("Guided setup")
        self.guided_btn.setObjectName("primary")
        self.guided_btn.clicked.connect(self._guided)
        row.addWidget(self.guided_btn)
        self.eval_btn = QPushButton("Check accuracy")
        self.eval_btn.clicked.connect(self._evaluate)
        row.addWidget(self.eval_btn)
        row.addStretch(1)
        lay.addLayout(row)

        self.banner = QFrame()
        self.banner.setStyleSheet(f"QFrame {{ background: {PANEL_2}; border-radius: 12px; }}")
        bl = QHBoxLayout(self.banner)
        self.banner_big = QLabel("A")
        self.banner_big.setStyleSheet("font-size: 34pt; font-weight: 700; min-width: 90px;")
        self.banner_big.setAlignment(Qt.AlignCenter)
        bl.addWidget(self.banner_big)
        bv = QVBoxLayout()
        self.banner_title = QLabel("")
        self.banner_title.setStyleSheet("font-size: 13pt; font-weight: 600;")
        self.banner_text = QLabel("")
        self.banner_text.setWordWrap(True)
        bv.addWidget(self.banner_title)
        bv.addWidget(self.banner_text)
        bl.addLayout(bv, 1)
        self.banner_stop = QPushButton("Stop")
        self.banner_stop.clicked.connect(self.engine.cancel_recording)
        bl.addWidget(self.banner_stop)
        self.banner.setVisible(False)
        lay.addWidget(self.banner)

        grid = QGridLayout()
        grid.setSpacing(8)
        self.tiles: dict[str, QPushButton] = {}
        labels = list(LETTERS) + ["_rest"]
        for i, label in enumerate(labels):
            b = self._tile(label)
            grid.addWidget(b, i // 7, i % 7)
        lay.addLayout(grid)
        opt = QLabel("Optional signs (pick any handshape you don't use for letters):")
        opt.setObjectName("muted")
        lay.addWidget(opt)
        orow = QHBoxLayout()
        for label in ("_space", "_backspace", "_send", "_clear"):
            orow.addWidget(self._tile(label))
        orow.addStretch(1)
        lay.addLayout(orow)

        sel = QHBoxLayout()
        self.sel_label = QLabel("")
        sel.addWidget(self.sel_label, 1)
        self.rec_btn = QPushButton("Record")
        self.rec_btn.clicked.connect(lambda: self._record(self.selected))
        sel.addWidget(self.rec_btn)
        self.del_btn = QPushButton("Delete recordings")
        self.del_btn.clicked.connect(self._delete_selected)
        sel.addWidget(self.del_btn)
        lay.addLayout(sel)
        self.report = QLabel("")
        self.report.setWordWrap(True)
        self.report.setTextFormat(Qt.RichText)
        lay.addWidget(self.report)
        lay.addStretch(1)
        self._refresh_tiles()
        self._update_vrc_chip()

    def _tile(self, label: str) -> QPushButton:
        b = QPushButton(tile_text(label))
        b.setObjectName("tile")
        b.setCheckable(True)
        b.setToolTip(display_name(label) + (" (moving letter)" if label in MOTION_LETTERS else ""))
        b.clicked.connect(lambda _=False, l=label: self._select(l))
        self.tiles[label] = b
        return b

    # ---------------------------------------------------------- actions
    def _select(self, label: str) -> None:
        self.selected = label
        self._refresh_tiles()

    def _record(self, label: str) -> None:
        if not self._ready_to_record():
            return
        self.engine.record(label)

    def _ready_to_record(self) -> bool:
        if self.engine.enabled:
            self._toggle_signing()
        return True

    def _guided(self) -> None:
        if not self._ready_to_record():
            return
        n = self.engine.guided()
        if n == 0:
            QMessageBox.information(self, APP_NAME, "Every sign already has recordings. Select a tile and press "
                                                    "Record to add more, or Check accuracy to see what to improve.")

    def _delete_selected(self) -> None:
        label = self.selected
        n = self.counts.get(label, 0)
        if not n:
            return
        if QMessageBox.question(self, APP_NAME, f"Delete all {n} recordings of {display_name(label)}?") \
                != QMessageBox.Yes:
            return
        self.engine.delete_label(label)

    def _toggle_signing(self) -> None:
        on = not self.engine.enabled
        if on and not any(l for l in self.model_labels if not l.startswith("_")):
            QMessageBox.information(self, APP_NAME, "Teach SignChat your handshapes first: press Guided setup.")
            return
        if on and self.engine.recording:
            self.engine.cancel_recording()
        self.engine.set_enabled(on)

    def _evaluate(self) -> None:
        self.report.setText("Checking…")
        threading.Thread(target=lambda: self.events.put({"type": "report", "data": self.engine.evaluate()}),
                         daemon=True).start()

    def _sim_typed(self, text: str) -> None:
        for ch in text:
            if ch.isalpha() or ch == " ":
                self._sim_press(ch.upper())
        self.sim_box.clear()

    def _sim_press(self, key: str) -> None:
        if self.source is not None and hasattr(self.source, "press"):
            self.source.press(key)

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self.settings, self)
        if dlg.exec() != QDialog.Accepted:
            return
        new = dlg.result_settings()
        old = self.settings
        self.settings = new
        new.save()
        restart_source = (new.source, new.hand, new.camera_index, new.hud_enabled, new.hud_distance_m,
                          new.hud_drop_m) != (old.source, old.hand, old.camera_index, old.hud_enabled,
                                              old.hud_distance_m, old.hud_drop_m)
        if (new.osc_host, new.osc_port) != (old.osc_host, old.osc_port):
            self.sender = self._make_sender()
            self.engine.set_sender(self.sender)
        self.engine.apply_settings(new)
        if restart_source:
            self._stop_source()
            self.engine.set_enabled(False)
            self.engine.cancel_recording()
            self.engine.set_source_kind(new.source)
            self._start_source()
        if (new.osc_toggle_enabled, new.osc_listen_port, new.osc_toggle_param) != \
                (old.osc_toggle_enabled, old.osc_listen_port, old.osc_toggle_param):
            self._start_listener()
        self._update_vrc_chip()

    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export recordings", f"{self.profile.name}.npz",
                                              "SignChat recordings (*.npz)")
        if path:
            self.profile.save(path)

    def _import(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import recordings", "", "SignChat recordings (*.npz)")
        if not path:
            return
        try:
            other = Profile.load(path)
        except Exception as exc:
            QMessageBox.warning(self, APP_NAME, f"Couldn't read that file: {exc}")
            return
        added = self.engine.import_takes(other.takes)
        QMessageBox.information(self, APP_NAME, f"Added {added} recordings.")

    def _reset(self) -> None:
        if QMessageBox.question(self, APP_NAME, "Delete every recording for this input type?") != QMessageBox.Yes:
            return
        self.engine.delete_all()

    def _open_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(app_dir())))

    # ----------------------------------------------------------- updates
    def _update_vrc_chip(self) -> None:
        s = self.settings
        self.vrc_chip.setText(f"VRChat OSC → {s.osc_host}:{s.osc_port}")
        self.vrc_chip.setToolTip("In VRChat: Action Menu → Options → OSC → Enabled")

    def _refresh_tiles(self) -> None:
        for label, b in self.tiles.items():
            n = self.counts.get(label, 0)
            want = (self.settings.takes_motion if label in MOTION_LETTERS
                    else 1 if label == "_rest" else self.settings.takes_static)
            state = "empty" if n == 0 else "done" if n >= want else "partial"
            b.setProperty("state", state)
            b.setChecked(label == self.selected)
            b.style().unpolish(b)
            b.style().polish(b)
        n = self.counts.get(self.selected, 0)
        hint = ""
        if self.selected in MOTION_LETTERS:
            hint = " · do the movement once after GO, then stop"
        elif self.selected == "_rest":
            hint = " · relax and move your hand naturally (helps ignore non-signs)"
        self.sel_label.setText(f"<b>{display_name(self.selected)}</b> — {n} recording{'s' if n != 1 else ''}{hint}")
        self.del_btn.setEnabled(n > 0)

    def _drain(self) -> None:
        latest_pred = None
        for _ in range(500):
            try:
                ev = self.events.get_nowait()
            except queue.Empty:
                break
            t = ev["type"]
            if t == "pred":
                latest_pred = ev
            elif t == "text":
                self.message.setText(ev["text"])
                self.count_label.setText(f"{len(ev['text'])} / 144")
            elif t == "commit":
                self.ring.flash = 4
            elif t == "sent" and ev.get("final"):
                self.sent_list.insertItem(0, ev["text"])
            elif t == "enabled":
                self.start_btn.setText("Stop signing" if ev["on"] else "Start signing")
                self.start_btn.setProperty("on", "true" if ev["on"] else "false")
                self.start_btn.style().unpolish(self.start_btn)
                self.start_btn.style().polish(self.start_btn)
            elif t == "model":
                self.counts = ev["counts"]
                self.model_labels = ev["labels"]
                self._refresh_tiles()
            elif t == "take":
                self.counts = ev["counts"]
                self._refresh_tiles()
                color = GREEN if ev["ok"] else RED
                self.report.setText(f"<span style='color:{color}'>{display_name(ev['label'])}: {ev['message']}</span>")
            elif t == "rec":
                self._show_rec(ev)
            elif t == "haptic":
                if self.source is not None and self.source.has_haptics:
                    self.source.haptic(ev["pattern"])
            elif t == "status":
                color = {"ok": GREEN, "waiting": AMBER, "error": RED}.get(ev["state"], MUTED)
                self.status_chip.setText(f"<span style='color:{color}'>●</span> {ev['message']}")
                self.status_chip.setToolTip(ev["message"])
            elif t == "toggle":
                if ev["on"] != self.engine.enabled:
                    self._toggle_signing()
            elif t == "report":
                self._show_report(ev["data"])
            elif t == "notice":
                self.report.setText(f"<span style='color:{AMBER}'>{ev['text']}</span>")
        if latest_pred is not None:
            self._show_pred(latest_pred)
        if self.ring.flash:
            self.ring.flash -= 1
            self.ring.update()
        if self.preview_image is not None and self.preview.isVisible():
            self.preview.setPixmap(QPixmap.fromImage(self.preview_image))

    def _show_pred(self, ev: dict) -> None:
        if not ev.get("tracked"):
            self.ring.set_state("–", 0, MUTED)
            mine = "right" if self.settings.hand == "right" else "left"
            other = "left" if mine == "right" else "right"
            self.pred_label.setText(f"No {mine} hand detected.")
            self.hand_label.setText(f"I can see your {other} hand. If you fingerspell with it, change "
                                    f"'I fingerspell with my' in Settings." if ev.get("other_hand") else "")
            return
        label = ev.get("label")
        enabled = self.engine.enabled
        if not self.model_labels:
            self.ring.set_state("·", 0, MUTED)
            self.pred_label.setText("No recordings yet. Press Guided setup to teach SignChat your handshapes.")
        elif label is None or label.startswith("_"):
            shown = "·" if label in (None, "_rest") else tile_text(label)
            self.ring.set_state(shown, ev.get("hold", 0), MUTED)
            raw = ev.get("raw")
            guess = f" (closest: {tile_text(raw)})" if raw and raw != "_rest" and label is None else ""
            self.pred_label.setText(("Not a sign I know" + guess) if label is None else "Resting")
        else:
            self.ring.set_state(label, ev.get("hold", 0), GREEN if enabled else BLUE)
            self.pred_label.setText("Hold still to type it." if enabled else "Press Start signing to type.")
        q = ev.get("quality") or ""
        parts = [f"tracking: {q}"] if q else []
        if not ev.get("active", True):
            parts.append("hand lowered")
        self.hand_label.setText(" · ".join(parts))

    def _show_rec(self, ev: dict) -> None:
        phase = ev.get("phase")
        if not phase:
            q = ev.get("queue", 0)
            self.banner.setVisible(q > 0)
            if q:
                self.banner_big.setText("…")
                self.banner_title.setText(f"{q} recording{'s' if q != 1 else ''} to go")
                self.banner_text.setText("Waiting for hand tracking. Make sure your hand is tracked."
                                         if ev.get("waiting") else "Next sign coming up…")
            return
        self.banner.setVisible(True)
        label = ev["label"]
        self.banner_big.setText(tile_text(label))
        take = f"{ev['take_no']}/{ev['takes_total']}"
        left = ev.get("queue", 0)
        tail = f" · {left} more after this" if left else ""
        if phase == "prompt":
            self.banner_title.setText(f"Next: {display_name(label)}  ({take})")
            self.banner_text.setText("Get your hand ready." + tail)
        elif phase == "countdown":
            self.banner_title.setText(f"Get ready: {display_name(label)}  ({take}) · {int(ev['remaining']) + 1}")
            self.banner_text.setText(("Relax your hand." if label == "_rest" else "Form the handshape now.") + tail)
        else:
            if ev.get("kind") == "motion":
                msg = f"Do the {label} movement once now, then stop."
            elif label == "_rest":
                msg = "Relax and move your hand around naturally."
            else:
                msg = "Hold it still…"
            self.banner_title.setText(f"Recording {display_name(label)}  ({take}) · {int(ev['progress'] * 100)}%")
            self.banner_text.setText(msg + tail)

    def _show_report(self, data: dict) -> None:
        rep = data.get("static")
        if rep is None:
            self.report.setText("Record some signs first.")
            return
        lines = [f"<b>Accuracy: {rep.overall * 100:.0f}%</b> (each recording tested against the others)"]
        worst = [w for w in rep.worst(6) if w[1] < 0.9]
        if worst:
            items = []
            for label, acc, conf in worst:
                c = f" → mistaken for <b>{tile_text(conf)}</b>" if conf else " → often not recognised"
                items.append(f"{tile_text(label)} {acc * 100:.0f}%{c}")
            lines.append("Needs work: " + "; ".join(items))
            lines.append("<span style='color:%s'>Tip: re-record those, or make the two handshapes more different "
                         "(VR controllers can't show every finger detail).</span>" % MUTED)
        else:
            lines.append("No letters are getting mixed up. 🎉")
        untestable = [l for l in rep.untestable if l != "_rest"]
        if untestable:
            lines.append("Need a 2nd recording to test: " + ", ".join(tile_text(l) for l in untestable))
        motion = data.get("motion") or {}
        if motion:
            lines.append("Moving letters: " + ", ".join(f"{k} {h}/{n}" for k, (h, n) in sorted(motion.items())))
        self.report.setText("<br>".join(lines))

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self._stop_source()
        if self.listener is not None:
            self.listener.stop()
        try:
            self.settings.save()
        except OSError:
            pass
        super().closeEvent(event)


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    for role, color in ((QPalette.Window, BG), (QPalette.Base, PANEL_2), (QPalette.Text, TEXT),
                        (QPalette.WindowText, TEXT), (QPalette.Button, PANEL_2), (QPalette.ButtonText, TEXT),
                        (QPalette.Highlight, "#33405a"), (QPalette.HighlightedText, TEXT),
                        (QPalette.ToolTipBase, PANEL), (QPalette.ToolTipText, TEXT)):
        pal.setColor(role, QColor(color))
    app.setPalette(pal)
    app.setStyleSheet(STYLE)


def main() -> None:
    log_path = setup_logging()
    log.info("%s %s starting (log: %s)", APP_NAME, __version__, log_path)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    apply_theme(app)
    win = MainWindow()
    win.show()
    smoke = os.environ.get("SIGNCHAT_SMOKE_EXIT_MS")  # used by the build check
    if smoke:
        QTimer.singleShot(int(smoke), app.quit)
    code = app.exec()
    win.close()
    log.info("%s exiting", APP_NAME)
    sys.exit(code)


if __name__ == "__main__":
    main()
