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
    QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QListView, QListWidget, QListWidgetItem, QMainWindow, QMenu, QMessageBox, QPushButton,
    QRadioButton, QScrollArea, QSizePolicy, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget, QToolButton,
    QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from . import APP_NAME, __version__
from . import packs as packs_mod
from .config import Settings
from .engine import LETTERS, MOTION_LETTERS, Engine, display_name
from .osc_in import OscToggleListener
from .osc_out import ChatboxSender, ttsvw_sink
from .paths import app_dir, logs_dir
from .profile import Profile
from .sources import SOURCES, create_source
from .vocab import CATEGORIES, DICTIONARIES, STARTER_BY_GLOSS, STARTER_VOCAB, default_text, normalize_gloss, \
    validate_gloss

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
QPushButton#tile[state="pack"] {{ border: 2px dashed {BLUE}; }}
QPushButton#tile:checked {{ background: #33405a; }}
QListWidget {{ background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 10px; padding: 6px; }}
QListWidget::item {{ padding: 4px 2px; }}
QListWidget#recent {{ padding: 4px; }}
QListWidget#recent::item {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px; padding: 4px 9px;
    margin: 2px; }}
QListWidget#recent::item:hover {{ border-color: {AMBER}; }}
QTableWidget {{ background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 10px; gridline-color: {BORDER}; }}
QTableWidget::item:selected {{ background: #33405a; }}
QHeaderView::section {{ background: {PANEL}; color: {MUTED}; border: none; border-bottom: 1px solid {BORDER};
    padding: 5px; }}
QTreeWidget {{ background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 10px; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 8px; padding: 6px 14px;
    margin-right: 6px; }}
QTabBar::tab:selected {{ background: #33405a; border-color: #4a5266; }}
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

        def note(form, text_):
            lbl = QLabel(text_)
            lbl.setObjectName("muted")
            lbl.setWordWrap(True)
            lbl.setOpenExternalLinks(True)
            form.addRow(lbl)

        f = group("Hand tracking")
        combo(f, "source", "Input", list(SOURCES.items()))
        combo(f, "hand", "My dominant hand (fingerspelling, one-handed signs)",
              [("right", "Right hand"), ("left", "Left hand")])
        ispin(f, "camera_index", "Webcam number", 0, 9)

        f = group("VRChat")
        text(f, "osc_host", "VRChat address")
        ispin(f, "osc_port", "VRChat OSC port", 1, 65535)
        check(f, "live_captions", "Show text in the chatbox while I sign")
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

        f = group("Words && phrases")
        check(f, "signs_enabled", "Recognise whole words and phrases")
        dspin(f, "sign_tolerance", "Tolerance (lower = stricter, fewer wrong words)", 0.6, 1.6, 0.05)
        dspin(f, "sign_record_s", "Record a word for", 2.0, 6.0, 0.5, " s")
        dspin(f, "phrase_record_s", "Record a phrase for", 3.0, 10.0, 0.5, " s")
        ispin(f, "takes_sign", "Recordings per word or phrase", 1, 8)

        f = group("TTS Voice Wizard")
        check(f, "ttsvw_enabled", "Send finished messages to TTS Voice Wizard instead of the chatbox")
        combo(f, "ttsvw_speak", "TTS Voice Wizard should", [(True, "Speak them (and show them, per its settings)"),
                                                            (False, "Only show them (no voice)")])
        text(f, "ttsvw_host", "TTS Voice Wizard address")
        ispin(f, "ttsvw_port", "TTS Voice Wizard OSC listener port", 1, 65535)
        note(f, "In TTS Voice Wizard turn on its OSC listener (port 4026 by default). Its own settings then "
                "decide the voice, translation and what goes to the VRChat chatbox, so the two apps don't "
                "overwrite each other. You may want to turn off <i>Show text in the chatbox while I sign</i> "
                "above, so SignChat's live text doesn't replace its messages.")

        f = group("Template packs")
        text(f, "pack_index_url", "Community pack list (https)")

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


def run_in_background(owner, fn, done) -> None:
    """Run fn() in a thread and call done(result, error) on the UI thread."""
    box: dict = {}

    def work():
        try:
            box["result"] = fn()
        except Exception as exc:  # reported to the user by `done`
            box["error"] = exc

    th = threading.Thread(target=work, daemon=True)
    th.start()
    timer = QTimer(owner)
    timer.setInterval(100)

    def poll():
        if th.is_alive():
            return
        timer.stop()
        timer.deleteLater()
        done(box.get("result"), box.get("error"))

    timer.timeout.connect(poll)
    timer.start()


def dictionary_links() -> str:
    return " · ".join(f"<a href='{url}' style='color:{BLUE}'>{name}</a>" for name, url in DICTIONARIES)


class SignDialog(QDialog):
    """Add a word or phrase, or edit one."""

    def __init__(self, parent=None, entry=None, sensitivity: float = 1.0, existing: set | None = None):
        super().__init__(parent)
        self.editing = entry is not None
        self.existing = existing or set()
        self.setWindowTitle(f"Edit {entry.gloss}" if self.editing else "Add a word or phrase")
        self.setMinimumWidth(520)
        lay = QVBoxLayout(self)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self.gloss = QLineEdit(entry.gloss if entry else "")
        self.gloss.setPlaceholderText("e.g. THANK-YOU or YOUR-NAME-WHAT")
        self.gloss.setEnabled(not self.editing)
        form.addRow("Sign (gloss)", self.gloss)
        self.text = QLineEdit(entry.text if entry else "")
        self.text.setPlaceholderText("what it types, e.g. what's your name?")
        self.text.setMaxLength(100)
        form.addRow("Types", self.text)
        self.phrase = QCheckBox("It's a phrase: several signs in a row (gets a longer recording)")
        self.phrase.setChecked(bool(entry and entry.kind == "phrase"))
        form.addRow(self.phrase)
        self.hands = QComboBox()
        for value, label in (("auto", "Work it out from my recordings"), ("one", "One hand"), ("two", "Both hands")):
            self.hands.addItem(label, value)
        self.hands.setCurrentIndex(max(0, self.hands.findData(entry.hands if entry else "auto")))
        form.addRow("Hands", self.hands)
        self.sens = None
        if self.editing:
            self.sens = QDoubleSpinBox()
            self.sens.setRange(0.4, 1.6)
            self.sens.setSingleStep(0.05)
            self.sens.setValue(sensitivity)
            form.addRow("Sensitivity (lower = must be signed more exactly)", self.sens)
        lay.addLayout(form)
        hint = QLabel("Write the gloss the usual way: capital letters, hyphens between the signs of a phrase. "
                      "It's only a name; what gets typed is the text below it. Look signs up in an ASL "
                      "dictionary: " + dictionary_links())
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        hint.setOpenExternalLinks(True)
        lay.addWidget(hint)
        self.error = QLabel("")
        self.error.setStyleSheet(f"color: {RED};")
        lay.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._ok)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self._text_touched = self.editing
        self.text.textEdited.connect(lambda _: setattr(self, "_text_touched", True))
        self.gloss.textEdited.connect(self._gloss_changed)

    def _gloss_changed(self, value: str) -> None:
        g = normalize_gloss(value)
        known = STARTER_BY_GLOSS.get(g)
        if not self._text_touched:
            self.text.setText(known.text if known else default_text(g))
        if known is not None:
            self.phrase.setChecked(known.kind == "phrase")
        self.error.setText("")

    def _ok(self) -> None:
        g = normalize_gloss(self.gloss.text())
        err = validate_gloss(g)
        if not err and not self.editing and g in self.existing:
            err = f"{g} is already in your list."
        if err:
            self.error.setText(err)
            return
        self.accept()

    def values(self) -> dict:
        g = normalize_gloss(self.gloss.text())
        return {"gloss": g, "text": self.text.text().strip() or default_text(g),
                "kind": "phrase" if self.phrase.isChecked() else "word", "hands": self.hands.currentData(),
                "sensitivity": self.sens.value() if self.sens is not None else None}


class CommonSignsDialog(QDialog):
    def __init__(self, parent=None, existing: set | None = None):
        super().__init__(parent)
        existing = existing or set()
        self.setWindowTitle("Add common signs")
        self.resize(560, 640)
        lay = QVBoxLayout(self)
        intro = QLabel("Pick the words and phrases you want SignChat to learn. You'll record each one yourself, "
                       "the way you sign it. SignChat doesn't teach signs; look them up in an ASL dictionary "
                       f"({dictionary_links()}) or, better, learn them from Deaf signers.")
        intro.setWordWrap(True)
        intro.setOpenExternalLinks(True)
        lay.addWidget(intro)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Sign", "Types"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.items: list[QTreeWidgetItem] = []
        for cat in CATEGORIES:
            parent_item = QTreeWidgetItem([cat])
            parent_item.setFlags(parent_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
            parent_item.setCheckState(0, Qt.Unchecked)
            self.tree.addTopLevelItem(parent_item)
            for e in STARTER_VOCAB:
                if e.category != cat:
                    continue
                it = QTreeWidgetItem([e.gloss, e.text])
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(0, Qt.Checked if e.gloss in existing else Qt.Unchecked)
                if e.gloss in existing:
                    it.setDisabled(True)
                    it.setToolTip(0, "Already in your list")
                parent_item.addChild(it)
                self.items.append(it)
            parent_item.setExpanded(True)
        lay.addWidget(self.tree, 1)
        self.existing = existing
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Add selected")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def selected(self) -> list[str]:
        return [it.text(0) for it in self.items
                if it.checkState(0) == Qt.Checked and it.text(0) not in self.existing]


class FixDialog(QDialog):
    """"That's not what I signed": correct a letter or sign, and learn from it."""

    def __init__(self, parent, token: dict, sign_choices: list[str]):
        super().__init__(parent)
        self.setWindowTitle("Fix a mistake")
        self.setMinimumWidth(520)
        lay = QVBoxLayout(self)
        what = f"the letter <b>{token['label']}</b>" if token["kind"] == "letter" else \
            f"<b>{token['text']}</b> (the sign {token['label']})"
        head = QLabel(f"SignChat typed {what}. What did you really sign?")
        head.setWordWrap(True)
        lay.addWidget(head)
        self.group = QButtonGroup(self)
        self.nothing = QRadioButton("Nothing, I wasn't signing that")
        self.group.addButton(self.nothing)
        lay.addWidget(self.nothing)
        row = QHBoxLayout()
        self.letter = QRadioButton("The letter")
        self.group.addButton(self.letter)
        self.letter_box = QComboBox()
        self.letter_box.addItems(list(LETTERS))
        row.addWidget(self.letter)
        row.addWidget(self.letter_box)
        row.addStretch(1)
        lay.addLayout(row)
        row = QHBoxLayout()
        self.sign = QRadioButton("The word/phrase")
        self.group.addButton(self.sign)
        self.sign_box = QComboBox()
        self.sign_box.addItems(sign_choices)
        self.sign.setEnabled(bool(sign_choices))
        row.addWidget(self.sign)
        row.addWidget(self.sign_box, 1)
        lay.addLayout(row)
        self.letter_box.activated.connect(lambda _: self.letter.setChecked(True))
        self.sign_box.activated.connect(lambda _: self.sign.setChecked(True))
        self.nothing.setChecked(True)
        note = QLabel("SignChat learns from this: a letter or word you correct is saved as a new recording, and a "
                      "word that was typed by mistake must be signed more exactly from now on.")
        note.setObjectName("muted")
        note.setWordWrap(True)
        lay.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def choice(self) -> str | None:
        if self.letter.isChecked():
            return self.letter_box.currentText()
        if self.sign.isChecked():
            return self.sign_box.currentText()
        return None


TRACKING_CHOICES = ("Valve Index controllers", "Quest hand tracking (Steam Link / Virtual Desktop)",
                    "Quest / Touch controllers", "Other")


class CreatePackDialog(QDialog):
    def __init__(self, parent, has_letters: bool, has_signs: bool):
        super().__init__(parent)
        self.setWindowTitle("Make a template pack")
        self.setMinimumWidth(520)
        lay = QVBoxLayout(self)
        intro = QLabel("A template pack is a copy of your recordings that other people can use until they record "
                       "their own. It only contains hand positions (no audio, video or personal data).")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        form = QFormLayout()
        self.name = QLineEdit()
        self.name.setPlaceholderText("e.g. Everyday ASL (Index)")
        form.addRow("Pack name", self.name)
        self.author = QLineEdit()
        form.addRow("Your name (optional)", self.author)
        self.desc = QLineEdit()
        form.addRow("Description (optional)", self.desc)
        self.tracking = QComboBox()
        self.tracking.setEditable(True)
        self.tracking.addItems(list(TRACKING_CHOICES))
        form.addRow("Recorded with", self.tracking)
        self.letters = QCheckBox("Include my letter recordings")
        self.letters.setChecked(has_letters)
        self.letters.setEnabled(has_letters)
        form.addRow(self.letters)
        self.signs = QCheckBox("Include my word and phrase recordings")
        self.signs.setChecked(has_signs)
        self.signs.setEnabled(has_signs)
        form.addRow(self.signs)
        lay.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Save pack…")
        buttons.accepted.connect(self._ok)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def _ok(self) -> None:
        if not self.name.text().strip():
            self.name.setFocus()
            return
        if not (self.letters.isChecked() or self.signs.isChecked()):
            return
        self.accept()


class PacksDialog(QDialog):
    """Install, choose and make template packs."""

    def __init__(self, parent, settings: Settings, profile: Profile, source_kind: str):
        super().__init__(parent)
        self.settings, self.profile, self.source_kind = settings, profile, source_kind
        self.active = list(settings.active_packs)
        self.setWindowTitle("Template packs")
        self.resize(640, 520)
        lay = QVBoxLayout(self)
        intro = QLabel("Template packs are recordings made by other signers, so you can start signing without "
                       "recording everything yourself. Your own recordings always win: as soon as you record a "
                       "letter or word, the pack's version of it isn't used any more. Packs work best with the "
                       "same kind of tracking they were recorded with.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.list = QListWidget()
        self.list.itemChanged.connect(self._toggled)
        self.list.currentRowChanged.connect(self._show_info)
        lay.addWidget(self.list, 1)
        self.info = QLabel("")
        self.info.setObjectName("muted")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        row = QHBoxLayout()
        for text, fn in (("Install from file…", self._install), ("Community packs…", self._community),
                         ("Remove", self._remove), ("Make a pack from my recordings…", self._create)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        lay.addLayout(row)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.accept)
        lay.addWidget(buttons)
        self._refresh()

    def _refresh(self) -> None:
        self.infos = packs_mod.list_installed()
        self.list.blockSignals(True)
        self.list.clear()
        for info in self.infos:
            text = f"{info.name}  ·  {info.summary()}"
            if info.tracking:
                text += f"  ·  {info.tracking}"
            it = QListWidgetItem(text)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if info.id in self.active else Qt.Unchecked)
            it.setData(Qt.UserRole, info.id)
            if info.source != self.source_kind:
                it.setToolTip(f"Recorded with {info.source}; you're using {self.source_kind}, so it isn't used.")
            self.list.addItem(it)
        self.list.blockSignals(False)
        if not self.infos:
            self.info.setText("No packs installed yet.")

    def _toggled(self, item: QListWidgetItem) -> None:
        pid = item.data(Qt.UserRole)
        if item.checkState() == Qt.Checked and pid not in self.active:
            self.active.append(pid)
        elif item.checkState() != Qt.Checked and pid in self.active:
            self.active.remove(pid)

    def _show_info(self, row: int) -> None:
        if 0 <= row < len(self.infos):
            i = self.infos[row]
            parts = [i.description] if i.description else []
            if i.author:
                parts.append(f"By {i.author}.")
            if i.quality:
                parts.append(f"Hand tracking: {i.quality}.")
            if i.letters:
                parts.append("Letters: " + "".join(l for l in i.letters if len(l) == 1))
            if i.signs:
                parts.append("Words: " + ", ".join(i.signs[:30]) + (" …" if len(i.signs) > 30 else ""))
            self.info.setText(" ".join(parts))

    def _install(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Install a template pack", "",
                                              "SignChat packs and recordings (*.npz)")
        if not path:
            return
        try:
            info = packs_mod.install_file(path)
        except (packs_mod.PackError, OSError) as exc:
            QMessageBox.warning(self, APP_NAME, str(exc))
            return
        if info.id not in self.active:
            self.active.append(info.id)
        self._refresh()

    def _remove(self) -> None:
        it = self.list.currentItem()
        if it is None:
            return
        pid = it.data(Qt.UserRole)
        if QMessageBox.question(self, APP_NAME, f"Remove the pack {it.text().split('  ·')[0]}?") != QMessageBox.Yes:
            return
        packs_mod.remove(pid)
        if pid in self.active:
            self.active.remove(pid)
        self._refresh()

    def _community(self) -> None:
        dlg = CommunityPacksDialog(self, self.settings.pack_index_url)
        dlg.exec()
        for pid in dlg.installed:
            if pid not in self.active:
                self.active.append(pid)
        self._refresh()

    def _create(self) -> None:
        mine = [t for t in self.profile.takes if t.source == self.source_kind]
        has_letters = any(t.kind != "sign" for t in mine)
        has_signs = any(t.kind == "sign" for t in mine)
        if not mine:
            QMessageBox.information(self, APP_NAME, "Record some letters or words first.")
            return
        dlg = CreatePackDialog(self, has_letters, has_signs)
        if dlg.exec() != QDialog.Accepted:
            return
        name = dlg.name.text().strip()
        default = str(Path.home() / f"{packs_mod.make_id(name)}.npz")
        path, _ = QFileDialog.getSaveFileName(self, "Save template pack", default, "SignChat pack (*.npz)")
        if not path:
            return
        labels = sorted({t.label for t in mine if (t.kind == "sign" and dlg.signs.isChecked())
                         or (t.kind != "sign" and dlg.letters.isChecked())})
        try:
            packs_mod.create_pack(self.profile, labels, name, dlg.author.text().strip(), dlg.desc.text().strip(),
                                  dlg.tracking.currentText().strip(), source=self.source_kind, dest=path)
        except (packs_mod.PackError, OSError) as exc:
            QMessageBox.warning(self, APP_NAME, str(exc))
            return
        QMessageBox.information(self, APP_NAME, f"Saved {path}.\n\nShare the file with anyone, or offer it to "
                                                "everyone by adding it to the community pack list (see packs/README "
                                                "in the SignChat repository).")


class CommunityPacksDialog(QDialog):
    def __init__(self, parent, url: str):
        super().__init__(parent)
        self.setWindowTitle("Community template packs")
        self.resize(620, 440)
        self.installed: list[str] = []
        self.entries: list = []
        lay = QVBoxLayout(self)
        self.status = QLabel("Loading the pack list…")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._show)
        lay.addWidget(self.list, 1)
        self.info = QLabel("")
        self.info.setObjectName("muted")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        row = QHBoxLayout()
        self.get_btn = QPushButton("Download and use")
        self.get_btn.setEnabled(False)
        self.get_btn.clicked.connect(self._download)
        row.addWidget(self.get_btn)
        row.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        lay.addLayout(row)
        run_in_background(self, lambda: packs_mod.fetch_index(url), self._loaded)

    def _loaded(self, entries, error) -> None:
        if error is not None:
            self.status.setText(f"Couldn't load the pack list: {error}")
            return
        self.entries = entries or []
        self.status.setText("Packs shared by the community:" if self.entries else
                            "No community packs yet. Make one from your recordings and share it!")
        for e in self.entries:
            self.list.addItem(f"{e.name}  ·  {e.contents or ''}  ·  {e.tracking or ''}".rstrip(" ·"))
        self.get_btn.setEnabled(bool(self.entries))

    def _show(self, row: int) -> None:
        if 0 <= row < len(self.entries):
            e = self.entries[row]
            size = f" ({e.size / 1e6:.1f} MB)" if e.size else ""
            self.info.setText(f"{e.description} {('By ' + e.author + '.') if e.author else ''}{size}")

    def _download(self) -> None:
        row = self.list.currentRow()
        if not 0 <= row < len(self.entries):
            return
        entry = self.entries[row]
        self.get_btn.setEnabled(False)
        self.status.setText(f"Downloading {entry.name}…")

        def done(info, error):
            self.get_btn.setEnabled(True)
            if error is not None:
                self.status.setText(f"Download failed: {error}")
                return
            self.installed.append(info.id)
            self.status.setText(f"Installed {info.name} ({info.summary()}). It's switched on.")

        run_in_background(self, lambda: packs_mod.download(entry), done)


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
        self.pack_counts: dict[str, int] = {}
        self.model_labels: list[str] = []
        self.sign_labels: list[str] = []
        self._recent_dirty = False

        self.setWindowTitle(f"{APP_NAME} – ASL to the VRChat chatbox")
        self.setWindowIcon(make_icon())
        self.resize(1240, 820)
        self._build()
        self.engine.set_source_kind(self.settings.source)
        self._load_packs()
        self._start_source()
        self._start_listener()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._drain)
        self.timer.start(33)

    # ------------------------------------------------------------ plumbing
    def _make_sender(self) -> ChatboxSender:
        s = self.settings
        sink = ttsvw_sink(s.ttsvw_host, s.ttsvw_port, s.ttsvw_speak) if s.ttsvw_enabled else None
        return ChatboxSender(s.osc_host, s.osc_port, s.live_interval_s, s.sound_on_send, final_sink=sink)

    def _load_packs(self) -> None:
        loaded = packs_mod.load_packs(self.settings.active_packs)
        missing = [p for p in self.settings.active_packs if p not in {x.info.id for x in loaded}]
        if missing:
            self.settings.active_packs = [p for p in self.settings.active_packs if p not in missing]
        self.engine.set_packs(loaded)

    def _start_source(self) -> None:
        self.source = create_source(self.settings.source, self.settings, self.engine.on_update, self._on_status)
        self.source.hud_provider = self.engine.hud_state
        if hasattr(self.source, "preview_cb"):
            self.source.preview_cb = self._on_preview
        self.source.start()
        self.sim_panel.setVisible(self.settings.source == "sim")
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

    @staticmethod
    def _muted(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("muted")
        lbl.setWordWrap(True)
        lbl.setOpenExternalLinks(True)
        return lbl

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
        packs_btn = QPushButton("Template packs")
        packs_btn.clicked.connect(self._open_packs)
        top.addWidget(packs_btn)
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
        self.pred_label = QLabel("Fingerspell (hold a handshape still) or sign words you've taught SignChat.")
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
        self.message.setMinimumHeight(110)
        self.message.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        lay.addWidget(self.message)
        self.count_label = QLabel("0 / 144")
        self.count_label.setObjectName("muted")
        self.count_label.setAlignment(Qt.AlignRight)
        lay.addWidget(self.count_label)
        btns = QHBoxLayout()
        for text, fn in (("Send now", self.engine.send_now), ("Delete last", self.engine.backspace),
                         ("Clear", self.engine.clear)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            btns.addWidget(b)
        lay.addLayout(btns)
        lay.addWidget(self._muted("Typed recently (click one to fix a mistake; SignChat learns from it):"))
        self.recent = QListWidget()
        self.recent.setObjectName("recent")
        self.recent.setFlow(QListView.LeftToRight)
        self.recent.setWrapping(True)
        self.recent.setResizeMode(QListView.Adjust)
        self.recent.setFixedHeight(76)
        self.recent.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.recent.itemClicked.connect(self._fix_clicked)
        lay.addWidget(self.recent)

        self.sim_panel = QWidget()
        sl = QVBoxLayout(self.sim_panel)
        sl.setContentsMargins(0, 0, 0, 0)
        self.sim_box = QLineEdit()
        self.sim_box.setPlaceholderText("Simulated hands: type letters here (space = pause, Enter = lower hands)")
        self.sim_box.textEdited.connect(self._sim_typed)
        self.sim_box.returnPressed.connect(lambda: self._sim_press("\n"))
        sl.addWidget(self.sim_box)
        srow = QHBoxLayout()
        self.sim_sign = QComboBox()
        self.sim_sign.setToolTip("The simulated signs are made-up movements, not real ASL.")
        srow.addWidget(self.sim_sign, 1)
        sim_go = QPushButton("Sign it (simulated)")
        sim_go.clicked.connect(self._sim_sign)
        srow.addWidget(sim_go)
        sl.addLayout(srow)
        lay.addWidget(self.sim_panel)
        self.preview = QLabel()
        self.preview.setFixedHeight(180)
        self.preview.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.preview)
        sent_h = QLabel("Sent")
        sent_h.setObjectName("muted")
        lay.addWidget(sent_h)
        self.sent_list = QListWidget()
        lay.addWidget(self.sent_list, 1)

        # ---- teach panel
        teach, lay = self._panel()
        cols.addWidget(teach, 6)
        hrow = QHBoxLayout()
        h = QLabel("Teach SignChat")
        h.setObjectName("h2")
        hrow.addWidget(h)
        hrow.addStretch(1)
        self.eval_btn = QPushButton("Check accuracy")
        self.eval_btn.clicked.connect(self._evaluate)
        hrow.addWidget(self.eval_btn)
        lay.addLayout(hrow)

        self.banner = QFrame()
        self.banner.setStyleSheet(f"QFrame {{ background: {PANEL_2}; border-radius: 12px; }}")
        bl = QHBoxLayout(self.banner)
        self.banner_big = QLabel("A")
        self.banner_big.setAlignment(Qt.AlignCenter)
        self.banner_big.setMinimumWidth(90)
        self.banner_big.setMaximumWidth(260)
        self.banner_big.setWordWrap(True)
        bl.addWidget(self.banner_big)
        bv = QVBoxLayout()
        self.banner_title = QLabel("")
        self.banner_title.setStyleSheet("font-size: 13pt; font-weight: 600;")
        self.banner_title.setWordWrap(True)
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

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_letters_tab(), "Letters")
        self.tabs.addTab(self._build_words_tab(), "Words && phrases")
        lay.addWidget(self.tabs, 1)

        self.report = QLabel("")
        self.report.setWordWrap(True)
        self.report.setTextFormat(Qt.RichText)
        lay.addWidget(self.report)
        self._refresh_tiles()
        self._refresh_words()
        self._update_vrc_chip()

    def _build_letters_tab(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 8, 0, 0)
        lay.addWidget(self._muted("Record each letter twice (J and Z three times, with their movement). It takes a "
                                  "few minutes and adapts to your hand, your hardware and your signing style. Use the "
                                  "handshapes you normally use in VR. More recordings of a letter make it more "
                                  "reliable; 'Fix a mistake' adds them while you chat."))
        row = QHBoxLayout()
        self.guided_btn = QPushButton("Guided setup")
        self.guided_btn.setObjectName("primary")
        self.guided_btn.clicked.connect(self._guided)
        row.addWidget(self.guided_btn)
        row.addStretch(1)
        lay.addLayout(row)
        grid = QGridLayout()
        grid.setSpacing(8)
        self.tiles: dict[str, QPushButton] = {}
        for i, label in enumerate(list(LETTERS) + ["_rest"]):
            grid.addWidget(self._tile(label), i // 7, i % 7)
        lay.addLayout(grid)
        lay.addWidget(self._muted("Optional signs (pick any handshape you don't use for letters):"))
        orow = QHBoxLayout()
        for label in ("_space", "_backspace", "_send", "_clear"):
            orow.addWidget(self._tile(label))
        orow.addStretch(1)
        lay.addLayout(orow)
        sel = QHBoxLayout()
        self.sel_label = QLabel("")
        self.sel_label.setWordWrap(True)
        sel.addWidget(self.sel_label, 1)
        self.rec_btn = QPushButton("Record")
        self.rec_btn.clicked.connect(lambda: self._record(self.selected))
        sel.addWidget(self.rec_btn)
        self.del_btn = QPushButton("Delete recordings")
        self.del_btn.clicked.connect(self._delete_selected)
        sel.addWidget(self.del_btn)
        lay.addLayout(sel)
        lay.addStretch(1)
        return page

    def _build_words_tab(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 8, 0, 0)
        lay.addWidget(self._muted(
            "Teach SignChat whole signs and phrases. Record each one 3 times the way you normally sign it: hands "
            "down, sign it once after GO, hands down again. A phrase you record in one go (like NICE-MEET-YOU) types "
            "its own text and replaces the separate words. Need to look a sign up? " + dictionary_links()))
        row = QHBoxLayout()
        b = QPushButton("Add common signs…")
        b.clicked.connect(self._add_common)
        row.addWidget(b)
        b = QPushButton("Add a sign…")
        b.clicked.connect(self._add_sign)
        row.addWidget(b)
        self.words_guided = QPushButton("Record missing")
        self.words_guided.setObjectName("primary")
        self.words_guided.clicked.connect(self._guided_signs)
        row.addWidget(self.words_guided)
        row.addStretch(1)
        lay.addLayout(row)
        self.words = QTableWidget(0, 4)
        self.words.setHorizontalHeaderLabels(["Sign", "Types", "Hands", "Recordings"])
        self.words.verticalHeader().setVisible(False)
        self.words.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.words.setSelectionMode(QAbstractItemView.SingleSelection)
        self.words.setEditTriggers(QAbstractItemView.NoEditTriggers)
        hdr = self.words.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.words.itemSelectionChanged.connect(self._word_selected)
        self.words.cellDoubleClicked.connect(lambda r, c: self._edit_sign())
        lay.addWidget(self.words, 1)
        self.words_empty = self._muted("No words yet. Press Add common signs… or Add a sign…, then Record missing.")
        lay.addWidget(self.words_empty)
        sel = QHBoxLayout()
        self.word_label = QLabel("")
        self.word_label.setWordWrap(True)
        sel.addWidget(self.word_label, 1)
        self.word_buttons = []
        for text, fn in (("Record", self._record_sign), ("Edit…", self._edit_sign),
                         ("Delete recordings", self._delete_sign_takes), ("Remove", self._remove_sign)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            sel.addWidget(b)
            self.word_buttons.append(b)
        lay.addLayout(sel)
        return page

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
            QMessageBox.information(self, APP_NAME, "Every letter already has recordings. Select a tile and press "
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

    # words & phrases
    def _selected_gloss(self) -> str | None:
        rows = self.words.selectionModel().selectedRows() if self.words.selectionModel() else []
        if not rows:
            return None
        item = self.words.item(rows[0].row(), 0)
        return item.data(Qt.UserRole) if item is not None else None

    def _add_common(self) -> None:
        dlg = CommonSignsDialog(self, set(self.engine.vocab))
        if dlg.exec() == QDialog.Accepted and dlg.selected():
            n = self.engine.add_common_signs(dlg.selected())
            self.report.setText(f"Added {n} sign{'s' if n != 1 else ''}. Press <b>Record missing</b> to record them.")

    def _add_sign(self) -> None:
        dlg = SignDialog(self, existing=set(self.engine.vocab))
        if dlg.exec() != QDialog.Accepted:
            return
        v = dlg.values()
        err, gloss = self.engine.add_sign(v["gloss"], v["text"], v["kind"], v["hands"])
        if err:
            QMessageBox.warning(self, APP_NAME, err)
            return
        if QMessageBox.question(self, APP_NAME, f"Record {gloss} now ({self.settings.takes_sign} times)?") \
                == QMessageBox.Yes:
            self._ready_to_record()
            self.engine.record(gloss)

    def _guided_signs(self) -> None:
        if not self.engine.vocab:
            self._add_common()
            if not self.engine.vocab:
                return
        self._ready_to_record()
        n = self.engine.guided_signs()
        if n == 0:
            QMessageBox.information(self, APP_NAME, "Every word in your list already has its recordings. Select one "
                                                    "and press Record to add more.")

    def _record_sign(self) -> None:
        g = self._selected_gloss()
        if g:
            self._ready_to_record()
            self.engine.record(g)

    def _edit_sign(self) -> None:
        g = self._selected_gloss()
        if not g:
            return
        dlg = SignDialog(self, self.engine.entry(g), self.engine.sign_sensitivity(g))
        if dlg.exec() == QDialog.Accepted:
            v = dlg.values()
            self.engine.update_sign(g, v["text"], v["kind"], v["hands"], v["sensitivity"])

    def _delete_sign_takes(self) -> None:
        g = self._selected_gloss()
        n = self.counts.get(g, 0) if g else 0
        if not n:
            return
        if QMessageBox.question(self, APP_NAME, f"Delete your {n} recordings of {g}?") == QMessageBox.Yes:
            self.engine.delete_label(g)

    def _remove_sign(self) -> None:
        g = self._selected_gloss()
        if not g:
            return
        if g not in self.engine.vocab:
            QMessageBox.information(self, APP_NAME, f"{g} comes from a template pack. Switch the pack off in "
                                                    "Template packs to stop using it.")
            return
        n = self.counts.get(g, 0)
        extra = f" and delete your {n} recordings of it" if n else ""
        if QMessageBox.question(self, APP_NAME, f"Remove {g} from your list{extra}?") == QMessageBox.Yes:
            self.engine.remove_sign(g)

    def _word_selected(self) -> None:
        g = self._selected_gloss()
        for b in self.word_buttons:
            b.setEnabled(g is not None)
        if g is None:
            self.word_label.setText("")
            return
        e = self.engine.entry(g)
        n, p = self.counts.get(g, 0), self.pack_counts.get(g, 0)
        rec = f"{n} of your recordings" + (f", {p} from a template pack" if p else "")
        self.word_label.setText(f"<b>{g}</b> types “{e.text}” · {rec}")

    # fixing mistakes
    def _fix_clicked(self, item: QListWidgetItem) -> None:
        tok = item.data(Qt.UserRole)
        if not tok:
            return
        choices = sorted({r["gloss"] for r in self.engine.sign_rows()})
        dlg = FixDialog(self, tok, choices)
        if dlg.exec() != QDialog.Accepted:
            return
        msg = self.engine.fix_token(tok["id"], dlg.choice())
        self.report.setText(f"<span style='color:{GREEN}'>{msg}</span>")
        self._refresh_recent()

    def _refresh_recent(self) -> None:
        self.recent.clear()
        for tok in self.engine.recent_tokens(14):
            text = tok["text"] if tok["kind"] == "word" else tok["label"]
            if tok.get("fixed") is not None:
                text = f"{text} → {tok['fixed'] or '✕'}"
            it = QListWidgetItem(text)
            it.setData(Qt.UserRole, tok)
            it.setToolTip("A letter" if tok["kind"] == "letter" else f"The sign {tok['label']}")
            if not tok.get("in_message"):
                it.setForeground(QColor(MUTED))
            self.recent.addItem(it)
        self.recent.scrollToBottom()

    def _toggle_signing(self) -> None:
        on = not self.engine.enabled
        if on and not any(l for l in self.model_labels if not l.startswith("_")) and not self.sign_labels:
            QMessageBox.information(self, APP_NAME, "Teach SignChat first: press Guided setup (letters) or add and "
                                                    "record some words, or switch on a template pack.")
            return
        if on and self.engine.recording:
            self.engine.cancel_recording()
        self.engine.set_enabled(on)

    def _evaluate(self) -> None:
        self.report.setText("Checking… (this can take a little while with many words)")
        self.eval_btn.setEnabled(False)
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

    def _sim_sign(self) -> None:
        g = self.sim_sign.currentText()
        if g and self.source is not None and hasattr(self.source, "perform_sign"):
            self.source.perform_sign(g, self.engine.entry(g).kind)

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self.settings, self)
        if dlg.exec() != QDialog.Accepted:
            return
        new = dlg.result_settings()
        new.active_packs = list(self.settings.active_packs)
        old = self.settings
        self.settings = new
        new.save()
        restart_source = (new.source, new.hand, new.camera_index, new.hud_enabled, new.hud_distance_m,
                          new.hud_drop_m) != (old.source, old.hand, old.camera_index, old.hud_enabled,
                                              old.hud_distance_m, old.hud_drop_m)
        sender_keys = ("osc_host", "osc_port", "ttsvw_enabled", "ttsvw_host", "ttsvw_port", "ttsvw_speak")
        if any(getattr(new, k) != getattr(old, k) for k in sender_keys):
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
        self._refresh_tiles()
        self._refresh_words()

    def _open_packs(self) -> None:
        dlg = PacksDialog(self, self.settings, self.profile, self.engine.source_kind)
        dlg.exec()
        if dlg.active != self.settings.active_packs:
            self.settings.active_packs = list(dlg.active)
            self.settings.save()
        self._load_packs()

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
        added = self.engine.import_takes(other.takes, other.meta)
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
        if s.ttsvw_enabled:
            self.vrc_chip.setText(f"Messages → TTS Voice Wizard {s.ttsvw_host}:{s.ttsvw_port}")
            self.vrc_chip.setToolTip("Finished messages go to TTS Voice Wizard's OSC listener; it speaks them and/or "
                                     "puts them in the chatbox with its own settings.")
        else:
            self.vrc_chip.setText(f"VRChat OSC → {s.osc_host}:{s.osc_port}")
            self.vrc_chip.setToolTip("In VRChat: Action Menu → Options → OSC → Enabled")

    def _refresh_tiles(self) -> None:
        for label, b in self.tiles.items():
            n = self.counts.get(label, 0)
            want = (self.settings.takes_motion if label in MOTION_LETTERS
                    else 1 if label == "_rest" else self.settings.takes_static)
            state = "done" if n >= want else "partial" if n else "pack" if self.pack_counts.get(label) else "empty"
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
        if not n and self.pack_counts.get(self.selected):
            hint += " · using a template pack until you record it"
        self.sel_label.setText(f"<b>{display_name(self.selected)}</b> — {n} recording{'s' if n != 1 else ''}{hint}")
        self.del_btn.setEnabled(n > 0)

    def _refresh_words(self) -> None:
        keep = self._selected_gloss()
        rows = self.engine.sign_rows()
        want = self.settings.takes_sign
        self.words.setRowCount(len(rows))
        for i, r in enumerate(rows):
            g = QTableWidgetItem(r["gloss"] + ("  (phrase)" if r["kind"] == "phrase" else ""))
            g.setData(Qt.UserRole, r["gloss"])
            if not r["mine"]:
                g.setToolTip("From a template pack")
            txt = QTableWidgetItem(r["text"])
            if r["hands"] == "auto":
                hands = "—" if r["two_handed"] is None else ("both" if r["two_handed"] else "one")
            else:
                hands = {"one": "one (set)", "two": "both (set)"}[r["hands"]]
            hd = QTableWidgetItem(hands)
            rec_text = str(r["own"]) + (f" + {r['pack']} pack" if r["pack"] else "")
            rc = QTableWidgetItem(rec_text)
            color = GREEN if r["own"] >= want else BLUE if r["pack"] else AMBER if r["own"] else MUTED
            rc.setForeground(QColor(color))
            for c, it in enumerate((g, txt, hd, rc)):
                self.words.setItem(i, c, it)
            if r["gloss"] == keep:
                self.words.selectRow(i)
        self.words_empty.setVisible(not rows)
        self._word_selected()
        current = self.sim_sign.currentText()
        names = sorted({r["gloss"] for r in rows} | {e.gloss for e in STARTER_VOCAB})
        self.sim_sign.blockSignals(True)
        self.sim_sign.clear()
        self.sim_sign.addItems(names)
        if current in names:
            self.sim_sign.setCurrentText(current)
        self.sim_sign.blockSignals(False)

    def _drain(self) -> None:
        latest_pred = None
        words_dirty = False
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
                self._recent_dirty = True
            elif t == "commit":
                self.ring.flash = 4
                self._recent_dirty = True
                if ev.get("sign"):
                    self.pred_label.setText(f"Signed: {ev['label']} → “{ev['text']}”")
            elif t == "sent" and ev.get("final"):
                self.sent_list.insertItem(0, ev["text"])
                self._recent_dirty = True
            elif t == "enabled":
                self.start_btn.setText("Stop signing" if ev["on"] else "Start signing")
                self.start_btn.setProperty("on", "true" if ev["on"] else "false")
                self.start_btn.style().unpolish(self.start_btn)
                self.start_btn.style().polish(self.start_btn)
            elif t == "model":
                self.counts = ev["counts"]
                self.pack_counts = ev.get("pack_counts", {})
                self.model_labels = ev["labels"]
                self.sign_labels = ev.get("signs", [])
                self._refresh_tiles()
                words_dirty = True
            elif t == "take":
                self.counts = ev["counts"]
                self._refresh_tiles()
                words_dirty = True
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
                self.eval_btn.setEnabled(True)
                self._show_report(ev["data"])
            elif t == "notice":
                self.report.setText(f"<span style='color:{AMBER}'>{ev['text']}</span>")
        if words_dirty:
            self._refresh_words()
        if self._recent_dirty:
            self._recent_dirty = False
            self._refresh_recent()
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
            self.hand_label.setText(f"I can see your {other} hand. If it's your dominant hand, change "
                                    f"'My dominant hand' in Settings." if ev.get("other_hand") else "")
            return
        label = ev.get("label")
        enabled = self.engine.enabled
        if not self.model_labels and not self.sign_labels:
            self.ring.set_state("·", 0, MUTED)
            self.pred_label.setText("Nothing recorded yet. Press Guided setup (letters) or add words in "
                                    "Words & phrases.")
        elif label is None or label.startswith("_"):
            shown = "·" if label in (None, "_rest") else tile_text(label)
            self.ring.set_state(shown, ev.get("hold", 0), MUTED)
            raw = ev.get("raw")
            guess = f" (closest letter: {tile_text(raw)})" if raw and raw != "_rest" and label is None else ""
            if not self.pred_label.text().startswith("Signed:") or not enabled:
                self.pred_label.setText(("Not a letter I know" + guess) if label is None else "Resting")
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
                self.banner_big.setStyleSheet("font-size: 34pt; font-weight: 700;")
                self.banner_big.setText("…")
                self.banner_title.setText(f"{q} recording{'s' if q != 1 else ''} to go")
                self.banner_text.setText("Waiting for hand tracking. Make sure your hand is tracked."
                                         if ev.get("waiting") else "Next one coming up…")
            return
        self.banner.setVisible(True)
        label = ev["label"]
        is_sign = ev.get("kind") == "sign"
        big = label if is_sign else tile_text(label)
        size = 34 if len(big) <= 3 else 20 if len(big) <= 8 else 14
        self.banner_big.setStyleSheet(f"font-size: {size}pt; font-weight: 700;")
        self.banner_big.setText(big)
        take = f"{ev['take_no']}/{ev['takes_total']}"
        left = ev.get("queue", 0)
        tail = f" · {left} more after this" if left else ""
        name = label if is_sign else display_name(label)
        if is_sign:
            text = self.engine.entry(label).text
            if phase == "prompt":
                self.banner_title.setText(f"Next: {name}  ({take})")
                self.banner_text.setText(f"Types “{text}”. Put your hands down; after GO sign it once the way you "
                                         "normally do, then put your hands down again." + tail)
            elif phase == "countdown":
                self.banner_title.setText(f"Get ready: {name}  ({take}) · {int(ev['remaining']) + 1}")
                self.banner_text.setText("Hands down, get ready…" + tail)
            else:
                self.banner_title.setText(f"Recording {name}  ({take}) · {int(ev['progress'] * 100)}%")
                self.banner_text.setText(f"Sign {name} now, then lower your hands." + tail)
            return
        if phase == "prompt":
            self.banner_title.setText(f"Next: {name}  ({take})")
            self.banner_text.setText("Get your hand ready." + tail)
        elif phase == "countdown":
            self.banner_title.setText(f"Get ready: {name}  ({take}) · {int(ev['remaining']) + 1}")
            self.banner_text.setText(("Relax your hand." if label == "_rest" else "Form the handshape now.") + tail)
        else:
            if ev.get("kind") == "motion":
                msg = f"Do the {label} movement once now, then stop."
            elif label == "_rest":
                msg = "Relax and move your hand around naturally."
            else:
                msg = "Hold it still…"
            self.banner_title.setText(f"Recording {name}  ({take}) · {int(ev['progress'] * 100)}%")
            self.banner_text.setText(msg + tail)

    def _show_report(self, data: dict) -> None:
        rep, signs = data.get("static"), data.get("signs")
        if rep is None and signs is None:
            self.report.setText("Record some letters or words first.")
            return
        lines = []
        if rep is not None:
            lines.append(f"<b>Letters: {rep.overall * 100:.0f}%</b> (each recording tested against the others)")
            worst = [w for w in rep.worst(6) if w[1] < 0.9]
            if worst:
                items = []
                for label, acc, conf in worst:
                    c = f" → mistaken for <b>{tile_text(conf)}</b>" if conf else " → often not recognised"
                    items.append(f"{tile_text(label)} {acc * 100:.0f}%{c}")
                lines.append("Needs work: " + "; ".join(items))
                lines.append("<span style='color:%s'>Tip: re-record those, or make the two handshapes more "
                             "different (VR controllers can't show every finger detail).</span>" % MUTED)
            else:
                lines.append("No letters are getting mixed up. 🎉")
            untestable = [l for l in rep.untestable if l != "_rest"]
            if untestable:
                lines.append("Need a 2nd recording to test: " + ", ".join(tile_text(l) for l in untestable))
            motion = data.get("motion") or {}
            if motion:
                lines.append("Moving letters: " + ", ".join(f"{k} {h}/{n}" for k, (h, n) in sorted(motion.items())))
        if signs is not None and signs.per_label:
            lines.append(f"<b>Words &amp; phrases: {signs.overall * 100:.0f}%</b>")
            worst = [w for w in signs.worst(6) if w[1] < 0.9]
            if worst:
                items = []
                for label, acc, conf in worst:
                    c = f" → mistaken for <b>{conf}</b>" if conf else " → often not recognised"
                    items.append(f"{label} {acc * 100:.0f}%{c}")
                lines.append("Needs work: " + "; ".join(items))
            else:
                lines.append("No words are getting mixed up.")
        if signs is not None and signs.untestable:
            lines.append("Need more of your own recordings to test: " + ", ".join(signs.untestable[:12]))
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
