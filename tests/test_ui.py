"""The window opens, every dialog builds, and the main buttons work (no screen needed)."""

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from signchat import app as ui  # noqa: E402
from signchat.config import Settings  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ui.apply_theme(a)
    return a


def pump(qapp, sec):
    end = time.time() + sec
    while time.time() < end:
        qapp.processEvents()
        time.sleep(0.01)


def test_window_and_dialogs(qapp, isolated_home):
    w = ui.MainWindow(Settings(source="sim", countdown_s=1.0))
    try:
        w.show()
        pump(qapp, 0.5)
        w.engine.add_common_signs(["HELLO", "NICE-MEET-YOU"])
        pump(qapp, 0.2)
        assert w.words.rowCount() == 2 and w.words.item(1, 0).data(ui.Qt.UserRole) == "NICE-MEET-YOU"
        w.words.selectRow(0)
        assert "HELLO" in w.word_label.text()
        for dlg in (ui.SignDialog(w, existing={"HELLO"}), ui.SignDialog(w, w.engine.entry("HELLO"), 0.8),
                    ui.CommonSignsDialog(w, {"HELLO"}), ui.SettingsDialog(w.settings, w),
                    ui.PacksDialog(w, w.settings, w.profile, "sim"),
                    ui.FixDialog(w, {"id": 1, "kind": "word", "text": "hello", "label": "HELLO"}, ["HELLO"])):
            dlg.show()
            pump(qapp, 0.05)
            dlg.close()
        d = ui.SignDialog(w, existing={"HELLO"})
        d.gloss.setText("hello")
        d._ok()
        assert "already" in d.error.text()
        d.gloss.setText("see you later")
        d._gloss_changed("see you later")
        assert d.values()["gloss"] == "SEE-YOU-LATER"
        s = ui.SettingsDialog(w.settings, w).result_settings()
        assert s.ttsvw_speak is True and s.signs_enabled is True
        w._record_sign()                      # records the selected sign (HELLO): the banner shows it
        pump(qapp, 2.0)
        assert w.banner.isVisible() and "HELLO" in w.banner_title.text()
        w.engine.cancel_recording()
        pump(qapp, 0.2)
    finally:
        w.close()
        pump(qapp, 0.2)
