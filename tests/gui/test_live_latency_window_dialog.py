"""Focused coverage for the non-modal latency-window editor context."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QWidget

from monstim_gui.core.ui_theme import apply_application_theme
from monstim_gui.dialogs.latency import LatencyWindowsDialog
from monstim_gui.dialogs.preferences import MWaveWindowNamesEditor
from monstim_signals.core import LatencyWindow, SessionAnnot
from monstim_signals.domain.session import Session


class _ConfigRepo:
    def read_config(self):
        return {"m_wave_window_names": ["M-wave", "M_response"]}


class _GUI(QWidget):
    def __init__(self, session):
        super().__init__()
        self.current_experiment = None
        self.current_dataset = None
        self.current_session = session


def _session(session_id: str, start: float) -> Session:
    session = Session.__new__(Session)
    session.id = session_id
    session.channel_names = ["Ch 1"]
    annotation = SessionAnnot.create_empty()
    annotation.latency_windows = [LatencyWindow(name="M-wave", color="blue", start_times=[start], durations=[2.0])]
    session.annot = annotation
    return session


def test_live_dialog_refreshes_session_draft_and_context():
    first = _session("S1", 1.0)
    gui = _GUI(first)
    dialog = LatencyWindowsDialog(first, gui, config_repo=_ConfigRepo())

    assert not bool(dialog.windowFlags() & Qt.WindowType.WindowStaysOnTopHint)

    assert dialog.apply_level_combo.currentData() == "session"
    assert "Session annotation" in dialog.value_source_label.text()
    assert "S1" in dialog.value_source_label.text()
    assert dialog.editor.windows()[0].start_times == [1.0]
    dialog.editor.table.selectRow(0)
    QApplication.processEvents()
    assert dialog.editor.m_wave_name_note.text() == "(M-max Compatible Window)"
    assert not dialog.editor.m_wave_name_note.isHidden()
    assert dialog.editor.global_radio.isChecked()
    assert dialog.editor.start_details.currentIndex() == 0
    dialog.editor.per_channel_radio.setChecked(True)
    assert dialog.editor.start_details.currentIndex() == 1
    assert dialog.editor.model.rows[0].per_channel

    dialog.editor.name_edit.setText("H-reflex")
    dialog.editor._apply_single_details()
    assert dialog.editor.m_wave_name_note.isHidden()

    gui.current_session = _session("S2", 7.5)
    dialog.refresh_from_current_selection()

    assert "Session" in dialog.active_context_label.text()
    assert "S2" in dialog.active_context_label.text()
    assert "Session annotation" in dialog.value_source_label.text()
    assert dialog.editor.windows()[0].start_times == [7.5]
    dialog.close()


def test_latency_window_text_fields_select_existing_value_on_focus():
    session = _session("S1", 1.0)
    gui = _GUI(session)
    dialog = LatencyWindowsDialog(session, gui, config_repo=_ConfigRepo())
    dialog.show()
    dialog.editor.table.selectRow(0)
    QApplication.processEvents()

    fields = [
        dialog.editor.name_edit,
        dialog.editor.duration_spin.lineEdit(),
        dialog.editor.global_start_spin.lineEdit(),
        dialog.editor.nudge_amount.lineEdit(),
    ]
    dialog.editor.per_channel_radio.setChecked(True)
    QApplication.processEvents()
    fields.append(dialog.editor.channel_table.cellWidget(0, 1).lineEdit())

    for field in fields:
        field.setFocus()
        QApplication.processEvents()
        assert field.selectedText() == field.text()

    dialog.close()


def test_per_channel_timing_editors_have_space_for_spin_controls():
    apply_application_theme(QApplication.instance())
    session = _session("S1", 1.0)
    session.channel_names = ["Ch0", "Ch1"]
    session.annot.latency_windows[0].start_times = [1.4, 1.5]
    gui = _GUI(session)
    dialog = LatencyWindowsDialog(session, gui, config_repo=_ConfigRepo())
    dialog.resize(dialog.minimumSize())
    dialog.show()
    dialog.editor.table.selectRow(0)
    dialog.editor.per_channel_radio.setChecked(True)
    QApplication.processEvents()

    table = dialog.editor.channel_table
    spinbox = table.cellWidget(0, 1)

    assert table.horizontalScrollBar().maximum() == 0
    assert spinbox.geometry().right() < table.viewport().rect().right()
    assert spinbox.width() > 0
    assert spinbox.height() >= dialog.editor.CHANNEL_TABLE_ROW_HEIGHT - 11
    dialog.close()


def test_m_wave_window_names_editor_restores_defaults_and_allows_empty_list():
    editor = MWaveWindowNamesEditor(["Protocol M"], ["M-wave", "M_response"])

    assert editor.validate() == ["Protocol M"]
    editor.set_names([])
    assert editor.validate() == []
    assert not editor.empty_notice.isHidden()

    editor.restore_shipped_defaults()
    assert editor.validate() == ["M-wave", "M_response"]
