"""Reusable, model-driven editor for ordered latency-window collections."""

from __future__ import annotations

import copy
import json
import uuid
from dataclasses import dataclass

from PySide6.QtCore import QAbstractTableModel, QByteArray, QItemSelectionModel, QModelIndex, Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QShortcut, QUndoCommand, QUndoStack
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QStyledItemDelegate,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from monstim_gui.core.clipboard import LatencyWindowClipboard
from monstim_gui.dialogs.base import COLOR_OPTIONS
from monstim_signals.core import LatencyWindow


@dataclass
class _EditorRow:
    uid: str
    window: LatencyWindow
    per_channel: bool = False


class _WindowSnapshotCommand(QUndoCommand):
    def __init__(self, editor, before, after, text):
        super().__init__(text)
        self.editor = editor
        self.before = copy.deepcopy(before)
        self.after = copy.deepcopy(after)
        self._first_redo = True

    def undo(self):
        self.editor._restore_history_snapshot(self.before)

    def redo(self):
        if self._first_redo:
            self._first_redo = False
            return
        self.editor._restore_history_snapshot(self.after)


class LatencyWindowTableModel(QAbstractTableModel):
    """Ordered draft windows with robust multi-row internal drag/drop."""

    columns = ("", "#", "Name", "Start\nms", "Duration\nms", "Color")
    MIME_TYPE = "application/x-monstim-latency-window-rows"
    rows_reordered = Signal(list)

    def __init__(self, channel_count: int, parent=None):
        super().__init__(parent)
        self.channel_count = max(1, channel_count)
        self.rows: list[_EditorRow] = []

    def rowCount(self, parent=None):
        return 0 if parent is not None and parent.isValid() else len(self.rows)

    def columnCount(self, parent=None):
        return 0 if parent is not None and parent.isValid() else len(self.columns)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self.columns[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = self.rows[index.row()]
        window = row.window
        col = index.column()
        if role == Qt.ItemDataRole.UserRole:
            return row.uid
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            if col == 0:
                return "⠿"
            if col == 1:
                return index.row() + 1
            if col == 2:
                return window.name
            if col == 3:
                if not row.per_channel:
                    return window.start_times[0]
                return f"Per-channel {min(window.start_times):.2f}–{max(window.start_times):.2f}"  # noqa: RUF001
            if col == 4:
                return window.durations[0]
            if col == 5:
                return window.color.replace("tab:", "")
        if role == Qt.ItemDataRole.ForegroundRole and col == 5:
            return QColor(window.color.replace("tab:", ""))
        if role == Qt.ItemDataRole.ToolTipRole and col == 5:
            return f"{window.color.replace('tab:', '')}; edit colour in the detail pane"
        if role == Qt.ItemDataRole.ToolTipRole and col == 3 and row.per_channel:
            return "Per-channel start times. The quick stepper shifts every channel equally and preserves their differences."
        if role == Qt.ItemDataRole.TextAlignmentRole and col in (0, 1):
            return Qt.AlignmentFlag.AlignCenter
        return None

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemFlag.ItemIsDropEnabled
        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsDragEnabled | Qt.ItemFlag.ItemIsDropEnabled
        if index.column() in (2, 3, 4, 5):
            flags |= Qt.ItemFlag.ItemIsEditable
        return flags

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if role != Qt.ItemDataRole.EditRole or not index.isValid():
            return False
        window = self.rows[index.row()].window
        if index.column() == 2:
            window.name = str(value).strip() or "Window"
        elif index.column() == 3:
            value = float(value)
            if not self.rows[index.row()].per_channel:
                window.start_times = [value] * self.channel_count
            else:
                # Per-channel quick edits are relative nudges, not destructive broadcasts.
                window.start_times = [start + value for start in window.start_times]
        elif index.column() == 4:
            window.durations = [max(0.0, float(value))] * self.channel_count
        elif index.column() == 5:
            window.color = str(value)
        else:
            return False
        self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole])
        return True

    def set_windows(self, windows: list[LatencyWindow]):
        self.beginResetModel()
        self.rows = [_EditorRow(str(uuid.uuid4()), self._normalise(copy.deepcopy(window)), not self._is_global(window)) for window in windows]
        self.endResetModel()

    def windows(self) -> list[LatencyWindow]:
        return [copy.deepcopy(row.window) for row in self.rows]

    def append_window(self, window: LatencyWindow):
        row = len(self.rows)
        self.beginInsertRows(QModelIndex(), row, row)
        self.rows.append(_EditorRow(str(uuid.uuid4()), self._normalise(copy.deepcopy(window)), not self._is_global(window)))
        self.endInsertRows()

    def remove_rows(self, row_numbers: list[int]):
        for row in sorted(set(row_numbers), reverse=True):
            self.beginRemoveRows(QModelIndex(), row, row)
            self.rows.pop(row)
            self.endRemoveRows()

    def mimeTypes(self):
        return [self.MIME_TYPE]

    def mimeData(self, indexes):
        unique_rows = sorted({index.row() for index in indexes})
        mime = super().mimeData(indexes)
        mime.setData(self.MIME_TYPE, QByteArray(json.dumps([self.rows[row].uid for row in unique_rows]).encode()))
        return mime

    def supportedDropActions(self):
        return Qt.DropAction.MoveAction

    def dropMimeData(self, data, action, row, column, parent):
        if action == Qt.DropAction.IgnoreAction:
            return True
        if action != Qt.DropAction.MoveAction or not data.hasFormat(self.MIME_TYPE):
            return False
        dragged = set(json.loads(bytes(data.data(self.MIME_TYPE)).decode()))
        selected = [item for item in self.rows if item.uid in dragged]
        if not selected:
            return False
        target = parent.row() if parent.isValid() else row
        if target < 0:
            target = len(self.rows)
        source_indexes = [i for i, item in enumerate(self.rows) if item.uid in dragged]
        target -= sum(i < target for i in source_indexes)
        remaining = [item for item in self.rows if item.uid not in dragged]
        target = max(0, min(target, len(remaining)))
        self.beginResetModel()
        self.rows = remaining[:target] + selected + remaining[target:]
        self.endResetModel()
        self.rows_reordered.emit([item.uid for item in selected])
        return True

    def _normalise(self, window: LatencyWindow) -> LatencyWindow:
        start = window.start_times[0] if window.start_times else 0.0
        duration = window.durations[0] if window.durations else 1.0
        if len(window.start_times) != self.channel_count:
            window.start_times = [start] * self.channel_count
        if len(window.durations) != self.channel_count:
            window.durations = [duration] * self.channel_count
        return window

    @staticmethod
    def _is_global(window: LatencyWindow) -> bool:
        return not window.start_times or max(window.start_times) - min(window.start_times) <= 1e-9


class _TimingDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        spin = QDoubleSpinBox(parent)
        spin.setDecimals(2)
        spin.setSingleStep(0.05)
        spin.setRange(-1000.0 if index.column() == 3 else 0.0, 1000.0)
        spin.setSuffix(" ms")
        return spin

    def setEditorData(self, editor, index):
        value = index.data(Qt.ItemDataRole.EditRole)
        if index.column() == 3 and isinstance(value, str):
            editor.setValue(0.0)
            editor.setPrefix("Shift all: ")
            editor.setToolTip("Nudge every channel by this amount while preserving per-channel differences.")
        else:
            editor.setValue(float(value))

    def setModelData(self, editor, model, index):
        model.setData(index, editor.value(), Qt.ItemDataRole.EditRole)


class _ColorDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        for color in COLOR_OPTIONS:
            combo.addItem(color.replace("tab:", ""), color)
        return combo

    def setEditorData(self, editor, index):
        color = index.model().rows[index.row()].window.color
        editor.setCurrentIndex(max(0, editor.findData(color)))

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentData(), Qt.ItemDataRole.EditRole)


class LatencyWindowEditor(QWidget):
    """Compact list, drag reordering, quick edits, and a persistent detail inspector."""

    changed = Signal()

    def __init__(self, channel_names: list[str], parent=None, *, compact: bool = False):
        super().__init__(parent)
        self.channel_names = channel_names or ["Default"]
        self.compact = compact
        self.model = LatencyWindowTableModel(len(self.channel_names), self)
        self.undo_stack = QUndoStack(self)
        self._history_replaying = False
        self._loading_windows = False
        self._last_snapshot: list[LatencyWindow] = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        toolbar = QGridLayout() if self.compact else QHBoxLayout()
        toolbar_buttons = []
        for label, slot in (
            ("Undo", self.undo_stack.undo),
            ("Redo", self.undo_stack.redo),
            ("Add", self.add_window),
            ("Duplicate", self.duplicate_selected),
            ("Delete", self.delete_selected),
            ("Copy", self.copy_selected),
            ("Copy All", self.copy_all),
            ("Paste", self.paste),
        ):
            button = QPushButton(label)
            button.clicked.connect(slot)
            if label == "Undo":
                self.undo_button = button
            elif label == "Redo":
                self.redo_button = button
            if self.compact:
                toolbar_buttons.append(button)
            else:
                toolbar.addWidget(button)
        if self.compact:
            # Four compact controls per row keep the preferences editor within its narrow tab.
            for position, button in enumerate(toolbar_buttons):
                toolbar.addWidget(button, position // 4, position % 4)
        else:
            toolbar.addStretch()
        layout.addLayout(toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setDragEnabled(True)
        self.table.setAcceptDrops(True)
        self.table.viewport().setAcceptDrops(True)
        self.table.setDropIndicatorShown(True)
        self.table.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.table.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.table.setItemDelegateForColumn(3, _TimingDelegate(self.table))
        self.table.setItemDelegateForColumn(4, _TimingDelegate(self.table))
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(2, 150)
        self.table.setColumnWidth(3, 60)
        self.table.setColumnWidth(4, 60)
        self.table.setColumnWidth(5, 76)
        if self.compact:
            self.table.setColumnWidth(2, 100)
            self.table.setColumnWidth(3, 55)
            self.table.setColumnWidth(4, 55)
            self.table.setColumnWidth(5, 70)
        self.table.setItemDelegateForColumn(5, _ColorDelegate(self.table))
        header.setFixedHeight(42)
        splitter.addWidget(self.table)

        self.inspector = QStackedWidget()
        self.inspector.addWidget(QLabel("Select a latency window to edit its details."))
        self.inspector.addWidget(self._single_inspector())
        self.inspector.addWidget(self._multi_inspector())
        self.inspector.setMinimumWidth(0 if self.compact else 260)
        splitter.addWidget(self.inspector)
        if self.compact:
            splitter.setOrientation(Qt.Orientation.Vertical)
            splitter.setSizes([220, 300])
        else:
            splitter.setSizes([450, 400])
        layout.addWidget(splitter, 1)

        self.table.selectionModel().selectionChanged.connect(self._update_inspector)
        self.model.dataChanged.connect(lambda *_: self._record_history("Edit latency window"))
        self.model.modelReset.connect(lambda: self._record_history("Reorder latency windows"))
        self.model.rows_reordered.connect(self._restore_selection_after_reorder)
        self.undo_stack.canUndoChanged.connect(self.undo_button.setEnabled)
        self.undo_stack.canRedoChanged.connect(self.redo_button.setEnabled)
        self.undo_button.setEnabled(False)
        self.redo_button.setEnabled(False)
        QShortcut(QKeySequence.StandardKey.Undo, self, activated=self.undo_stack.undo)
        QShortcut(QKeySequence.StandardKey.Redo, self, activated=self.undo_stack.redo)

    def _single_inspector(self):
        panel = QWidget()
        form = QFormLayout(panel)
        self.name_edit = QLineEdit()
        self.color_combo = QComboBox()
        for color in COLOR_OPTIONS:
            self.color_combo.addItem(color.replace("tab:", ""), color)
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.0, 1000.0)
        self.duration_spin.setDecimals(2)
        self.duration_spin.setSingleStep(0.1)
        self.duration_spin.setSuffix(" ms")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Global", "Per-channel"])
        self.global_start_spin = QDoubleSpinBox()
        self.global_start_spin.setRange(-1000.0, 1000.0)
        self.global_start_spin.setDecimals(2)
        self.global_start_spin.setSingleStep(0.1)
        self.global_start_spin.setSuffix(" ms")
        self.channel_table = QTableWidget(0, 2)
        self.channel_table.setHorizontalHeaderLabels(["Channel", "Start time (ms)"])
        self.channel_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        form.addRow("Name", self.name_edit)
        form.addRow("Color", self.color_combo)
        form.addRow("Duration", self.duration_spin)
        form.addRow("Start mode", self.mode_combo)
        form.addRow("Global start", self.global_start_spin)
        form.addRow("Per-channel starts", self.channel_table)
        self.global_start_label = form.labelForField(self.global_start_spin)
        self.channel_table_label = form.labelForField(self.channel_table)
        self.name_edit.editingFinished.connect(self._apply_single_details)
        self.color_combo.currentIndexChanged.connect(self._apply_single_details)
        self.duration_spin.valueChanged.connect(self._apply_single_details)
        self.global_start_spin.valueChanged.connect(self._apply_single_details)
        self.mode_combo.currentIndexChanged.connect(self._change_mode)
        return panel

    def _multi_inspector(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel("Multiple windows selected. Nudge their timing without losing per-channel differences."))
        self.nudge_amount = QDoubleSpinBox()
        self.nudge_amount.setRange(0.01, 1000.0)
        self.nudge_amount.setValue(0.1)
        self.nudge_amount.setDecimals(2)
        self.nudge_amount.setSuffix(" ms")
        start_row = QHBoxLayout()
        start_row.addWidget(QLabel("Start time:"))
        start_row.addWidget(self.nudge_amount)
        for label, delta in (("−", -1), ("+", 1)):  # noqa: RUF001
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, sign=delta: self._nudge_selected(3, sign))
            start_row.addWidget(button)
        layout.addLayout(start_row)
        duration_row = QHBoxLayout()
        duration_row.addWidget(QLabel("Duration:"))
        for label, delta in (("−", -1), ("+", 1)):  # noqa: RUF001
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, sign=delta: self._nudge_selected(4, sign))
            duration_row.addWidget(button)
        duration_row.addStretch()
        layout.addLayout(duration_row)
        layout.addStretch()
        return panel

    def set_windows(self, windows: list[LatencyWindow]):
        self._loading_windows = True
        self.model.set_windows(windows)
        self._loading_windows = False
        self._last_snapshot = self.windows()
        self.undo_stack.clear()
        self._update_inspector()

    def windows(self) -> list[LatencyWindow]:
        return self.model.windows()

    def _record_history(self, text: str):
        if self._loading_windows or self._history_replaying:
            return
        current = self.windows()
        if current == self._last_snapshot:
            return
        self._history_replaying = True
        self.undo_stack.push(_WindowSnapshotCommand(self, self._last_snapshot, current, text))
        self._history_replaying = False
        self._last_snapshot = current
        self.changed.emit()

    def _restore_history_snapshot(self, windows: list[LatencyWindow]):
        selected_rows = self._selected_rows()
        self._history_replaying = True
        self.model.set_windows(windows)
        self._last_snapshot = self.windows()
        self._history_replaying = False
        selection = self.table.selectionModel()
        selection.clearSelection()
        for row in selected_rows:
            if row < self.model.rowCount():
                selection.select(self.model.index(row, 0), QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
        self._update_inspector()
        self.changed.emit()

    def add_window(self):
        self.model.append_window(LatencyWindow(f"Window {self.model.rowCount() + 1}", "black", [0.0], [1.0], ":"))
        self._record_history("Add latency window")

    def duplicate_selected(self):
        rows = self._selected_rows()
        for row in rows:
            window = copy.deepcopy(self.model.rows[row].window)
            window.name = self._unique_window_name(window.name)
            self.model.append_window(window)
        if rows:
            self._record_history("Duplicate latency window")

    def delete_selected(self):
        rows = self._selected_rows()
        if rows:
            self.model.remove_rows(rows)
            self._record_history("Delete latency window")

    def copy_selected(self):
        windows = [self.model.rows[row].window for row in self._selected_rows()]
        if len(windows) == 1:
            LatencyWindowClipboard.set_single(windows[0])
        elif windows:
            LatencyWindowClipboard.set_multiple(windows)

    def copy_all(self):
        if self.model.rows:
            LatencyWindowClipboard.set_multiple(self.windows())

    def paste(self):
        mode, data = LatencyWindowClipboard.get_most_recent()
        if mode == "none":
            return
        if mode == "multiple":
            if (
                self.model.rows
                and QMessageBox.question(
                    self,
                    "Replace windows?",
                    "Pasting multiple windows replaces the current list.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                != QMessageBox.StandardButton.Yes
            ):
                return
            self.model.set_windows(data)
        else:
            window = data
            if any(row.window.name == window.name for row in self.model.rows):
                window.name = self._unique_window_name(window.name)
            self.model.append_window(window)
        self._record_history("Paste latency windows")

    def _selected_rows(self):
        return sorted({index.row() for index in self.table.selectionModel().selectedRows()})

    def _update_inspector(self, *_):
        rows = self._selected_rows()
        if len(rows) != 1:
            self.inspector.setCurrentIndex(2 if rows else 0)
            return
        self._detail_row = rows[0]
        window = self.model.rows[self._detail_row].window
        for widget in (self.name_edit, self.color_combo, self.duration_spin, self.mode_combo, self.global_start_spin):
            widget.blockSignals(True)
        self.name_edit.setText(window.name)
        self.color_combo.setCurrentIndex(max(0, self.color_combo.findData(window.color)))
        self.duration_spin.setValue(window.durations[0])
        global_mode = not self.model.rows[self._detail_row].per_channel
        self.mode_combo.setCurrentIndex(0 if global_mode else 1)
        self.global_start_spin.setValue(window.start_times[0])
        for widget in (self.name_edit, self.color_combo, self.duration_spin, self.mode_combo, self.global_start_spin):
            widget.blockSignals(False)
        self.channel_table.blockSignals(True)
        self.channel_table.setRowCount(len(self.channel_names))
        for i, (channel, start) in enumerate(zip(self.channel_names, window.start_times, strict=True)):
            self.channel_table.setItem(i, 0, QTableWidgetItem(channel))
            spin = QDoubleSpinBox()
            spin.setRange(-1000, 1000)
            spin.setDecimals(2)
            spin.setSingleStep(0.1)
            spin.setValue(start)
            spin.valueChanged.connect(self._apply_channel_details)
            self.channel_table.setCellWidget(i, 1, spin)
        self.channel_table.setVisible(not global_mode)
        self.global_start_spin.setVisible(global_mode)
        self.global_start_label.setVisible(global_mode)
        self.channel_table_label.setVisible(not global_mode)
        self.channel_table.blockSignals(False)
        self.inspector.setCurrentIndex(1)

    def _apply_single_details(self, *_):
        if not hasattr(self, "_detail_row"):
            return
        window = self.model.rows[self._detail_row].window
        window.name = self.name_edit.text().strip() or "Window"
        window.color = self.color_combo.currentData()
        window.durations = [self.duration_spin.value()] * self.model.channel_count
        if self.mode_combo.currentIndex() == 0:
            window.start_times = [self.global_start_spin.value()] * self.model.channel_count
        self.model.dataChanged.emit(self.model.index(self._detail_row, 2), self.model.index(self._detail_row, 5))
        self.changed.emit()

    def _apply_channel_details(self, *_):
        if not hasattr(self, "_detail_row") or self.mode_combo.currentIndex() == 0:
            return
        window = self.model.rows[self._detail_row].window
        window.start_times = [self.channel_table.cellWidget(i, 1).value() for i in range(self.channel_table.rowCount())]
        self.model.dataChanged.emit(self.model.index(self._detail_row, 3), self.model.index(self._detail_row, 3))
        self.changed.emit()

    def _change_mode(self, index):
        if not hasattr(self, "_detail_row"):
            return
        window = self.model.rows[self._detail_row].window
        if index == 0:
            window.start_times = [self.global_start_spin.value()] * self.model.channel_count
        self.model.rows[self._detail_row].per_channel = index == 1
        self.model.dataChanged.emit(self.model.index(self._detail_row, 3), self.model.index(self._detail_row, 3))
        self._update_inspector()

    def _nudge_selected(self, column, sign):
        for row in self._selected_rows():
            window = self.model.rows[row].window
            if column == 3:
                value = sign * self.nudge_amount.value()
                if not self.model.rows[row].per_channel:
                    value += window.start_times[0]
            else:
                value = window.durations[0] + sign * self.nudge_amount.value()
            self.model.setData(self.model.index(row, column), value)
        self.changed.emit()

    def _restore_selection_after_reorder(self, selected_uids):
        selection = self.table.selectionModel()
        selection.clearSelection()
        for row, item in enumerate(self.model.rows):
            if item.uid in selected_uids:
                selection.select(self.model.index(row, 0), QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
        self.changed.emit()
        self._update_inspector()

    def _unique_window_name(self, base):
        names = {row.window.name for row in self.model.rows}
        number = 2
        candidate = f"{base} ({number})"
        while candidate in names:
            number += 1
            candidate = f"{base} ({number})"
        return candidate
