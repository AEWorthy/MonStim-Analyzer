"""Reusable profile-library browser independent of profile persistence."""

from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGridLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from monstim_gui.managers.profile_manager import ProfileRecord


class ProfileLibraryWidget(QWidget):
    """Displays profile records and emits intent; its owner performs writes."""

    selected = Signal(object)
    add_requested = Signal()
    duplicate_requested = Signal(object)
    delete_requested = Signal(object)
    import_requested = Signal()
    export_requested = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(280)
        self.setMaximumWidth(360)
        title = QLabel("Profile Library", self)
        title.setObjectName("profileLibraryTitle")
        description = QLabel("Choose a profile to inspect its defaults and analysis overrides.", self)
        description.setObjectName("profileLibraryDescription")
        description.setWordWrap(True)
        self.list_widget = QTreeWidget(self)
        self.list_widget.setObjectName("profileLibraryTable")
        self.list_widget.setHeaderLabels(["Profile", "Source"])
        self.list_widget.setRootIsDecorated(False)
        self.list_widget.setUniformRowHeights(True)
        self.list_widget.setAlternatingRowColors(True)
        self.list_widget.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list_widget.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.list_widget.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.list_widget.setStyleSheet("QTreeWidget::item { height: 28px; padding: 3px 5px; }")
        self.source_label = QLabel("Select a profile", self)
        self.source_label.setObjectName("profileLibraryStatus")
        self.source_label.setWordWrap(True)
        self.add_button = QPushButton("New", self)
        self.duplicate_button = QPushButton("Duplicate", self)
        self.delete_button = QPushButton("Delete", self)
        self.import_button = QPushButton("Import…", self)
        self.export_button = QPushButton("Export…", self)
        self.add_button.setToolTip("Create a new user-owned analysis profile.")
        self.duplicate_button.setToolTip("Create an editable user copy of the selected profile.")
        self.delete_button.setToolTip("Delete the selected user profile when Settings are applied.")
        self.import_button.setToolTip("Import a portable analysis-profile YAML file.")
        self.export_button.setToolTip("Export the selected profile as a portable YAML file.")
        controls = QGridLayout()
        controls.setHorizontalSpacing(6)
        controls.setVerticalSpacing(6)
        controls.addWidget(self.add_button, 0, 0)
        controls.addWidget(self.duplicate_button, 0, 1)
        controls.addWidget(self.import_button, 1, 0)
        controls.addWidget(self.export_button, 1, 1)
        controls.addWidget(self.delete_button, 2, 0, 1, 2)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(title)
        layout.addWidget(description)
        layout.addWidget(self.list_widget, 1)
        layout.addWidget(self.source_label)
        layout.addLayout(controls)
        self.list_widget.currentItemChanged.connect(self._emit_selected)
        self.add_button.clicked.connect(self.add_requested)
        self.duplicate_button.clicked.connect(lambda: self.duplicate_requested.emit(self.current_record()))
        self.delete_button.clicked.connect(lambda: self.delete_requested.emit(self.current_record()))
        self.import_button.clicked.connect(self.import_requested)
        self.export_button.clicked.connect(lambda: self.export_requested.emit(self.current_record()))
        self._update_actions()

    def set_records(self, records: Iterable[ProfileRecord], selected_path: str | None = None) -> None:
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        selected_item = None
        for record in records:
            item = QTreeWidgetItem([record.name, "Built-in" if record.read_only else "User"])
            item.setData(0, Qt.ItemDataRole.UserRole, record)
            item.setToolTip(0, f"{record.source} profile\n{record.data.get('description', '')}")
            self.list_widget.addTopLevelItem(item)
            if record.path == selected_path:
                selected_item = item
        first_item = self.list_widget.topLevelItem(0) if self.list_widget.topLevelItemCount() else None
        self.list_widget.setCurrentItem(selected_item or first_item)
        self.list_widget.blockSignals(False)
        self._emit_selected()

    def current_record(self) -> ProfileRecord | None:
        item = self.list_widget.currentItem()
        return item.data(0, Qt.ItemDataRole.UserRole) if item else None

    def _emit_selected(self, *_args) -> None:
        record = self.current_record()
        if record:
            state = "Built-in profile — duplicate to customize" if record.read_only else "User profile — editable"
            self.source_label.setText(state)
        else:
            self.source_label.setText("Select a profile")
        self._update_actions()
        self.selected.emit(record)

    def _update_actions(self) -> None:
        record = self.current_record()
        enabled = record is not None
        self.duplicate_button.setEnabled(enabled)
        self.export_button.setEnabled(enabled)
        self.delete_button.setEnabled(bool(record and not record.read_only))
