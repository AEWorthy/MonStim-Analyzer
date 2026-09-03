# monstim_gui/dialogs/bulk_export_dialog.py
"""
BulkExportDialog - wizard-style dialog for the Bulk Data Export feature.

The dialog collects:
  - Data Export Level    : Dataset or Experiment
  - Objects       : hierarchical collapsible experiment / dataset checkboxes
  - Data types    : Average Reflex Curves, Longform Reflex Amplitudes, M-max, Max H-reflex
  - Methods       : rms, auc, peak_to_trough, average_rectified, average_unrectified
  - Export options: Normalize to M-max
  - Channels      : per-channel checkboxes
  - Output path   : directory chooser

On acceptance a :class:`BulkExportWorker` QThread is launched; progress is
shown via a QProgressDialog.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from monstim_gui.managers.bulk_export_manager import (
    DATA_TYPE_LABELS,
    METHOD_LABELS,
    BulkExportConfig,
    run_bulk_export,
)

if TYPE_CHECKING:
    from monstim_gui.gui_main import MonstimGUI

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Worker thread
# ─────────────────────────────────────────────────────────────────────────────


class BulkExportWorker(QThread):
    """Runs :func:`run_bulk_export` in a background thread.

    Signals
    -------
    progress(current: int, total: int, message: str)
    finished(written_files: list[str])
    error(message: str)
    """

    progress = Signal(int, int, str)
    finished_export = Signal(list)
    error = Signal(str)

    def __init__(self, config: BulkExportConfig, parent=None):
        super().__init__(parent)
        self._config = config
        self._canceled = threading.Event()

    def cancel(self) -> None:
        self._canceled.set()

    @property
    def is_canceled(self) -> bool:
        return self._canceled.is_set()

    def run(self) -> None:
        try:
            written = run_bulk_export(
                self._config,
                progress_callback=lambda cur, tot, msg: self.progress.emit(cur, tot, msg),
                is_canceled=self._canceled.is_set,
            )
            self.finished_export.emit(written)
        except Exception as exc:
            logger.exception("BulkExportWorker encountered an unexpected error.")
            self.error.emit(str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Progress window with rolling log
# ─────────────────────────────────────────────────────────────────────────────


class BulkExportProgressWindow(QDialog):
    """Application-modal progress window shown while a bulk export runs.

    Shows a progress bar and a scrolling plain-text log so the user can follow
    each loading/writing step in real time.
    """

    canceled = Signal()

    def __init__(self, total: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bulk Export - In Progress")
        self.setMinimumSize(500, 340)
        self._total = total
        self._done = False

        root = QVBoxLayout(self)
        root.setSpacing(6)

        # Status label
        self._status_lbl = QLabel(f"Starting export of {total} object(s)\u2026")
        root.addWidget(self._status_lbl)

        # Progress bar
        self._bar = QProgressBar()
        self._bar.setRange(0, max(total, 1))
        self._bar.setValue(0)
        self._bar.setTextVisible(True)
        root.addWidget(self._bar)

        # Scrolling log
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(200)
        self._log.setStyleSheet("QPlainTextEdit { font-family: Consolas, 'Courier New', monospace; font-size: 8pt; }")
        root.addWidget(self._log, 1)

        # Cancel button row
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFixedWidth(90)
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addWidget(self._cancel_btn)
        root.addLayout(btn_row)

    # ── public API ────────────────────────────────────────────────────────

    def update_progress(self, cur: int, tot: int, msg: str) -> None:
        """Append *msg* to the log and advance the progress bar."""
        ts = datetime.now(UTC).astimezone().strftime("%H:%M:%S")
        self._bar.setMaximum(max(tot, 1))
        self._bar.setValue(cur)
        self._status_lbl.setText(f"Progress: {cur} / {tot}")
        self._log.appendPlainText(f"[{ts}]  {msg}")
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def mark_done(self) -> None:
        """Switch the Cancel button to Close once the export finishes."""
        self._done = True
        self._cancel_btn.setText("Close")
        self._cancel_btn.setEnabled(True)
        try:
            self._cancel_btn.clicked.disconnect()
        except RuntimeError:
            logger.warning("Could not disconnect cancel button clicked signal; it may have already been disconnected.")
        self._cancel_btn.clicked.connect(self.accept)

    # ── cancel / close ────────────────────────────────────────────────────

    def _on_cancel(self) -> None:
        if self._done:
            self.accept()
            return
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.setText("Canceling\u2026")
        ts = datetime.now(UTC).astimezone().strftime("%H:%M:%S")
        self._log.appendPlainText(f"[{ts}]  Cancellation requested\u2026")
        self.canceled.emit()

    def closeEvent(self, event) -> None:
        if self._done:
            event.accept()
        else:
            self._on_cancel()
            event.ignore()  # keep open until worker finishes


# ─────────────────────────────────────────────────────────────────────────────
# Helper widget: Collapsible experiment group with checkbox children
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _DatasetStatus:
    """Lightweight dataset metadata needed by the export selector."""

    dataset_id: str
    display_name: str
    is_completed: bool | None
    is_excluded: bool = False
    incomplete_active_session_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ExperimentStatus:
    """Lightweight experiment metadata needed by the export selector."""

    is_completed: bool | None
    datasets: list[_DatasetStatus]


def _completion_label_text(is_completed: bool | None) -> str:
    if is_completed is True:
        return "Complete"
    if is_completed is False:
        return "Incomplete"
    return "Unknown"


def _completion_label_stylesheet(is_completed: bool | None) -> str:
    base = "QLabel { border: 1px solid %s; border-radius: 7px; padding: 1px 7px; font-weight: 600; color: %s; background: %s;}"
    if is_completed is True:
        return base % ("#8fd19e", "#176b2c", "#e7f6ea")
    if is_completed is False:
        return base % ("#f1a8a8", "#b42318", "#fdeaea")
    return base % ("#c7c7c7", "#555555", "#f1f1f1")


def _make_completion_badge(is_completed: bool | None, tooltip_prefix: str, parent: QWidget | None = None) -> QLabel:
    label = QLabel(_completion_label_text(is_completed), parent)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setMinimumWidth(88)
    label.setStyleSheet(_completion_label_stylesheet(is_completed))
    label.setToolTip(f"{tooltip_prefix}: {_completion_label_text(is_completed)}")
    return label


def _dataset_completion_summary(datasets: list[_DatasetStatus]) -> str:
    if not datasets:
        return "No datasets"
    known = [ds for ds in datasets if ds.is_completed is not None]
    complete = sum(1 for ds in known if ds.is_completed)
    incomplete = sum(1 for ds in known if ds.is_completed is False)
    unknown = len(datasets) - len(known)
    parts = [f"{complete} complete", f"{incomplete} incomplete"]
    if unknown:
        parts.append(f"{unknown} unknown")
    incomplete_session_datasets = sum(bool(ds.incomplete_active_session_ids) for ds in datasets)
    if incomplete_session_datasets:
        noun = "dataset has" if incomplete_session_datasets == 1 else "datasets have"
        parts.append(f"{incomplete_session_datasets} {noun} incomplete sessions")
    return ", ".join(parts)


def _incomplete_session_warning(status: _DatasetStatus) -> QLabel | None:
    """Return an actionable badge when active sessions would be skipped.

    Dataset exclusion is deliberately ignored here: excluded sessions are
    already intentionally omitted from every export.  The badge only reports
    sessions that remain active but lack their own completion marker.
    """
    session_ids = status.incomplete_active_session_ids
    if not session_ids:
        return None
    label = QLabel(f"{len(session_ids)} session{'s' if len(session_ids) != 1 else ''} incomplete")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setStyleSheet(
        "QLabel { border: 1px solid #d97706; border-radius: 7px; padding: 1px 7px; "
        "font-weight: 600; color: #92400e; background: #fef3c7; }"
    )
    label.setToolTip(
        "Completed data only will omit these non-excluded sessions until they are marked complete: "
        + ", ".join(session_ids)
    )
    return label


class _ExperimentGroup(QWidget):
    """A collapsible card showing one experiment with its dataset checkboxes."""

    def __init__(
        self,
        expt_name: str,
        experiment_completed: bool | None,
        datasets: list[_DatasetStatus],
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.expt_name = expt_name
        self.experiment_completed = experiment_completed
        self.dataset_statuses = datasets

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 2, 0, 2)
        outer.setSpacing(2)

        # ── header row: collapse arrow + experiment checkbox ──────────────
        header_row = QWidget()
        header_layout = QHBoxLayout(header_row)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)

        self._toggle_btn = QToolButton()
        self._toggle_btn.setArrowType(Qt.ArrowType.RightArrow)
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setChecked(False)
        self._toggle_btn.setFixedWidth(20)
        self._toggle_btn.clicked.connect(self._on_toggle)
        header_layout.addWidget(self._toggle_btn)

        self._expt_cb = QCheckBox(expt_name)
        self._expt_cb.setTristate(False)
        self._expt_cb.setChecked(False)
        self._expt_cb.setToolTip(f"Dataset completion summary: {_dataset_completion_summary(datasets)}")
        self._expt_cb.stateChanged.connect(self._on_expt_checked)
        header_layout.addWidget(self._expt_cb, 1)

        self._expt_status_lbl = _make_completion_badge(experiment_completed, "Experiment status", self)
        header_layout.addWidget(self._expt_status_lbl)

        outer.addWidget(header_row)

        # ── children container (datasets) ─────────────────────────────────
        self._children_widget = QWidget()
        children_layout = QVBoxLayout(self._children_widget)
        children_layout.setContentsMargins(28, 0, 0, 4)
        children_layout.setSpacing(2)

        self._dataset_cbs: list[QCheckBox] = []
        self._dataset_ids_by_cb: dict[QCheckBox, str] = {}
        self._dataset_status_by_cb: dict[QCheckBox, _DatasetStatus] = {}
        self._dataset_row_by_cb: dict[QCheckBox, QWidget] = {}
        for ds_status in datasets:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)

            cb = QCheckBox(ds_status.display_name)
            cb.setToolTip(ds_status.dataset_id)
            cb.setChecked(False)
            cb.stateChanged.connect(self._on_child_changed)
            row_layout.addWidget(cb, 1)
            row_layout.addWidget(_make_completion_badge(ds_status.is_completed, "Dataset status", row))

            incomplete_sessions_warning = _incomplete_session_warning(ds_status)
            if incomplete_sessions_warning is not None:
                row_layout.addWidget(incomplete_sessions_warning)

            if ds_status.is_excluded:
                excluded_lbl = QLabel("Excluded")
                excluded_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                excluded_lbl.setMinimumWidth(70)
                excluded_lbl.setStyleSheet("QLabel { border: 1px solid #c7c7c7; border-radius: 7px; padding: 1px 7px; color: #555555; }")
                excluded_lbl.setToolTip("This dataset is marked as excluded in the experiment metadata")
                row_layout.addWidget(excluded_lbl)

            children_layout.addWidget(row)
            self._dataset_cbs.append(cb)
            self._dataset_ids_by_cb[cb] = ds_status.dataset_id
            self._dataset_status_by_cb[cb] = ds_status
            self._dataset_row_by_cb[cb] = row

        self._children_widget.setVisible(False)
        outer.addWidget(self._children_widget)

        # In dataset mode the children are meaningful; in experiment mode hide them
        self._dataset_mode = True
        self._completed_only = False

    # ── internal slots ────────────────────────────────────────────────────

    def _on_toggle(self, checked: bool) -> None:
        self._toggle_btn.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
        if self._dataset_mode:
            self._children_widget.setVisible(checked)

    def _on_expt_checked(self, state: int) -> None:
        checked = state == Qt.CheckState.Checked.value
        if self._dataset_mode:
            for cb in self._visible_dataset_cbs():
                cb.blockSignals(True)
                cb.setChecked(checked)
                cb.blockSignals(False)

    def _on_child_changed(self) -> None:
        """Update experiment-level checkbox based on children."""
        visible_cbs = self._visible_dataset_cbs()
        states = [cb.isChecked() for cb in visible_cbs]
        self._expt_cb.blockSignals(True)
        if states and all(states):
            self._expt_cb.setCheckState(Qt.CheckState.Checked)
        elif any(states):
            self._expt_cb.setCheckState(Qt.CheckState.PartiallyChecked)
        else:
            self._expt_cb.setCheckState(Qt.CheckState.Unchecked)
        self._expt_cb.blockSignals(False)

    # ── public API ────────────────────────────────────────────────────────

    def set_dataset_mode(self, enabled: bool) -> None:
        """Switch between dataset-level (children shown) and experiment-level."""
        self._dataset_mode = enabled
        if not enabled:
            self._children_widget.setVisible(False)
            self._toggle_btn.setChecked(False)
            self._toggle_btn.setArrowType(Qt.ArrowType.RightArrow)
            self._toggle_btn.setEnabled(False)
            # Experiment checkbox controls whether this experiment is exported
            self._expt_cb.setTristate(False)
        else:
            self._toggle_btn.setEnabled(True)
            # Re-sync tristate to child state
            self._on_child_changed()

    @property
    def is_expt_checked(self) -> bool:
        return self._expt_cb.checkState() != Qt.CheckState.Unchecked

    def set_completed_only(self, enabled: bool) -> None:
        """Hide incomplete data at every selectable level and clear its selection."""
        self._completed_only = enabled
        experiment_is_visible = not enabled or self.experiment_completed is True
        self.setVisible(experiment_is_visible)
        if not experiment_is_visible:
            self._expt_cb.blockSignals(True)
            self._expt_cb.setChecked(False)
            self._expt_cb.blockSignals(False)
        for cb in self._dataset_cbs:
            status = self._dataset_status_by_cb[cb]
            is_visible = experiment_is_visible and (not enabled or status.is_completed is True)
            row = self._dataset_row_by_cb[cb]
            row.setVisible(is_visible)
            if not is_visible and cb.isChecked():
                cb.blockSignals(True)
                cb.setChecked(False)
                cb.blockSignals(False)
        self._on_child_changed()

    def _visible_dataset_cbs(self) -> list[QCheckBox]:
        if not self._completed_only:
            return list(self._dataset_cbs)
        return [cb for cb in self._dataset_cbs if self._dataset_status_by_cb[cb].is_completed is True]

    @property
    def selected_dataset_ids(self) -> list[str]:
        """Return selected dataset IDs (only meaningful in dataset mode)."""
        if not self._dataset_mode:
            return []
        return [self._dataset_ids_by_cb[cb] for cb in self._visible_dataset_cbs() if cb.isChecked()]


# ─────────────────────────────────────────────────────────────────────────────
# Main dialog
# ─────────────────────────────────────────────────────────────────────────────


class BulkExportDialog(QDialog):
    """Multi-section configuration dialog for bulk data export."""

    _NARROW_LAYOUT_MAX_WIDTH = 719

    def __init__(self, gui: MonstimGUI, parent: QWidget | None = None):
        super().__init__(parent or gui)
        self.gui = gui
        self.setWindowTitle("Bulk Data Export")
        self.setMinimumSize(480, 320)
        self.resize(960, 640)

        self._expt_groups: list[_ExperimentGroup] = []
        self._layout_mode: str | None = None
        self._layout_update_pending = False
        self._options_pane_collapsed = False

        self._build_ui()
        self._populate_object_tree()
        self._populate_channels()
        self._set_default_method()
        self._refresh_readiness()
        self._update_responsive_layout()

    # ─────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # ── Output directory ─────────────────────────────────────────────
        path_box = QGroupBox("Output Directory")
        path_layout = QHBoxLayout(path_box)
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Select an output folder…")
        default_out = str(getattr(self.gui, "export_path", "") or "")
        if default_out:
            self._path_edit.setText(default_out)
        self._path_edit.textChanged.connect(self._refresh_readiness)
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse_output)
        self._toggle_options_btn = QToolButton()
        self._toggle_options_btn.setText("Hide Options")
        self._toggle_options_btn.setToolTip("Show or hide the export options pane")
        self._toggle_options_btn.clicked.connect(self._toggle_options_pane)
        path_layout.addWidget(self._path_edit, 1)
        path_layout.addWidget(browse_btn)
        path_layout.addWidget(self._toggle_options_btn)
        root.addWidget(path_box)

        # ── Selection pane ───────────────────────────────────────────────
        self._selection_pane = QWidget()
        selection_layout = QVBoxLayout(self._selection_pane)
        selection_layout.setContentsMargins(0, 0, 0, 0)
        selection_layout.setSpacing(8)

        level_box = QGroupBox("Data Export Level")
        level_layout = QHBoxLayout(level_box)
        self._rb_dataset = QRadioButton("Dataset")
        self._rb_experiment = QRadioButton("Experiment")
        self._rb_dataset.setChecked(True)
        self._level_group = QButtonGroup(self)
        self._level_group.addButton(self._rb_dataset)
        self._level_group.addButton(self._rb_experiment)
        level_layout.addWidget(self._rb_dataset)
        level_layout.addWidget(self._rb_experiment)
        level_layout.addStretch()
        self._rb_dataset.toggled.connect(self._on_level_changed)
        self._rb_dataset.toggled.connect(self._refresh_readiness)
        selection_layout.addWidget(level_box)

        obj_box = QGroupBox("Select Objects to Export")
        obj_box_layout = QVBoxLayout(obj_box)
        obj_box_layout.setContentsMargins(6, 6, 6, 6)

        # Select-all / deselect-all toolbar
        sel_row = QWidget()
        sel_layout = QHBoxLayout(sel_row)
        sel_layout.setContentsMargins(0, 0, 0, 0)
        sel_btn_all = QPushButton("Select All")
        sel_btn_none = QPushButton("Deselect All")
        sel_btn_all.setFixedHeight(22)
        sel_btn_none.setFixedHeight(22)
        sel_btn_all.clicked.connect(lambda: self._set_all_objects(True))
        sel_btn_none.clicked.connect(lambda: self._set_all_objects(False))
        sel_layout.addWidget(sel_btn_all)
        sel_layout.addWidget(sel_btn_none)
        self._cb_completed_only = QCheckBox("Completed data only")
        self._cb_completed_only.setToolTip(
            "Export only data explicitly marked Complete at every level: experiments, datasets, and sessions. "
            "Incomplete or unknown experiment cards and dataset rows are hidden. Dataset warnings identify active sessions that would be omitted."
        )
        self._cb_completed_only.toggled.connect(self._on_completed_only_changed)
        self._cb_completed_only.toggled.connect(self._refresh_readiness)
        sel_layout.addWidget(self._cb_completed_only)
        sel_layout.addStretch()
        obj_box_layout.addWidget(sel_row)

        # Scroll area holding the experiment groups
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setMinimumHeight(200)
        self._tree_container = QWidget()
        self._tree_layout = QVBoxLayout(self._tree_container)
        self._tree_layout.setContentsMargins(2, 2, 2, 2)
        self._tree_layout.setSpacing(1)
        self._tree_layout.addStretch()  # placeholder; groups inserted before this
        self._scroll_area.setWidget(self._tree_container)
        obj_box_layout.addWidget(self._scroll_area)
        selection_layout.addWidget(obj_box, 1)

        # ── Export options pane ──────────────────────────────────────────
        self._options_pane = QScrollArea()
        self._options_pane.setWidgetResizable(True)
        self._options_pane.setFrameShape(QScrollArea.Shape.NoFrame)
        options_content = QWidget()
        options_layout = QVBoxLayout(options_content)
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_layout.setSpacing(8)

        dtype_box = QGroupBox("Data Types")
        dtype_layout = QVBoxLayout(dtype_box)
        self._dtype_cbs: dict[str, QCheckBox] = {}
        for key, label in DATA_TYPE_LABELS.items():
            cb = QCheckBox(label)
            cb.setChecked(False)
            cb.toggled.connect(self._refresh_readiness)
            dtype_layout.addWidget(cb)
            self._dtype_cbs[key] = cb
        options_layout.addWidget(dtype_box)

        method_box = QGroupBox("Calculation Methods")
        method_layout = QVBoxLayout(method_box)
        self._method_cbs: dict[str, QCheckBox] = {}
        for key, label in METHOD_LABELS.items():
            cb = QCheckBox(label)
            cb.setChecked(False)
            cb.toggled.connect(self._refresh_readiness)
            method_layout.addWidget(cb)
            self._method_cbs[key] = cb
        options_layout.addWidget(method_box)

        opts_box = QGroupBox("Export Options")
        opts_layout = QVBoxLayout(opts_box)
        self._cb_normalize_mmax = QCheckBox("Normalize amplitudes to M-max")
        self._cb_normalize_mmax.setChecked(False)
        self._cb_normalize_mmax.setToolTip(
            "Adds *_norm_mmax_* columns alongside raw amplitude columns in the "
            "Avg Reflex Curves, Max H-Reflex, and Longform Reflex Amplitudes sheets.\n"
            "⚠ Requires M-max latency windows to be defined for all selected objects."
        )
        opts_layout.addWidget(self._cb_normalize_mmax)
        options_layout.addWidget(opts_box)

        chan_box = QGroupBox("Channels")
        self._chan_layout = QVBoxLayout(chan_box)
        self._channel_cbs: list[QCheckBox] = []  # populated in _populate_channels
        options_layout.addWidget(chan_box)
        options_layout.addStretch()
        self._options_pane.setWidget(options_content)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setChildrenCollapsible(True)
        self._splitter.setHandleWidth(6)
        self._splitter.addWidget(self._selection_pane)
        self._splitter.addWidget(self._options_pane)
        self._splitter.setStretchFactor(0, 3)
        self._splitter.setStretchFactor(1, 2)
        self._splitter.setSizes([570, 350])
        root.addWidget(self._splitter, 1)

        self._tabs = QTabWidget()
        self._tabs.setVisible(False)
        root.addWidget(self._tabs, 1)

        self._readiness_lbl = QLabel()
        self._readiness_lbl.setWordWrap(True)
        self._readiness_lbl.setStyleSheet("QLabel { color: #555555; }")
        root.addWidget(self._readiness_lbl)

        btn_box = QDialogButtonBox()
        self._export_btn = btn_box.addButton("Export", QDialogButtonBox.ButtonRole.AcceptRole)
        btn_box.addButton(QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        root.addWidget(btn_box)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "_splitter") and not self._layout_update_pending:
            # Reparenting visible widgets from resizeEvent corrupts Qt's active
            # layout pass on some platforms. Apply the mode change afterwards.
            self._layout_update_pending = True
            QTimer.singleShot(0, self._update_responsive_layout)

    def _update_responsive_layout(self) -> None:
        """Use tabs when two side-by-side panes no longer fit comfortably."""
        self._layout_update_pending = False
        mode = "tabs" if self.width() <= self._NARROW_LAYOUT_MAX_WIDTH else "splitter"
        if mode == self._layout_mode:
            return

        if self._layout_mode is None and mode == "splitter":
            self._layout_mode = mode
            return

        if mode == "tabs":
            self._splitter.hide()
            self._selection_pane.setParent(None)
            self._options_pane.setParent(None)
            self._tabs.addTab(self._selection_pane, "Selection")
            self._tabs.addTab(self._options_pane, "Export Options")
            self._tabs.show()
            self._toggle_options_btn.hide()
        else:
            selected_tab = self._tabs.currentIndex()
            self._tabs.hide()
            self._tabs.removeTab(self._tabs.indexOf(self._selection_pane))
            self._tabs.removeTab(self._tabs.indexOf(self._options_pane))
            self._splitter.addWidget(self._selection_pane)
            self._splitter.addWidget(self._options_pane)
            # A non-active QTabWidget page is hidden. Reset both child
            # visibilities before exposing the splitter again.
            self._selection_pane.show()
            self._options_pane.show()
            self._splitter.show()
            self._toggle_options_btn.show()
            self._set_options_pane_visibility()
            if selected_tab == 1 and not self._options_pane_collapsed:
                self._options_pane.setFocus()
        self._layout_mode = mode

    def _toggle_options_pane(self) -> None:
        """Collapse or restore the options pane without changing its state."""
        if self._layout_mode != "splitter":
            return
        self._options_pane_collapsed = not self._options_pane_collapsed
        self._set_options_pane_visibility()

    def _set_options_pane_visibility(self) -> None:
        """Apply the stored options-pane state after the active layout is stable."""
        visible = not self._options_pane_collapsed
        self._options_pane.setVisible(visible)
        self._toggle_options_btn.setText("Hide Options" if visible else "Show Options")
        if not visible:
            self._splitter.setSizes([max(self.width(), 1), 0])
        else:
            self._splitter.setSizes([570, 350])

    def _refresh_readiness(self) -> None:
        """Provide immediate, non-blocking feedback about export requirements."""
        dataset_mode = self._rb_dataset.isChecked()
        selected_count = 0
        for group in self._expt_groups:
            if not group.is_expt_checked:
                continue
            selected_count += len(group.selected_dataset_ids) if dataset_mode else 1
        data_types = sum(cb.isChecked() for cb in self._dtype_cbs.values())
        methods = sum(cb.isChecked() for cb in self._method_cbs.values())
        channels = sum(cb.isChecked() for cb in self._channel_cbs)
        missing = []
        if not selected_count:
            missing.append("select object(s)")
        if not data_types:
            missing.append("choose data type(s)")
        if not methods:
            missing.append("choose calculation method(s)")
        if not channels:
            missing.append("choose channel(s)")
        if not self._path_edit.text().strip():
            missing.append("choose an output directory")

        summary = f"{selected_count} object(s) selected · {data_types} data type(s) · {methods} method(s) · {channels} channel(s)"
        if missing:
            self._readiness_lbl.setText(f"Ready to export: {summary}. Still needed: {', '.join(missing)}.")
        else:
            self._readiness_lbl.setText(f"Ready to export: {summary}.")
        self._export_btn.setEnabled(not missing)

    # ─────────────────────────────────────────────────────────────────────
    # Population helpers
    # ─────────────────────────────────────────────────────────────────────

    def _populate_object_tree(self) -> None:
        """Populate experiment groups from gui.expts_dict."""
        expts_dict: dict[str, str] = getattr(self.gui, "expts_dict", {})

        # Remove any existing groups (before the trailing stretch)
        while self._tree_layout.count() > 1:
            item = self._tree_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        self._expt_groups.clear()

        for expt_name, expt_path_str in sorted(expts_dict.items()):
            status = self._discover_experiment_status(expt_path_str)
            group = _ExperimentGroup(expt_name, status.is_completed, status.datasets)
            group._expt_cb.stateChanged.connect(self._refresh_readiness)
            for dataset_cb in group._dataset_cbs:
                dataset_cb.stateChanged.connect(self._refresh_readiness)
            self._tree_layout.insertWidget(self._tree_layout.count() - 1, group)
            self._expt_groups.append(group)

        if not self._expt_groups:
            empty_lbl = QLabel("No experiments found. Import data first.")
            empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._tree_layout.insertWidget(0, empty_lbl)

    @staticmethod
    def _discover_experiment_status(expt_path_str: str) -> _ExperimentStatus:
        """Return lightweight experiment/dataset completion status for the chooser."""
        try:
            folder = Path(expt_path_str)
            from monstim_signals.io.repositories import ExperimentRepository

            metadata = ExperimentRepository(folder).get_metadata()
            excluded_ids = set(metadata.get("excluded_datasets") or [])
            dataset_statuses = []
            for ds_meta in metadata.get("datasets") or []:
                ds_id = str(ds_meta.get("id") or "")
                if not ds_id:
                    continue
                dataset_statuses.append(
                    _DatasetStatus(
                        dataset_id=ds_id,
                        display_name=str(ds_meta.get("formatted_name") or ds_id),
                        is_completed=bool(ds_meta.get("is_completed", False)),
                        is_excluded=ds_id in excluded_ids,
                        incomplete_active_session_ids=tuple(ds_meta.get("incomplete_active_session_ids") or ()),
                    )
                )

            if not dataset_statuses:
                dataset_statuses = [
                    _DatasetStatus(dataset_id=p.name, display_name=p.name, is_completed=None) for p in sorted(folder.iterdir()) if p.is_dir()
                ]

            return _ExperimentStatus(
                is_completed=bool(metadata.get("is_completed", False)),
                datasets=sorted(dataset_statuses, key=lambda ds: ds.display_name.casefold()),
            )
        except Exception:
            logger.exception("Could not read completion status for experiment path %s", expt_path_str)
            try:
                folder = Path(expt_path_str)
                datasets = [_DatasetStatus(dataset_id=p.name, display_name=p.name, is_completed=None) for p in sorted(folder.iterdir()) if p.is_dir()]
            except Exception:
                logger.exception("Could not read datasets for experiment path %s", expt_path_str)
                datasets = []
            return _ExperimentStatus(is_completed=None, datasets=datasets)

    def _populate_channels(self) -> None:
        """Add per-channel checkboxes from the currently loaded experiment."""
        # Clear any existing checkboxes
        while self._chan_layout.count():
            item = self._chan_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        self._channel_cbs.clear()

        # Try to get channel names from the loaded experiment, fall back to config
        channel_names: list[str] = []
        try:
            ch = getattr(self.gui, "channel_names", [])
            if ch:
                channel_names = list(ch)
        except Exception:
            logger.exception("Could not read channel names from GUI; falling back to config.")

        if not channel_names:
            try:
                from monstim_signals.core import load_config

                channel_names = load_config().get("default_channel_names", [])
            except Exception:
                channel_names = []
                logger.exception("Could not read default channel names from config; falling back to Ch0.")

        if not channel_names:
            channel_names = ["Ch0"]

        for i in range(len(channel_names)):
            cb = QCheckBox(f"Ch{i}")
            cb.setChecked(False)
            cb.toggled.connect(self._refresh_readiness)
            self._chan_layout.addWidget(cb)
            self._channel_cbs.append(cb)
        self._chan_layout.addStretch()

    def _set_default_method(self) -> None:
        """Pre-select the default analysis method from the current experiment."""
        try:
            default = None
            if self.gui.current_session:
                default = self.gui.current_session.default_method
            elif self.gui.current_dataset:
                default = self.gui.current_dataset.default_method
            elif self.gui.current_experiment:
                default = self.gui.current_experiment.default_method
            if default and default in self._method_cbs:
                # Ensure the default is checked; leave all others as-is
                self._method_cbs[default].setChecked(True)
        except Exception:
            logger.exception("Could not determine default method from GUI; leaving all methods unchecked.")

    # ─────────────────────────────────────────────────────────────────────
    # Slots
    # ─────────────────────────────────────────────────────────────────────

    def _on_level_changed(self, dataset_mode_active: bool) -> None:
        for group in self._expt_groups:
            group.set_dataset_mode(dataset_mode_active)
            group.set_completed_only(self._cb_completed_only.isChecked())
        self._refresh_readiness()

    def _set_all_objects(self, checked: bool) -> None:
        for group in self._expt_groups:
            if not group.isHidden():
                group._expt_cb.setChecked(checked)
        self._refresh_readiness()

    def _on_completed_only_changed(self, checked: bool) -> None:
        for group in self._expt_groups:
            group.set_completed_only(checked)
        self._refresh_readiness()

    def _browse_output(self) -> None:
        current = self._path_edit.text().strip() or str(getattr(self.gui, "export_path", ""))
        chosen = QFileDialog.getExistingDirectory(self, "Select Output Directory", current)
        if chosen:
            self._path_edit.setText(chosen)

    def _on_accept(self) -> None:
        """Validate selections, build config, launch worker."""
        config = self._build_config()
        if config is None:
            return  # validation failed

        self.hide()

        total = sum(max(len(v), 1) for v in config.selected_objects.values())
        progress_win = BulkExportProgressWindow(total, parent=self.parent() or self)
        progress_win.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress_win.show()

        worker = BulkExportWorker(config, parent=self)
        progress_win.canceled.connect(worker.cancel)

        def _on_progress(cur: int, tot: int, msg: str) -> None:
            progress_win.update_progress(cur, tot, msg)

        def _on_finished(written: list[str]) -> None:
            progress_win.mark_done()
            worker.deleteLater()
            if worker.is_canceled:
                QMessageBox.information(
                    progress_win,
                    "Bulk Export Canceled",
                    f"Export canceled. {len(written)} fully written file(s) were kept; no incomplete workbook was saved.",
                )
            elif not written:
                QMessageBox.warning(
                    progress_win,
                    "Bulk Export",
                    "Export completed but no files were written.\nCheck the application log for details.",
                )
            else:
                msg = f"Export complete.\n\nWritten {len(written)} file(s) to:\n{config.output_path}"
                box = QMessageBox(progress_win)
                box.setWindowTitle("Bulk Export Complete")
                box.setText(msg)
                box.setIcon(QMessageBox.Icon.Information)
                open_btn = box.addButton("Open Folder", QMessageBox.ButtonRole.ActionRole)
                box.addButton(QMessageBox.StandardButton.Ok)
                box.exec()
                if box.clickedButton() is open_btn:
                    from PySide6.QtCore import QUrl
                    from PySide6.QtGui import QDesktopServices

                    QDesktopServices.openUrl(QUrl.fromLocalFile(config.output_path))
            progress_win.accept()
            self.accept()

        def _on_error(msg: str) -> None:
            progress_win.mark_done()
            worker.deleteLater()
            QMessageBox.critical(
                progress_win,
                "Bulk Export Error",
                f"An unexpected error occurred during export:\n\n{msg}",
            )
            progress_win.accept()
            self.show()

        worker.progress.connect(_on_progress)
        worker.finished_export.connect(_on_finished)
        worker.error.connect(_on_error)
        worker.start()

    # ─────────────────────────────────────────────────────────────────────
    # Config collection and validation
    # ─────────────────────────────────────────────────────────────────────

    def _build_config(self) -> BulkExportConfig | None:
        """Collect UI state into a :class:`BulkExportConfig`; return None if invalid."""
        data_level = "dataset" if self._rb_dataset.isChecked() else "experiment"

        # Selected objects
        selected_objects: dict[str, list[str]] = {}
        for group in self._expt_groups:
            if not group.is_expt_checked:
                continue
            if data_level == "dataset":
                ds_ids = group.selected_dataset_ids
                if ds_ids:
                    selected_objects[group.expt_name] = ds_ids
            else:
                selected_objects[group.expt_name] = []

        if not selected_objects:
            QMessageBox.warning(self, "Validation", "Please select at least one object to export.")
            return None

        # Data types
        data_types = [k for k, cb in self._dtype_cbs.items() if cb.isChecked()]
        if not data_types:
            QMessageBox.warning(self, "Validation", "Please select at least one data type.")
            return None

        # Methods
        methods = [k for k, cb in self._method_cbs.items() if cb.isChecked()]
        if not methods:
            QMessageBox.warning(self, "Validation", "Please select at least one calculation method.")
            return None

        # Channels
        channel_indices = [i for i, cb in enumerate(self._channel_cbs) if cb.isChecked()]
        if not channel_indices:
            QMessageBox.warning(self, "Validation", "Please select at least one channel.")
            return None

        # Output path
        output_path = self._path_edit.text().strip()
        if not output_path:
            QMessageBox.warning(self, "Validation", "Please specify an output directory.")
            return None
        if not os.path.isdir(output_path):
            reply = QMessageBox.question(
                self,
                "Create Directory?",
                f"The directory does not exist:\n{output_path}\n\nCreate it now?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                try:
                    os.makedirs(output_path, exist_ok=True)
                except Exception as exc:
                    logger.exception("Could not create directory")
                    QMessageBox.critical(self, "Error", f"Could not create directory:\n{exc}")
                    return None
            else:
                return None

        expts_dict: dict[str, str] = getattr(self.gui, "expts_dict", {})
        normalize_to_mmax = self._cb_normalize_mmax.isChecked()

        return BulkExportConfig(
            data_level=data_level,
            selected_objects=selected_objects,
            data_types=data_types,
            methods=methods,
            channel_indices=channel_indices,
            output_path=output_path,
            normalize_to_mmax=normalize_to_mmax,
            completed_only=self._cb_completed_only.isChecked(),
            experiment_paths={name: str(expts_dict.get(name, "")) for name in selected_objects},
        )
