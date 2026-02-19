# monstim_gui/dialogs/bulk_export_dialog.py
"""
BulkExportDialog – wizard-style dialog for the Bulk Data Export feature.

The dialog collects:
  - Data level    : Dataset or Experiment
  - Objects       : hierarchical collapsible experiment / dataset checkboxes
  - Data types    : Average Reflex Curves, M-max, Max H-reflex
  - Methods       : rms, auc, peak_to_trough, average_rectified, average_unrectified
  - Plot options  : Normalize to M-max
  - Channels      : per-channel checkboxes
  - Output path   : directory chooser

On acceptance a :class:`BulkExportWorker` QThread is launched; progress is
shown via a QProgressDialog.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QThread, Signal
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
    QSpinBox,
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
        self._canceled = False

    def cancel(self) -> None:
        self._canceled = True

    def run(self) -> None:
        try:
            written = run_bulk_export(
                self._config,
                progress_callback=lambda cur, tot, msg: self.progress.emit(cur, tot, msg),
                is_canceled=lambda: self._canceled,
            )
            self.finished_export.emit(written)
        except Exception as exc:
            logging.exception("BulkExportWorker encountered an unexpected error.")
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
        self.setWindowTitle("Bulk Export – In Progress")
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
        ts = datetime.now().strftime("%H:%M:%S")
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
            pass
        self._cancel_btn.clicked.connect(self.accept)

    # ── cancel / close ────────────────────────────────────────────────────

    def _on_cancel(self) -> None:
        if self._done:
            self.accept()
            return
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.setText("Canceling\u2026")
        ts = datetime.now().strftime("%H:%M:%S")
        self._log.appendPlainText(f"[{ts}]  Cancellation requested\u2026")
        self.canceled.emit()

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._done:
            event.accept()
        else:
            self._on_cancel()
            event.ignore()  # keep open until worker finishes


# ─────────────────────────────────────────────────────────────────────────────
# Helper widget: Collapsible experiment group with checkbox children
# ─────────────────────────────────────────────────────────────────────────────


class _ExperimentGroup(QWidget):
    """A collapsible card showing one experiment with its dataset checkboxes."""

    def __init__(self, expt_name: str, dataset_ids: list[str], parent: QWidget | None = None):
        super().__init__(parent)
        self.expt_name = expt_name
        self.dataset_ids = dataset_ids

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
        self._expt_cb.stateChanged.connect(self._on_expt_checked)
        header_layout.addWidget(self._expt_cb, 1)

        outer.addWidget(header_row)

        # ── children container (datasets) ─────────────────────────────────
        self._children_widget = QWidget()
        children_layout = QVBoxLayout(self._children_widget)
        children_layout.setContentsMargins(28, 0, 0, 4)
        children_layout.setSpacing(2)

        self._dataset_cbs: list[QCheckBox] = []
        for ds_id in dataset_ids:
            cb = QCheckBox(ds_id)
            cb.setChecked(False)
            cb.stateChanged.connect(self._on_child_changed)
            children_layout.addWidget(cb)
            self._dataset_cbs.append(cb)

        self._children_widget.setVisible(False)
        outer.addWidget(self._children_widget)

        # In dataset mode the children are meaningful; in experiment mode hide them
        self._dataset_mode = True

    # ── internal slots ────────────────────────────────────────────────────

    def _on_toggle(self, checked: bool) -> None:
        self._toggle_btn.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
        if self._dataset_mode:
            self._children_widget.setVisible(checked)

    def _on_expt_checked(self, state: int) -> None:
        checked = state == Qt.CheckState.Checked.value
        if self._dataset_mode:
            for cb in self._dataset_cbs:
                cb.blockSignals(True)
                cb.setChecked(checked)
                cb.blockSignals(False)

    def _on_child_changed(self) -> None:
        """Update experiment-level checkbox based on children."""
        states = [cb.isChecked() for cb in self._dataset_cbs]
        self._expt_cb.blockSignals(True)
        if all(states):
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

    @property
    def selected_dataset_ids(self) -> list[str]:
        """Return selected dataset IDs (only meaningful in dataset mode)."""
        if not self._dataset_mode:
            return []
        return [cb.text() for cb in self._dataset_cbs if cb.isChecked()]


# ─────────────────────────────────────────────────────────────────────────────
# Main dialog
# ─────────────────────────────────────────────────────────────────────────────


class BulkExportDialog(QDialog):
    """Multi-section configuration dialog for bulk data export."""

    def __init__(self, gui: "MonstimGUI", parent: QWidget | None = None):
        super().__init__(parent or gui)
        self.gui = gui
        self.setWindowTitle("Bulk Data Export")
        self.setMinimumWidth(520)
        self.setMinimumHeight(640)

        self._expt_groups: list[_ExperimentGroup] = []

        self._build_ui()
        self._populate_object_tree()
        self._populate_channels()
        self._set_default_method()

    # ─────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # ── 1. Data level ─────────────────────────────────────────────────
        level_box = QGroupBox("Data Level")
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
        root.addWidget(level_box)

        # ── 2. Object selection tree ──────────────────────────────────────
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
        root.addWidget(obj_box)

        # ── 3. Data types ─────────────────────────────────────────────────
        dtype_box = QGroupBox("Data Types")
        dtype_layout = QHBoxLayout(dtype_box)
        self._dtype_cbs: dict[str, QCheckBox] = {}
        for key, label in DATA_TYPE_LABELS.items():
            cb = QCheckBox(label)
            cb.setChecked(False)
            dtype_layout.addWidget(cb)
            self._dtype_cbs[key] = cb
        dtype_layout.addStretch()
        root.addWidget(dtype_box)

        # ── 4. Methods ────────────────────────────────────────────────────
        method_box = QGroupBox("Calculation Methods")
        method_layout = QHBoxLayout(method_box)
        self._method_cbs: dict[str, QCheckBox] = {}
        for key, label in METHOD_LABELS.items():
            cb = QCheckBox(label)
            cb.setChecked(False)
            method_layout.addWidget(cb)
            self._method_cbs[key] = cb
        method_layout.addStretch()
        root.addWidget(method_box)

        # ── 5. Plot Options ───────────────────────────────────────────────
        opts_box = QGroupBox("Plot Options")
        opts_layout = QVBoxLayout(opts_box)
        self._cb_normalize_mmax = QCheckBox("Normalize amplitudes to M-max")
        self._cb_normalize_mmax.setChecked(False)
        self._cb_normalize_mmax.setToolTip(
            "Adds *_norm_mmax_* columns alongside raw amplitude columns in the "
            "Avg Reflex Curves and Max H-Reflex sheets.\n"
            "⚠ Requires M-max latency windows to be defined for all selected objects."
        )
        opts_layout.addWidget(self._cb_normalize_mmax)
        workers_row = QWidget()
        workers_layout = QHBoxLayout(workers_row)
        workers_layout.setContentsMargins(0, 2, 0, 0)
        workers_layout.setSpacing(6)
        workers_layout.addWidget(QLabel("Parallel workers:"))
        self._sb_workers = QSpinBox()
        self._sb_workers.setRange(1, 16)
        self._sb_workers.setValue(1)
        self._sb_workers.setFixedWidth(60)
        self._sb_workers.setToolTip(
            "Number of datasets to load and process simultaneously.\n"
            "Values > 1 can significantly speed up large exports but use more RAM.\n"
            "Applies to dataset-level exports only."
        )
        workers_layout.addWidget(self._sb_workers)
        workers_layout.addStretch()
        opts_layout.addWidget(workers_row)
        root.addWidget(opts_box)

        # ── 6. Channels ───────────────────────────────────────────────────
        chan_box = QGroupBox("Channels")
        self._chan_layout = QHBoxLayout(chan_box)
        self._channel_cbs: list[QCheckBox] = []  # populated in _populate_channels
        root.addWidget(chan_box)

        # ── 7. Output path ───────────────────────────────────────────────
        path_box = QGroupBox("Output Directory")
        path_layout = QHBoxLayout(path_box)
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Select an output folder…")
        default_out = str(getattr(self.gui, "output_path", "") or "")
        if default_out:
            self._path_edit.setText(default_out)
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse_output)
        path_layout.addWidget(self._path_edit, 1)
        path_layout.addWidget(browse_btn)
        root.addWidget(path_box)

        # ── 8. Button box ───────────────────────────────────────────────
        btn_box = QDialogButtonBox()
        self._export_btn = btn_box.addButton("Export", QDialogButtonBox.ButtonRole.AcceptRole)
        btn_box.addButton(QDialogButtonBox.StandardButton.Cancel)
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        root.addWidget(btn_box)

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
            dataset_ids = self._discover_dataset_ids(expt_path_str)
            group = _ExperimentGroup(expt_name, dataset_ids)
            self._tree_layout.insertWidget(self._tree_layout.count() - 1, group)
            self._expt_groups.append(group)

        if not self._expt_groups:
            empty_lbl = QLabel("No experiments found. Import data first.")
            empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._tree_layout.insertWidget(0, empty_lbl)

    @staticmethod
    def _discover_dataset_ids(expt_path_str: str) -> list[str]:
        """Return sorted dataset folder names inside an experiment directory."""
        try:
            folder = Path(expt_path_str)
            return sorted(p.name for p in folder.iterdir() if p.is_dir())
        except Exception:
            return []

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
            pass

        if not channel_names:
            try:
                from monstim_signals.core import load_config

                channel_names = load_config().get("default_channel_names", [])
            except Exception:
                channel_names = []

        if not channel_names:
            channel_names = ["Ch0"]

        for i in range(len(channel_names)):
            cb = QCheckBox(f"Ch{i}")
            cb.setChecked(False)
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
            pass

    # ─────────────────────────────────────────────────────────────────────
    # Slots
    # ─────────────────────────────────────────────────────────────────────

    def _on_level_changed(self, dataset_mode_active: bool) -> None:
        for group in self._expt_groups:
            group.set_dataset_mode(dataset_mode_active)

    def _set_all_objects(self, checked: bool) -> None:
        for group in self._expt_groups:
            group._expt_cb.setChecked(checked)

    def _browse_output(self) -> None:
        current = self._path_edit.text().strip() or str(getattr(self.gui, "output_path", ""))
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
            if not written:
                QMessageBox.warning(
                    progress_win,
                    "Bulk Export",
                    "Export completed but no files were written.\n" "Check the application log for details.",
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
                    QMessageBox.critical(self, "Error", f"Could not create directory:\n{exc}")
                    return None
            else:
                return None

        expts_dict: dict[str, str] = getattr(self.gui, "expts_dict", {})
        normalize_to_mmax = self._cb_normalize_mmax.isChecked()
        max_workers = self._sb_workers.value()

        return BulkExportConfig(
            data_level=data_level,
            selected_objects=selected_objects,
            data_types=data_types,
            methods=methods,
            channel_indices=channel_indices,
            output_path=output_path,
            normalize_to_mmax=normalize_to_mmax,
            max_workers=max_workers,
            experiment_paths={name: str(expts_dict.get(name, "")) for name in selected_objects},
        )
