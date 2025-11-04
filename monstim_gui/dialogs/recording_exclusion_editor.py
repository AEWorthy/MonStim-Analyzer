"""
Recording Exclusion Editor Dialog
Allows users to exclude recordings based on various criteria like stimulus amplitude.
Designed to be extensible for future criteria-based exclusion.
"""

import logging
from typing import TYPE_CHECKING, List, Set

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QIcon
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QCheckBox,
    QFileDialog,
    QWidget,
)

import json
from typing import Dict, Any

if TYPE_CHECKING:
    from monstim_gui.gui_main import MonstimGUI
    from monstim_signals.domain import Session


class RecordingExclusionEditor(QDialog):
    """
    Extensible dialog for excluding recordings based on various criteria.
    Currently supports stimulus amplitude thresholds, designed for future expansion.
    """

    exclusions_applied = pyqtSignal()  # Signal emitted when exclusions are applied

    def __init__(self, parent: "MonstimGUI"):
        super().__init__(parent)
        self.gui = parent
        self.current_session = parent.current_session
        self.current_dataset = parent.current_dataset
        self.current_experiment = parent.current_experiment

        # Store original exclusion state for cancel/reset functionality
        self.original_excluded_recordings = (
            set(self.current_session.excluded_recordings.copy()) if self.current_session else set()
        )

        # Track preview exclusions (not yet applied)
        self.preview_excluded_recordings: Set[str] = set()

        self.setup_ui()
        self.load_data()

    # TODO: Exclusion editor enhancements
    # - Add thumbnail/sparkline previews for each recording in the preview table so users
    #   can get a quick visual cue when deciding to exclude recordings.
    # - Allow multi-row selection in the preview table with a "Toggle Exclusion" button
    #   to manually override the automatic criteria. This should integrate with the
    #   Command pattern (BulkRecordingExclusionCommand) so actions are undoable.
    # - Implement additional exclusion criteria (SNR, baseline drift, flatline, line-noise
    #   50/60Hz content, artifact detection) and an "Auto-flag low quality" quick action.
    # - Provide Save/Load exclusion profile support so users can persist criteria sets
    #   and reapply them across experiments.

    def setup_ui(self):
        """Set up the dialog UI."""
        self.setWindowTitle("Recording Exclusion Editor")
        self.setModal(True)
        self.resize(800, 600)

        # Main layout
        main_layout = QVBoxLayout(self)

        # Create splitter for criteria and preview
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # Left side: Exclusion criteria tabs
        criteria_widget = self.create_criteria_widget()
        splitter.addWidget(criteria_widget)

        # Right side: Recording preview table
        preview_widget = self.create_preview_widget()
        splitter.addWidget(preview_widget)

        # Set splitter proportions
        splitter.setSizes([300, 500])

        # Button layout
        button_layout = QHBoxLayout()

        # Apply level selection
        self.level_combo = QComboBox()
        self.level_combo.addItem("Current Session Only", "session")
        if self.current_dataset:
            self.level_combo.addItem("Entire Dataset", "dataset")
        if self.current_experiment:
            self.level_combo.addItem("Entire Experiment", "experiment")

        button_layout.addWidget(QLabel("Apply to:"))
        button_layout.addWidget(self.level_combo)
        button_layout.addStretch()

        # Control buttons
        self.preview_button = QPushButton("Preview")
        self.preview_button.clicked.connect(self.update_preview)
        button_layout.addWidget(self.preview_button)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_criteria)
        button_layout.addWidget(self.reset_button)

        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_exclusions)
        button_layout.addWidget(self.apply_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)

        main_layout.addLayout(button_layout)

        # Connect level change to update preview
        self.level_combo.currentTextChanged.connect(self.update_preview)

    def create_criteria_widget(self) -> QWidget:
        """Create the criteria selection widget with tabs for extensibility."""
        criteria_widget = QWidget()
        layout = QVBoxLayout(criteria_widget)

        # Create tab widget for different types of criteria
        self.criteria_tabs = QTabWidget()
        layout.addWidget(self.criteria_tabs)

        # Add stimulus amplitude tab
        stimulus_tab = self.create_stimulus_amplitude_tab()
        self.criteria_tabs.addTab(stimulus_tab, "Stimulus Amplitude")

        # Future tabs can be added here:
        # - Recording quality metrics
        # - Channel-specific criteria
        # - Time-based criteria
        # - Custom user-defined criteria

        return criteria_widget

    def create_stimulus_amplitude_tab(self) -> QWidget:
        """Create the stimulus amplitude exclusion criteria tab."""
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)

        # Enable/disable group
        self.stimulus_group = QGroupBox("Exclude recordings by stimulus amplitude")
        self.stimulus_group.setCheckable(True)
        self.stimulus_group.setChecked(False)
        group_layout = QFormLayout(self.stimulus_group)

        # Threshold settings
        self.threshold_type_combo = QComboBox()
        self.threshold_type_combo.addItem("Above threshold", "above")
        self.threshold_type_combo.addItem("Below threshold", "below")
        self.threshold_type_combo.addItem("Outside range", "outside")
        self.threshold_type_combo.addItem("Inside range", "inside")
        group_layout.addRow("Exclude recordings:", self.threshold_type_combo)

        # Primary threshold
        self.threshold_spinbox = QDoubleSpinBox()
        self.threshold_spinbox.setRange(0.0, 100.0)
        self.threshold_spinbox.setSingleStep(0.1)
        self.threshold_spinbox.setSuffix(" V")
        self.threshold_spinbox.setDecimals(2)
        self.threshold_spinbox.setValue(1.0)
        group_layout.addRow("Threshold:", self.threshold_spinbox)

        # Secondary threshold (for range-based exclusions)
        self.threshold2_spinbox = QDoubleSpinBox()
        self.threshold2_spinbox.setRange(0.0, 100.0)
        self.threshold2_spinbox.setSingleStep(0.1)
        self.threshold2_spinbox.setSuffix(" V")
        self.threshold2_spinbox.setDecimals(2)
        self.threshold2_spinbox.setValue(5.0)
        self.threshold2_spinbox.setVisible(False)
        group_layout.addRow("Upper threshold:", self.threshold2_spinbox)

        # Show/hide secondary threshold based on type
        def update_threshold_visibility():
            is_range = self.threshold_type_combo.currentData() in ["outside", "inside"]
            self.threshold2_spinbox.setVisible(is_range)
            if is_range:
                group_layout.labelForField(self.threshold_spinbox).setText("Lower threshold:")
            else:
                group_layout.labelForField(self.threshold_spinbox).setText("Threshold:")

        self.threshold_type_combo.currentTextChanged.connect(update_threshold_visibility)

        # Connect changes to auto-preview
        self.stimulus_group.toggled.connect(self.update_preview)
        self.threshold_type_combo.currentTextChanged.connect(self.update_preview)
        self.threshold_spinbox.valueChanged.connect(self.update_preview)
        self.threshold2_spinbox.valueChanged.connect(self.update_preview)

        layout.addWidget(self.stimulus_group)
        layout.addStretch()

        return tab_widget

    def create_preview_widget(self) -> QWidget:
        """Create the recording preview table widget."""
        preview_widget = QWidget()
        layout = QVBoxLayout(preview_widget)

        # Header
        header_label = QLabel("Recording Preview")
        header_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        layout.addWidget(header_label)

        # Table for recordings (add Preview column)
        self.recordings_table = QTableWidget()
        # Columns: Preview, Recording ID, Session, Stimulus, Status
        self.recordings_table.setColumnCount(5)
        self.recordings_table.setHorizontalHeaderLabels(["Preview", "Recording ID", "Session", "Stimulus (V)", "Status"])

        # Configure table
        header = self.recordings_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)

        self.recordings_table.setAlternatingRowColors(True)
        self.recordings_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        layout.addWidget(self.recordings_table)

        # Summary label
        self.summary_label = QLabel()
        layout.addWidget(self.summary_label)

    # Toggle exclusion button for manual override (undoable)
    toggle_layout = QHBoxLayout()
    self.toggle_exclusion_button = QPushButton("Toggle Exclusion")
    self.toggle_exclusion_button.clicked.connect(self.toggle_selected_exclusions)
    toggle_layout.addWidget(self.toggle_exclusion_button)
    toggle_layout.addStretch()
    layout.addLayout(toggle_layout)

        return preview_widget

    def load_data(self):
        """Load initial data and populate preview."""
        if not self.current_session:
            QMessageBox.warning(self, "No Session", "No session is currently selected.")
            return

        self.update_preview()

    def get_sessions_for_level(self) -> List["Session"]:
        """Get list of sessions based on selected application level."""
        level = self.level_combo.currentData()

        if level == "session":
            return [self.current_session] if self.current_session else []
        elif level == "dataset":
            return self.current_dataset.sessions if self.current_dataset else []
        elif level == "experiment":
            if self.current_experiment:
                sessions = []
                for dataset in self.current_experiment.datasets:
                    sessions.extend(dataset.sessions)
                return sessions
            return []
        else:
            return []

    def should_exclude_recording(self, recording) -> bool:
        """
        Check if a recording should be excluded based on current criteria.
        This method is designed to be extensible for future criteria types.
        """
        # Check stimulus amplitude criteria
        if self.stimulus_group.isChecked():
            stimulus_value = recording.stim_amplitude
            threshold_type = self.threshold_type_combo.currentData()
            threshold1 = self.threshold_spinbox.value()
            threshold2 = self.threshold2_spinbox.value()

            if threshold_type == "above":
                return stimulus_value > threshold1
            elif threshold_type == "below":
                return stimulus_value < threshold1
            elif threshold_type == "outside":
                return stimulus_value < threshold1 or stimulus_value > threshold2
            elif threshold_type == "inside":
                return threshold1 <= stimulus_value <= threshold2

        # Future criteria can be added here
        # e.g., recording quality, signal-to-noise ratio, etc.

        return False

    def compute_quality_metrics(self, recording) -> Dict[str, Any]:
        """Compute simple quality metrics for a recording.

        Returns a dict with keys: snr, baseline_drift, flatline, line_noise. Values
        are numeric or None if computation unavailable.
        """
        # Attempt to robustly fetch waveform data from likely attributes
        waveform = None
        for attr in ("waveform", "signal", "data", "trace", "samples"):
            waveform = getattr(recording, attr, None)
            if waveform is not None:
                break

        # If waveform is callable (method) call it
        try:
            if callable(waveform):
                waveform = waveform()
        except Exception:
            waveform = None

        # Ensure waveform is a sequence of numbers
        try:
            import numpy as _np

            arr = _np.asarray(waveform) if waveform is not None else None
        except Exception:
            arr = None

        metrics = {"snr": None, "baseline_drift": None, "flatline": None, "line_noise": None}

        if arr is None or arr.size == 0:
            return metrics

        # Simple RMS-based SNR: estimate noise from first 10% of trace
        try:
            n = arr.size
            noise_seg = arr[: max(1, n // 10)]
            signal_seg = arr[n // 10 :]
            noise_rms = float((_np.mean(noise_seg ** 2)) ** 0.5)
            signal_rms = float((_np.mean(signal_seg ** 2)) ** 0.5)
            metrics["snr"] = signal_rms / (noise_rms + 1e-12)
        except Exception:
            metrics["snr"] = None

        # Baseline drift: absolute difference between median of first and last 10%
        try:
            first_med = float(_np.median(arr[: max(1, n // 10)]))
            last_med = float(_np.median(arr[-max(1, n // 10) :]))
            metrics["baseline_drift"] = abs(last_med - first_med)
        except Exception:
            metrics["baseline_drift"] = None

        # Flatline: low variance
        try:
            metrics["flatline"] = float(_np.std(arr))
        except Exception:
            metrics["flatline"] = None

        # Line noise detection: rudimentary via fft energy near 50/60 Hz
        try:
            fs = getattr(recording, "sampling_rate", None) or getattr(recording, "fs", None) or 1000.0
            freqs = _np.fft.rfftfreq(n, 1.0 / float(fs))
            fft = _np.abs(_np.fft.rfft(arr))
            # look for energy near 50 and 60 Hz
            def band_energy(target):
                mask = (freqs > (target - 1.0)) & (freqs < (target + 1.0))
                return float(_np.sum(fft[mask]))

            e50 = band_energy(50)
            e60 = band_energy(60)
            metrics["line_noise"] = max(e50, e60)
        except Exception:
            metrics["line_noise"] = None

        return metrics

    def generate_sparkline_icon(self, recording, width=80, height=24) -> QIcon:
        """Generate a small sparkline icon for a recording if waveform is available.

        Returns a QIcon. Falls back to a simple placeholder pixmap.
        """
        # Attempt to get waveform similar to compute_quality_metrics
        waveform = None
        for attr in ("waveform", "signal", "data", "trace", "samples"):
            waveform = getattr(recording, attr, None)
            if waveform is not None:
                break

        try:
            if callable(waveform):
                waveform = waveform()
        except Exception:
            waveform = None

        try:
            import numpy as _np

            arr = _np.asarray(waveform) if waveform is not None else None
        except Exception:
            arr = None

        pix = QPixmap(width, height)
        pix.fill(QColor("transparent"))
        painter = QPainter(pix)
        pen = QPen(QColor("#2b85d8"))
        pen.setWidth(1)
        painter.setPen(pen)

        if arr is None or arr.size == 0:
            # draw placeholder line
            painter.drawLine(2, height // 2, width - 2, height // 2)
            painter.end()
            return QIcon(pix)

        # downsample to width points
        try:
            n = arr.size
            if n <= width:
                ys = arr
            else:
                import numpy as _np

                idx = _np.linspace(0, n - 1, width).astype(int)
                ys = arr[idx]

            # normalize to [2, height-2]
            ymin, ymax = float(ys.min()), float(ys.max())
            rng = ymax - ymin if ymax != ymin else 1.0
            points = []
            for i, v in enumerate(ys):
                x = int(i * (width - 4) / max(1, len(ys) - 1)) + 2
                y = int((1.0 - (float(v) - ymin) / rng) * (height - 4)) + 2
                points.append((x, y))

            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                painter.drawLine(x1, y1, x2, y2)
        except Exception:
            painter.drawLine(2, height // 2, width - 2, height // 2)

        painter.end()
        return QIcon(pix)

    # TODO: Exclusion criteria extensibility
    # - When adding new criteria (SNR, RMS outliers, baseline drift), provide a
    #   unified evaluation pipeline that returns (bool, reason_dict) for preview so
    #   tooltips can explain why a recording was flagged.

    def update_preview(self):
        """Update the preview table based on current criteria."""
        sessions = self.get_sessions_for_level()
        if not sessions:
            self.recordings_table.setRowCount(0)
            self.summary_label.setText("No sessions available.")
            return

        # Clear previous preview
        self.preview_excluded_recordings.clear()

        # Collect all recordings and their exclusion status
        recordings_data = []
        for session in sessions:
            for recording in session.get_all_recordings(include_excluded=True):  # Include all recordings, even excluded ones
                will_exclude = self.should_exclude_recording(recording)
                current_status = recording.id in session.excluded_recordings

                # Track what would be excluded after applying criteria
                if will_exclude:
                    self.preview_excluded_recordings.add(recording.id)

                status = "Will exclude" if will_exclude else ("Excluded" if current_status else "Included")

                recordings_data.append(
                    {
                        "recording": recording,
                        "session_id": session.id,
                        "stimulus": recording.stim_amplitude,
                        "status": status,
                        "will_exclude": will_exclude,
                        "currently_excluded": current_status,
                    }
                )

        # Update table
        self.recordings_table.setRowCount(len(recordings_data))

        for row, data in enumerate(recordings_data):
            # Recording ID
            item = QTableWidgetItem(data["recording"].id)
            if data["will_exclude"]:
                item.setBackground(Qt.GlobalColor.lightGray)
            self.recordings_table.setItem(row, 0, item)

            # Session
            item = QTableWidgetItem(data["session_id"])
            if data["will_exclude"]:
                item.setBackground(Qt.GlobalColor.lightGray)
            self.recordings_table.setItem(row, 1, item)

            # Stimulus
            item = QTableWidgetItem(f"{data['stimulus']:.3f}")
            if data["will_exclude"]:
                item.setBackground(Qt.GlobalColor.lightGray)
            self.recordings_table.setItem(row, 2, item)

            # Status
            item = QTableWidgetItem(data["status"])
            if data["will_exclude"]:
                item.setBackground(Qt.GlobalColor.lightGray)
            self.recordings_table.setItem(row, 3, item)

        # Update summary
        total_recordings = len(recordings_data)
        currently_excluded = sum(1 for d in recordings_data if d["currently_excluded"])
        will_exclude = sum(1 for d in recordings_data if d["will_exclude"])

        summary_text = f"Total recordings: {total_recordings} | "
        summary_text += f"Currently excluded: {currently_excluded} | "
        summary_text += f"Will exclude with criteria: {will_exclude}"

        self.summary_label.setText(summary_text)

    def reset_criteria(self):
        """Reset all criteria to default values."""
        # Reset stimulus amplitude criteria
        self.stimulus_group.setChecked(False)
        self.threshold_type_combo.setCurrentIndex(0)
        self.threshold_spinbox.setValue(1.0)
        self.threshold2_spinbox.setValue(5.0)

        # Future criteria resets can be added here

        self.update_preview()

    def apply_exclusions(self):
        """Apply the exclusion criteria to the selected level."""
        sessions = self.get_sessions_for_level()
        if not sessions:
            QMessageBox.warning(self, "No Sessions", "No sessions available to apply exclusions to.")
            return

        # Count what will change
        total_exclusions = 0
        total_inclusions = 0

        for session in sessions:
            for recording in session.get_all_recordings(include_excluded=True):
                should_exclude = self.should_exclude_recording(recording)
                currently_excluded = recording.id in session.excluded_recordings

                if should_exclude and not currently_excluded:
                    total_exclusions += 1
                elif not should_exclude and currently_excluded:
                    total_inclusions += 1

        if total_exclusions == 0 and total_inclusions == 0:
            QMessageBox.information(self, "No Changes", "No recordings need to be changed based on current criteria.")
            return

        # Confirm with user
        level_name = self.level_combo.currentText()
        msg = f"Apply exclusion criteria to {level_name}?\n\n"
        msg += f"• {total_exclusions} recordings will be excluded\n"
        msg += f"• {total_inclusions} recordings will be included\n\n"
        msg += "This action can be undone."

        reply = QMessageBox.question(
            self,
            "Confirm Exclusions",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Apply exclusions using command pattern for undo support
        try:
            # Prefer using the command/undo system so exclusions are reversible.
            # TODO: Ensure the command_invoker is always available in the GUI and
            # avoid falling back to non-undoable changes. If the invoker is missing,
            # grey out the Apply button instead of silently applying without undo.
            from monstim_gui.commands import BulkRecordingExclusionCommand

            # Create command with all the changes
            changes = []
            for session in sessions:
                session_changes = []
                for recording in session.get_all_recordings(include_excluded=True):
                    should_exclude = self.should_exclude_recording(recording)
                    currently_excluded = recording.id in session.excluded_recordings

                    if should_exclude != currently_excluded:
                        session_changes.append({"recording_id": recording.id, "exclude": should_exclude})

                if session_changes:
                    changes.append({"session": session, "changes": session_changes})

            if changes:
                command = BulkRecordingExclusionCommand(self.gui, changes)
                self.gui.command_invoker.execute(command)

                self.exclusions_applied.emit()
                self.accept()

                self.gui.status_bar.showMessage(
                    f"Applied exclusion criteria: {total_exclusions} excluded, {total_inclusions} included", 5000
                )

        except ImportError:
            # Fallback: if command class is not importable, raise a clear error
            # instead of silently applying non-undoable changes. This prevents
            # accidental data loss and encourages wiring the command system.
            logging.error("BulkRecordingExclusionCommand not available - Command pattern not installed or import failed.")
            QMessageBox.critical(
                self,
                "Internal Error",
                "Cannot apply exclusions because the undo/redo command system is not available. Please restart the app or contact support.",
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply exclusions:\n{str(e)}")
            logging.error(f"Error applying recording exclusions: {e}")
