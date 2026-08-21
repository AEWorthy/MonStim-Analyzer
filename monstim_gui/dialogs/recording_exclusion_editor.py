"""
Recording Exclusion Editor Dialog
Allows users to exclude recordings based on various criteria like stimulus amplitude.
Designed to be extensible for future criteria-based exclusion.
"""

import datetime
import json
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
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
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from monstim_signals.domain.recording import Recording

if TYPE_CHECKING:
    from monstim_gui.gui_main import MonstimGUI
    from monstim_signals.domain import Session


logger = logging.getLogger(__name__)


class RecordingExclusionEditor(QDialog):
    """
    Extensible dialog for excluding recordings based on various criteria.
    Currently supports stimulus amplitude thresholds, designed for future expansion.
    """

    exclusions_applied = Signal()  # Signal emitted when exclusions are applied

    def __init__(self, parent: MonstimGUI):
        super().__init__(parent)
        self.gui = parent
        self.current_session = parent.current_session
        self.current_dataset = parent.current_dataset
        self.current_experiment = parent.current_experiment

        # Snapshot exclusions at dialog-open time.  They are a protected baseline
        # until the user explicitly marks a recording included in this dialog.
        self.initial_exclusion_states: dict[tuple[str, str], bool] = {}
        self._capture_initial_exclusion_states()

        # Track preview exclusions (not yet applied)
        self.preview_excluded_recordings: set[str] = set()

        # Decisions are staged until Apply.  Keys include the session because
        # recording ids are only guaranteed to be unique within a session.
        self.manual_decisions: dict[tuple[str, str], bool] = {}
        self.auto_flagged_recordings: dict[tuple[str, str], dict[str, Any]] = {}
        self._quality_cache: dict[tuple[str, str, tuple[int, int] | None, int], dict[str, float | None]] = {}
        self._sparkline_cache: dict[tuple[str, str, tuple[int, int] | None, int], QIcon] = {}
        self._preview_trace_cache: dict[tuple[str, str, tuple[int, int] | None, int], np.ndarray | None] = {}
        self._preview_y_range: tuple[float, float] | None = None
        self.preview_channel_index: int | None = None
        self._detail_preview_dialog: QDialog | None = None
        self._sort_column = 1
        self._sort_order = Qt.SortOrder.AscendingOrder

        self.setup_ui()
        self.load_data()

    def setup_ui(self):
        """Set up the dialog UI."""
        self.setWindowTitle("Recording Exclusion Editor")
        self.setModal(True)
        self.resize(800, 600)

        # Main layout
        main_layout = QVBoxLayout(self)

        # Create splitter for criteria and preview
        self.editor_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.editor_splitter)

        # Left side: Exclusion criteria tabs
        criteria_widget = self.create_criteria_widget()
        self.editor_splitter.addWidget(criteria_widget)

        # Right side: Recording preview table
        preview_widget = self.create_preview_widget()
        self.editor_splitter.addWidget(preview_widget)

        # Set splitter proportions
        self.editor_splitter.setSizes([300, 500])

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

        # Save/Load profile buttons
        self.save_profile_button = QPushButton("Save Profile")
        self.save_profile_button.clicked.connect(self.save_profile)
        button_layout.addWidget(self.save_profile_button)

        self.load_profile_button = QPushButton("Load Profile")
        self.load_profile_button.clicked.connect(self.load_profile)
        button_layout.addWidget(self.load_profile_button)

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
        self._configure_tooltips()

    @staticmethod
    def _set_tooltip(widget, text: str) -> None:
        """Give every interactive control matching hover and status help."""
        widget.setToolTip(text)
        widget.setStatusTip(text)

    @staticmethod
    def _set_combo_item_tooltips(combo: QComboBox, descriptions: list[str]) -> None:
        for index, description in enumerate(descriptions):
            combo.setItemData(index, description, Qt.ItemDataRole.ToolTipRole)

    def _configure_tooltips(self) -> None:
        """Document units and decision effects at the point of use."""
        self._set_tooltip(self.level_combo, "Choose the sessions included in this review and final bulk commit.")
        self._set_combo_item_tooltips(
            self.level_combo,
            [
                "Evaluate and apply changes only to the selected session.",
                "Evaluate and apply changes to every session in the selected dataset.",
                "Evaluate and apply changes to every session in the selected experiment.",
            ],
        )
        self._set_tooltip(self.preview_button, "Recalculate the preview using the current criteria without changing any recordings.")
        self._set_tooltip(self.save_profile_button, "Save criteria and range settings for reuse with another experiment.")
        self._set_tooltip(self.load_profile_button, "Load saved criteria and range settings; this only updates the preview until Apply.")
        self._set_tooltip(self.reset_button, "Restore default criteria and discard staged manual and automatic flags.")
        self._set_tooltip(self.apply_button, "Commit the reviewed decisions through the undoable bulk exclusion command.")
        self._set_tooltip(self.cancel_button, "Close without committing staged decisions.")

        self._set_tooltip(self.stimulus_group, "Enable stimulus-amplitude based exclusion.")
        self._set_tooltip(self.threshold_type_combo, "Select how the stimulus-amplitude threshold is interpreted.")
        self._set_combo_item_tooltips(
            self.threshold_type_combo,
            [
                "Flag recordings whose stimulus amplitude is greater than the threshold.",
                "Flag recordings whose stimulus amplitude is less than the threshold.",
                "Flag recordings outside the lower and upper stimulus limits.",
                "Flag recordings inside the lower and upper stimulus limits.",
            ],
        )
        self._set_tooltip(self.threshold_spinbox, "Stimulus-amplitude limit in volts.")
        self._set_tooltip(self.threshold2_spinbox, "Upper stimulus-amplitude limit in volts for range rules.")

        self._set_tooltip(self.quality_group, "Enable automatic exclusion based on waveform quality metrics.")
        self._set_tooltip(self.snr_spin, "Flag traces whose RMS signal-to-noise ratio is below this unitless ratio.")
        self._set_tooltip(self.drift_spin, "Flag traces whose first-versus-last baseline median changes by more than this voltage (V).")
        self._set_tooltip(self.flatline_spin, "Flag traces whose standard deviation, measured in volts, is below this value.")
        self._set_tooltip(
            self.line_noise_spin,
            "Flag traces whose unnormalised 50/60 Hz FFT-band magnitude exceeds this value; 0 disables this rule.",
        )
        self._set_tooltip(
            self.burst_duration_spin,
            "Flag continuous above-baseline activity longer than this duration in milliseconds; short APs should not trigger it.",
        )
        self._set_tooltip(
            self.outlier_z_spin,
            "Robust median/MAD distance used to flag unusually large RMS, peak-to-peak, or burst-duration values within a session.",
        )
        self._set_tooltip(self.range_combo, "Choose which part of each recording is measured and displayed.")
        self._set_combo_item_tooltips(
            self.range_combo,
            [
                "Use the active analysis profile's pre-stimulus and response window.",
                "Use every sample in the recording.",
                "Use the custom start and end times relative to the stimulus.",
            ],
        )
        self._set_tooltip(self.range_start_spin, "Custom window start in milliseconds relative to the stimulus.")
        self._set_tooltip(self.range_end_spin, "Custom window end in milliseconds relative to the stimulus.")
        self._set_tooltip(self.auto_flag_button, "Run the visible quality thresholds now and stage the resulting flags for review.")
        self._set_tooltip(self.clear_auto_flags_button, "Discard all staged automatic quality flags without changing manual decisions or recordings.")

        self._set_tooltip(self.preview_range_combo, "Choose the time range shown in waveform snippets; this does not change quality calculations.")
        self._set_combo_item_tooltips(
            self.preview_range_combo,
            [
                "Show the active analysis profile's pre-stimulus and response window.",
                "Show every sample in each recording.",
                "Show the custom start and end times relative to the stimulus.",
            ],
        )
        self._set_tooltip(self.preview_start_spin, "Snippet start in milliseconds relative to the stimulus.")
        self._set_tooltip(self.preview_end_spin, "Snippet end in milliseconds relative to the stimulus.")
        self._set_tooltip(self.preview_y_scale_combo, "Choose independent normalization or one shared y-axis range for every visible snippet.")
        self._set_combo_item_tooltips(
            self.preview_y_scale_combo,
            [
                "Normalize each snippet to fill its own preview cell; best for waveform shape review.",
                "Use the same y-axis range for all snippets; best for comparing absolute amplitudes.",
            ],
        )
        self._set_tooltip(self.preview_filter_combo, "Limit visible rows without changing the proposed or committed decisions.")
        self._set_combo_item_tooltips(
            self.preview_filter_combo,
            [
                "Show every recording in scope.",
                "Show recordings with an automatic flag or pending exclusion.",
                "Show recordings currently excluded from analysis.",
                "Show recordings currently included and not pending exclusion.",
            ],
        )
        self._set_tooltip(self.toggle_exclusion_button, "Stage exclusion for all selected rows; Apply commits it undoably.")
        self._set_tooltip(self.include_button, "Stage inclusion for all selected rows, overriding automatic flags until cleared.")
        self._set_tooltip(
            self.clear_manual_button,
            "Remove staged manual decisions so automatic criteria apply again; exclusions present when this dialog opened remain protected.",
        )
        self._set_tooltip(self.export_report_button, "Export the current review state, reasons, metrics, and pending decisions as JSON.")
        self._set_tooltip(self.recordings_table, "Select one or more rows to stage a manual decision. Reasons and metrics explain automatic flags.")
        header_help = [
            "A normalized waveform preview for the selected evaluation range.",
            "Recording identifier within its session.",
            "Session containing this recording.",
            "Stimulus amplitude in volts.",
            "Current or staged inclusion/exclusion state.",
            "High severity indicates a burst or robust session outlier.",
            "Criteria that caused the automatic flag.",
            "Computed quality metrics; voltage-derived values are in volts except duration, which is milliseconds.",
        ]
        for index, text in enumerate(header_help):
            self.recordings_table.horizontalHeaderItem(index).setToolTip(text)
        self._set_tooltip(self.preview_channel_previous_button, "Use the previous channel for both waveform previews and quality metrics.")
        self._set_tooltip(self.preview_channel_next_button, "Use the next channel for both waveform previews and quality metrics.")
        self._set_tooltip(
            self.preview_channel_label,
            "The selected channel is used for previews and flag metrics. Auto uses the active plot selection, then an EMG channel, then channel 0.",
        )

        for widget in self.findChildren(QWidget):
            widget.installEventFilter(self)

    def eventFilter(self, watched, event):
        if isinstance(watched, QComboBox) and event.type() == QEvent.Type.Wheel:
            return True
        table = getattr(self, "recordings_table", None)
        is_mouse_press = event.type() == QEvent.Type.MouseButtonPress
        is_empty_table_click = table is not None and watched is table.viewport() and is_mouse_press and not table.indexAt(event.pos()).isValid()
        if is_empty_table_click:
            table.clearSelection()
            self._close_detail_preview()
            return True
        return super().eventFilter(watched, event)

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

        # Add quality metrics tab
        quality_tab = self.create_quality_tab()
        self.criteria_tabs.addTab(quality_tab, "Quality")

        # TODO: Future tabs can be added here:
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

    def create_quality_tab(self) -> QWidget:
        """Create quality-based exclusion criteria tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.quality_group = QGroupBox("Exclude recordings by quality metrics")
        self.quality_group.setCheckable(True)
        self.quality_group.setChecked(False)
        form = QFormLayout(self.quality_group)

        # SNR threshold (exclude if SNR below)
        self.snr_spin = QDoubleSpinBox()
        self.snr_spin.setRange(0.0, 1000.0)
        self.snr_spin.setDecimals(2)
        self.snr_spin.setValue(2.0)
        form.addRow("Min SNR:", self.snr_spin)

        # Baseline drift threshold (exclude if drift above)
        self.drift_spin = QDoubleSpinBox()
        self.drift_spin.setRange(0.0, 100.0)
        self.drift_spin.setDecimals(4)
        self.drift_spin.setValue(0.20)
        self.drift_spin.setSuffix(" V")
        form.addRow("Max baseline drift:", self.drift_spin)

        # Flatline threshold (exclude if std below)
        self.flatline_spin = QDoubleSpinBox()
        self.flatline_spin.setRange(0.0, 100.0)
        self.flatline_spin.setDecimals(6)
        self.flatline_spin.setValue(1e-6)
        form.addRow("Min std (flatline):", self.flatline_spin)

        # Line noise energy threshold
        self.line_noise_spin = QDoubleSpinBox()
        self.line_noise_spin.setRange(0.0, 1e9)
        self.line_noise_spin.setDecimals(3)
        self.line_noise_spin.setValue(0.0)
        form.addRow("Max line-noise energy:", self.line_noise_spin)

        self.burst_duration_spin = QDoubleSpinBox()
        self.burst_duration_spin.setRange(0.1, 10000.0)
        self.burst_duration_spin.setSingleStep(1.0)
        self.burst_duration_spin.setDecimals(1)
        self.burst_duration_spin.setValue(10.0)
        self.burst_duration_spin.setSuffix(" ms")
        form.addRow("Max sustained burst duration:", self.burst_duration_spin)

        self.outlier_z_spin = QDoubleSpinBox()
        self.outlier_z_spin.setRange(1.0, 20.0)
        self.outlier_z_spin.setSingleStep(0.5)
        self.outlier_z_spin.setValue(5.0)
        form.addRow("Session outlier sensitivity:", self.outlier_z_spin)

        self.range_combo = QComboBox()
        self.range_combo.addItem("Analysis profile window", "profile")
        self.range_combo.addItem("Full recording", "full")
        self.range_combo.addItem("Custom window", "custom")
        form.addRow("Evaluate range:", self.range_combo)

        self.range_start_spin = QDoubleSpinBox()
        self.range_start_spin.setRange(0.0, 1e6)
        self.range_start_spin.setSuffix(" ms")
        self.range_start_spin.setVisible(False)
        form.addRow("Custom start:", self.range_start_spin)
        self.range_start_label = form.labelForField(self.range_start_spin)
        self.range_start_label.setVisible(False)
        self.range_end_spin = QDoubleSpinBox()
        self.range_end_spin.setRange(0.0, 1e6)
        self.range_end_spin.setValue(20.0)
        self.range_end_spin.setSuffix(" ms")
        self.range_end_spin.setVisible(False)
        form.addRow("Custom end:", self.range_end_spin)
        self.range_end_label = form.labelForField(self.range_end_spin)
        self.range_end_label.setVisible(False)

        def update_range_controls():
            is_custom = self.range_combo.currentData() == "custom"
            self.range_start_spin.setVisible(is_custom)
            self.range_end_spin.setVisible(is_custom)
            self.range_start_label.setVisible(is_custom)
            self.range_end_label.setVisible(is_custom)
            self._clear_preview_caches()
            self.update_preview()

        self.range_combo.currentIndexChanged.connect(update_range_controls)

        # Auto-flag button
        h = QHBoxLayout()
        self.auto_flag_button = QPushButton("Auto-flag low quality")
        self.auto_flag_button.clicked.connect(self.auto_flag_low_quality)
        h.addWidget(self.auto_flag_button)
        self.clear_auto_flags_button = QPushButton("Clear Flags")
        self.clear_auto_flags_button.clicked.connect(self.clear_auto_flags)
        h.addWidget(self.clear_auto_flags_button)
        h.addStretch()
        layout.addWidget(self.quality_group)
        layout.addLayout(h)

        # Connect to preview updates
        self.quality_group.toggled.connect(self.update_preview)
        self.snr_spin.valueChanged.connect(self.update_preview)
        self.drift_spin.valueChanged.connect(self.update_preview)
        self.flatline_spin.valueChanged.connect(self.update_preview)
        self.line_noise_spin.valueChanged.connect(self.update_preview)
        self.burst_duration_spin.valueChanged.connect(self.update_preview)
        self.outlier_z_spin.valueChanged.connect(self.update_preview)
        self.range_start_spin.valueChanged.connect(self._range_value_changed)
        self.range_end_spin.valueChanged.connect(self._range_value_changed)

        return tab

    def create_preview_widget(self) -> QWidget:
        """Create the recording preview table widget."""
        preview_widget = QWidget()
        self.preview_widget = preview_widget
        layout = QVBoxLayout(preview_widget)

        # Header
        header_layout = QHBoxLayout()
        header_label = QLabel("Recording Preview")
        header_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        self.preview_channel_previous_button = QToolButton()
        self.preview_channel_previous_button.setText("◀")
        self.preview_channel_previous_button.clicked.connect(lambda: self._change_preview_channel(-1))
        header_layout.addWidget(self.preview_channel_previous_button)
        self.preview_channel_label = QLabel("Channel: Auto")
        header_layout.addWidget(self.preview_channel_label)
        self.preview_channel_next_button = QToolButton()
        self.preview_channel_next_button.setText("▶")
        self.preview_channel_next_button.clicked.connect(lambda: self._change_preview_channel(1))
        header_layout.addWidget(self.preview_channel_next_button)
        layout.addLayout(header_layout)

        preview_controls = QFormLayout()
        self.preview_range_combo = QComboBox()
        self.preview_range_combo.addItem("Analysis profile window", "profile")
        self.preview_range_combo.addItem("Whole recording", "full")
        self.preview_range_combo.addItem("Custom time section", "custom")
        preview_controls.addRow("Preview snippet:", self.preview_range_combo)
        self.preview_start_spin = QDoubleSpinBox()
        self.preview_start_spin.setRange(0.0, 1e6)
        self.preview_start_spin.setSuffix(" ms")
        self.preview_start_spin.setVisible(False)
        preview_controls.addRow("Preview start:", self.preview_start_spin)
        self.preview_start_label = preview_controls.labelForField(self.preview_start_spin)
        self.preview_start_label.setVisible(False)
        self.preview_end_spin = QDoubleSpinBox()
        self.preview_end_spin.setRange(0.0, 1e6)
        self.preview_end_spin.setValue(20.0)
        self.preview_end_spin.setSuffix(" ms")
        self.preview_end_spin.setVisible(False)
        preview_controls.addRow("Preview end:", self.preview_end_spin)
        self.preview_end_label = preview_controls.labelForField(self.preview_end_spin)
        self.preview_end_label.setVisible(False)
        self.preview_y_scale_combo = QComboBox()
        self.preview_y_scale_combo.addItem("Scale each snippet", "individual")
        self.preview_y_scale_combo.addItem("Use one shared scale", "unified")
        preview_controls.addRow("Preview y-axis:", self.preview_y_scale_combo)
        layout.addLayout(preview_controls)

        def update_preview_controls():
            is_custom = self.preview_range_combo.currentData() == "custom"
            self.preview_start_spin.setVisible(is_custom)
            self.preview_end_spin.setVisible(is_custom)
            self.preview_start_label.setVisible(is_custom)
            self.preview_end_label.setVisible(is_custom)
            self._clear_preview_caches()
            self.update_preview()

        self.preview_range_combo.currentIndexChanged.connect(update_preview_controls)
        self.preview_start_spin.valueChanged.connect(self._preview_setting_changed)
        self.preview_end_spin.valueChanged.connect(self._preview_setting_changed)
        self.preview_y_scale_combo.currentIndexChanged.connect(self._preview_setting_changed)

        # Table for recordings (add Preview column)
        self.recordings_table = QTableWidget()
        self.recordings_table.setColumnCount(8)
        self.recordings_table.setHorizontalHeaderLabels(["Preview", "Rec.\nID", "Session", "Stim.\n(V)", "Status", "Severity", "Reasons", "Metrics"])

        # Configure table
        header = self.recordings_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.recordings_table.setColumnWidth(0, 66)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.Stretch)
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setFixedHeight(42)
        header.setSectionsClickable(True)
        header.setSortIndicatorShown(True)
        header.setSortIndicator(self._sort_column, self._sort_order)
        header.sortIndicatorChanged.connect(self._remember_sort_order)

        self.recordings_table.setAlternatingRowColors(True)
        self.recordings_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.recordings_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.recordings_table.setIconSize(QSize(60, 30))
        self.recordings_table.setSortingEnabled(True)
        self.recordings_table.verticalHeader().setDefaultSectionSize(36)
        self.recordings_table.itemSelectionChanged.connect(self._show_selected_recording_detail)
        self.recordings_table.viewport().installEventFilter(self)

        layout.addWidget(self.recordings_table)

        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Show:"))
        self.preview_filter_combo = QComboBox()
        self.preview_filter_combo.addItem("All recordings", "all")
        self.preview_filter_combo.addItem("Flagged or pending", "flagged")
        self.preview_filter_combo.addItem("Excluded", "excluded")
        self.preview_filter_combo.addItem("Included", "included")
        self.preview_filter_combo.currentIndexChanged.connect(self._apply_preview_filter)
        filter_layout.addWidget(self.preview_filter_combo)
        filter_layout.addStretch()
        layout.addLayout(filter_layout)

        # Summary label
        self.summary_label = QLabel()
        layout.addWidget(self.summary_label)

        # Manual decisions remain staged until Apply so they can be reviewed with
        # automatic flags.  The final commit is one undoable bulk command.
        toggle_layout = QHBoxLayout()
        self.toggle_exclusion_button = QPushButton("Mark Excluded")
        self.toggle_exclusion_button.clicked.connect(self.toggle_selected_exclusions)
        toggle_layout.addWidget(self.toggle_exclusion_button)
        self.include_button = QPushButton("Mark Included")
        self.include_button.clicked.connect(lambda: self._set_selected_manual_decision(False))
        toggle_layout.addWidget(self.include_button)
        self.clear_manual_button = QPushButton("Clear Manual Decision")
        self.clear_manual_button.clicked.connect(self._clear_selected_manual_decisions)
        toggle_layout.addWidget(self.clear_manual_button)
        self.export_report_button = QPushButton("Export Report")
        self.export_report_button.clicked.connect(self.export_curation_report)
        toggle_layout.addWidget(self.export_report_button)
        toggle_layout.addStretch()
        layout.addLayout(toggle_layout)

        return preview_widget

    def _remember_sort_order(self, column: int, order: Qt.SortOrder) -> None:
        self._sort_column = column
        self._sort_order = order

    def _create_detail_preview_dialog(self) -> QDialog:
        """Create the separate full-recording reference window used by selected rows."""
        detail_dialog = QDialog(self)
        detail_dialog.setWindowTitle("Recording Detail")
        detail_dialog.setModal(False)
        detail_dialog.resize(320, 420)
        detail_layout = QVBoxLayout(detail_dialog)
        self._detail_preview_title = QLabel("Select a recording to inspect it in detail")
        self._detail_preview_title.setStyleSheet("font-weight: bold;")
        detail_layout.addWidget(self._detail_preview_title)
        self._detail_preview_context = QLabel()
        self._detail_preview_context.setWordWrap(True)
        detail_layout.addWidget(self._detail_preview_context)
        self._detail_preview_plot = pg.PlotWidget()
        self._detail_preview_plot.setLabel("left", "Voltage", units="V")
        self._detail_preview_plot.showGrid(x=True, y=True, alpha=0.2)
        self._set_tooltip(
            self._detail_preview_plot,
            "Full selected-channel trace. The shaded region is the table preview snippet; the table may use a different y-scale.",
        )
        detail_layout.addWidget(self._detail_preview_plot)
        self._detail_preview_dialog = detail_dialog
        return detail_dialog

    def _range_value_changed(self):
        self._clear_preview_caches()
        self.update_preview()

    def _clear_preview_caches(self):
        self._quality_cache.clear()
        self._sparkline_cache.clear()
        self._preview_trace_cache.clear()

    def _preview_setting_changed(self):
        self._sparkline_cache.clear()
        self._preview_trace_cache.clear()
        self.update_preview()

    def load_data(self):
        """Load initial data and populate preview."""
        if not self.current_session:
            QMessageBox.warning(self, "No Session", "No session is currently selected.")
            return

        self.update_preview()

    def _capture_initial_exclusion_states(self) -> None:
        """Capture the opening state for every session the dialog may review."""
        sessions = []
        if self.current_experiment is not None:
            for dataset in self.current_experiment.datasets:
                sessions.extend(dataset.sessions)
        elif self.current_dataset is not None:
            sessions.extend(self.current_dataset.sessions)
        elif self.current_session is not None:
            sessions.append(self.current_session)

        for session in sessions:
            excluded = session.excluded_recordings
            for recording in session.get_all_recordings(include_excluded=True):
                self.initial_exclusion_states[(str(session.id), str(recording.id))] = recording.id in excluded

    def get_sessions_for_level(self) -> list[Session]:
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

    def _get_active_profile_data(self) -> dict[str, Any]:
        """Return the active analysis profile payload, if available."""
        active_profile_data = getattr(self.gui, "active_profile_data", None)
        if isinstance(active_profile_data, dict):
            return active_profile_data

        current_session = getattr(self, "current_session", None)
        config = getattr(current_session, "_config", None) if current_session is not None else None
        if isinstance(config, dict):
            return config

        return {}

    def _get_preview_channel_indices(self, recording: Recording) -> list[int]:
        """Return the explicitly selected channel, or the documented automatic choice."""
        if self.preview_channel_index is not None and 0 <= self.preview_channel_index < recording.num_channels:
            return [self.preview_channel_index]
        return self._get_automatic_preview_channel_indices(recording)

    def _get_automatic_preview_channel_indices(self, recording: Recording) -> list[int]:
        """Prefer the active plot channel, then an EMG channel, then channel zero."""
        try:
            selected_channels = getattr(getattr(self.gui, "plot_widget", None), "persistent_channel_selection", [])
            selected_channels = [int(channel_idx) for channel_idx in selected_channels if 0 <= int(channel_idx) < recording.num_channels]
            if selected_channels:
                return selected_channels
        except Exception:
            pass

        channel_indices = [idx for idx, channel_type in enumerate(recording.channel_types) if str(channel_type).strip().lower().startswith("emg")]
        return channel_indices or [0]

    def _change_preview_channel(self, direction: int) -> None:
        """Move through the available channels without mutating the main plot selection."""
        entries = getattr(self, "_last_recordings_data", [])
        if entries:
            recording = entries[0]["recording"]
        elif self.current_session:
            recordings = self.current_session.get_all_recordings(include_excluded=True)
            recording = recordings[0] if recordings else None
        else:
            recording = None
        if recording is None or recording.num_channels <= 0:
            return
        current = self.preview_channel_index
        if current is None:
            automatic = self._get_automatic_preview_channel_indices(recording)
            current = automatic[0] if automatic else 0
        self.preview_channel_index = (current + direction) % recording.num_channels
        self._clear_preview_caches()
        self.update_preview()

    def _update_preview_channel_label(self, recording: Recording | None) -> None:
        if recording is None:
            self.preview_channel_label.setText("Channel: Auto")
            return
        if self.preview_channel_index is None:
            index = self._get_automatic_preview_channel_indices(recording)[0]
            self.preview_channel_label.setText(f"Channel: Auto (Ch {index + 1})")
            return
        channel_type = ""
        if self.preview_channel_index < len(recording.channel_types):
            channel_type = str(recording.channel_types[self.preview_channel_index])
        suffix = f" ({channel_type})" if channel_type else ""
        self.preview_channel_label.setText(f"Channel: Ch {self.preview_channel_index + 1}{suffix}")

    def _get_recording_trace(self, recording: Recording, session, ch_idx: int) -> np.ndarray | None:
        """Return the filtered session trace for this recording when available."""
        try:
            all_recordings = session.get_all_recordings(include_excluded=True)
            recording_index = next(idx for idx, rec in enumerate(all_recordings) if rec.id == recording.id)
            filtered_recordings = getattr(session, "all_recordings_filtered", None)
            if filtered_recordings is not None and recording_index < len(filtered_recordings):
                return np.asarray(filtered_recordings[recording_index][:, ch_idx]).squeeze()
        except Exception:
            pass

        try:
            return np.asarray(recording.raw_view(ch=ch_idx, t=slice(None))).squeeze()
        except Exception:
            return None

    def _get_preview_time_window(self, recording: Recording, session=None) -> tuple[int, int] | None:
        """Return the sample window used by the active EMG analysis profile.

        This mirrors SessionPlotterPyQtGraph.get_time_axis():
        start = stim_start - pre_stim_time
        end = stim_start + time_window
        """
        mode = self.range_combo.currentData()
        if mode == "full":
            return None

        profile_data = self._get_active_profile_data()
        analysis_params = profile_data.get("analysis_parameters", {}) if isinstance(profile_data, dict) else {}
        if not isinstance(analysis_params, dict):
            analysis_params = {}

        try:
            source_session = session if session is not None else self.current_session
            stim_start_ms = float(getattr(source_session, "stim_start", 0.0))
            scan_rate = float(getattr(recording, "scan_rate", 0.0))
            if mode == "custom":
                start_ms = stim_start_ms + float(self.range_start_spin.value())
                end_ms = stim_start_ms + float(self.range_end_spin.value())
            else:
                pre_stim_time_ms = float(analysis_params.get("pre_stim_time", getattr(source_session, "pre_stim_time_ms", 0.0)))
                time_window_ms = float(analysis_params.get("time_window", getattr(source_session, "time_window_ms", 0.0)))
                start_ms = stim_start_ms - pre_stim_time_ms
                end_ms = stim_start_ms + time_window_ms
        except Exception:
            return None

        if scan_rate <= 0:
            return None

        start_sample = int(start_ms * scan_rate / 1000.0)
        end_sample = int(end_ms * scan_rate / 1000.0)

        start_sample = max(0, start_sample)
        end_sample = min(recording.num_samples, end_sample)

        if end_sample <= start_sample:
            return None

        return start_sample, end_sample

    def _get_preview_snippet_time_window(self, recording: Recording, session=None) -> tuple[int, int] | None:
        """Return the independent time window selected for waveform snippets."""
        mode = self.preview_range_combo.currentData()
        if mode == "full":
            return None

        source_session = session if session is not None else self.current_session
        try:
            scan_rate = float(getattr(recording, "scan_rate", 0.0))
            stim_start_ms = float(getattr(source_session, "stim_start", 0.0))
            if mode == "custom":
                start_ms = stim_start_ms + float(self.preview_start_spin.value())
                end_ms = stim_start_ms + float(self.preview_end_spin.value())
            else:
                profile_data = self._get_active_profile_data()
                parameters = profile_data.get("analysis_parameters", {}) if isinstance(profile_data, dict) else {}
                start_ms = stim_start_ms - float(parameters.get("pre_stim_time", getattr(source_session, "pre_stim_time_ms", 0.0)))
                end_ms = stim_start_ms + float(parameters.get("time_window", getattr(source_session, "time_window_ms", 0.0)))
        except TypeError, ValueError:
            return None

        if scan_rate <= 0:
            return None
        start_sample = max(0, int(start_ms * scan_rate / 1000.0))
        end_sample = min(recording.num_samples, int(end_ms * scan_rate / 1000.0))
        return (start_sample, end_sample) if end_sample > start_sample else None

    def _get_preview_samples(self, recording: Recording, session) -> np.ndarray | None:
        """Load one snippet once so unified scaling does not require repeated I/O."""
        channel_indices = self._get_preview_channel_indices(recording)
        ch_idx = channel_indices[0] if channel_indices else 0
        window = self._get_preview_snippet_time_window(recording, session)
        cache_key = (str(getattr(session, "id", "")), str(recording.id), window, ch_idx)
        if cache_key not in self._preview_trace_cache:
            try:
                trace = self._get_recording_trace(recording, session, ch_idx)
                if trace is None:
                    self._preview_trace_cache[cache_key] = None
                else:
                    samples = trace[window[0] : window[1]] if window else trace
                    self._preview_trace_cache[cache_key] = np.asarray(samples).squeeze()
            except Exception:
                self._preview_trace_cache[cache_key] = None
        return self._preview_trace_cache[cache_key]

    def _recording_key(self, recording: Recording, session) -> tuple[str, str]:
        return str(session.id), str(recording.id)

    def _evaluation_for_recording(self, recording: Recording, session, metrics: dict[str, float | None], outlier_metrics: set[str]) -> dict[str, Any]:
        """Return an explainable automatic exclusion decision for one recording."""
        reasons: list[str] = []
        if self.stimulus_group.isChecked():
            stimulus_value = recording.stim_amplitude
            threshold_type = self.threshold_type_combo.currentData()
            threshold1 = self.threshold_spinbox.value()
            threshold2 = self.threshold2_spinbox.value()

            match threshold_type:
                case "above":
                    if stimulus_value > threshold1:
                        reasons.append(f"stimulus > {threshold1:g} V")
                case "below":
                    if stimulus_value < threshold1:
                        reasons.append(f"stimulus < {threshold1:g} V")
                case "outside":
                    if stimulus_value < threshold1 or stimulus_value > threshold2:
                        reasons.append(f"stimulus outside {threshold1:g}-{threshold2:g} V")
                case "inside":
                    if threshold1 <= stimulus_value <= threshold2:
                        reasons.append(f"stimulus inside {threshold1:g}-{threshold2:g} V")

        if self.quality_group.isChecked():
            checks = (
                ("snr", lambda value: value < self.snr_spin.value(), "low SNR"),
                ("baseline_drift", lambda value: value > self.drift_spin.value(), "baseline drift"),
                ("flatline", lambda value: value < self.flatline_spin.value(), "flatline"),
                ("line_noise", lambda value: self.line_noise_spin.value() > 0 and value > self.line_noise_spin.value(), "line noise"),
                ("burst_duration_ms", lambda value: value > self.burst_duration_spin.value(), "sustained burst"),
            )
            for name, predicate, label in checks:
                value = metrics.get(name)
                if value is not None and predicate(value):
                    reasons.append(f"{label} ({value:.3g})")
            for metric_name in sorted(outlier_metrics):
                reasons.append(f"session {metric_name.replace('_', ' ')} outlier")

        severity = "high" if any("burst" in reason or "outlier" in reason for reason in reasons) else ("medium" if reasons else "")
        return {"flagged": bool(reasons), "reasons": reasons, "severity": severity, "metrics": metrics}

    def compute_quality_metrics(self, recording: Recording, session=None) -> dict[str, float | None]:
        """Compute simple quality metrics for a recording.

        Returns a dict with keys: snr, baseline_drift, flatline, line_noise. Values
        are numeric or None if computation unavailable.
        """
        source_session = session if session is not None else self.current_session
        channel_indices = self._get_preview_channel_indices(recording)
        ch_idx = channel_indices[0] if channel_indices else 0
        window = self._get_preview_time_window(recording, source_session)
        cache_key = (str(getattr(source_session, "id", "")), str(recording.id), window, ch_idx)
        cached = self._quality_cache.get(cache_key)
        if cached is not None:
            return cached

        arr = None
        try:
            sig = self._get_recording_trace(recording, source_session, ch_idx)
            if sig is not None:
                arr = np.asarray(sig if window is None else sig[window[0] : window[1]]).squeeze()
        except Exception:
            arr = None

        metrics: dict[str, float | None] = {
            "snr": None,
            "baseline_drift": None,
            "flatline": None,
            "line_noise": None,
            "rms": None,
            "peak_to_peak": None,
            "burst_duration_ms": None,
        }

        if arr is None or arr.size == 0:
            self._quality_cache[cache_key] = metrics
            return metrics

        # Simple RMS-based SNR: estimate noise from first 10% of trace
        try:
            n = arr.size
            noise_seg = arr[: max(1, n // 10)]
            signal_seg = arr[n // 10 :]
            noise_rms = float((np.mean(noise_seg**2)) ** 0.5)
            signal_rms = float((np.mean(signal_seg**2)) ** 0.5)
            metrics["snr"] = signal_rms / (noise_rms + 1e-12)
        except Exception:
            metrics["snr"] = None

        # Baseline drift: absolute difference between median of first and last 10%
        try:
            first_med = float(np.median(arr[: max(1, n // 10)]))
            last_med = float(np.median(arr[-max(1, n // 10) :]))
            metrics["baseline_drift"] = abs(last_med - first_med)
        except Exception:
            metrics["baseline_drift"] = None

        # Flatline: low variance
        try:
            metrics["flatline"] = float(np.std(arr))
            metrics["rms"] = float(np.sqrt(np.mean(arr**2)))
            metrics["peak_to_peak"] = float(np.ptp(arr))
            absolute = np.abs(arr)
            median_abs = float(np.median(absolute))
            absolute_mad = float(np.median(np.abs(absolute - median_abs)))
            activity_threshold = median_abs + 6.0 * max(absolute_mad, 1e-12)
            active = absolute > activity_threshold
            if active.any():
                transitions = np.diff(np.r_[False, active, False].astype(np.int8))
                run_lengths = np.flatnonzero(transitions == -1) - np.flatnonzero(transitions == 1)
                scan_rate = float(getattr(recording, "scan_rate", 0.0))
                if scan_rate > 0:
                    metrics["burst_duration_ms"] = float(run_lengths.max() * 1000.0 / scan_rate)
        except Exception:
            metrics["flatline"] = None

        # Line noise detection: rudimentary via fft energy near 50/60 Hz
        try:
            fs = getattr(recording, "scan_rate", None) or getattr(recording, "sampling_rate", None) or getattr(recording, "fs", None) or 1000.0
            # Bounded FFT keeps full-recording curation responsive on long files.
            fft_arr = arr if n <= 50000 else arr[np.linspace(0, n - 1, 50000).astype(int)]
            freqs = np.fft.rfftfreq(fft_arr.size, 1.0 / float(fs))
            fft = np.abs(np.fft.rfft(fft_arr))

            # look for energy near 50 and 60 Hz
            def band_energy(target):
                mask = (freqs > (target - 1.0)) & (freqs < (target + 1.0))
                return float(np.sum(fft[mask]))

            e50 = band_energy(50)
            e60 = band_energy(60)
            metrics["line_noise"] = max(e50, e60)
        except Exception:
            metrics["line_noise"] = None

        self._quality_cache[cache_key] = metrics
        return metrics

    def generate_sparkline_icon(self, recording, session=None, width=120, height=30) -> QIcon:
        """Generate a small sparkline icon for a recording if waveform is available.

        Returns a QIcon. Falls back to a simple placeholder pixmap.
        """
        source_session = session if session is not None else self.current_session
        channel_indices = self._get_preview_channel_indices(recording)
        ch_idx = channel_indices[0] if channel_indices else 0
        window = self._get_preview_snippet_time_window(recording, source_session)
        y_range = self._preview_y_range if self.preview_y_scale_combo.currentData() == "unified" else None
        cache_key = (str(getattr(source_session, "id", "")), str(recording.id), window, ch_idx, y_range)
        cached = self._sparkline_cache.get(cache_key)
        if cached is not None:
            return cached

        # Use the same kind of EMG trace shown in the normal session EMG plot.
        arr = self._get_preview_samples(recording, source_session)

        pix = QPixmap(width, height)
        pix.fill(QColor("transparent"))
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor("#2b85d8"))
        pen.setWidth(2)
        painter.setPen(pen)

        if arr is None or arr.size == 0:
            # draw placeholder line
            painter.drawLine(2, height // 2, width - 2, height // 2)
            painter.end()
            icon = QIcon(pix)
            self._sparkline_cache[cache_key] = icon
            return icon

        # Use a min/max envelope rather than point sampling.  Point sampling can
        # entirely miss the narrow stimulation spikes visible in the detail plot.
        try:
            ys_source = np.asarray(arr, dtype=float).reshape(-1)
            ys_source = ys_source[np.isfinite(ys_source)]
            if ys_source.size == 0:
                painter.drawLine(2, height // 2, width - 2, height // 2)
                painter.end()
                return QIcon(pix)

            n = ys_source.size
            if n <= width:
                ys = ys_source
            else:
                edges = np.linspace(0, n, width + 1, dtype=int)
                mins = np.empty(width)
                maxs = np.empty(width)
                for index in range(width):
                    segment = ys_source[edges[index] : max(edges[index] + 1, edges[index + 1])]
                    mins[index] = np.min(segment)
                    maxs[index] = np.max(segment)
                ys = np.empty(width * 2)
                ys[0::2] = mins
                ys[1::2] = maxs

            # Normalize each preview trace to the available cell height.
            ymin, ymax = y_range if y_range is not None else (float(np.min(ys_source)), float(np.max(ys_source)))
            rng = ymax - ymin if ymax != ymin else 1.0
            points = []
            for i, v in enumerate(ys):
                x = int(i * (width - 6) / max(1, len(ys) - 1)) + 3
                y = int((1.0 - (float(v) - ymin) / rng) * (height - 6)) + 3
                points.append((x, y))

            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                painter.drawLine(x1, y1, x2, y2)
        except Exception:
            painter.drawLine(2, height // 2, width - 2, height // 2)

        painter.end()
        icon = QIcon(pix)
        self._sparkline_cache[cache_key] = icon
        return icon

    def update_preview(self):
        """Evaluate and render a reviewable, explainable curation preview."""
        sessions = self.get_sessions_for_level()
        if not sessions:
            self.recordings_table.setRowCount(0)
            self.summary_label.setText("No sessions available.")
            self._update_preview_channel_label(None)
            return

        self.preview_excluded_recordings.clear()
        recordings_data = []
        for session in sessions:
            session_records = session.get_all_recordings(include_excluded=True)
            metrics_by_recording = {recording.id: self.compute_quality_metrics(recording, session) for recording in session_records}
            outliers = self._session_outliers(metrics_by_recording)
            for recording in session_records:
                key = self._recording_key(recording, session)
                evaluation = self._evaluation_for_recording(recording, session, metrics_by_recording[recording.id], outliers.get(recording.id, set()))
                auto_flag = self.auto_flagged_recordings.get(key)
                if auto_flag:
                    evaluation["reasons"] = [*evaluation["reasons"], *auto_flag["reasons"]]
                    evaluation["flagged"] = True
                    evaluation["severity"] = "high"
                current_status = recording.id in session.excluded_recordings
                initial_excluded = self.initial_exclusion_states.setdefault(key, current_status)
                manual_decision = self.manual_decisions.get(key)
                will_exclude = manual_decision if manual_decision is not None else (initial_excluded or evaluation["flagged"])

                if will_exclude:
                    self.preview_excluded_recordings.add(f"{session.id}:{recording.id}")

                if manual_decision is True:
                    status = "Manual exclude"
                elif manual_decision is False:
                    status = "Manual include"
                elif initial_excluded:
                    status = "Existing exclusion"
                elif will_exclude:
                    status = "Will exclude"
                else:
                    status = "Included"

                recordings_data.append(
                    {
                        "recording": recording,
                        "session": session,
                        "session_id": session.id,
                        "stimulus": recording.stim_amplitude,
                        "status": status,
                        "will_exclude": bool(will_exclude),
                        "currently_excluded": current_status,
                        "evaluation": evaluation,
                        "manual_decision": manual_decision,
                    }
                )

        # Store last recordings for use by toggle operations
        self._last_recordings_data = recordings_data
        self._update_preview_channel_label(recordings_data[0]["recording"] if recordings_data else None)

        # Populate with sorting disabled so rows remain aligned while item data is
        # attached. Sorting is restored after all columns are complete.
        self.recordings_table.setSortingEnabled(False)
        self.recordings_table.setRowCount(len(recordings_data))
        self._preview_y_range = self._compute_preview_y_range(recordings_data)

        for row, data in enumerate(recordings_data):
            # Preview icon
            icon = self.generate_sparkline_icon(data["recording"], data["session"], width=60) if data["recording"] is not None else QIcon()
            item = QTableWidgetItem()
            item.setIcon(icon)
            self._style_preview_item(item, data)
            self.recordings_table.setItem(row, 0, item)

            # Recording ID
            item = QTableWidgetItem(data["recording"].id)
            item.setData(Qt.ItemDataRole.UserRole, row)
            self._style_preview_item(item, data)
            self.recordings_table.setItem(row, 1, item)

            # Session
            item = QTableWidgetItem(data["session_id"])
            self._style_preview_item(item, data)
            self.recordings_table.setItem(row, 2, item)

            # Stimulus
            item = QTableWidgetItem(f"{data['stimulus']:.3f}")
            self._style_preview_item(item, data)
            self.recordings_table.setItem(row, 3, item)

            # Status
            item = QTableWidgetItem(data["status"])
            self._style_preview_item(item, data)
            self.recordings_table.setItem(row, 4, item)

            evaluation = data["evaluation"]
            severity_item = QTableWidgetItem(evaluation["severity"])
            self._style_preview_item(severity_item, data)
            self.recordings_table.setItem(row, 5, severity_item)
            reasons = "; ".join(evaluation["reasons"]) or "—"
            reasons_item = QTableWidgetItem(reasons)
            reasons_item.setToolTip(reasons)
            self._style_preview_item(reasons_item, data)
            self.recordings_table.setItem(row, 6, reasons_item)
            metric_text = self._format_metrics(evaluation["metrics"])
            metrics_item = QTableWidgetItem(metric_text)
            metrics_item.setToolTip(metric_text)
            self._style_preview_item(metrics_item, data)
            self.recordings_table.setItem(row, 7, metrics_item)

        self.recordings_table.setSortingEnabled(True)
        self.recordings_table.sortItems(self._sort_column, self._sort_order)

        # Update summary
        total_recordings = len(recordings_data)
        currently_excluded = sum(1 for d in recordings_data if d["currently_excluded"])
        will_exclude = sum(1 for d in recordings_data if d["will_exclude"])

        summary_text = f"Total recordings: {total_recordings} | "
        summary_text += f"Currently excluded: {currently_excluded} | "
        summary_text += f"Pending exclusion: {will_exclude} | Range: {self.range_combo.currentText()}"

        self.summary_label.setText(summary_text)
        self._apply_preview_filter()

    def _compute_preview_y_range(self, recordings_data: list[dict[str, Any]]) -> tuple[float, float] | None:
        """Return a shared preview scale, loading each selected snippet at most once."""
        if self.preview_y_scale_combo.currentData() != "unified":
            return None
        samples = []
        for entry in recordings_data:
            trace = self._get_preview_samples(entry["recording"], entry["session"])
            if trace is not None:
                finite = np.asarray(trace, dtype=float).reshape(-1)
                finite = finite[np.isfinite(finite)]
                if finite.size:
                    samples.append((float(np.min(finite)), float(np.max(finite))))
        if not samples:
            return None
        ymin = min(pair[0] for pair in samples)
        ymax = max(pair[1] for pair in samples)
        return (ymin, ymax) if ymax > ymin else (ymin - 0.5, ymax + 0.5)

    def _session_outliers(self, metrics_by_recording: dict[str, dict[str, float | None]]) -> dict[str, set[str]]:
        """Find robust per-session RMS and burst outliers without expensive pairwise work."""
        result = {recording_id: set() for recording_id in metrics_by_recording}
        for metric_name in ("rms", "peak_to_peak", "burst_duration_ms"):
            values = np.asarray([metrics[metric_name] for metrics in metrics_by_recording.values() if metrics[metric_name] is not None], dtype=float)
            if values.size < 4:
                continue
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            scale = max(1.4826 * mad, abs(median) * 0.05, 1e-12)
            limit = median + self.outlier_z_spin.value() * scale
            for recording_id, metrics in metrics_by_recording.items():
                value = metrics[metric_name]
                if value is not None and value > limit:
                    result[recording_id].add(metric_name)
        return result

    @staticmethod
    def _format_metrics(metrics: dict[str, float | None]) -> str:
        names = {
            "snr": "SNR",
            "baseline_drift": "drift",
            "flatline": "std",
            "line_noise": "line",
            "rms": "RMS",
            "peak_to_peak": "p-p",
            "burst_duration_ms": "burst ms",
        }
        return ", ".join(f"{names[name]}={value:.3g}" for name, value in metrics.items() if value is not None)

    @staticmethod
    def _style_preview_item(item: QTableWidgetItem, data: dict[str, Any]) -> None:
        if data["manual_decision"] is False:
            item.setBackground(QColor("#d9edf7"))
        elif data["will_exclude"]:
            item.setBackground(QColor("#f2dede"))

    def _apply_preview_filter(self):
        mode = self.preview_filter_combo.currentData()
        for row in range(self.recordings_table.rowCount()):
            id_item = self.recordings_table.item(row, 1)
            data_index = id_item.data(Qt.ItemDataRole.UserRole) if id_item is not None else None
            if not isinstance(data_index, int) or data_index >= len(getattr(self, "_last_recordings_data", [])):
                continue
            data = self._last_recordings_data[data_index]
            show = mode == "all"
            if mode == "flagged":
                show = data["will_exclude"] or bool(data["evaluation"]["flagged"])
            elif mode == "excluded":
                show = data["currently_excluded"]
            elif mode == "included":
                show = not data["currently_excluded"] and not data["will_exclude"]
            self.recordings_table.setRowHidden(row, not show)

    def _selected_entries(self) -> list[dict[str, Any]]:
        if not hasattr(self, "_last_recordings_data") or not self._last_recordings_data:
            return []
        selected_rows = sorted({idx.row() for idx in self.recordings_table.selectionModel().selectedRows()})
        entries = []
        for row in selected_rows:
            id_item = self.recordings_table.item(row, 1)
            data_index = id_item.data(Qt.ItemDataRole.UserRole) if id_item is not None else None
            if isinstance(data_index, int) and data_index < len(self._last_recordings_data):
                entries.append(self._last_recordings_data[data_index])
        return entries

    def _show_selected_recording_detail(self) -> None:
        """Open/update a lightweight full-recording plot for the selected row."""
        entries = self._selected_entries()
        if not entries:
            self._close_detail_preview()
            return
        entry = entries[0]
        recording = entry["recording"]
        session = entry["session"]
        channel_indices = self._get_preview_channel_indices(recording)
        channel_index = channel_indices[0] if channel_indices else 0
        trace = self._get_recording_trace(recording, session, channel_index)
        if trace is None:
            return
        samples = np.asarray(trace, dtype=float).reshape(-1)
        finite = np.isfinite(samples)
        if not finite.any():
            return

        dialog = self._detail_preview_dialog or self._create_detail_preview_dialog()
        plot = self._detail_preview_plot
        plot.clear()
        max_points = 8000
        if samples.size > max_points:
            indices = np.linspace(0, samples.size - 1, max_points).astype(int)
            display_samples = samples[indices]
        else:
            indices = np.arange(samples.size)
            display_samples = samples

        scan_rate = float(getattr(recording, "scan_rate", 0.0))
        if scan_rate > 0:
            x_values = indices * 1000.0 / scan_rate
            plot.setLabel("bottom", "Time", units="ms")
        else:
            x_values = indices
            plot.setLabel("bottom", "Sample")
        plot.plot(x_values, display_samples, pen=pg.mkPen("#2b85d8", width=1))

        snippet = self._get_preview_snippet_time_window(recording, session)
        if snippet is not None and scan_rate > 0:
            region = pg.LinearRegionItem(
                values=(snippet[0] * 1000.0 / scan_rate, snippet[1] * 1000.0 / scan_rate),
                movable=False,
                brush=pg.mkBrush(43, 133, 216, 35),
            )
            region.setZValue(10)
            plot.addItem(region)

        channel_name = f"Ch {channel_index + 1}"
        if channel_index < len(recording.channel_types):
            channel_name += f" ({recording.channel_types[channel_index]})"
        self._detail_preview_title.setText(f"{entry['session_id']} / {recording.id} — {channel_name}")
        self._detail_preview_context.setText(
            "Full selected-channel trace. The shaded band is the waveform-preview snippet; "
            "the table sparkline uses that band only and may use an independent y-scale."
        )
        self._position_detail_preview()
        dialog.show()
        dialog.raise_()

    def _close_detail_preview(self) -> None:
        if self._detail_preview_dialog is not None and self._detail_preview_dialog.isVisible():
            self._detail_preview_dialog.close()

    def _position_detail_preview(self) -> None:
        if self._detail_preview_dialog is None:
            return
        editor_top_right = self.frameGeometry().topRight()
        self._detail_preview_dialog.move(editor_top_right.x() + 8, editor_top_right.y())

    def moveEvent(self, event) -> None:
        super().moveEvent(event)
        if self._detail_preview_dialog is not None and self._detail_preview_dialog.isVisible():
            self._position_detail_preview()

    def done(self, result: int) -> None:
        """Ensure the non-modal detail sidecar never outlives this editor."""
        self._close_detail_preview()
        super().done(result)

    def closeEvent(self, event) -> None:
        self._close_detail_preview()
        super().closeEvent(event)

    def _set_selected_manual_decision(self, exclude: bool):
        entries = self._selected_entries()
        if not entries:
            QMessageBox.information(self, "No Selection", "Select one or more recordings first.")
            return
        for entry in entries:
            self.manual_decisions[self._recording_key(entry["recording"], entry["session"])] = exclude
        self.update_preview()

    def _clear_selected_manual_decisions(self):
        entries = self._selected_entries()
        if not entries:
            QMessageBox.information(self, "No Selection", "Select one or more recordings first.")
            return
        for entry in entries:
            self.manual_decisions.pop(self._recording_key(entry["recording"], entry["session"]), None)
        self.update_preview()

    def toggle_selected_exclusions(self):
        """Stage an explicit exclusion for the selected rows."""
        self._set_selected_manual_decision(True)

    def auto_flag_low_quality(self):
        """Run quality checks now and stage explainable, reviewable auto-flags."""
        self.auto_flagged_recordings.clear()
        sessions = self.get_sessions_for_level()
        if not sessions:
            QMessageBox.information(self, "No Sessions", "No sessions available to auto-flag.")
            return

        for session in sessions:
            recordings = session.get_all_recordings(include_excluded=True)
            metrics_by_recording = {recording.id: self.compute_quality_metrics(recording, session) for recording in recordings}
            outliers = self._session_outliers(metrics_by_recording)
            for recording in recordings:
                metrics = metrics_by_recording[recording.id]
                # The quick action intentionally evaluates the visible thresholds
                # even before the normal automatic-quality group is enabled.
                original_enabled = self.quality_group.isChecked()
                if not original_enabled:
                    self.quality_group.blockSignals(True)
                    self.quality_group.setChecked(True)
                    self.quality_group.blockSignals(False)
                evaluation = self._evaluation_for_recording(recording, session, metrics, outliers.get(recording.id, set()))
                if not original_enabled:
                    self.quality_group.blockSignals(True)
                    self.quality_group.setChecked(False)
                    self.quality_group.blockSignals(False)
                if evaluation["flagged"]:
                    self.auto_flagged_recordings[self._recording_key(recording, session)] = {
                        "reasons": evaluation["reasons"],
                        "metrics": metrics,
                    }

        # Refresh preview to show manual flags
        self.update_preview()

    def clear_auto_flags(self):
        """Discard the currently staged automatic flags while preserving manual review work."""
        if not self.auto_flagged_recordings:
            return
        self.auto_flagged_recordings.clear()
        self.update_preview()

    def export_curation_report(self):
        """Export the current review state, including reasons and measurements."""
        path, _ = QFileDialog.getSaveFileName(self, "Export Curation Report", filter="JSON Files (*.json)")
        if not path:
            return
        report = []
        for entry in getattr(self, "_last_recordings_data", []):
            report.append(
                {
                    "session_id": entry["session_id"],
                    "recording_id": entry["recording"].id,
                    "stimulus_v": entry["stimulus"],
                    "status": entry["status"],
                    "pending_exclusion": entry["will_exclude"],
                    "severity": entry["evaluation"]["severity"],
                    "reasons": entry["evaluation"]["reasons"],
                    "metrics": entry["evaluation"]["metrics"],
                    "range": self.range_combo.currentData(),
                }
            )
        try:
            with open(path, "w", encoding="utf-8") as file:
                json.dump(report, file, indent=2)
            self.gui.status_bar.showMessage(f"Exported curation report: {path}", 5000)
        except OSError as error:
            logger.error("Failed to export curation report", exc_info=True)
            QMessageBox.critical(self, "Export Failed", f"Could not export the curation report:\n{error}")

    def save_profile(self):
        """Save current criteria to a JSON profile."""
        profile = {
            "stimulus": {
                "enabled": bool(self.stimulus_group.isChecked()),
                "type": self.threshold_type_combo.currentData(),
                "threshold1": float(self.threshold_spinbox.value()),
                "threshold2": float(self.threshold2_spinbox.value()),
            },
            "quality": {
                "enabled": bool(self.quality_group.isChecked()),
                "snr": float(self.snr_spin.value()),
                "drift": float(self.drift_spin.value()),
                "flatline": float(self.flatline_spin.value()),
                "line_noise": float(self.line_noise_spin.value()),
                "burst_duration_ms": float(self.burst_duration_spin.value()),
                "outlier_z": float(self.outlier_z_spin.value()),
            },
            "range": {
                "mode": self.range_combo.currentData(),
                "start_ms": float(self.range_start_spin.value()),
                "end_ms": float(self.range_end_spin.value()),
            },
            "preview": {
                "mode": self.preview_range_combo.currentData(),
                "start_ms": float(self.preview_start_spin.value()),
                "end_ms": float(self.preview_end_spin.value()),
                "y_scale": self.preview_y_scale_combo.currentData(),
                "channel_index": self.preview_channel_index,
            },
        }

        path, _ = QFileDialog.getSaveFileName(self, "Save Exclusion Profile", filter="JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(profile, f, indent=2)
            self.gui.status_bar.showMessage(f"Saved exclusion profile: {path}", 5000)
        except Exception as e:
            logger.error(f"Failed to save profile: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save profile:\n{e}")

    def load_profile(self):
        """Load an exclusion profile from JSON and apply to UI (preview only)."""
        path, _ = QFileDialog.getOpenFileName(self, "Load Exclusion Profile", filter="JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                profile = json.load(f)

            stim = profile.get("stimulus", {})
            self.stimulus_group.setChecked(bool(stim.get("enabled", False)))
            # find corresponding index for threshold type
            t = stim.get("type", "above")
            idx = 0
            for i in range(self.threshold_type_combo.count()):
                if self.threshold_type_combo.itemData(i) == t:
                    idx = i
                    break
            self.threshold_type_combo.setCurrentIndex(idx)
            self.threshold_spinbox.setValue(float(stim.get("threshold1", 1.0)))
            self.threshold2_spinbox.setValue(float(stim.get("threshold2", 5.0)))

            q = profile.get("quality", {})
            self.quality_group.setChecked(bool(q.get("enabled", False)))
            self.snr_spin.setValue(float(q.get("snr", 2.0)))
            self.drift_spin.setValue(float(q.get("drift", 0.20)))
            self.flatline_spin.setValue(float(q.get("flatline", 1e-6)))
            self.line_noise_spin.setValue(float(q.get("line_noise", 0.0)))
            self.burst_duration_spin.setValue(float(q.get("burst_duration_ms", 10.0)))
            self.outlier_z_spin.setValue(float(q.get("outlier_z", 5.0)))

            range_profile = profile.get("range", {})
            mode = range_profile.get("mode", "profile")
            self.range_combo.setCurrentIndex(max(0, self.range_combo.findData(mode)))
            self.range_start_spin.setValue(float(range_profile.get("start_ms", 0.0)))
            self.range_end_spin.setValue(float(range_profile.get("end_ms", 20.0)))

            preview_profile = profile.get("preview", {})
            self.preview_range_combo.setCurrentIndex(max(0, self.preview_range_combo.findData(preview_profile.get("mode", "profile"))))
            self.preview_start_spin.setValue(float(preview_profile.get("start_ms", 0.0)))
            self.preview_end_spin.setValue(float(preview_profile.get("end_ms", 20.0)))
            self.preview_y_scale_combo.setCurrentIndex(max(0, self.preview_y_scale_combo.findData(preview_profile.get("y_scale", "individual"))))
            channel_index = preview_profile.get("channel_index")
            self.preview_channel_index = int(channel_index) if isinstance(channel_index, int) else None

            self.update_preview()
            self.gui.status_bar.showMessage(f"Loaded exclusion profile: {path}", 5000)
        except Exception as e:
            logger.error(f"Failed to load profile: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load profile:\n{e}")

    def reset_criteria(self):
        """Reset all criteria to default values."""
        # Reset stimulus amplitude criteria
        self.stimulus_group.setChecked(False)
        self.threshold_type_combo.setCurrentIndex(0)
        self.threshold_spinbox.setValue(1.0)
        self.threshold2_spinbox.setValue(5.0)
        self.quality_group.setChecked(False)
        self.snr_spin.setValue(2.0)
        self.drift_spin.setValue(0.20)
        self.flatline_spin.setValue(1e-6)
        self.line_noise_spin.setValue(0.0)
        self.burst_duration_spin.setValue(10.0)
        self.outlier_z_spin.setValue(5.0)
        self.range_combo.setCurrentIndex(0)
        self.range_start_spin.setValue(0.0)
        self.range_end_spin.setValue(20.0)
        self.preview_range_combo.setCurrentIndex(0)
        self.preview_start_spin.setValue(0.0)
        self.preview_end_spin.setValue(20.0)
        self.preview_y_scale_combo.setCurrentIndex(0)
        self.preview_channel_index = None
        self.manual_decisions.clear()
        self.auto_flagged_recordings.clear()
        self._clear_preview_caches()

        self.update_preview()

    def apply_exclusions(self):
        """Commit the reviewed automatic and manual decisions as one undoable action."""
        if not getattr(self, "_last_recordings_data", None):
            QMessageBox.warning(self, "No Sessions", "No sessions available to apply exclusions to.")
            return

        changes_by_session: dict[Any, list[dict[str, Any]]] = {}
        total_exclusions = total_inclusions = 0
        for entry in self._last_recordings_data:
            should_exclude = entry["will_exclude"]
            currently_excluded = entry["currently_excluded"]
            if should_exclude == currently_excluded:
                continue
            evaluation = entry["evaluation"]
            curation = {
                "source": "manual" if entry["manual_decision"] is not None else "automatic",
                "decision": "exclude" if should_exclude else "include",
                "reasons": evaluation["reasons"],
                "metrics": {key: value for key, value in evaluation["metrics"].items() if value is not None},
                "range": self.range_combo.currentData(),
                "timestamp": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
                "rule_version": 1,
            }
            changes_by_session.setdefault(entry["session"], []).append(
                {"recording_id": entry["recording"].id, "exclude": should_exclude, "curation": curation}
            )
            if should_exclude:
                total_exclusions += 1
            else:
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

            changes = [{"session": session, "changes": session_changes} for session, session_changes in changes_by_session.items()]

            if changes:
                command = BulkRecordingExclusionCommand(self.gui, changes)
                self.gui.command_invoker.execute(command)

                self.exclusions_applied.emit()
                self.accept()

                self.gui.status_bar.showMessage(f"Applied exclusion criteria: {total_exclusions} excluded, {total_inclusions} included", 5000)

        except ImportError:
            # Fallback: if command class is not importable, raise a clear error
            # instead of silently applying non-undoable changes. This prevents
            # accidental data loss and encourages wiring the command system.
            logger.error("BulkRecordingExclusionCommand not available - Command pattern not installed or import failed.")
            QMessageBox.critical(
                self,
                "Internal Error",
                "Cannot apply exclusions because the undo/redo command system is not available. Please restart the app or contact support.",
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply exclusions:\n{e!s}")
            logger.error(f"Error applying recording exclusions: {e}")
