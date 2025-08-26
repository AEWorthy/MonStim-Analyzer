from typing import TYPE_CHECKING, List

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIntValidator
from PyQt6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from monstim_gui.core.responsive_widgets import ResponsiveComboBox
from monstim_gui.core.utils.custom_gui_elements import FloatLineEdit

from .plotting_cycler import RecordingCyclerWidget

if TYPE_CHECKING:
    from monstim_gui import MonstimGUI

    from .plotting_widget import PlotWidget

# TODO: "Reflex Amplitude Calculation Method" is too long to fit in the combo box.
# Consider abbreviating or using a tooltip.


# Base class for plot options
class BasePlotOptions(QWidget):
    def __init__(self, parent: "PlotWidget"):
        super().__init__(parent)
        self.gui_main = parent.parent
        self.layout: QVBoxLayout = QVBoxLayout(self)
        self.layout.setSpacing(0)  # No spacing between widgets
        self.layout.setContentsMargins(4, 4, 4, 4)  # Set smaller margins
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.create_options()

    def create_form_layout(self):
        """Create a standardized form layout with consistent styling"""
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(4)  # Slightly increased vertical spacing for better readability
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)  # Keep everything on one row
        return form

    def create_options(self):
        # To be implemented by subclasses
        raise NotImplementedError("Subclasses must implement create_options()")

    def get_options(self):
        # To be implemented by subclasses
        raise NotImplementedError("Subclasses must implement get_options()")

    def set_options(self, options):
        # To be implemented by subclasses
        raise NotImplementedError("Subclasses must implement set_options()")


class ChannelSelectorWidget(QGroupBox):
    def __init__(self, gui_main: "MonstimGUI", parent=None):
        super().__init__("Channel Selector", parent)

        # Figure out how many channels we should allow for the current view
        view = gui_main.plot_widget.view
        if view == "session":
            emg_data = gui_main.current_session
        elif view == "dataset":
            emg_data = gui_main.current_dataset
        elif view == "experiment":
            emg_data = gui_main.current_experiment
        else:
            emg_data = None

        max_ch = getattr(emg_data, "num_channels", 0)

        # Set up a grid layout with proper spacing and margins
        grid = QGridLayout()
        grid.setSpacing(6)  # Increased spacing between checkboxes
        grid.setContentsMargins(8, 8, 8, 8)  # Better padding to prevent border clipping

        # Set size policy to make it as compact as possible
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.checkboxes: List[QCheckBox] = []
        total = (max_ch + 5) // 6 * 6  # Round up to the nearest multiple of 6
        for idx in range(total):
            cb = QCheckBox(f"{idx}")
            # Only enable the ones your data actually has
            cb.setEnabled(idx < max_ch)
            cb.setChecked(idx < max_ch)
            row, col = divmod(idx, 6)
            # Center the checkboxes in their cells for better alignment
            grid.addWidget(cb, row, col, alignment=Qt.AlignmentFlag.AlignCenter)
            self.checkboxes.append(cb)

        self.setLayout(grid)

    def get_selected_channels(self) -> List[int]:
        return [i for i, cb in enumerate(self.checkboxes) if cb.isChecked()]

    def set_selected_channels(self, selected: List[int]):
        for i, cb in enumerate(self.checkboxes):
            if cb.isEnabled():
                cb.setChecked(i in selected)


# EMG Options
class EMGOptions(BasePlotOptions):
    def create_options(self):
        # Data type options box
        form = self.create_form_layout()

        self.data_type_combo = ResponsiveComboBox()
        self.data_type_combo.addItems(["filtered", "raw", "rectified_raw", "rectified_filtered"])
        form.addRow("Select Data Type:", self.data_type_combo)

        # flags / legend / colormap
        self.all_windows_checkbox = QCheckBox()
        self.all_windows_checkbox.setToolTip("If checked, all latency windows will be shown in the plot.")
        self.latency_legend_checkbox = QCheckBox()
        self.latency_legend_checkbox.setToolTip("If checked, the latency window legend will be shown in the plot.")
        self.plot_colormap_checkbox = QCheckBox()
        self.plot_colormap_checkbox.setToolTip("If checked, a colormap legend will be shown to the side of the plot.")
        form.addRow("Show Flags:", self.all_windows_checkbox)
        self.all_windows_checkbox.setChecked(True)
        self.all_windows_checkbox.stateChanged.connect(self._on_all_windows_toggled)
        self._on_all_windows_toggled(self.all_windows_checkbox.checkState())
        form.addRow("Show Legend:", self.latency_legend_checkbox)
        self.latency_legend_checkbox.setChecked(True)
        form.addRow("Show Colormap:", self.plot_colormap_checkbox)
        self.plot_colormap_checkbox.setChecked(True)
        self.interactive_cursor_checkbox = QCheckBox()
        self.interactive_cursor_checkbox.setToolTip("If checked, an interactive crosshair cursor will be shown in the plot.")
        self.interactive_cursor_checkbox.setChecked(False)
        form.addRow("Show Interactive Cursor:", self.interactive_cursor_checkbox)

        # Create the channel selector
        self.channel_selector = ChannelSelectorWidget(self.gui_main, parent=self)

        # Add widgets to layout with proper spacing
        options_widget = QWidget()
        options_widget.setLayout(form)
        options_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.layout.addWidget(options_widget)
        self.layout.addWidget(self.channel_selector)
        self.layout.addStretch(1)

    def _on_all_windows_toggled(self, state):
        # Enable or disable the latency legend checkbox based on the state of the all_windows_checkbox
        enabled = state == Qt.CheckState.Checked or state == 2
        self.latency_legend_checkbox.setChecked(enabled)
        self.latency_legend_checkbox.setEnabled(enabled)

    def get_options(self):
        return {
            "channel_indices": self.channel_selector.get_selected_channels(),
            "data_type": self.data_type_combo.currentText(),
            "all_flags": self.all_windows_checkbox.isChecked(),
            "plot_legend": self.latency_legend_checkbox.isChecked(),
            "plot_colormap": self.plot_colormap_checkbox.isChecked(),
            "interactive_cursor": self.interactive_cursor_checkbox.isChecked(),
        }

    def set_options(self, options):
        if "channel_indices" in options:
            self.channel_selector.set_selected_channels(options["channel_indices"])
        if "data_type" in options:
            index = self.data_type_combo.findText(options["data_type"])
            if index >= 0:
                self.data_type_combo.setCurrentIndex(index)
        if "all_flags" in options:
            self.all_windows_checkbox.setChecked(options["all_flags"])
        if "plot_legend" in options:
            self.latency_legend_checkbox.setChecked(options["plot_legend"])
        if "plot_colormap" in options:
            self.plot_colormap_checkbox.setChecked(options["plot_colormap"])
        if "interactive_cursor" in options:
            self.interactive_cursor_checkbox.setChecked(options["interactive_cursor"])


class SingleEMGRecordingOptions(BasePlotOptions):
    def create_options(self):
        form = self.create_form_layout()

        self.data_type_combo = ResponsiveComboBox()
        self.data_type_combo.addItems(["filtered", "raw", "rectified_raw", "rectified_filtered"])
        form.addRow("Select Data Type:", self.data_type_combo)

        # Create and add checkboxes
        self.all_windows_checkbox = QCheckBox()
        self.all_windows_checkbox.setToolTip("If checked, all analysis windows will be shown in the plot.")
        self.latency_legend_checkbox = QCheckBox()
        self.latency_legend_checkbox.setToolTip("If checked, a legend for the latency markers will be shown in the plot.")
        self.plot_colormap_checkbox = QCheckBox()
        self.plot_colormap_checkbox.setToolTip("If checked, a colormap will be shown in the plot.")
        self.fixed_y_axis_checkbox = QCheckBox()
        self.fixed_y_axis_checkbox.setToolTip("If checked, the y-axis will be fixed to a range of [-1, 1].")
        self.interactive_cursor_checkbox = QCheckBox()
        self.interactive_cursor_checkbox.setToolTip("If checked, an interactive crosshair cursor will be shown in the plot.")

        # Add checkboxes to form
        form.addRow("Show Flags:", self.all_windows_checkbox)
        self.all_windows_checkbox.setChecked(True)
        self.all_windows_checkbox.stateChanged.connect(self._on_all_windows_toggled)
        self._on_all_windows_toggled(self.all_windows_checkbox.checkState())

        form.addRow("Show Legend:", self.latency_legend_checkbox)
        self.latency_legend_checkbox.setChecked(True)

        form.addRow("Show Colormap:", self.plot_colormap_checkbox)
        self.plot_colormap_checkbox.setChecked(True)

        form.addRow("Fixed Y-Axis:", self.fixed_y_axis_checkbox)
        self.fixed_y_axis_checkbox.setChecked(True)  # Set the initial state to True

        form.addRow("Show Interactive Cursor:", self.interactive_cursor_checkbox)
        self.interactive_cursor_checkbox.setChecked(False)

        # Create the recording cycler widget and add it to the form
        self.recording_cycler = RecordingCyclerWidget(self)

        # Create the channel selector widget and add it to the form
        self.channel_selector = ChannelSelectorWidget(self.gui_main, parent=self)

        # Add widgets to layout with proper spacing and organization
        options_widget = QWidget()
        options_widget.setLayout(form)
        options_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.layout.addWidget(options_widget)
        self.layout.addWidget(self.recording_cycler)
        self.layout.addWidget(self.channel_selector)
        self.layout.addStretch(1)

    def _on_all_windows_toggled(self, state):
        # Enable or disable the latency legend checkbox based on the state of the all_windows_checkbox
        enabled = state == Qt.CheckState.Checked or state == 2
        self.latency_legend_checkbox.setChecked(enabled)
        self.latency_legend_checkbox.setEnabled(enabled)

    def get_options(self):
        return {
            "channel_indices": self.channel_selector.get_selected_channels(),
            "data_type": self.data_type_combo.currentText(),
            "all_flags": self.all_windows_checkbox.isChecked(),
            "plot_legend": self.latency_legend_checkbox.isChecked(),
            "recording_index": self.recording_cycler.get_current_recording(),
            "fixed_y_axis": self.fixed_y_axis_checkbox.isChecked(),
            "plot_colormap": self.plot_colormap_checkbox.isChecked(),
            "interactive_cursor": self.interactive_cursor_checkbox.isChecked(),
        }

    def set_options(self, options):
        if "channel_indices" in options:
            self.channel_selector.set_selected_channels(options["channel_indices"])
        if "data_type" in options:
            index = self.data_type_combo.findText(options["data_type"])
            if index >= 0:
                self.data_type_combo.setCurrentIndex(index)
        if "all_flags" in options:
            self.all_windows_checkbox.setChecked(options["all_flags"])
        if "plot_legend" in options:
            self.latency_legend_checkbox.setChecked(options["plot_legend"])
        if "recording_index" in options:
            self.recording_cycler.recording_spinbox.setValue(options["recording_index"])
        if "fixed_y_axis" in options:
            self.fixed_y_axis_checkbox.setChecked(options["fixed_y_axis"])
        if "plot_colormap" in options:
            self.plot_colormap_checkbox.setChecked(options["plot_colormap"])
        if "interactive_cursor" in options:
            self.interactive_cursor_checkbox.setChecked(options["interactive_cursor"])


class SessionReflexCurvesOptions(BasePlotOptions):
    def create_options(self):
        form = self.create_form_layout()

        self.method_combo = ResponsiveComboBox()
        self.method_combo.addItems(["rms", "average_rectified", "average_unrectified", "peak_to_trough"])
        self.method_combo.setCurrentIndex(0)  # Set the initial selection to "rms"
        form.addRow("Reflex Amplitude Calculation Method:", self.method_combo)

        # Checkboxes
        self.relative_to_mmax_checkbox = QCheckBox()
        self.relative_to_mmax_checkbox.setToolTip(
            "If checked, the reflex amplitudes will be calculated relative to the M-max value."
        )
        form.addRow("Relative to M-max:", self.relative_to_mmax_checkbox)
        self.relative_to_mmax_checkbox.setChecked(True)

        self.show_legend_checkbox = QCheckBox()
        self.show_legend_checkbox.setToolTip("If checked, the plot legend will be shown.")
        form.addRow("Show Plot Legend:", self.show_legend_checkbox)
        self.show_legend_checkbox.setChecked(True)

        self.interactive_cursor_checkbox = QCheckBox()
        self.interactive_cursor_checkbox.setToolTip("If checked, an interactive crosshair cursor will be shown in the plot.")
        form.addRow("Show Interactive Cursor:", self.interactive_cursor_checkbox)
        self.interactive_cursor_checkbox.setChecked(False)

        # Create the channel selector
        self.channel_selector = ChannelSelectorWidget(self.gui_main, parent=self)

        # Add widgets to layout with proper spacing
        options_widget = QWidget()
        options_widget.setLayout(form)
        options_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.layout.addWidget(options_widget)
        self.layout.addWidget(self.channel_selector)
        self.layout.addStretch(1)

    def get_options(self):
        return {
            "channel_indices": self.channel_selector.get_selected_channels(),
            "method": self.method_combo.currentText(),
            "relative_to_mmax": self.relative_to_mmax_checkbox.isChecked(),
            "plot_legend": self.show_legend_checkbox.isChecked(),
            "interactive_cursor": self.interactive_cursor_checkbox.isChecked(),
        }

    def set_options(self, options):
        if "channel_indices" in options:
            self.channel_selector.set_selected_channels(options["channel_indices"])
        if "method" in options:
            index = self.method_combo.findText(options["method"])
            if index >= 0:
                self.method_combo.setCurrentIndex(index)
        if "relative_to_mmax" in options:
            self.relative_to_mmax_checkbox.setChecked(options["relative_to_mmax"])
        if "plot_legend" in options:
            self.show_legend_checkbox.setChecked(options["plot_legend"])
        if "interactive_cursor" in options:
            self.interactive_cursor_checkbox.setChecked(options["interactive_cursor"])


class AverageReflexCurvesOptions(BasePlotOptions):
    def create_options(self):
        form = self.create_form_layout()

        self.method_combo = ResponsiveComboBox()
        self.method_combo.addItems(["rms", "average_rectified", "average_unrectified", "peak_to_trough"])
        self.method_combo.setCurrentIndex(0)  # Set the initial selection to "rms"
        form.addRow("Reflex Amplitude Calculation Method:", self.method_combo)

        # Checkboxes
        self.relative_to_mmax_checkbox = QCheckBox()
        self.relative_to_mmax_checkbox.setToolTip(
            "If checked, the reflex amplitudes will be calculated relative to the M-max value."
        )
        self.show_legend_checkbox = QCheckBox()
        self.show_legend_checkbox.setToolTip("If checked, the plot legend will be shown.")
        form.addRow("Relative to M-max:", self.relative_to_mmax_checkbox)
        self.relative_to_mmax_checkbox.setChecked(True)
        form.addRow("Show Plot Legend:", self.show_legend_checkbox)
        self.show_legend_checkbox.setChecked(True)

        # Create the channel selector
        self.channel_selector = ChannelSelectorWidget(self.gui_main, parent=self)

        # Add widgets to layout with proper spacing
        options_widget = QWidget()
        options_widget.setLayout(form)
        options_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.layout.addWidget(options_widget)
        self.layout.addWidget(self.channel_selector)
        self.layout.addStretch(1)

    def get_options(self):
        return {
            "channel_indices": self.channel_selector.get_selected_channels(),
            "method": self.method_combo.currentText(),
            "relative_to_mmax": self.relative_to_mmax_checkbox.isChecked(),
            "plot_legend": self.show_legend_checkbox.isChecked(),
        }

    def set_options(self, options):
        if "channel_indices" in options:
            self.channel_selector.set_selected_channels(options["channel_indices"])
        if "method" in options:
            index = self.method_combo.findText(options["method"])
            if index >= 0:
                self.method_combo.setCurrentIndex(index)
        if "relative_to_mmax" in options:
            self.relative_to_mmax_checkbox.setChecked(options["relative_to_mmax"])
        if "plot_legend" in options:
            self.show_legend_checkbox.setChecked(options["plot_legend"])


class AverageSessionReflexOptions(BasePlotOptions):
    def create_options(self):
        form = self.create_form_layout()

        self.method_combo = ResponsiveComboBox()
        self.method_combo.addItems(["rms", "average_rectified", "average_unrectified", "peak_to_trough"])
        self.method_combo.setCurrentIndex(0)  # Set the initial selection to "rms"
        form.addRow("Reflex Amplitude Calculation Method:", self.method_combo)

        # Checkboxes
        self.relative_to_mmax_checkbox = QCheckBox()
        self.relative_to_mmax_checkbox.setToolTip(
            "If checked, the reflex amplitudes will be calculated relative to the M-max value."
        )
        self.show_legend_checkbox = QCheckBox()
        self.show_legend_checkbox.setToolTip("If checked, the plot legend will be shown.")
        self.interactive_cursor_checkbox = QCheckBox()
        self.interactive_cursor_checkbox.setToolTip("If checked, an interactive crosshair cursor will be shown in the plot.")

        form.addRow("Relative to M-max:", self.relative_to_mmax_checkbox)
        self.relative_to_mmax_checkbox.setChecked(False)  # Default to False
        form.addRow("Show Plot Legend:", self.show_legend_checkbox)
        self.show_legend_checkbox.setChecked(True)
        form.addRow("Show Interactive Cursor:", self.interactive_cursor_checkbox)
        self.interactive_cursor_checkbox.setChecked(False)

        # Create the channel selector
        self.channel_selector = ChannelSelectorWidget(self.gui_main, parent=self)

        # Add widgets to layout with proper spacing
        options_widget = QWidget()
        options_widget.setLayout(form)
        options_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.layout.addWidget(options_widget)
        self.layout.addWidget(self.channel_selector)
        self.layout.addStretch(1)

    def get_options(self):
        return {
            "channel_indices": self.channel_selector.get_selected_channels(),
            "method": self.method_combo.currentText(),
            "relative_to_mmax": self.relative_to_mmax_checkbox.isChecked(),
            "plot_legend": self.show_legend_checkbox.isChecked(),
            "interactive_cursor": self.interactive_cursor_checkbox.isChecked(),
        }

    def set_options(self, options):
        if "channel_indices" in options:
            self.channel_selector.set_selected_channels(options["channel_indices"])
        if "method" in options:
            index = self.method_combo.findText(options["method"])
            if index >= 0:
                self.method_combo.setCurrentIndex(index)
        if "relative_to_mmax" in options:
            self.relative_to_mmax_checkbox.setChecked(options["relative_to_mmax"])
        if "plot_legend" in options:
            self.show_legend_checkbox.setChecked(options["plot_legend"])
        if "interactive_cursor" in options:
            self.interactive_cursor_checkbox.setChecked(options["interactive_cursor"])


class MMaxOptions(BasePlotOptions):
    def create_options(self):
        form = self.create_form_layout()

        self.method_combo = ResponsiveComboBox()
        self.method_combo.addItems(["rms", "average_rectified", "average_unrectified", "peak_to_trough"])
        self.method_combo.setCurrentIndex(0)  # Set the initial selection to "rms"
        form.addRow("Reflex Amplitude Calculation Method:", self.method_combo)

        # Checkboxes
        self.interactive_cursor_checkbox = QCheckBox()
        self.interactive_cursor_checkbox.setToolTip("If checked, an interactive crosshair cursor will be shown in the plot.")
        self.interactive_cursor_checkbox.setChecked(False)
        form.addRow("Show Interactive Cursor:", self.interactive_cursor_checkbox)

        # Create the channel selector
        self.channel_selector = ChannelSelectorWidget(self.gui_main, parent=self)

        # Add widgets to layout with proper spacing
        options_widget = QWidget()
        options_widget.setLayout(form)
        options_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.layout.addWidget(options_widget)
        self.layout.addWidget(self.channel_selector)
        self.layout.addStretch(1)

    def get_options(self):
        return {
            "channel_indices": self.channel_selector.get_selected_channels(),
            "method": self.method_combo.currentText(),
            "interactive_cursor": self.interactive_cursor_checkbox.isChecked(),
        }

    def set_options(self, options):
        if "channel_indices" in options:
            self.channel_selector.set_selected_channels(options["channel_indices"])
        if "method" in options:
            index = self.method_combo.findText(options["method"])
            if index >= 0:
                self.method_combo.setCurrentIndex(index)
        if "interactive_cursor" in options:
            self.interactive_cursor_checkbox.setChecked(options["interactive_cursor"])


class MaxHReflexOptions(BasePlotOptions):
    def create_options(self):
        form = self.create_form_layout()

        self.method_combo = ResponsiveComboBox()
        self.method_combo.addItems(["rms", "average_rectified", "average_unrectified", "peak_to_trough"])
        self.method_combo.setCurrentIndex(0)  # Set the initial selection to "rms"
        form.addRow("Reflex Amplitude Calculation Method:", self.method_combo)

        # Checkboxes
        self.relative_to_mmax_checkbox = QCheckBox()
        self.relative_to_mmax_checkbox.setToolTip(
            "If checked, the reflex amplitudes will be calculated relative to the M-max value."
        )
        form.addRow("Relative to M-max:", self.relative_to_mmax_checkbox)
        self.relative_to_mmax_checkbox.setChecked(False)

        self.max_stim_value = FloatLineEdit(default_value=10.0)
        self.max_stim_value.setPlaceholderText("(float)")
        self.max_stim_value.setToolTip(
            "Maximum value of the stimulus (in V) that will be used to calculate the average reflex amplitudes."
        )
        self.max_stim_value.setMaximumWidth(80)
        form.addRow("Max Stimulus Value:", self.max_stim_value)

        self.bin_margin_input = QLineEdit()
        self.bin_margin_input.setValidator(QIntValidator())
        self.bin_margin_input.setText("0")
        self.bin_margin_input.setPlaceholderText("(integer)")
        self.bin_margin_input.setToolTip(
            "Number of bins to add to the left and right of the maximum stimulus value to add nerby datapoints to the average reflex calculation."
        )
        form.addRow("Bin Margin:", self.bin_margin_input)

        # Create the channel selector
        self.channel_selector = ChannelSelectorWidget(self.gui_main, parent=self)

        # Add widgets to layout with proper spacing
        options_widget = QWidget()
        options_widget.setLayout(form)
        options_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.layout.addWidget(options_widget)
        self.layout.addWidget(self.channel_selector)
        self.layout.addStretch(1)

    def get_options(self):
        return {
            "channel_indices": self.channel_selector.get_selected_channels(),
            "method": self.method_combo.currentText(),
            "relative_to_mmax": self.relative_to_mmax_checkbox.isChecked(),
            "max_stim_value": self.max_stim_value.get_value(),
            "bin_margin": int(self.bin_margin_input.text()),
        }

    def set_options(self, options):
        if "channel_indices" in options:
            self.channel_selector.set_selected_channels(options["channel_indices"])
        if "method" in options:
            index = self.method_combo.findText(options["method"])
            if index >= 0:
                self.method_combo.setCurrentIndex(index)
        if "relative_to_mmax" in options:
            self.relative_to_mmax_checkbox.setChecked(bool(options["relative_to_mmax"]))
        if "max_stim_value" in options:
            self.max_stim_value.set_value(float(options["max_stim_value"]))
        if "bin_margin" in options:
            self.bin_margin_input.setText(str(int(options["bin_margin"])))


# No longer used, but kept for backwards compatibility
class SuspectedHReflexesOptions(BasePlotOptions):
    """Deprecated plot type, kept for backwards compatibility"""

    def create_options(self):
        # H threshold option
        h_threshold_layout = QHBoxLayout()
        self.h_threshold_label = QLabel("H Threshold:")
        self.h_threshold_input = FloatLineEdit(default_value=0.5)
        self.h_threshold_input.setPlaceholderText("H-relex Threshold (mV):")
        h_threshold_layout.addWidget(self.h_threshold_label)
        h_threshold_layout.addWidget(self.h_threshold_input)
        self.layout.addLayout(h_threshold_layout)

        # H-reflex calculation method option
        h_method_layout = QHBoxLayout()
        self.h_method_label = QLabel("H-reflex Calculation Method:")
        self.h_method_combo = ResponsiveComboBox()
        self.h_method_combo.addItems(["rms", "average_rectified", "average_unrectified", "peak_to_trough"])
        self.h_method_combo.setCurrentIndex(0)  # Set the initial selection to "rms"
        h_method_layout.addWidget(self.h_method_label)
        h_method_layout.addWidget(self.h_method_combo)
        self.layout.addLayout(h_method_layout)

        # plot_legend option (checkbox)
        self.plot_legend_label = QLabel("Show Plot Legend:")
        self.plot_legend_checkbox = QCheckBox()
        self.plot_legend_checkbox.setChecked(True)  # Set the initial state to True
        self.layout.addWidget(self.plot_legend_label)
        self.layout.addWidget(self.plot_legend_checkbox)

        # Channel selection
        self.channel_selector = ChannelSelectorWidget(self.gui_main, parent=self)
        self.layout.addWidget(self.channel_selector)

    def get_options(self):
        return {
            "channel_indices": self.channel_selector.get_selected_channels(),
            "h_threshold": self.h_threshold_input.get_value(),
            "method": self.h_method_combo.currentText(),
            "plot_legend": self.plot_legend_checkbox.isChecked(),
        }

    def set_options(self, options):
        if "channel_indices" in options:
            self.channel_selector.set_selected_channels(options["channel_indices"])
        if "h_threshold" in options:
            self.h_threshold_input.set_value(options["h_threshold"])
        if "method" in options:
            index = self.h_method_combo.findText(options["method"])
            if index >= 0:
                self.h_method_combo.setCurrentIndex(index)
        if "plot_legend" in options:
            self.plot_legend_checkbox.setChecked(options["plot_legend"])
