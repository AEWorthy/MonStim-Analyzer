from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox, QLineEdit, QGridLayout, QFormLayout, QGroupBox
from PyQt6.QtGui import QIntValidator
from monstim_gui.core.utils.custom_gui_elements import FloatLineEdit
from monstim_gui.widgets.plotting.plotting_cycler import RecordingCyclerWidget
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from monstim_gui import MonstimGUI
    from .plotting_widget import PlotWidget

# Things to do:
# - change flag system to a more general system of latency windows.
# - Add option to show or hide the plot legend for emg plots
# - Add manual mmax option

# Base class for plot options
class BasePlotOptions(QWidget):
    def __init__(self, parent : 'PlotWidget'):
        super().__init__(parent)
        self.gui_main = parent.parent
        self.layout = QVBoxLayout(self)
        self.create_options()

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
    def __init__(self, gui_main: 'MonstimGUI', parent=None):
        super().__init__("Channels", parent)
        # figure out how many channels we should allow
        view = gui_main.plot_widget.view
        if view == "session":
            emg_data = gui_main.current_session
        elif view == "dataset":
            emg_data = gui_main.current_dataset
        elif view == "experiment":
            emg_data = gui_main.current_experiment
        else:
            emg_data = None

        max_ch = getattr(emg_data, 'num_channels', 0)

        # set up a 2×3 grid
        grid = QGridLayout()
        grid.setSpacing(6)
        grid.setContentsMargins(4, 4, 4, 4)

        self.checkboxes: List[QCheckBox] = []
        total = (max_ch + 5) // 6 * 6  # Round up to the nearest multiple of 6
        for idx in range(total):
            cb = QCheckBox(f"{idx}")
            # only enable the ones your data actually has
            cb.setEnabled(idx < max_ch)
            cb.setChecked(idx < max_ch)
            row, col = divmod(idx, 6)
            grid.addWidget(cb, row, col, alignment=Qt.AlignmentFlag.AlignLeft)
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
        ## Data type options box
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(4)

        self.data_type_combo = QComboBox()
        self.data_type_combo.addItems([
            "filtered", "raw", "rectified_raw", "rectified_filtered"
        ])
        form.addRow("Select Data Type:", self.data_type_combo)

        # flags / legend / colormap
        self.all_windows_checkbox    = QCheckBox()
        self.all_windows_checkbox.setToolTip("If checked, all latency windows will be shown in the plot.")
        self.latency_legend_checkbox = QCheckBox()
        self.latency_legend_checkbox.setToolTip("If checked, the latency window legend will be shown in the plot.")
        self.plot_colormap_checkbox  = QCheckBox()
        self.plot_colormap_checkbox.setToolTip("If checked, a colormap legend will be shown to the side of the plot.")
        form.addRow("Show Flags:",   self.all_windows_checkbox)
        self.all_windows_checkbox.setChecked(True)
        self.all_windows_checkbox.stateChanged.connect(self._on_all_windows_toggled)
        self._on_all_windows_toggled(self.all_windows_checkbox.checkState())
        form.addRow("Show Legend:",  self.latency_legend_checkbox)
        self.latency_legend_checkbox.setChecked(True)
        form.addRow("Show Colormap:",self.plot_colormap_checkbox)
        self.plot_colormap_checkbox.setChecked(True)

        self.layout.addLayout(form)

        # channel selectors at the bottom
        self.channel_selector = ChannelSelectorWidget(self.gui_main, parent=self)
        self.layout.addWidget(self.channel_selector)
    
    def _on_all_windows_toggled(self, state):
        # Enable or disable the latency legend checkbox based on the state of the all_windows_checkbox
        enabled = (state == Qt.CheckState.Checked or state == 2)
        self.latency_legend_checkbox.setChecked(enabled)
        self.latency_legend_checkbox.setEnabled(enabled)
    
    def get_options(self):
        return {
            "channel_indices" : self.channel_selector.get_selected_channels(),
            "data_type": self.data_type_combo.currentText(),
            "all_flags": self.all_windows_checkbox.isChecked(),
            "plot_legend": self.latency_legend_checkbox.isChecked(),
            "plot_colormap": self.plot_colormap_checkbox.isChecked()
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

class SingleEMGRecordingOptions(BasePlotOptions):
    def create_options(self):
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(4)

        self.data_type_combo = QComboBox()
        self.data_type_combo.addItems([
            "filtered", "raw", "rectified_raw", "rectified_filtered"
        ])
        form.addRow("Select Data Type:", self.data_type_combo)

        # flags / legend / colormap
        self.all_windows_checkbox    = QCheckBox()
        self.all_windows_checkbox.setToolTip("If checked, all latency windows will be shown in the plot.")
        self.latency_legend_checkbox = QCheckBox()
        self.latency_legend_checkbox.setToolTip("If checked, the latency window legend will be shown in the plot.")
        self.plot_colormap_checkbox  = QCheckBox()
        self.plot_colormap_checkbox.setToolTip("If checked, a colormap legend will be shown to the side of the plot.")
        self.fixed_y_axis_checkbox = QCheckBox()
        self.fixed_y_axis_checkbox.setToolTip("If checked, the y-axes for all channels will be fixed to the maximum y-axis value.")
        form.addRow("Show Flags:",   self.all_windows_checkbox)
        self.all_windows_checkbox.setChecked(True)
        self.all_windows_checkbox.stateChanged.connect(self._on_all_windows_toggled)
        self._on_all_windows_toggled(self.all_windows_checkbox.checkState())
        form.addRow("Show Legend:",  self.latency_legend_checkbox)
        self.latency_legend_checkbox.setChecked(True)
        form.addRow("Show Colormap:",self.plot_colormap_checkbox)
        self.plot_colormap_checkbox.setChecked(True)
        form.addRow("Fixed Y-Axis:", self.fixed_y_axis_checkbox)
        self.fixed_y_axis_checkbox.setChecked(True)  # Set the initial state to True

        self.layout.addLayout(form)

        
        # Recording Cycler
        self.recording_cycler = RecordingCyclerWidget(self)
        self.layout.addWidget(self.recording_cycler)
        # Channel selection
        self.channel_selector = ChannelSelectorWidget(self.gui_main, parent=self)
        self.layout.addWidget(self.channel_selector)

    def _on_all_windows_toggled(self, state):
        # Enable or disable the latency legend checkbox based on the state of the all_windows_checkbox
        enabled = (state == Qt.CheckState.Checked or state == 2)
        self.latency_legend_checkbox.setChecked(enabled)
        self.latency_legend_checkbox.setEnabled(enabled)
    
    def get_options(self):
        return {
            "channel_indices" : self.channel_selector.get_selected_channels(),
            "data_type": self.data_type_combo.currentText(),
            "all_flags": self.all_windows_checkbox.isChecked(),
            "plot_legend": self.latency_legend_checkbox.isChecked(),
            "recording_index": self.recording_cycler.get_current_recording(),
            "fixed_y_axis": self.fixed_y_axis_checkbox.isChecked(),
            "plot_colormap": self.plot_colormap_checkbox.isChecked()
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
            self.fixed_y_axis_checkbox.setChecked(options["fixed_y_axis"]),
        if "plot_colormap" in options:
            self.plot_colormap_checkbox.setChecked(options["plot_colormap"])

class ReflexCurvesOptions(BasePlotOptions):
    def create_options(self):
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(4)

        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "rms", "average_rectified", "average_unrectified", "peak_to_trough"
        ])
        self.method_combo.setCurrentIndex(0)  # Set the initial selection to "rms"
        form.addRow("Reflex Amplitude Calculation Method:", self.method_combo)

        # Checkboxes
        self.relative_to_mmax_checkbox    = QCheckBox()
        self.relative_to_mmax_checkbox.setToolTip("If checked, the reflex amplitudes will be calculated relative to the M-max value.")
        self.show_legend_checkbox = QCheckBox()
        self.show_legend_checkbox.setToolTip("If checked, the plot legend will be shown.")
        form.addRow("Relative to M-max:",   self.relative_to_mmax_checkbox)
        self.relative_to_mmax_checkbox.setChecked(False)
        form.addRow("Show Plot Legend:", self.show_legend_checkbox)
        self.show_legend_checkbox.setChecked(True)

        self.layout.addLayout(form)

        # Channel selection
        self.channel_selector = ChannelSelectorWidget(self.gui_main, parent=self)
        self.layout.addWidget(self.channel_selector)

    def get_options(self):
        return {
            "channel_indices" : self.channel_selector.get_selected_channels(),
            "method": self.method_combo.currentText(),
            "relative_to_mmax": self.relative_to_mmax_checkbox.isChecked(),
            "plot_legend": self.show_legend_checkbox.isChecked()
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

class AverageReflexCurvesOptions(BasePlotOptions):
    def create_options(self):
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(4)

        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "rms", "average_rectified", "average_unrectified", "peak_to_trough"
        ])
        self.method_combo.setCurrentIndex(0)  # Set the initial selection to "rms"
        form.addRow("Reflex Amplitude Calculation Method:", self.method_combo)

        # Checkboxes
        self.relative_to_mmax_checkbox    = QCheckBox()
        self.relative_to_mmax_checkbox.setToolTip("If checked, the reflex amplitudes will be calculated relative to the M-max value.")
        self.show_legend_checkbox = QCheckBox()
        self.show_legend_checkbox.setToolTip("If checked, the plot legend will be shown.")
        form.addRow("Relative to M-max:",   self.relative_to_mmax_checkbox)
        self.relative_to_mmax_checkbox.setChecked(False)
        form.addRow("Show Plot Legend:", self.show_legend_checkbox)
        self.show_legend_checkbox.setChecked(True)

        self.layout.addLayout(form)

        # Channel selection
        self.channel_selector = ChannelSelectorWidget(self.gui_main, parent=self)
        self.layout.addWidget(self.channel_selector)

    def get_options(self):
        return {
            "channel_indices" : self.channel_selector.get_selected_channels(),
            "method": self.method_combo.currentText(),
            "relative_to_mmax": self.relative_to_mmax_checkbox.isChecked(),
            "plot_legend": self.show_legend_checkbox.isChecked()
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

class MMaxOptions(BasePlotOptions):
    def create_options(self):
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(4)

        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "rms", "average_rectified", "average_unrectified", "peak_to_trough"
        ])
        self.method_combo.setCurrentIndex(0)  # Set the initial selection to "rms"
        form.addRow("Reflex Amplitude Calculation Method:", self.method_combo)

        self.layout.addLayout(form)

        # channel selection
        self.channel_selector = ChannelSelectorWidget(self.gui_main, parent=self)
        self.layout.addWidget(self.channel_selector)

    def get_options(self):
        return {
            "channel_indices" : self.channel_selector.get_selected_channels(),
            "method": self.method_combo.currentText()
        }
    
    def set_options(self, options):
        if "channel_indices" in options:
            self.channel_selector.set_selected_channels(options["channel_indices"])
        if "method" in options:
            index = self.method_combo.findText(options["method"])
            if index >= 0:
                self.method_combo.setCurrentIndex(index)

class MaxHReflexOptions(BasePlotOptions):
    def create_options(self):
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(4)

        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "rms", "average_rectified", "average_unrectified", "peak_to_trough"
        ])
        self.method_combo.setCurrentIndex(0)  # Set the initial selection to "rms"
        form.addRow("Reflex Amplitude Calculation Method:", self.method_combo)

        # Checkboxes
        self.relative_to_mmax_checkbox    = QCheckBox()
        self.relative_to_mmax_checkbox.setToolTip("If checked, the reflex amplitudes will be calculated relative to the M-max value.")
        form.addRow("Relative to M-max:",   self.relative_to_mmax_checkbox)
        self.relative_to_mmax_checkbox.setChecked(False)

        self.max_stim_value = FloatLineEdit(default_value=10.0)
        self.max_stim_value.setPlaceholderText("(float)")
        self.max_stim_value.setToolTip("Maximum value of the stimulus (in V) that will be used to calculate the average reflex amplitudes.")
        self.max_stim_value.setMaximumWidth(80)
        form.addRow("Max Stimulus Value:", self.max_stim_value)

        self.bin_margin_input = QLineEdit()
        self.bin_margin_input.setValidator(QIntValidator())
        self.bin_margin_input.setText("0")
        self.bin_margin_input.setPlaceholderText("(integer)")
        self.bin_margin_input.setToolTip("Number of bins to add to the left and right of the maximum stimulus value to add nerby datapoints to the average reflex calculation.")
        form.addRow("Bin Margin:", self.bin_margin_input)

        self.layout.addLayout(form)


        # Channel selection
        self.channel_selector = ChannelSelectorWidget(self.gui_main, parent=self)
        self.layout.addWidget(self.channel_selector)

    def get_options(self):
        return {
            "channel_indices" : self.channel_selector.get_selected_channels(),
            "method": self.method_combo.currentText(),
            "relative_to_mmax": self.relative_to_mmax_checkbox.isChecked(),
            "max_stim_value": self.max_stim_value.get_value(),
            "bin_margin": int(self.bin_margin_input.text())
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
    '''Deprecated plot type, kept for backwards compatibility'''
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
        self.h_method_combo = QComboBox()
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
            "channel_indices" : self.channel_selector.get_selected_channels(),
            "h_threshold": self.h_threshold_input.get_value(),
            "method": self.h_method_combo.currentText(),
            "plot_legend": self.plot_legend_checkbox.isChecked()
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