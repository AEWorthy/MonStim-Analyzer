from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox, QLineEdit, QGridLayout
from PyQt6.QtGui import QIntValidator
from .custom_gui_elements import FloatLineEdit
from .plotting_cycler import RecordingCyclerWidget
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from monstim_gui import EMGAnalysisGUI
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
        pass

    def get_options(self):
        # To be implemented by subclasses
        pass

    def set_options(self, options):
        # To be implemented by subclasses
        pass

class ChannelSelectorLayout(QHBoxLayout):
    def __init__(self, gui_main : 'EMGAnalysisGUI'):
        self.gui_main = gui_main
        super().__init__()
        plot_widget_view = gui_main.plot_widget.view # current level of analysis (session, dataset, experiment)
        match plot_widget_view:
            case "session":
                self.emg_data = gui_main.current_session
            case "dataset":
                self.emg_data = gui_main.current_dataset
            case "experiment":
                self.emg_data = gui_main.current_experiment
            case _:
                self.emg_data = None

        self.max_num_channels = self.emg_data.num_channels if self.emg_data is not None else 0
        
        self.channel_checkboxes : List[QCheckBox] = []
        grid_layout = QGridLayout()
        for i in range(6):
            checkbox = QCheckBox(f"Channel {i}")
            if i >= self.max_num_channels:
                checkbox.setChecked(False)
                checkbox.setEnabled(False)
            else:
                checkbox.setChecked(True)
                checkbox.setEnabled(True)
            row = i // 3
            col = i % 3
            grid_layout.addWidget(checkbox, row, col)
            self.channel_checkboxes.append(checkbox)
        
        self.addLayout(grid_layout)
        
    def get_selected_channels(self):
        return [i for i in range(6) if self.channel_checkboxes[i].isChecked()]
    
    def set_selected_channels(self, selected_channels):
        for i in range(self.max_num_channels):
            self.channel_checkboxes[i].setChecked(i in selected_channels)
        
# EMG Options
class EMGOptions(BasePlotOptions):
    def create_options(self):
        ## Data type options box
        data_type_layout = QHBoxLayout()
        self.data_type_label = QLabel("Select EMG Data Type:")
        self.data_type_combo = QComboBox()
        self.data_type_combo.addItems(["filtered", "raw", "rectified_raw", "rectified_filtered"])
        data_type_layout.addWidget(self.data_type_label)
        data_type_layout.addWidget(self.data_type_combo)
        self.layout.addLayout(data_type_layout)

        ## Latency Window options
        latency_windows_layout = QVBoxLayout()

        # First row
        first_row_layout = QHBoxLayout()
        self.all_windows_label = QLabel("Show Latency Window Flags:")
        self.all_windows_checkbox = QCheckBox()
        self.all_windows_checkbox.setChecked(True)
        first_row_layout.addWidget(self.all_windows_label)
        first_row_layout.addWidget(self.all_windows_checkbox)
        latency_windows_layout.addLayout(first_row_layout)

        # Second row
        second_row_layout = QHBoxLayout()
        self.latency_legend_label = QLabel("Show Latency Window Legend:")
        self.latency_legend_checkbox = QCheckBox()
        self.latency_legend_checkbox.setChecked(True)  # Set the initial state to True
        second_row_layout.addWidget(self.latency_legend_label)
        second_row_layout.addWidget(self.latency_legend_checkbox)
        latency_windows_layout.addLayout(second_row_layout)

        # Third row
        third_row_layout = QHBoxLayout()
        self.plot_colormap_legend = QLabel("Show Colormap:")
        self.plot_colormap_checkbox = QCheckBox()
        self.plot_colormap_checkbox.setChecked(True)  # Set the initial state to True
        third_row_layout.addWidget(self.plot_colormap_legend)
        third_row_layout.addWidget(self.plot_colormap_checkbox)
        latency_windows_layout.addLayout(third_row_layout)

        self.all_windows_checkbox.stateChanged.connect(self._on_all_windows_toggled)
        self._on_all_windows_toggled(self.all_windows_checkbox.checkState())

        # Add the latency_windows_layout to the main layout
        self.layout.addLayout(latency_windows_layout)

        # Channel selection
        self.channel_selector = ChannelSelectorLayout(self.gui_main)
        self.layout.addLayout(self.channel_selector)
    
    def _on_all_windows_toggled(self, state):
        # Enable or disable the latency legend checkbox based on the state of the all_windows_checkbox
        enabled = (state == Qt.CheckState.Checked or state == 2)
        self.latency_legend_label.setEnabled(enabled)
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
        ## Data type options box
        data_type_layout = QHBoxLayout()
        self.data_type_label = QLabel("Select EMG Data Type:")
        self.data_type_combo = QComboBox()
        self.data_type_combo.addItems(["filtered", "raw", "rectified_raw", "rectified_filtered"])
        data_type_layout.addWidget(self.data_type_label)
        data_type_layout.addWidget(self.data_type_combo)
        self.layout.addLayout(data_type_layout)

        ## Latency Window options
        latency_windows_layout = QVBoxLayout()

        # First row
        first_row_layout = QHBoxLayout()
        self.all_windows_label = QLabel("Show All Latency Windows:")
        self.all_windows_checkbox = QCheckBox()
        self.all_windows_checkbox.setChecked(True)
        first_row_layout.addWidget(self.all_windows_label)
        first_row_layout.addWidget(self.all_windows_checkbox)
        latency_windows_layout.addLayout(first_row_layout)

        # Second row
        second_row_layout = QHBoxLayout()
        self.latency_legend_label = QLabel("Show Latency Window Legend:")
        self.latency_legend_checkbox = QCheckBox()
        self.latency_legend_checkbox.setChecked(True)  # Set the initial state to True
        second_row_layout.addWidget(self.latency_legend_label)
        second_row_layout.addWidget(self.latency_legend_checkbox)
        latency_windows_layout.addLayout(second_row_layout)

        # Third row
        third_row_layout = QHBoxLayout()
        self.plot_colormap_legend = QLabel("Show Colormap:")
        self.plot_colormap_checkbox = QCheckBox()
        self.plot_colormap_checkbox.setChecked(True)  # Set the initial state to True
        third_row_layout.addWidget(self.plot_colormap_legend)
        third_row_layout.addWidget(self.plot_colormap_checkbox)
        latency_windows_layout.addLayout(third_row_layout)

        # Fourth row
        fourth_row_layout = QHBoxLayout()
        self.fixed_y_axis_label = QLabel("Fixed Y-Axis:")
        self.fixed_y_axis_checkbox = QCheckBox()
        self.fixed_y_axis_checkbox.setChecked(True)  # Set the initial state to False
        fourth_row_layout.addWidget(self.fixed_y_axis_label)
        fourth_row_layout.addWidget(self.fixed_y_axis_checkbox)
        latency_windows_layout.addLayout(fourth_row_layout)

        # Add the latency_windows_layout to the main layout
        self.layout.addLayout(latency_windows_layout)
        # Recording Cycler
        self.recording_cycler = RecordingCyclerWidget(self)
        self.layout.addWidget(self.recording_cycler)
        # Channel selection
        self.channel_selector = ChannelSelectorLayout(self.gui_main)
        self.layout.addLayout(self.channel_selector)

        # Set the initial state of the checkboxes based on the current state of all_windows_checkbox
        self.all_windows_checkbox.stateChanged.connect(self._on_all_windows_toggled)
        self._on_all_windows_toggled(self.all_windows_checkbox.checkState())

    def _on_all_windows_toggled(self, state):
        # Enable or disable the latency legend checkbox based on the state of the all_windows_checkbox
        enabled = (state == Qt.CheckState.Checked or state == 2)
        self.latency_legend_label.setEnabled(enabled)
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
        # method option
        method_layout = QHBoxLayout()
        self.method_label = QLabel("Reflex Amplitude Calculation Method:")
        self.method_combo = QComboBox()
        self.method_combo.addItems(["rms", "average_rectified", "average_unrectified", "peak_to_trough"])
        self.method_combo.setCurrentIndex(0)  # Set the initial selection to "rms"
        method_layout.addWidget(self.method_label)
        method_layout.addWidget(self.method_combo)
        self.layout.addLayout(method_layout)

        # relative_to_mmax checkbox
        relative_to_mmax_layout = QHBoxLayout()
        self.relative_to_mmax_label = QLabel("Relative to M-max:")
        self.relative_to_mmax_checkbox = QCheckBox()
        self.relative_to_mmax_checkbox.setChecked(False)  # Set the initial state to False
        relative_to_mmax_layout.addWidget(self.relative_to_mmax_label)
        relative_to_mmax_layout.addWidget(self.relative_to_mmax_checkbox)
        self.layout.addLayout(relative_to_mmax_layout)

        # plot_legend option (checkbox)
        plot_legend_layout = QHBoxLayout()
        self.plot_legend_label = QLabel("Show Plot Legend:")
        self.plot_legend_checkbox = QCheckBox()
        self.plot_legend_checkbox.setChecked(True)  # Set the initial state to True
        plot_legend_layout.addWidget(self.plot_legend_label)
        plot_legend_layout.addWidget(self.plot_legend_checkbox)
        self.layout.addLayout(plot_legend_layout)

        # Channel selection
        self.channel_selector = ChannelSelectorLayout(self.gui_main)
        self.layout.addLayout(self.channel_selector)

    def get_options(self):
        return {
            "channel_indices" : self.channel_selector.get_selected_channels(),
            "method": self.method_combo.currentText(),
            "relative_to_mmax": self.relative_to_mmax_checkbox.isChecked(),
            "plot_legend": self.plot_legend_checkbox.isChecked()
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
            self.plot_legend_checkbox.setChecked(options["plot_legend"])

class AverageReflexCurvesOptions(BasePlotOptions):
    def create_options(self):
        # method option
        method_layout = QHBoxLayout()
        self.method_label = QLabel("Reflex Amplitude Calculation Method:")
        self.method_combo = QComboBox()
        self.method_combo.addItems(["rms", "average_rectified", "average_unrectified", "peak_to_trough"])
        self.method_combo.setCurrentIndex(0)  # Set the initial selection to "rms"
        method_layout.addWidget(self.method_label)
        method_layout.addWidget(self.method_combo)
        self.layout.addLayout(method_layout)

        # relative_to_mmax checkbox
        relative_to_mmax_layout = QHBoxLayout()
        self.relative_to_mmax_label = QLabel("Relative to M-max:")
        self.relative_to_mmax_checkbox = QCheckBox()
        self.relative_to_mmax_checkbox.setChecked(False)  # Set the initial state to False
        relative_to_mmax_layout.addWidget(self.relative_to_mmax_label)
        relative_to_mmax_layout.addWidget(self.relative_to_mmax_checkbox)
        self.layout.addLayout(relative_to_mmax_layout)

        # plot_legend option (checkbox)
        plot_legend_layout = QHBoxLayout()
        self.plot_legend_label = QLabel("Show Plot Legend:")
        self.plot_legend_checkbox = QCheckBox()
        self.plot_legend_checkbox.setChecked(True)  # Set the initial state to True
        plot_legend_layout.addWidget(self.plot_legend_label)
        plot_legend_layout.addWidget(self.plot_legend_checkbox)
        self.layout.addLayout(plot_legend_layout)

        # Channel selection
        self.channel_selector = ChannelSelectorLayout(self.gui_main)
        self.layout.addLayout(self.channel_selector)

    def get_options(self):
        return {
            "channel_indices" : self.channel_selector.get_selected_channels(),
            "method": self.method_combo.currentText(),
            "relative_to_mmax": self.relative_to_mmax_checkbox.isChecked(),
            "plot_legend": self.plot_legend_checkbox.isChecked()
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
            self.plot_legend_checkbox.setChecked(options["plot_legend"])

class MMaxOptions(BasePlotOptions):
    def create_options(self):
        # method option
        method_layout = QHBoxLayout()
        self.method_label = QLabel("Reflex Amplitude Calculation Method:")
        self.method_combo = QComboBox()
        self.method_combo.addItems(["rms", "average_rectified", "average_unrectified", "peak_to_trough"])
        self.method_combo.setCurrentIndex(0)  # Set the initial selection to "rms"
        method_layout.addWidget(self.method_label)
        method_layout.addWidget(self.method_combo)
        self.layout.addLayout(method_layout)

        # channel selection
        self.channel_selector = ChannelSelectorLayout(self.gui_main)
        self.layout.addLayout(self.channel_selector)

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
        # method option
        method_layout = QHBoxLayout()
        self.method_label = QLabel("Reflex Amplitude Calculation Method:")
        self.method_combo = QComboBox()
        self.method_combo.addItems(["rms", "average_rectified", "average_unrectified", "peak_to_trough"])
        self.method_combo.setCurrentIndex(0)  # Set the initial selection to "rms"
        method_layout.addWidget(self.method_label)
        method_layout.addWidget(self.method_combo)
        self.layout.addLayout(method_layout)

        # relative_to_mmax checkbox
        relative_to_mmax_layout = QHBoxLayout()
        self.relative_to_mmax_label = QLabel("Relative to M-max:")
        self.relative_to_mmax_checkbox = QCheckBox()
        self.relative_to_mmax_checkbox.setChecked(False)  # Set the initial state to False
        relative_to_mmax_layout.addWidget(self.relative_to_mmax_label)
        relative_to_mmax_layout.addWidget(self.relative_to_mmax_checkbox)
        self.layout.addLayout(relative_to_mmax_layout)

        # Max stim value option - float
        max_stim_value_layout = QHBoxLayout()
        self.max_stim_value_label = QLabel("Max Stim Value:")
        self.max_stim_value_input = FloatLineEdit(default_value=10.0)
        self.max_stim_value_input.setPlaceholderText("(float)")
        self.max_stim_value_input.setToolTip("Maximum value of the stimulus (in V) that will be used to calculate the average reflex amplitudes.")
        self.max_stim_value_input.setMaximumWidth(80)
        max_stim_value_layout.addWidget(self.max_stim_value_label)
        max_stim_value_layout.addWidget(self.max_stim_value_input)
        max_stim_value_layout.addStretch(2)
        self.layout.addLayout(max_stim_value_layout)

        # bin margin option (integer)
        bin_margin_layout = QHBoxLayout()
        self.bin_margin_label = QLabel("Bin Margin:")
        self.bin_margin_input = QLineEdit()
        self.bin_margin_input.setValidator(QIntValidator())
        self.bin_margin_input.setText("0")
        self.bin_margin_input.setPlaceholderText("(integer)")
        self.bin_margin_input.setToolTip("Number of bins to add to the left and right of the maximum stimulus value to add nerby datapoints to the average reflex calculation.")
        self.bin_margin_input.setMaximumWidth(80)
        bin_margin_layout.addWidget(self.bin_margin_label)
        bin_margin_layout.addWidget(self.bin_margin_input)
        bin_margin_layout.addStretch(0)
        self.layout.addLayout(bin_margin_layout)

        # Channel selection
        self.channel_selector = ChannelSelectorLayout(self.gui_main)
        self.layout.addLayout(self.channel_selector)

    def get_options(self):
        return {
            "channel_indices" : self.channel_selector.get_selected_channels(),
            "method": self.method_combo.currentText(),
            "relative_to_mmax": self.relative_to_mmax_checkbox.isChecked(),
            "max_stim_value": self.max_stim_value_input.get_value(),
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
            self.max_stim_value_input.set_value(float(options["max_stim_value"]))
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
        self.channel_selector = ChannelSelectorLayout(self.gui_main)
        self.layout.addLayout(self.channel_selector)

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