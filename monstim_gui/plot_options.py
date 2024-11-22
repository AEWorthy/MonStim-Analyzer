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
        
        self.channel_checkboxes : List[QCheckBox] = []
        grid_layout = QGridLayout()
        for i in range(6):
            checkbox = QCheckBox(f"Channel {i}")
            checkbox.setChecked(True)
            if i >= self.emg_data.num_channels:
                checkbox.setChecked(False)
                checkbox.setEnabled(False)
            row = i // 3
            col = i % 3
            grid_layout.addWidget(checkbox, row, col)
            self.channel_checkboxes.append(checkbox)
        
        self.addLayout(grid_layout)
        
    def get_selected_channels(self):
        return [i for i in range(6) if self.channel_checkboxes[i].isChecked()]
    
    def set_selected_channels(self, selected_channels):
        for i in range(6):
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

        # Add the latency_windows_layout to the main layout
        self.layout.addLayout(latency_windows_layout)

        # Channel selection
        self.channel_selector = ChannelSelectorLayout(self.gui_main)
        self.layout.addLayout(self.channel_selector)
    
    def get_options(self):
        return {
            "channel_indices" : self.channel_selector.get_selected_channels(),
            "data_type": self.data_type_combo.currentText(),
            "all_flags": self.all_windows_checkbox.isChecked(),
            "plot_legend": self.latency_legend_checkbox.isChecked()
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
        self.fixed_y_axis_label = QLabel("Fixed Y-Axis:")
        self.fixed_y_axis_checkbox = QCheckBox()
        self.fixed_y_axis_checkbox.setChecked(True)  # Set the initial state to False
        third_row_layout.addWidget(self.fixed_y_axis_label)
        third_row_layout.addWidget(self.fixed_y_axis_checkbox)
        latency_windows_layout.addLayout(third_row_layout)

        # Add the latency_windows_layout to the main layout
        self.layout.addLayout(latency_windows_layout)

        # Recording Cycler
        self.recording_cycler = RecordingCyclerWidget(self)
        self.layout.addWidget(self.recording_cycler)

        # Channel selection
        self.channel_selector = ChannelSelectorLayout(self.gui_main)
        self.layout.addLayout(self.channel_selector)
    
    def get_options(self):
        return {
            "channel_indices" : self.channel_selector.get_selected_channels(),
            "data_type": self.data_type_combo.currentText(),
            "all_flags": self.all_windows_checkbox.isChecked(),
            "plot_legend": self.latency_legend_checkbox.isChecked(),
            "recording_index": self.recording_cycler.get_current_recording(),
            "fixed_y_axis": self.fixed_y_axis_checkbox.isChecked()
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

class SuspectedHReflexesOptions(BasePlotOptions):
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

        # bin margin option (integer)
        bin_margin_layout = QHBoxLayout()
        self.bin_margin_label = QLabel("Bin Margin:")
        self.bin_margin_input = QLineEdit()
        self.bin_margin_input.setValidator(QIntValidator())
        self.bin_margin_input.setText("0")
        self.bin_margin_input.setPlaceholderText("Bin Margin (integer)")
        bin_margin_layout.addWidget(self.bin_margin_label)
        bin_margin_layout.addWidget(self.bin_margin_input)
        self.layout.addLayout(bin_margin_layout)

        # Channel selection
        self.channel_selector = ChannelSelectorLayout(self.gui_main)
        self.layout.addLayout(self.channel_selector)

    def get_options(self):
        return {
            "channel_indices" : self.channel_selector.get_selected_channels(),
            "method": self.method_combo.currentText(),
            "relative_to_mmax": self.relative_to_mmax_checkbox.isChecked(),
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
            self.relative_to_mmax_checkbox.setChecked(options["relative_to_mmax"])
        if "bin_margin" in options:
            self.bin_margin_input.set_text(str(options["bin_margin"]))