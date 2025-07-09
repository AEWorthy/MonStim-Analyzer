from typing import TYPE_CHECKING
import logging
import copy
from PyQt6.QtWidgets import (QGroupBox, QVBoxLayout, QRadioButton, QButtonGroup, QFormLayout,
                             QComboBox, QHBoxLayout, QPushButton, QSizePolicy, QWidget)
from PyQt6.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from .plot_options import (EMGOptions, ReflexCurvesOptions, SingleEMGRecordingOptions,
                           MMaxOptions, AverageReflexCurvesOptions, MaxHReflexOptions)

if TYPE_CHECKING:
    from monstim_gui import MonstimGUI
    from .plot_options import BasePlotOptions
    from matplotlib import Figure, FigureCanvas

# Plotting Widget
class PlotWidget(QGroupBox):
    def restore_last_plot_type_and_options(self):
        """Restore the last selected plot type and as many options as possible for the current view."""
        # Determine the last plot type for the current view
        if not hasattr(self, 'last_plot_type'):
            return  # If last_plot_type is not initialized, do nothing
        last_plot_type = self.last_plot_type.get(self.view, None)
        available_types = list(self.plot_options[self.view].keys())
        if last_plot_type in available_types:
            self.plot_type_combo.setCurrentText(last_plot_type)
        else:
            self.plot_type_combo.setCurrentText(available_types[0])
            last_plot_type = available_types[0]
            self.last_plot_type[self.view] = last_plot_type

        # Re-initialize the options widget for the selected plot type
        self.update_plot_options()
        if self.current_option_widget:
            # Get the default options for the new widget
            default_options = self.current_option_widget.get_options()
            # Get the last saved options for this plot type
            last_options = self.last_options[self.view].get(last_plot_type, {})
            # Only keep keys that are still valid
            merged_options = {k: v for k, v in last_options.items() if k in default_options}
            # Fill in missing keys with defaults
            for k, v in default_options.items():
                if k not in merged_options:
                    merged_options[k] = v
            self.current_option_widget.set_options(merged_options)
    def __init__(self, parent: 'MonstimGUI'):
        super().__init__("Plotting", parent)
        self.current_option_widget: 'BasePlotOptions' = None
        self.parent : 'MonstimGUI' = parent
        self.layout : 'QVBoxLayout' = QVBoxLayout()
        self.layout.setContentsMargins(8, 8, 8, 8)
        self.layout.setSpacing(6)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(4)

        # Data Selection
        level_widget = QWidget()
        level_h = QHBoxLayout(level_widget)
        level_h.setContentsMargins(0, 0, 0, 0)
        self.view_group     = QButtonGroup(self)
        self.session_radio  = QRadioButton("Session")
        self.dataset_radio  = QRadioButton("Dataset")
        self.experiment_radio = QRadioButton("Experiment")
        for rb in (self.session_radio, self.dataset_radio, self.experiment_radio):
            self.view_group.addButton(rb)
            level_h.addWidget(rb)
        self.session_radio.setChecked(True)
        self.view = "session"
        self.session_radio.toggled.connect(self.on_view_changed)
        self.dataset_radio.toggled.connect(self.on_view_changed)
        form.addRow("Select Data Level to Plot:", level_widget)

        # Plot Type Selection Box
        self.plot_type_combo = QComboBox()
        form.addRow("Plot Type:", self.plot_type_combo)
        self.plot_type_combo.currentTextChanged.connect(self.on_plot_type_changed)

        self.layout.addLayout(form)

        # Dynamic Options Box
        self.options_box = QGroupBox("Options")
        self.options_box.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.MinimumExpanding
        )
        self.options_layout = QVBoxLayout(self.options_box)
        self.options_layout.setContentsMargins(6, 6, 6, 6)
        self.options_layout.setSpacing(6)
        self.layout.addWidget(self.options_box)

        # Create the buttons for plotting and extracting data
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.plot_button     = QPushButton("Plot")
        self.get_data_button = QPushButton("Plot & Extract Data")
        for btn in (self.plot_button, self.get_data_button):
            btn.setSizePolicy(QSizePolicy.Policy.Preferred,
                              QSizePolicy.Policy.Fixed)
            btn_row.addWidget(btn)
        self.plot_button.clicked.connect(self.parent.plot_controller.plot_data)
        self.get_data_button.clicked.connect(self.parent.plot_controller.get_raw_data)
        self.layout.addLayout(btn_row)


        # self.create_view_selection()
        self.create_additional_options()
        # self.import_canvas()
        self.setLayout(self.layout)

    def initialize_plot_widget(self):
        # Occurs after the data has been loaded. Called from EMGAnalysisGUI.
        self.plot_options = {
            "session": {
                "EMG": EMGOptions,
                "Single EMG Recordings": SingleEMGRecordingOptions,
                "Reflex Curves": ReflexCurvesOptions,
                "M-max": MMaxOptions
            },
            "dataset": {
                "Average Reflex Curves": AverageReflexCurvesOptions,
                "Max H-reflex": MaxHReflexOptions,
                "M-max": MMaxOptions
            },
            "experiment": {
                "Average Reflex Curves": AverageReflexCurvesOptions,
                "Max H-reflex": MaxHReflexOptions,
                "M-max": MMaxOptions
            }
        }
        
        # Store the last selected plot type and options for each view
        self.last_plot_type = {
            "session": "EMG",
            "dataset": "Average Reflex Curves",
            "experiment": "Average Reflex Curves"
        }
        self.last_options = {
            "session": {plot_type: {} for plot_type in self.plot_options["session"]},
            "dataset": {plot_type: {} for plot_type in self.plot_options["dataset"]},
            "experiment": {plot_type: {} for plot_type in self.plot_options["experiment"]}
        }

        # Initialize plot types and options
        self.update_plot_types()
        self.update_plot_options()

    def create_additional_options(self):
        self.additional_options_layout = QVBoxLayout()
        self.layout.addLayout(self.additional_options_layout)

    def create_plot_button(self):
        self.plot_button.clicked.connect(self.parent.plot_controller.plot_data)
        self.get_data_button.clicked.connect(self.parent.plot_controller.get_raw_data)
    
    def import_canvas(self):
        self.canvas : 'FigureCanvas' = self.parent.plot_pane.canvas
        self.figure : 'Figure' = self.parent.plot_pane.figure        

    def on_view_changed(self):
        match self.view_group.checkedButton():
            case self.session_radio:
                view = "session"
            case self.dataset_radio:
                view = "dataset"
            case self.experiment_radio:
                view = "experiment"
        
        if view == self.view:
            return

        self.save_current_options()
        self.view = view
        self.update_plot_types()
        self.update_plot_options()

    def on_plot_type_changed(self):
        self.save_current_options()
        plot_type = self.plot_type_combo.currentText()
        if plot_type == self.last_plot_type[self.view]:
            logging.debug(f"Plot type {plot_type} is already selected. No change needed. self.last_plot_type[self.view]: {self.last_plot_type[self.view]}")
            return
        self.last_plot_type[self.view] = plot_type
        self.update_plot_options()

    def on_data_selection_changed(self):
        try:
            self.update_plot_options()
        except AttributeError:
            pass

    def update_plot_types(self):
        self.plot_type_combo.blockSignals(True)
        self.plot_type_combo.clear()
        self.plot_type_combo.addItems(self.plot_options[self.view].keys())
        self.plot_type_combo.setCurrentText(self.last_plot_type[self.view])
        self.plot_type_combo.blockSignals(False)

    def update_plot_options(self):
        if self.current_option_widget:
            self.options_layout.removeWidget(self.current_option_widget)
            self.current_option_widget.deleteLater()     

        plot_type = self.plot_type_combo.currentText()

        if plot_type in self.plot_options[self.view]:
            self.current_option_widget = self.plot_options[self.view][plot_type](self)
            self.options_layout.addWidget(self.current_option_widget)

            if plot_type in self.last_options[self.view]:
                options = copy.deepcopy(self.last_options[self.view][plot_type])
                logging.debug(f"Using last options for {self.view} - {plot_type}: {options}")
                self.current_option_widget.set_options(options)
            else:
                logging.debug(f"No last options found for {self.view} - {plot_type}. Using default options.")
        
        self.options_layout.update()
        if plot_type == "Single EMG Recordings":
            self.current_option_widget.recording_cycler.reset_max_recordings()

    def save_current_options(self):
       if self.current_option_widget:
            current_plot_type = self.plot_type_combo.currentText()
            current_options = self.current_option_widget.get_options()
            self.last_options[self.view][current_plot_type] = copy.deepcopy(current_options)
            logging.debug(f"Saved options for {self.view} - {current_plot_type}: {current_options}")

    def get_plot_options(self):
        if self.current_option_widget:
            return self.current_option_widget.get_options()
        return {}


class PlotPane(QGroupBox):
    def __init__(self, parent: 'MonstimGUI'):
        super().__init__("Plot Pane", parent)
        self.parent = parent
        self.layout = QVBoxLayout()

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure) # Type: FigureCanvas
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.setMinimumSize(400, 400)       
        self.layout.addWidget(self.canvas)
        
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.toolbar)
        self.setLayout(self.layout)
        logging.debug("Canvas created and added to layout.")
