from typing import TYPE_CHECKING
import logging
from PyQt6.QtWidgets import (QGroupBox, QVBoxLayout, QRadioButton, QButtonGroup,
                             QComboBox, QLabel, QHBoxLayout, QPushButton, QSizePolicy)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from .plot_options import (EMGOptions, SuspectedHReflexesOptions, ReflexCurvesOptions, SingleEMGRecordingOptions,
                           MMaxOptions, AverageReflexCurvesOptions, MaxHReflexOptions)

if TYPE_CHECKING:
    from monstim_gui import EMGAnalysisGUI
    from .plot_options import BasePlotOptions

# Plotting Widget
class PlotWidget(QGroupBox):
    def __init__(self, parent: 'EMGAnalysisGUI'):
        super().__init__("Plotting", parent)
        self.current_option_widget: 'BasePlotOptions' = None
        self.parent = parent # Type: EMGAnalysisGUI
        self.layout = QVBoxLayout() # Type: QVBoxLayout
        self.create_view_selection()
        self.create_plot_type_selection()
        self.create_additional_options()
        self.create_plot_button()
        self.import_canvas()
        self.setLayout(self.layout)

    def initialize_plot_widget(self):
        # Occurs after the data has been loaded. Called from EMGAnalysisGUI.
        self.plot_options = {
            "session": {
                "EMG": EMGOptions,
                "Single EMG Recordings": SingleEMGRecordingOptions,
                "Suspected H-reflexes": SuspectedHReflexesOptions,
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

    def create_view_selection(self):
        self.layout.addWidget(QLabel("Select Data Level to Plot:"))
        view_layout = QHBoxLayout()
        self.view_group = QButtonGroup(self)
        self.session_radio = QRadioButton("Session")
        self.dataset_radio = QRadioButton("Dataset")
        self.experiment_radio = QRadioButton("Experiment")
        self.view_group.addButton(self.session_radio)
        self.view_group.addButton(self.dataset_radio)
        self.view_group.addButton(self.experiment_radio)
        self.session_radio.setChecked(True)
        view_layout.addWidget(self.session_radio)
        view_layout.addWidget(self.dataset_radio)
        view_layout.addWidget(self.experiment_radio)
        self.layout.addLayout(view_layout)

        self.view = "session"
        self.session_radio.toggled.connect(self.on_view_changed)
        self.dataset_radio.toggled.connect(self.on_view_changed)

    def create_plot_type_selection(self):
        plot_type_layout = QHBoxLayout()
        self.plot_type_label = QLabel("Select Plot Type:")
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.currentTextChanged.connect(self.on_plot_type_changed)
        plot_type_layout.addWidget(self.plot_type_label)
        plot_type_layout.addWidget(self.plot_type_combo)
        self.layout.addLayout(plot_type_layout)

    def create_additional_options(self):
        self.additional_options_layout = QVBoxLayout()
        self.layout.addLayout(self.additional_options_layout)

    def create_plot_button(self):
        self.plot_button = QPushButton("Plot")
        self.plot_button.clicked.connect(self.parent.plot_data)
        self.layout.addWidget(self.plot_button)

        self.get_data_button = QPushButton("Plot/Extact Raw Data")
        self.get_data_button.clicked.connect(self.parent.get_raw_data)
        self.layout.addWidget(self.get_data_button)
    
    def import_canvas(self):
        self.canvas = self.parent.plot_pane.canvas
        self.figure = self.parent.plot_pane.figure        

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
        plot_type = self.plot_type_combo.currentText()
        if plot_type == self.last_plot_type[self.view]:
            return
        self.save_current_options()
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
        self.on_plot_type_changed()

    def update_plot_options(self):
        if self.current_option_widget:
            self.additional_options_layout.removeWidget(self.current_option_widget)
            self.current_option_widget.deleteLater()
            self.current_option_widget = None

        plot_type = self.plot_type_combo.currentText()

        if plot_type in self.plot_options[self.view]:
            self.current_option_widget = self.plot_options[self.view][plot_type](self)
            self.additional_options_layout.addWidget(self.current_option_widget)

            if plot_type in self.last_options[self.view]:
                self.current_option_widget.set_options(self.last_options[self.view][plot_type])
        
        self.additional_options_layout.update()
        
        if plot_type == "Single EMG Recordings":
            self.current_option_widget.recording_cycler.reset_max_recordings()

    def save_current_options(self):
       if self.current_option_widget:
            current_plot_type = self.last_plot_type[self.view]
            current_options = self.current_option_widget.get_options()
            self.last_options[self.view][current_plot_type] = current_options

    def get_plot_options(self):
        if self.current_option_widget:
            return self.current_option_widget.get_options()
        return {}


class PlotPane(QGroupBox):
    def __init__(self, parent: 'EMGAnalysisGUI'):
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
