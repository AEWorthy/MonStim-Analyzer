import logging
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QIntValidator, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStyle,
    QStyleOptionButton,
    QVBoxLayout,
    QWidget,
)

from monstim_gui.core.responsive_widgets import ResponsiveComboBox
from monstim_gui.core.utils.custom_gui_elements import FloatLineEdit

from .plotting_cycler import RecordingCyclerWidget

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from monstim_gui import MonstimGUI

    from .plotting_widget import PlotWidget

CALCULATION_METHODS = ["peak_to_trough", "extrema_ptt", "exclusive_extrema_ptt", "rms", "average_rectified", "average_unrectified", "auc"]
EXTREMA_METHODS = ["extrema_ptt", "exclusive_extrema_ptt"]
DATA_TYPES = ["filtered", "raw", "rectified_raw", "rectified_filtered"]


class OptionToggleButton(QPushButton):
    """Accessible toggle used for binary plot-display options."""

    def __init__(self, text: str, tooltip: str, parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setToolTip(tooltip)
        self.setProperty("plotOptionToggle", True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)


class ChannelCheckBox(QCheckBox):
    """Channel checkbox with a visible check mark for the themed indicator."""

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.isChecked():
            return

        option = QStyleOptionButton()
        self.initStyleOption(option)
        indicator = self.style().subElementRect(QStyle.SubElement.SE_CheckBoxIndicator, option, self)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(
            QPen(
                Qt.GlobalColor.white,
                2,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
        )
        left = indicator.left() + indicator.width() * 0.27
        mid = indicator.top() + indicator.height() * 0.55
        right = indicator.right() - indicator.width() * 0.22
        painter.drawLine(left, mid, indicator.left() + indicator.width() * 0.46, indicator.bottom() - indicator.height() * 0.24)
        painter.drawLine(
            indicator.left() + indicator.width() * 0.46,
            indicator.bottom() - indicator.height() * 0.24,
            right,
            indicator.top() + indicator.height() * 0.25,
        )


class StableOptionGrid(QWidget):
    """Responsive grid that keeps a canonical option order without blank slots."""

    OPTION_ORDER = (
        "show_flags",
        "show_legend",
        "show_colormap",
        "interactive_cursor",
        "fixed_y_axis",
        "show_extrema",
        "relative_to_mmax",
        "show_density",
    )

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(2, 2, 2, 2)
        self.grid.setHorizontalSpacing(4)
        self.grid.setVerticalSpacing(4)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.buttons: dict[str, OptionToggleButton] = {}

    def add_option(self, key: str, text: str, tooltip: str) -> OptionToggleButton:
        if key not in self.OPTION_ORDER:
            raise ValueError(f"Unknown plot option: {key}")
        if key in self.buttons:
            raise ValueError(f"Duplicate plot option: {key}")
        button = OptionToggleButton(text, tooltip, self)
        self.buttons[key] = button
        self._relayout()
        return button

    def resizeEvent(self, event):
        self._relayout()
        super().resizeEvent(event)

    def _relayout(self):
        while self.grid.count():
            self.grid.takeAt(0)

        ordered_buttons = [self.buttons[key] for key in self.OPTION_ORDER if key in self.buttons]
        sparse_layout = len(ordered_buttons) <= 2
        column_count = 2 if sparse_layout else 3 if self.width() >= 430 else 2
        buttons_per_row = 1 if sparse_layout else column_count

        for column in range(3):
            self.grid.setColumnStretch(column, 1 if column < column_count else 0)

        for index, button in enumerate(ordered_buttons):
            row, column = divmod(index, buttons_per_row)
            self.grid.addWidget(button, row, column)


# Base class for plot options
class BasePlotOptions(QWidget):
    OPTION_SECTION_SPACING = 5

    def __init__(self, parent: PlotWidget):
        super().__init__(parent)
        self.gui_main = parent.parent
        self.layout: QVBoxLayout = QVBoxLayout(self)
        self.layout.setSpacing(2)  # Minimal spacing between widgets
        self.layout.setContentsMargins(8, 0, 8, 8)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.create_options()

    def create_form_layout(self):
        """Create a standardized form layout with consistent styling"""
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(2)  # Reduced vertical spacing for tighter layout
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)  # Keep everything on one row
        form.setContentsMargins(0, 0, 0, 0)
        return form

    def create_toggle_grid(self) -> StableOptionGrid:
        self.toggle_grid = StableOptionGrid(self)
        self.layout.addWidget(self.toggle_grid)
        return self.toggle_grid

    def add_toggle(self, key: str, text: str, tooltip: str) -> OptionToggleButton:
        if not hasattr(self, "toggle_grid"):
            self.create_toggle_grid()
        return self.toggle_grid.add_option(key, text, tooltip)

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
    def __init__(self, gui_main: MonstimGUI, parent=None):
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
        grid.setSpacing(2)  # Minimal spacing between checkboxes
        grid.setContentsMargins(2, 2, 2, 2)  # Minimal padding to minimize space

        # Match the width of the other settings boxes in the options panel.
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.checkboxes: list[QCheckBox] = []

        # Hide the widget if there are no channels to avoid taking up space
        if max_ch == 0:
            self.hide()
            return

        total = (max_ch + 5) // 6 * 6  # Round up to the nearest multiple of 6
        for col in range(6):
            grid.setColumnStretch(col, 1)
        for idx in range(total):
            cb = ChannelCheckBox(f"{idx}")
            cb.setObjectName("plotChannelSelector")
            cb.setProperty("channelIndex", idx)
            cb.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            # Only enable the ones your data actually has
            cb.setEnabled(idx < max_ch)
            cb.setChecked(idx < max_ch)
            row, col = divmod(idx, 6)
            # Center the checkboxes in their cells for better alignment
            grid.addWidget(cb, row, col, alignment=Qt.AlignmentFlag.AlignCenter)
            self.checkboxes.append(cb)

        self.setLayout(grid)

    def get_selected_channels(self) -> list[int]:
        return [i for i, cb in enumerate(self.checkboxes) if cb.isChecked()]

    def set_selected_channels(self, selected: list[int]):
        for i, cb in enumerate(self.checkboxes):
            if cb.isEnabled():
                cb.setChecked(i in selected)


# EMG Options
class EMGOptions(BasePlotOptions):
    def create_options(self):
        # Data type options box
        form = self.create_form_layout()

        self.data_type_combo = ResponsiveComboBox()
        self.data_type_combo.addItems(DATA_TYPES)
        form.addRow("Select Data Type:", self.data_type_combo)

        # flags / legend / colormap
        self.all_windows_checkbox = self.add_toggle("show_flags", "Show Flags", "Show all latency windows in the plot.")
        self.latency_legend_checkbox = self.add_toggle("show_legend", "Show Legend", "Show the latency-window legend in the plot.")
        self.plot_colormap_checkbox = self.add_toggle("show_colormap", "Show Colormap", "Show a colormap legend beside the plot.")
        self.all_windows_checkbox.setChecked(True)
        self.all_windows_checkbox.toggled.connect(self._on_all_windows_toggled)
        self._on_all_windows_toggled(self.all_windows_checkbox.isChecked())
        self.latency_legend_checkbox.setChecked(True)
        self.plot_colormap_checkbox.setChecked(True)
        self.interactive_cursor_checkbox = self.add_toggle(
            "interactive_cursor", "Interactive Cursor", "Show an interactive crosshair cursor in the plot."
        )
        self.interactive_cursor_checkbox.setChecked(False)
        self._add_extrema_controls(form)

        # Create the channel selector
        self.channel_selector = ChannelSelectorWidget(self.gui_main, parent=self)

        # Add widgets to layout with proper spacing
        options_widget = QWidget()
        options_widget.setLayout(form)
        options_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.layout.insertWidget(0, options_widget)
        self.layout.addWidget(self.channel_selector)

    def _on_all_windows_toggled(self, state):
        # Enable or disable the latency legend checkbox based on the state of the all_windows_checkbox
        enabled = state is True or state == Qt.CheckState.Checked or state == Qt.CheckState.Checked.value
        self.latency_legend_checkbox.setChecked(enabled)
        self.latency_legend_checkbox.setEnabled(enabled)

    def _add_extrema_controls(self, form):
        self.show_extrema_labels_checkbox = self.add_toggle(
            "show_extrema", "Show PTT Extrema", "Show selected PTT extrema on filtered, unrectified EMG traces."
        )
        self.extrema_method_combo = ResponsiveComboBox()
        self.extrema_method_combo.addItems(EXTREMA_METHODS)
        form.addRow("Extrema Method:", self.extrema_method_combo)
        self.show_extrema_labels_checkbox.toggled.connect(self._update_extrema_controls)
        self.data_type_combo.currentTextChanged.connect(self._update_extrema_controls)
        self._update_extrema_controls()

    def _update_extrema_controls(self):
        supported = self.data_type_combo.currentText() == "filtered"
        self.show_extrema_labels_checkbox.setEnabled(supported)
        if not supported:
            self.show_extrema_labels_checkbox.setChecked(False)
        self.extrema_method_combo.setEnabled(supported and self.show_extrema_labels_checkbox.isChecked())

    def get_options(self):
        return {
            "channel_indices": self.channel_selector.get_selected_channels(),
            "data_type": self.data_type_combo.currentText(),
            "all_flags": self.all_windows_checkbox.isChecked(),
            "plot_legend": self.latency_legend_checkbox.isChecked(),
            "plot_colormap": self.plot_colormap_checkbox.isChecked(),
            "interactive_cursor": self.interactive_cursor_checkbox.isChecked(),
            "show_extrema_labels": self.show_extrema_labels_checkbox.isChecked(),
            "extrema_label_method": self.extrema_method_combo.currentText(),
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
        if "show_extrema_labels" in options:
            self.show_extrema_labels_checkbox.setChecked(options["show_extrema_labels"])
        if "extrema_label_method" in options:
            index = self.extrema_method_combo.findText(options["extrema_label_method"])
            if index >= 0:
                self.extrema_method_combo.setCurrentIndex(index)
        self._update_extrema_controls()


class SingleEMGRecordingOptions(BasePlotOptions):
    def create_options(self):
        form = self.create_form_layout()

        self.data_type_combo = ResponsiveComboBox()
        self.data_type_combo.addItems(DATA_TYPES)
        form.addRow("Select Data Type:", self.data_type_combo)

        # Create and add checkboxes
        self.all_windows_checkbox = self.add_toggle("show_flags", "Show Flags", "Show all analysis windows in the plot.")
        self.latency_legend_checkbox = self.add_toggle("show_legend", "Show Legend", "Show the latency-window legend in the plot.")
        self.plot_colormap_checkbox = self.add_toggle("show_colormap", "Show Colormap", "Show a colormap beside the plot.")
        self.fixed_y_axis_checkbox = self.add_toggle("fixed_y_axis", "Fixed Y-Axis", "Fix the y-axis to a range of [-1, 1].")
        self.interactive_cursor_checkbox = self.add_toggle(
            "interactive_cursor", "Interactive Cursor", "Show an interactive crosshair cursor in the plot."
        )

        # Add checkboxes to form
        # Optional TODO: Add ability to set which flags to display
        self.all_windows_checkbox.setChecked(True)
        self.all_windows_checkbox.toggled.connect(self._on_all_windows_toggled)
        self._on_all_windows_toggled(self.all_windows_checkbox.isChecked())

        self.latency_legend_checkbox.setChecked(True)
        self.plot_colormap_checkbox.setChecked(True)
        self.fixed_y_axis_checkbox.setChecked(True)  # Set the initial state to True
        self.interactive_cursor_checkbox.setChecked(False)
        self._add_extrema_controls(form)

        # Create the recording cycler widget and add it to the form
        self.recording_cycler = RecordingCyclerWidget(self)

        # Create the channel selector widget and add it to the form
        self.channel_selector = ChannelSelectorWidget(self.gui_main, parent=self)

        # Add widgets to layout with proper spacing and organization
        options_widget = QWidget()
        options_widget.setLayout(form)
        options_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.layout.insertWidget(0, options_widget)
        self.layout.addSpacing(self.OPTION_SECTION_SPACING)
        self.layout.addWidget(self.recording_cycler)
        self.layout.addSpacing(self.OPTION_SECTION_SPACING)
        self.layout.addWidget(self.channel_selector)

    def _on_all_windows_toggled(self, state):
        # Enable or disable the latency legend checkbox based on the state of the all_windows_checkbox
        enabled = state is True or state == Qt.CheckState.Checked or state == Qt.CheckState.Checked.value
        self.latency_legend_checkbox.setChecked(enabled)
        self.latency_legend_checkbox.setEnabled(enabled)

    _add_extrema_controls = EMGOptions._add_extrema_controls
    _update_extrema_controls = EMGOptions._update_extrema_controls

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
            "show_extrema_labels": self.show_extrema_labels_checkbox.isChecked(),
            "extrema_label_method": self.extrema_method_combo.currentText(),
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
        if "show_extrema_labels" in options:
            self.show_extrema_labels_checkbox.setChecked(options["show_extrema_labels"])
        if "extrema_label_method" in options:
            index = self.extrema_method_combo.findText(options["extrema_label_method"])
            if index >= 0:
                self.extrema_method_combo.setCurrentIndex(index)
        self._update_extrema_controls()


class SessionReflexCurvesOptions(BasePlotOptions):
    def create_options(self):
        form = self.create_form_layout()

        self.method_combo = ResponsiveComboBox()
        self.method_combo.addItems(CALCULATION_METHODS)
        self.method_combo.setCurrentIndex(0)  # Set the initial selection to "rms"
        self.method_combo.setToolTip("Method used to calculate the average reflex amplitude.")
        form.addRow("Reflex Calc. Method:", self.method_combo)

        # Checkboxes
        self.relative_to_mmax_checkbox = self.add_toggle(
            "relative_to_mmax", "Relative to M-max", "Calculate reflex amplitudes relative to the M-max value."
        )
        self.relative_to_mmax_checkbox.setChecked(False)

        self.show_legend_checkbox = self.add_toggle("show_legend", "Show Legend", "Show the plot legend.")
        self.show_legend_checkbox.setChecked(True)

        self.interactive_cursor_checkbox = self.add_toggle(
            "interactive_cursor", "Interactive Cursor", "Show an interactive crosshair cursor in the plot."
        )
        self.interactive_cursor_checkbox.setChecked(False)

        # Create the channel selector
        self.channel_selector = ChannelSelectorWidget(self.gui_main, parent=self)

        # Add widgets to layout with proper spacing
        options_widget = QWidget()
        options_widget.setLayout(form)
        options_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.layout.insertWidget(0, options_widget)
        self.layout.addWidget(self.channel_selector)

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
        self.method_combo.addItems(CALCULATION_METHODS)
        self.method_combo.setCurrentIndex(0)  # Set the initial selection to "rms"
        self.method_combo.setToolTip("Method used to calculate the average reflex amplitude.")
        form.addRow("Reflex Calc. Method:", self.method_combo)

        # Checkboxes
        self.relative_to_mmax_checkbox = self.add_toggle(
            "relative_to_mmax", "Relative to M-max", "Calculate reflex amplitudes relative to the M-max value."
        )
        self.relative_to_mmax_checkbox.setChecked(False)
        self.show_legend_checkbox = self.add_toggle("show_legend", "Show Legend", "Show the plot legend.")
        self.show_legend_checkbox.setChecked(True)

        # Create the channel selector
        self.channel_selector = ChannelSelectorWidget(self.gui_main, parent=self)

        # Add widgets to layout with proper spacing
        options_widget = QWidget()
        options_widget.setLayout(form)
        options_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.layout.insertWidget(0, options_widget)
        self.layout.addWidget(self.channel_selector)

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


class LatencyWindowDistributionOptions(BasePlotOptions):
    """Options for dataset-level latency-window amplitude distribution plots."""

    def create_options(self):
        form = self.create_form_layout()

        # Method selection
        self.method_combo = ResponsiveComboBox()
        self.method_combo.addItems(CALCULATION_METHODS)
        self.method_combo.setCurrentIndex(0)
        form.addRow("Reflex Calc. Method:", self.method_combo)

        # Bins (integer)
        self.bins_spin = QSpinBox()
        self.bins_spin.setMinimum(5)
        self.bins_spin.setMaximum(1000)
        self.bins_spin.setValue(30)
        self.bins_spin.setToolTip("Number of bins to use for amplitude histogram (shared per channel)")
        form.addRow("Number of bins:", self.bins_spin)

        # Density checkbox
        self.density_checkbox = self.add_toggle("show_density", "Show Density", "Plot densities instead of raw counts.")

        # Legend checkbox
        self.plot_legend_checkbox = self.add_toggle("show_legend", "Show Legend", "Show the plot legend.")
        self.plot_legend_checkbox.setChecked(True)

        # Channel selector
        self.channel_selector = ChannelSelectorWidget(self.gui_main, parent=self)

        options_widget = QWidget()
        options_widget.setLayout(form)
        options_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.layout.insertWidget(0, options_widget)
        self.layout.addWidget(self.channel_selector)

    def get_options(self):
        return {
            "channel_indices": self.channel_selector.get_selected_channels(),
            "method": self.method_combo.currentText(),
            "bins": int(self.bins_spin.value()),
            "density": self.density_checkbox.isChecked(),
            "plot_legend": self.plot_legend_checkbox.isChecked(),
        }

    def set_options(self, options):
        if "channel_indices" in options:
            self.channel_selector.set_selected_channels(options["channel_indices"])
        if "method" in options:
            idx = self.method_combo.findText(options["method"])
            if idx >= 0:
                self.method_combo.setCurrentIndex(idx)
        if "bins" in options:
            try:
                self.bins_spin.setValue(int(options["bins"]))
            except Exception as e:
                # If bins value is invalid, log the error and set to default (30)
                logger.warning(f"Invalid bins value in set_options: {options.get('bins')!r} ({e}) - using default 30")
                self.bins_spin.setValue(30)
        if "density" in options:
            self.density_checkbox.setChecked(bool(options["density"]))
        if "plot_legend" in options:
            self.plot_legend_checkbox.setChecked(bool(options["plot_legend"]))


class AverageSessionReflexOptions(BasePlotOptions):
    def create_options(self):
        form = self.create_form_layout()

        self.method_combo = ResponsiveComboBox()
        self.method_combo.addItems(CALCULATION_METHODS)
        self.method_combo.setCurrentIndex(0)  # Set the initial selection to "rms"
        self.method_combo.setToolTip("Method used to calculate the average reflex amplitude.")
        form.addRow("Reflex Calc. Method:", self.method_combo)

        # Checkboxes
        self.relative_to_mmax_checkbox = self.add_toggle(
            "relative_to_mmax", "Relative to M-max", "Calculate reflex amplitudes relative to the M-max value."
        )
        self.relative_to_mmax_checkbox.setChecked(False)
        self.show_legend_checkbox = self.add_toggle("show_legend", "Show Legend", "Show the plot legend.")
        self.show_legend_checkbox.setChecked(True)
        self.interactive_cursor_checkbox = self.add_toggle(
            "interactive_cursor", "Interactive Cursor", "Show an interactive crosshair cursor in the plot."
        )
        self.interactive_cursor_checkbox.setChecked(False)

        # Create the channel selector
        self.channel_selector = ChannelSelectorWidget(self.gui_main, parent=self)

        # Add widgets to layout with proper spacing
        options_widget = QWidget()
        options_widget.setLayout(form)
        options_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.layout.insertWidget(0, options_widget)
        self.layout.addWidget(self.channel_selector)

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
        self.method_combo.addItems(CALCULATION_METHODS)
        self.method_combo.setCurrentIndex(0)  # Set the initial selection to "rms"
        self.method_combo.setToolTip("Method used to calculate the average reflex amplitude.")
        form.addRow("Reflex Calc. Method:", self.method_combo)

        # Checkboxes
        self.interactive_cursor_checkbox = self.add_toggle(
            "interactive_cursor", "Interactive Cursor", "Show an interactive crosshair cursor in the plot."
        )
        self.interactive_cursor_checkbox.setChecked(False)

        # Create the channel selector
        self.channel_selector = ChannelSelectorWidget(self.gui_main, parent=self)

        # Add widgets to layout with proper spacing
        options_widget = QWidget()
        options_widget.setLayout(form)
        options_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.layout.insertWidget(0, options_widget)
        self.layout.addWidget(self.channel_selector)

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
        self.method_combo.addItems(CALCULATION_METHODS)
        self.method_combo.setCurrentIndex(0)  # Set the initial selection to "rms"
        self.method_combo.setToolTip("Method used to calculate the average reflex amplitude.")
        form.addRow("Reflex Calc. Method:", self.method_combo)

        # Checkboxes
        self.relative_to_mmax_checkbox = self.add_toggle(
            "relative_to_mmax", "Relative to M-max", "Calculate reflex amplitudes relative to the M-max value."
        )
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

        self.layout.insertWidget(0, options_widget)
        self.layout.addWidget(self.channel_selector)

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
