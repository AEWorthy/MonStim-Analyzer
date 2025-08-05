import copy
import logging
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..core.responsive_widgets import ResponsiveComboBox, ResponsiveScrollArea
from ..core.ui_scaling import ui_scaling
from .plot_types import PLOT_OPTIONS_DICT

if TYPE_CHECKING:
    from monstim_gui import MonstimGUI

    from .plot_options import BasePlotOptions


# Plotting Widget
class PlotWidget(QGroupBox):
    def __init__(self, parent: "MonstimGUI"):
        super().__init__("Plotting", parent)
        self.current_option_widget: "BasePlotOptions" = None
        self.parent: "MonstimGUI" = parent
        self.layout: "QVBoxLayout" = QVBoxLayout()
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
        self.view_group = QButtonGroup(self)
        self.session_radio = QRadioButton("Session")
        self.dataset_radio = QRadioButton("Dataset")
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
        self.plot_type_combo = ResponsiveComboBox()
        form.addRow("Plot Type:", self.plot_type_combo)
        self.plot_type_combo.currentTextChanged.connect(self.on_plot_type_changed)

        self.layout.addLayout(form)

        # Dynamic Options Box with scroll area for long content
        self.options_box = QGroupBox("Options")
        self.options_box.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding
        )

        # Create scroll area for options content
        self.options_scroll = ResponsiveScrollArea()
        self.options_content = QWidget()
        self.options_layout = QVBoxLayout(self.options_content)
        self.options_layout.setContentsMargins(6, 6, 6, 6)
        self.options_layout.setSpacing(6)

        self.options_scroll.setWidget(self.options_content)

        # Layout for the group box
        options_box_layout = QVBoxLayout(self.options_box)
        options_box_layout.setContentsMargins(4, 4, 4, 4)
        options_box_layout.addWidget(self.options_scroll)

        self.layout.addWidget(self.options_box)

        # Create the buttons for plotting and extracting data
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.plot_button = QPushButton("Plot")
        self.get_data_button = QPushButton("Plot & Extract Data")
        for btn in (self.plot_button, self.get_data_button):
            btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
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
        self.plot_options = PLOT_OPTIONS_DICT

        # Store the last selected plot type and options for each view
        self.last_plot_type = {
            "session": "EMG",
            "dataset": "Average Reflex Curves",
            "experiment": "Average Reflex Curves",
        }
        self.last_options = {
            "session": {plot_type: {} for plot_type in self.plot_options["session"]},
            "dataset": {plot_type: {} for plot_type in self.plot_options["dataset"]},
            "experiment": {
                plot_type: {} for plot_type in self.plot_options["experiment"]
            },
        }

        # Persistent channel selection that carries across view and plot type changes
        self.persistent_channel_selection = []

        # Initialize plot types and options
        self.update_plot_types()
        self.update_plot_options()

    def create_additional_options(self):
        self.additional_options_layout = QVBoxLayout()
        self.layout.addLayout(self.additional_options_layout)

    def save_current_channel_selection(self):
        """Save the current channel selection to persistent storage."""
        if self.current_option_widget and hasattr(
            self.current_option_widget, "channel_selector"
        ):
            self.persistent_channel_selection = (
                self.current_option_widget.channel_selector.get_selected_channels()
            )

    def connect_channel_selection_updates(self):
        """Connect channel selector checkboxes to update persistent selection."""
        if self.current_option_widget and hasattr(
            self.current_option_widget, "channel_selector"
        ):
            for checkbox in self.current_option_widget.channel_selector.checkboxes:
                checkbox.stateChanged.connect(self.save_current_channel_selection)

    def on_view_changed(self):
        match self.view_group.checkedButton():
            case self.session_radio:
                new_view = "session"
            case self.dataset_radio:
                new_view = "dataset"
            case self.experiment_radio:
                new_view = "experiment"

        if new_view == self.view:
            return

        # Save current options for the current view and plot type before changing
        if self.current_option_widget and self.view:
            try:
                # Save the current channel selection to persistent storage
                self.save_current_channel_selection()

                current_plot_type = self.plot_type_combo.currentText()
                if current_plot_type:
                    current_options = self.current_option_widget.get_options()
                    # Deep copy to ensure no reference sharing
                    self.last_options[self.view][current_plot_type] = copy.deepcopy(
                        current_options
                    )
            except Exception as e:
                logging.warning(
                    f"Failed to save options for {self.view} - {current_plot_type}: {e}"
                )

        # Change to new view
        self.view = new_view

        # Block signals to prevent multiple updates
        self.plot_type_combo.blockSignals(True)
        self.plot_type_combo.clear()
        self.plot_type_combo.addItems(self.plot_options[self.view].keys())

        # Set the last plot type for this view
        last_plot_type = self.last_plot_type.get(self.view)
        if last_plot_type and last_plot_type in self.plot_options[self.view]:
            self.plot_type_combo.setCurrentText(last_plot_type)
        else:
            # Use first available plot type
            available_types = list(self.plot_options[self.view].keys())
            if available_types:
                self.plot_type_combo.setCurrentText(available_types[0])
                self.last_plot_type[self.view] = available_types[0]

        self.plot_type_combo.blockSignals(False)

        # Update the plot options
        self.update_plot_options()

    def on_plot_type_changed(self):
        plot_type = self.plot_type_combo.currentText()

        # Skip if plot_type is empty (happens during combo box updates)
        if not plot_type:
            return

        # Get the previous plot type before updating
        previous_plot_type = self.last_plot_type.get(self.view)

        # Skip if this is the same plot type (avoid unnecessary updates)
        if plot_type == previous_plot_type:
            return

        # Save current options for the PREVIOUS plot type before changing
        if self.current_option_widget and previous_plot_type:
            try:
                # Save the current channel selection to persistent storage
                self.save_current_channel_selection()

                current_options = self.current_option_widget.get_options()
                # Deep copy to ensure no reference sharing
                self.last_options[self.view][previous_plot_type] = copy.deepcopy(
                    current_options
                )
            except Exception as e:
                logging.warning(
                    f"Failed to save options for {self.view} - {previous_plot_type}: {e}"
                )

        # Update the last plot type and refresh the options widget
        self.last_plot_type[self.view] = plot_type
        self.update_plot_options()

    def on_data_selection_changed(self):
        """Called when the underlying data (session/dataset) changes.
        This should refresh the current plot options widget without changing saved options.
        """
        try:
            # Just refresh the current widget without changing saved options
            plot_type = self.plot_type_combo.currentText()
            if plot_type and self.current_option_widget:
                # Remove the current widget
                self.options_layout.removeWidget(self.current_option_widget)
                self.current_option_widget.deleteLater()
                self.current_option_widget = None

                # Create a new widget with the same plot type
                if plot_type in self.plot_options[self.view]:
                    self.current_option_widget = self.plot_options[self.view][
                        plot_type
                    ](self)
                    self.options_layout.addWidget(self.current_option_widget)

                    # Restore the saved options for this view and plot type
                    if (
                        plot_type in self.last_options[self.view]
                        and self.last_options[self.view][plot_type]
                    ):
                        try:
                            saved_options = self.last_options[self.view][plot_type]

                            # Filter options to only include those that are valid for this plot type
                            default_options = self.current_option_widget.get_options()
                            filtered_options = {
                                k: v
                                for k, v in saved_options.items()
                                if k in default_options
                            }

                            # Use persistent channel selection instead of view-specific selection
                            if "channel_indices" in filtered_options and hasattr(
                                self, "persistent_channel_selection"
                            ):
                                filtered_options["channel_indices"] = (
                                    self.persistent_channel_selection
                                )

                            self.current_option_widget.set_options(filtered_options)
                        except Exception as e:
                            logging.warning(
                                f"Failed to restore options for {self.view} - {plot_type}: {e}"
                            )
                    else:
                        # Apply persistent channel selection even if no other saved options exist
                        if hasattr(self, "persistent_channel_selection") and hasattr(
                            self.current_option_widget, "channel_selector"
                        ):
                            self.current_option_widget.channel_selector.set_selected_channels(
                                self.persistent_channel_selection
                            )

                    # Connect channel selection updates for real-time persistence
                    self.connect_channel_selection_updates()

                    self.options_layout.update()

                    # Handle special case for Single EMG Recordings
                    if plot_type == "Single EMG Recordings":
                        self.current_option_widget.recording_cycler.reset_max_recordings()

        except AttributeError:
            pass

    def update_plot_types(self):
        self.plot_type_combo.blockSignals(True)
        self.plot_type_combo.clear()
        self.plot_type_combo.addItems(self.plot_options[self.view].keys())
        self.plot_type_combo.setCurrentText(self.last_plot_type[self.view])
        self.plot_type_combo.blockSignals(False)

    def update_plot_options(self):
        plot_type = self.plot_type_combo.currentText()

        # Skip if plot_type is empty (happens during combo box updates)
        if not plot_type:
            return

        if self.current_option_widget:
            self.options_layout.removeWidget(self.current_option_widget)
            self.current_option_widget.deleteLater()
            self.current_option_widget = None

        if plot_type in self.plot_options[self.view]:
            # Create the new widget
            option_class = self.plot_options[self.view][plot_type]
            self.current_option_widget = option_class(self)
            self.options_layout.addWidget(self.current_option_widget)

            # Restore saved options if they exist
            if (
                plot_type in self.last_options[self.view]
                and self.last_options[self.view][plot_type]
            ):
                try:
                    saved_options = self.last_options[self.view][plot_type]

                    # Filter options to only include those that are valid for this plot type
                    # Get default options from the widget to see what keys are valid
                    default_options = self.current_option_widget.get_options()
                    filtered_options = {
                        k: v for k, v in saved_options.items() if k in default_options
                    }

                    # Use persistent channel selection instead of view-specific selection
                    if "channel_indices" in filtered_options and hasattr(
                        self, "persistent_channel_selection"
                    ):
                        filtered_options["channel_indices"] = (
                            self.persistent_channel_selection
                        )

                    self.current_option_widget.set_options(filtered_options)
                except Exception as e:
                    logging.warning(
                        f"Failed to restore options for {self.view} - {plot_type}: {e}"
                    )
            else:
                # Apply persistent channel selection even if no other saved options exist
                if hasattr(self, "persistent_channel_selection") and hasattr(
                    self.current_option_widget, "channel_selector"
                ):
                    self.current_option_widget.channel_selector.set_selected_channels(
                        self.persistent_channel_selection
                    )

            # Connect channel selection updates for real-time persistence
            self.connect_channel_selection_updates()

        self.options_layout.update()

        if plot_type == "Single EMG Recordings" and self.current_option_widget:
            self.current_option_widget.recording_cycler.reset_max_recordings()

    def get_plot_options(self):
        if self.current_option_widget:
            return self.current_option_widget.get_options()
        return {}
