import logging
import traceback
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from monstim_gui import MonstimGUI
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt

from ..core.utils.dataframe_exporter import DataFrameDialog


class PlotController:
    """Handle plotting and returning raw data."""

    def __init__(self, gui : 'MonstimGUI'):
        self.gui : 'MonstimGUI' = gui

    def plot_data(self, return_raw_data: bool = False):
        self.gui.plot_widget.canvas.show()
        plot_type_raw = self.gui.plot_widget.plot_type_combo.currentText()
        plot_type = self.gui.PLOT_TYPE_DICT.get(plot_type_raw)
        plot_options = self.gui.plot_widget.get_plot_options()
        raw_data = None

        if self.gui.plot_widget.session_radio.isChecked():
            level = "session"
            level_object = self.gui.current_session
        elif self.gui.plot_widget.dataset_radio.isChecked():
            level = "dataset"
            level_object = self.gui.current_dataset
        elif self.gui.plot_widget.experiment_radio.isChecked():
            level = "experiment"
            level_object = self.gui.current_experiment
        else:
            QMessageBox.warning(
                self.gui,
                "Warning",
                "Please select a level to plot data from (session, dataset, or experiment).",
            )
            logging.warning("No level selected for plotting data.")
            return

        if level_object is None:
            QMessageBox.warning(
                self.gui,
                "Warning",
                f"No {level} data exists to plot. Please try importing experiment data first.",
            )
            logging.warning(
                f"No {level} data exists to plot. Please try importing experiment data first."
            )
            return

        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            raw_data = level_object.plot(
                plot_type=plot_type,
                **plot_options,
                canvas=self.gui.plot_widget.canvas,
            )
        except Exception as e:
            QMessageBox.critical(self.gui, "Error", f"An error occurred: {e}")
            logging.error(f"An error occurred while plotting: {e}")
            logging.error(f"Plot type: {plot_type}, options: {plot_options}")
            logging.error(
                f"Current session: {self.gui.current_session}, current dataset: {self.gui.current_dataset}"
            )
            logging.error(traceback.format_exc())
        finally:
            QApplication.restoreOverrideCursor()

        logging.info(
            f"Plot Created. level: {level} type: {plot_type}, options: {plot_options}, return_raw_data: {return_raw_data}."
        )
        self.gui.plot_pane.layout.update()

        if return_raw_data:
            return raw_data
        return None

    def get_raw_data(self):
        raw_data = self.plot_data(return_raw_data=True)
        if raw_data is not None:
            dialog = DataFrameDialog(raw_data, self.gui)
            dialog.exec()
        else:
            QMessageBox.warning(self.gui, "Warning", "No data to display.")
