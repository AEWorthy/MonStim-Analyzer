import logging
from PyQt6.QtWidgets import QMessageBox

from monstim_signals.core.utils import format_report
from ..dialogs import CopyableReportDialog


class ReportManager:
    """Show various reports for the current selection."""

    def __init__(self, gui):
        self.gui = gui

    def show_session_report(self):
        logging.debug("Showing session parameters report.")
        if self.gui.current_session:
            report = self.gui.current_session.session_parameters()
            report = format_report(report)
            dialog = CopyableReportDialog("Session Report", report, self.gui)
            dialog.exec()
        else:
            QMessageBox.warning(self.gui, "Warning", "Please select a session first.")

    def show_dataset_report(self):
        logging.debug("Showing dataset parameters report.")
        if self.gui.current_dataset:
            report = self.gui.current_dataset.dataset_parameters()
            report = format_report(report)
            dialog = CopyableReportDialog("Dataset Report", report, self.gui)
            dialog.exec()
        else:
            QMessageBox.warning(self.gui, "Warning", "Please select a dataset first.")

    def show_experiment_report(self):
        logging.debug("Showing experiment parameters report.")
        if self.gui.current_experiment:
            report = self.gui.current_experiment.experiment_parameters()
            report = format_report(report)
            dialog = CopyableReportDialog("Experiment Report", report, self.gui)
            dialog.exec()
        else:
            QMessageBox.warning(self.gui, "Warning", "Please select an experiment first.")

    def show_mmax_report(self):
        logging.debug("Showing M-max report.")
        if self.gui.current_session:
            report = self.gui.current_session.m_max_report()
            report = format_report(report)
            dialog = CopyableReportDialog("M-max Report (method = RMS)", report, self.gui)
            dialog.exec()
        else:
            QMessageBox.warning(self.gui, "Warning", "Please select a session first.")
