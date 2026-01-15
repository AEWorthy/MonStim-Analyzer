import glob
import logging
import multiprocessing
import os
import re
import shutil
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING
from zipfile import ZIP_DEFLATED, ZipFile

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QInputDialog,
    QMessageBox,
    QProgressDialog,
)

from monstim_gui.core.application_state import app_state
from monstim_gui.io.experiment_loader import ExperimentLoadingThread
from monstim_signals.core import get_config_path, get_data_path, get_log_dir
from monstim_signals.io.csv_importer import (
    GUIExptImportingThread,
    MultiExptImportingThread,
)

if TYPE_CHECKING:
    from monstim_signals import Experiment

    from ..gui_main import MonstimGUI


class DataManager:
    """Handle loading and saving of experiment data."""

    def __init__(self, gui):
        self.gui: "MonstimGUI" = gui
        self.loading_completed_successfully = False

    # ------------------------------------------------------------------
    # experiment discovery
    def unpack_existing_experiments(self):
        logging.debug("Unpacking existing experiments.")
        if os.path.exists(self.gui.output_path):
            try:
                # In headless tests there may be no QApplication; guard cursor changes
                has_app = QApplication.instance() is not None
                if has_app:
                    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                self.gui.expts_dict = {
                    name: os.path.join(self.gui.output_path, name)
                    for name in os.listdir(self.gui.output_path)
                    if os.path.isdir(os.path.join(self.gui.output_path, name))
                }
                self.gui.expts_dict_keys = sorted(self.gui.expts_dict.keys())
                logging.debug("Existing experiments unpacked successfully.")
            except Exception as e:
                if has_app:
                    QApplication.restoreOverrideCursor()
                QMessageBox.critical(
                    self.gui,
                    "Error",
                    f"An error occurred while unpacking existing experiments: {e}",
                )
                logging.error(f"An error occurred while unpacking existing experiments: {e}")
                logging.error(traceback.format_exc())
            finally:
                if has_app:
                    QApplication.restoreOverrideCursor()

    # ------------------------------------------------------------------
    # import experiment from CSVs
    def import_expt_data(self):
        logging.info("Importing new experiment data from CSV files.")

        # Get the last used import directory, fallback to default data path
        last_import_path = app_state.get_last_import_path()
        if not last_import_path or not os.path.isdir(last_import_path):
            last_import_path = str(get_data_path())

        expt_path = QFileDialog.getExistingDirectory(self.gui, "Select Experiment Directory", last_import_path)
        expt_name = os.path.splitext(os.path.basename(expt_path))[0]

        if expt_path and expt_name:
            # Save the selected directory for next time
            app_state.save_last_import_path(os.path.dirname(expt_path))

            if os.path.exists(os.path.join(self.gui.output_path, expt_name)):
                overwrite = QMessageBox.question(
                    self.gui,
                    "Warning",
                    "This experiment already exists in your 'data' folder. Do you want to continue the importation process and overwrite the existing data?\n\nNote: This will also reset and changes you made to the datasets in this experiment (e.g., channel names, latency time windows, etc.)",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if overwrite == QMessageBox.StandardButton.Yes:
                    logging.info(f"Overwriting existing experiment '{expt_name}' in the output folder.")

                    # Close all existing data to prevent file locking issues
                    self.close_all_data()

                    # Use retry mechanism for robust deletion
                    try:
                        import gc
                        import time

                        gc.collect()

                        max_retries = 3
                        for retry in range(max_retries):
                            try:
                                shutil.rmtree(os.path.join(self.gui.output_path, expt_name))
                                logging.info(f"Deleted existing experiment '{expt_name}' in 'data' folder.")
                                break
                            except (OSError, PermissionError) as e:
                                if retry < max_retries - 1:
                                    logging.warning(f"Failed to delete '{expt_name}' on attempt {retry + 1}: {e}. Retrying...")
                                    time.sleep(0.5)
                                else:
                                    raise e
                    except Exception as e:
                        QMessageBox.critical(
                            self.gui,
                            "Error",
                            f"Failed to delete existing experiment '{expt_name}': {e}\n\nPlease close any applications that might be using these files and try again.",
                        )
                        logging.error(f"Failed to delete existing experiment '{expt_name}': {e}")
                        return
                else:
                    logging.info(f"User chose not to overwrite existing experiment '{expt_name}' in the output folder.")
                    QMessageBox.warning(
                        self.gui,
                        "Canceled",
                        "The importation of your data was canceled.",
                    )
                    return

            try:
                dataset_dirs_without_csv = []
                for dataset_dir in os.listdir(expt_path):
                    dataset_path = os.path.join(expt_path, dataset_dir)
                    if os.path.isdir(dataset_path):
                        validated_path, metadata = self.validate_dataset_name(dataset_path)
                        # Update the dataset_path to the validated path (in case it was renamed)
                        dataset_path = validated_path
                        csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
                        if not csv_files:
                            dataset_dirs_without_csv.append(os.path.basename(dataset_path))
                if dataset_dirs_without_csv:
                    raise FileNotFoundError(
                        f"The following dataset directories do not contain any .csv files: {dataset_dirs_without_csv}"
                    )
            except Exception as e:
                QMessageBox.critical(
                    self.gui,
                    "Error",
                    f"An error occurred while validating your experiment: {e}.\n\nImportation was canceled.",
                )
                logging.error(f"An error occurred while validating dataset names: {e}. Importation was canceled.")
                logging.error(traceback.format_exc())
                return

            progress_dialog = QProgressDialog("Processing...", "Cancel", 0, 100, self.gui)
            progress_dialog.setWindowTitle("Importing Data")
            progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
            progress_dialog.setAutoClose(False)
            progress_dialog.setAutoReset(False)
            progress_dialog.show()

            max_workers = max(1, multiprocessing.cpu_count() - 1)
            self.thread = GUIExptImportingThread(expt_name, expt_path, self.gui.output_path, max_workers=max_workers)
            self.thread.progress.connect(progress_dialog.setValue)

            self.thread.finished.connect(progress_dialog.close)
            self.thread.finished.connect(self._on_import_finished)
            self.thread.finished.connect(
                lambda: self.gui.status_bar.showMessage("Data processed and imported successfully.", 5000)
            )
            self.thread.finished.connect(lambda: logging.info("Data processed and imported successfully."))

            self.thread.error.connect(lambda e: QMessageBox.critical(self.gui, "Error", f"An error occurred: {e}"))
            self.thread.error.connect(lambda e: logging.error(f"An error occurred while importing CSVs: {e}"))
            self.thread.error.connect(lambda: logging.error(traceback.format_exc()))

            self.thread.canceled.connect(progress_dialog.close)
            self.thread.canceled.connect(lambda: self.gui.status_bar.showMessage("Data processing canceled.", 5000))
            self.thread.canceled.connect(lambda: logging.info("Data processing canceled."))
            self.thread.canceled.connect(self._on_import_finished)

            self.thread.start()
            progress_dialog.canceled.connect(self.thread.cancel)
        else:
            QMessageBox.warning(self.gui, "Warning", "You must select a CSV directory.")
            logging.warning("No CSV directory selected. Import canceled.")

    # ------------------------------------------------------------------
    # import multiple experiments from CSVs
    def import_multiple_expt_data(self):
        logging.info("Importing multiple experiment data from CSV files.")

        # Get the last used import directory, fallback to default data path
        last_import_path = app_state.get_last_import_path()
        if not last_import_path or not os.path.isdir(last_import_path):
            last_import_path = str(get_data_path())

        # Use QFileDialog to select multiple directories
        from PySide6.QtWidgets import QFileDialog

        root_path = QFileDialog.getExistingDirectory(
            self.gui,
            "Select Root Directory Containing Multiple Experiments",
            last_import_path,
        )

        if not root_path:
            QMessageBox.warning(self.gui, "Warning", "You must select a root directory.")
            logging.warning("No root directory selected. Import canceled.")
            return

        # Save the selected directory for next time
        app_state.save_last_import_path(root_path)

        # Find all subdirectories that could be experiments
        potential_experiments = []
        try:
            for item in os.listdir(root_path):
                item_path = os.path.join(root_path, item)
                if os.path.isdir(item_path):
                    # Check if this directory contains subdirectories with CSV files
                    has_datasets = False
                    for subitem in os.listdir(item_path):
                        subitem_path = os.path.join(item_path, subitem)
                        if os.path.isdir(subitem_path):
                            csv_files = [f for f in os.listdir(subitem_path) if f.endswith(".csv")]
                            if csv_files:
                                has_datasets = True
                                break
                    if has_datasets:
                        potential_experiments.append(item_path)
        except Exception as e:
            QMessageBox.critical(
                self.gui,
                "Error",
                f"An error occurred while scanning the directory: {e}",
            )
            logging.error(f"An error occurred while scanning directory: {e}")
            return

        if not potential_experiments:
            QMessageBox.warning(
                self.gui,
                "Warning",
                "No valid experiment directories found. Each experiment should contain subdirectories with CSV files.",
            )
            return

        # Show selection dialog
        from PySide6.QtWidgets import (
            QCheckBox,
            QDialog,
            QLabel,
            QPushButton,
            QScrollArea,
            QVBoxLayout,
            QWidget,
        )

        dialog = QDialog(self.gui)
        dialog.setWindowTitle("Select Experiments to Import")
        dialog.setModal(True)
        dialog.resize(600, 400)

        layout = QVBoxLayout()
        info_text = (
            f"Found {len(potential_experiments)} potential experiments in:\n{root_path}\n\nSelect which experiments to import:"
        )
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Create scrollable area for checkboxes
        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        checkboxes = {}
        for exp_path in potential_experiments:
            exp_name = os.path.basename(exp_path)
            # Count datasets for preview
            dataset_count = len([d for d in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, d))])
            checkbox = QCheckBox(f"{exp_name} ({dataset_count} datasets)")
            checkbox.setChecked(True)
            checkbox.setToolTip(f"Full path: {exp_path}")
            checkboxes[exp_path] = checkbox
            scroll_layout.addWidget(checkbox)

        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)

        # Add buttons
        from PySide6.QtWidgets import QHBoxLayout

        button_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_none_btn = QPushButton("Select None")
        import_btn = QPushButton("Import Selected")
        cancel_btn = QPushButton("Cancel")

        button_layout.addWidget(select_all_btn)
        button_layout.addWidget(select_none_btn)
        button_layout.addStretch()
        button_layout.addWidget(import_btn)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)
        dialog.setLayout(layout)

        # Connect buttons
        select_all_btn.clicked.connect(lambda: [cb.setChecked(True) for cb in checkboxes.values()])
        select_none_btn.clicked.connect(lambda: [cb.setChecked(False) for cb in checkboxes.values()])
        cancel_btn.clicked.connect(dialog.reject)
        import_btn.clicked.connect(dialog.accept)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        # Get selected experiments
        selected_experiments = [path for path, checkbox in checkboxes.items() if checkbox.isChecked()]

        if not selected_experiments:
            QMessageBox.warning(self.gui, "Warning", "No experiments selected.")
            return

        # Check for existing experiments and handle conflicts
        conflicts = []
        for exp_path in selected_experiments:
            exp_name = os.path.basename(exp_path)
            if os.path.exists(os.path.join(self.gui.output_path, exp_name)):
                conflicts.append(exp_name)

        if conflicts:
            conflict_msg = (
                f"The following experiments already exist:\n{', '.join(conflicts)}\n\nDo you want to overwrite them?"
            )
            overwrite = QMessageBox.question(
                self.gui,
                "Conflicts Found",
                conflict_msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if overwrite != QMessageBox.StandardButton.Yes:
                return
            else:
                # Close all existing data to prevent file locking issues
                self.close_all_data()

                # Create a progress dialog for deletion if there are conflicts
                if conflicts:
                    deletion_progress = QProgressDialog("Preparing for import...", "Cancel", 0, len(conflicts), self.gui)
                    deletion_progress.setWindowTitle("Removing Existing Experiments")
                    deletion_progress.setWindowModality(Qt.WindowModality.WindowModal)
                    deletion_progress.setAutoClose(False)
                    deletion_progress.setAutoReset(False)
                    deletion_progress.show()
                    QApplication.processEvents()

                # Remove existing experiments with progress tracking
                for i, exp_name in enumerate(conflicts):
                    if conflicts:  # Only update progress if we have the dialog
                        deletion_progress.setLabelText(f"Removing existing experiment: {exp_name}")
                        deletion_progress.setValue(i)
                        QApplication.processEvents()

                        if deletion_progress.wasCanceled():
                            deletion_progress.close()
                            QMessageBox.information(self.gui, "Cancelled", "Import operation was cancelled.")
                            return

                    existing_path = os.path.join(self.gui.output_path, exp_name)

                    # Ensure all files are closed before deletion
                    try:
                        # Additional safety: try to ensure no handles are open
                        import gc

                        gc.collect()

                        # Use a retry mechanism for Windows file locking issues
                        import time

                        max_retries = 3
                        for retry in range(max_retries):
                            try:
                                shutil.rmtree(existing_path)
                                logging.info(f"Deleted existing experiment '{exp_name}' for overwrite.")
                                break
                            except (OSError, PermissionError) as e:
                                if retry < max_retries - 1:
                                    logging.warning(f"Failed to delete '{exp_name}' on attempt {retry + 1}: {e}. Retrying...")
                                    time.sleep(0.5)  # Brief pause before retry
                                else:
                                    raise e
                    except Exception as e:
                        if conflicts:
                            deletion_progress.close()
                        QMessageBox.critical(
                            self.gui,
                            "Error",
                            f"Failed to delete existing experiment '{exp_name}': {e}\n\nPlease close any applications that might be using these files and try again.",
                        )
                        logging.error(f"Failed to delete existing experiment '{exp_name}': {e}")
                        return

                if conflicts:
                    deletion_progress.setValue(len(conflicts))
                    deletion_progress.close()

        # Validate all selected experiments
        try:
            for exp_path in selected_experiments:
                for dataset_dir in os.listdir(exp_path):
                    dataset_path = os.path.join(exp_path, dataset_dir)
                    if os.path.isdir(dataset_path):
                        validated_path, metadata = self.validate_dataset_name(dataset_path)
                        # Update the dataset_path to the validated path (in case it was renamed)
                        dataset_path = validated_path
                        csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
                        if not csv_files:
                            raise FileNotFoundError(
                                f"Dataset directory '{os.path.basename(dataset_path)}' in experiment '{os.path.basename(exp_path)}' does not contain any .csv files"
                            )
        except Exception as e:
            QMessageBox.critical(
                self.gui,
                "Error",
                f"An error occurred while validating experiments: {e}.\n\nImportation was canceled.",
            )
            logging.error(f"An error occurred while validating experiments: {e}. Importation was canceled.")
            return

        # Start multi-experiment import
        progress_dialog = QProgressDialog("Processing...", "Cancel", 0, 100, self.gui)
        progress_dialog.setWindowTitle("Importing Multiple Experiments")
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.setAutoClose(False)
        progress_dialog.setAutoReset(False)
        progress_dialog.show()

        max_workers = max(1, multiprocessing.cpu_count() - 1)
        self.multi_thread = MultiExptImportingThread(selected_experiments, self.gui.output_path, max_workers=max_workers)
        self.multi_thread.progress.connect(progress_dialog.setValue)
        self.multi_thread.status_update.connect(lambda msg: progress_dialog.setLabelText(msg))

        self.multi_thread.finished.connect(progress_dialog.close)
        self.multi_thread.finished.connect(self._on_import_finished)
        self.multi_thread.finished.connect(lambda: self.gui.data_selection_widget.experiment_combo.setCurrentIndex(0))
        self.multi_thread.finished.connect(lambda count: self._show_import_summary(count, len(selected_experiments)))
        self.multi_thread.finished.connect(
            lambda count: self.gui.status_bar.showMessage(f"{count} experiments imported successfully.", 5000)
        )
        self.multi_thread.finished.connect(lambda count: logging.info(f"{count} experiments imported successfully."))

        self.multi_thread.error.connect(lambda e: QMessageBox.critical(self.gui, "Error", f"An error occurred: {e}"))
        self.multi_thread.error.connect(
            lambda e: logging.error(f"An error occurred while importing multiple experiments: {e}")
        )

        self.multi_thread.canceled.connect(progress_dialog.close)
        self.multi_thread.canceled.connect(lambda: self.gui.status_bar.showMessage("Multi-experiment import canceled.", 5000))
        self.multi_thread.canceled.connect(lambda: logging.info("Multi-experiment import canceled."))
        self.multi_thread.canceled.connect(self._on_import_finished)

        self.multi_thread.start()
        progress_dialog.canceled.connect(self.multi_thread.cancel)

    def _show_import_summary(self, successful_count: int, total_count: int):
        """Show a summary dialog after multi-experiment import."""
        if successful_count == total_count:
            QMessageBox.information(
                self.gui,
                "Import Complete",
                f"Successfully imported all {successful_count} experiments!",
            )
        elif successful_count > 0:
            QMessageBox.warning(
                self.gui,
                "Import Partially Complete",
                f"Successfully imported {successful_count} out of {total_count} experiments.\n\n"
                f"Check the log files for details about any failed imports.",
            )
        else:
            QMessageBox.critical(
                self.gui,
                "Import Failed",
                f"Failed to import any of the {total_count} experiments.\n\n" f"Check the log files for error details.",
            )

    # ------------------------------------------------------------------
    def rename_experiment(self):
        """Rename the currently selected experiment using the robust rename_experiment_by_id method."""
        logging.info("Renaming experiment.")
        if not self.gui.current_experiment:
            QMessageBox.warning(self.gui, "Warning", "Please select an experiment first.")
            return

        old_name = self.gui.current_experiment.id
        new_name, ok = QInputDialog.getText(
            self.gui,
            "Rename Experiment",
            "Enter new experiment name:",
            text=old_name,
        )

        if ok and new_name:
            try:
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

                # Use the robust rename_experiment_by_id method
                self.rename_experiment_by_id(old_name, new_name)

                self.gui.status_bar.showMessage("Experiment renamed successfully.", 5000)
                logging.info(f"Experiment renamed from '{old_name}' to '{new_name}' successfully.")

            except Exception as e:
                # Handle all exceptions from rename_experiment_by_id
                QMessageBox.critical(self.gui, "Error", str(e))
                logging.error(f"Failed to rename experiment: {e}")
            finally:
                QApplication.restoreOverrideCursor()

    # ------------------------------------------------------------------
    def delete_current_experiment(self):
        logging.info("Deleting experiment.")
        if self.gui.current_experiment:
            delete = QMessageBox.warning(
                self.gui,
                "Delete Experiment",
                f"Are you sure you want to delete the experiment '{self.gui.current_experiment.id}'?\n\nWARNING: This action cannot be undone.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if delete == QMessageBox.StandardButton.Yes:
                current_expt_id = self.gui.current_experiment.id
                current_expt_path = os.path.join(self.gui.output_path, current_expt_id)

                # Close the experiment and all associated data
                self.gui.current_experiment.close()
                self.gui.current_experiment = None
                self.gui.current_dataset = None
                self.gui.current_session = None

                # Use retry mechanism for robust deletion
                try:
                    import gc
                    import time

                    gc.collect()

                    max_retries = 3
                    for retry in range(max_retries):
                        try:
                            shutil.rmtree(current_expt_path)
                            logging.info(f"Deleted experiment folder: {current_expt_path}.")
                            break
                        except (OSError, PermissionError) as e:
                            if retry < max_retries - 1:
                                logging.warning(
                                    f"Failed to delete '{current_expt_id}' on attempt {retry + 1}: {e}. Retrying..."
                                )
                                time.sleep(0.5)
                            else:
                                raise e
                except Exception as e:
                    QMessageBox.critical(
                        self.gui,
                        "Error",
                        f"Failed to delete experiment '{current_expt_id}': {e}\n\nPlease close any applications that might be using these files and try again.",
                    )
                    logging.error(f"Failed to delete experiment '{current_expt_id}': {e}")
                    return

                # After deleting experiment, refresh the list and reset selections
                if hasattr(self.gui, "data_selection_widget"):
                    self.gui.data_selection_widget.refresh()
                self.gui.status_bar.showMessage("Experiment deleted successfully.", 5000)

    # ------------------------------------------------------------------
    def reload_current_session(self):
        logging.info("Reloading current session.")
        if self.gui.current_dataset and self.gui.current_session:
            if self.gui.current_session.repo is not None:
                if self.gui.current_session.repo.session_js.exists():
                    try:
                        self.gui.current_session.repo.session_js.unlink()
                    except (OSError, PermissionError) as e:
                        logging.warning(f"Failed to unlink session JS file: {e}")
                        QMessageBox.warning(
                            self.gui,
                            "Warning",
                            f"Could not remove session cache file. Reload may not work as expected: {e}",
                        )
                new_sess = self.gui.current_session.repo.load(config=self.gui.config_repo.read_config())
                idx = self.gui.current_dataset.get_all_sessions(include_excluded=True).index(self.gui.current_session)
                self.gui.current_dataset.get_all_sessions(include_excluded=True)[idx] = new_sess
                self.gui.set_current_session(new_sess)
            self.gui.plot_widget.on_data_selection_changed()
            self.gui.status_bar.showMessage("Session reloaded successfully.", 5000)
            logging.info("Session reloaded successfully.")
        else:
            QMessageBox.warning(self.gui, "Warning", "Please select a session first.")

    def reload_current_dataset(self):
        logging.info("Reloading current dataset.")
        if self.gui.current_dataset:
            if self.gui.current_dataset.repo is not None:
                if self.gui.current_dataset.repo.dataset_js.exists():
                    try:
                        self.gui.current_dataset.repo.dataset_js.unlink()
                    except (OSError, PermissionError) as e:
                        logging.warning(f"Failed to unlink dataset JS file: {e}")
                        QMessageBox.warning(
                            self.gui,
                            "Warning",
                            f"Could not remove dataset cache file. Reload may not work as expected: {e}",
                        )
                else:
                    logging.warning(
                        f"Dataset JS file does not exist: {self.gui.current_dataset.repo.dataset_js}. Cannot unlink."
                    )

                for sess in self.gui.current_dataset.get_all_sessions(include_excluded=True):
                    if sess.repo and sess.repo.session_js.exists():
                        try:
                            sess.repo.session_js.unlink()
                        except (OSError, PermissionError) as e:
                            logging.warning(f"Failed to unlink session JS file for {sess.id}: {e}")
                    else:
                        logging.warning(
                            f"Session JS file does not exist: {sess.repo.session_js if sess.repo else 'No repo'}. Cannot unlink."
                        )

                new_ds = self.gui.current_dataset.repo.load(config=self.gui.config_repo.read_config())
                if self.gui.current_dataset.parent_experiment is not None:
                    # Update the dataset in the parent experiment's list if it exists.
                    logging.info(f"Reloading dataset in parent experiment: {self.gui.current_dataset.parent_experiment.id}.")
                    idx = self.gui.current_dataset.parent_experiment._all_datasets.index(self.gui.current_dataset)
                    self.gui.current_dataset.parent_experiment._all_datasets[idx] = new_ds
                self.gui.set_current_dataset(new_ds)

            self.gui.data_selection_widget.update(levels=("session",))
            self.gui.plot_widget.on_data_selection_changed()
            self.gui.status_bar.showMessage("Dataset reloaded successfully.", 5000)
            logging.info("Dataset reloaded successfully.")
        else:
            QMessageBox.warning(self.gui, "Warning", "Please select a dataset first.")

    def reload_current_experiment(self):
        if self.gui.current_experiment:
            logging.info(f"Reloading current experiment: {self.gui.current_experiment.id}.")
            if self.gui.current_experiment.repo is not None:
                if self.gui.current_experiment.repo.expt_js.exists():
                    try:
                        self.gui.current_experiment.repo.expt_js.unlink()
                    except (OSError, PermissionError) as e:
                        logging.warning(f"Failed to unlink experiment JS file: {e}")
                        QMessageBox.warning(
                            self.gui,
                            "Warning",
                            f"Could not remove experiment cache file. Reload may not work as expected: {e}",
                        )
                else:
                    logging.warning(
                        f"Experiment JS file does not exist: {self.gui.current_experiment.repo.expt_js}. Cannot unlink."
                    )

                for ds in self.gui.current_experiment._all_datasets:
                    if ds.repo and ds.repo.dataset_js.exists():
                        try:
                            ds.repo.dataset_js.unlink()
                        except (OSError, PermissionError) as e:
                            logging.warning(f"Failed to unlink dataset JS file for {ds.id}: {e}")
                    else:
                        logging.warning(
                            f"Dataset JS file does not exist: {ds.repo.dataset_js if ds.repo else 'No repo'}. Cannot unlink."
                        )

                    for sess in ds.get_all_sessions(include_excluded=True):
                        if sess.repo and sess.repo.session_js.exists():
                            try:
                                sess.repo.session_js.unlink()
                            except (OSError, PermissionError) as e:
                                logging.warning(f"Failed to unlink session JS file for {sess.id}: {e}")
                        else:
                            logging.warning(
                                f"Session JS file does not exist: {sess.repo.session_js if sess.repo else 'No repo'}. Cannot unlink."
                            )

                # Reload the experiment from the repository.
                logging.info(f"Reloading experiment from repository: {self.gui.current_experiment.repo.folder.name}.")
                new_expt = self.gui.current_experiment.repo.load(config=self.gui.config_repo.read_config())
                self.gui.set_current_experiment(new_expt)
            else:
                self.gui.set_current_experiment(None)
                logging.warning("No repository found for the current experiment. Cannot reload.")
            self.gui.set_current_dataset(None)
            self.gui.set_current_session(None)

            # After reloading experiment, refresh the list and restore the experiment selection
            self.unpack_existing_experiments()
            if hasattr(self.gui, "data_selection_widget"):
                self.gui.data_selection_widget.update()  # Will automatically restore selection based on current_experiment

            if self.gui.current_experiment:
                self.gui.current_experiment.reset_all_caches()
            self.gui.plot_widget.on_data_selection_changed()

            logging.debug("Experiment reloaded successfully.")
            self.gui.status_bar.showMessage("Experiment reloaded successfully.", 5000)

    def _on_import_finished(self):
        """Handle completion of import operations by refreshing the experiments list."""
        if hasattr(self.gui, "data_selection_widget"):
            self.gui.data_selection_widget.refresh()
        # Refresh index after import to speed up first load
        try:
            if hasattr(self.gui, "current_experiment") and self.gui.current_experiment is not None:
                exp_path = self.gui.current_experiment.repo.folder if self.gui.current_experiment.repo else None
                if exp_path:
                    from monstim_signals.io.experiment_index import ensure_fresh_index

                    ensure_fresh_index(self.gui.current_experiment.id, exp_path)
        except Exception:
            logging.debug("Non-fatal: index refresh after import failed.", exc_info=True)

    # ------------------------------------------------------------------
    def show_preferences_window(self):
        logging.debug("Showing preferences window.")
        from monstim_gui.dialogs.preferences import PreferencesDialog

        window = PreferencesDialog(get_config_path(), parent=self.gui)
        if window.exec() == QDialog.DialogCode.Accepted:
            # After closing preferences, refresh the profile selector in the main window
            if hasattr(self.gui, "refresh_profile_selector"):
                self.gui.refresh_profile_selector()
            self.gui.update_domain_configs()
            self.gui.status_bar.showMessage("Preferences applied successfully.", 5000)
            logging.debug("Preferences applied successfully.")
        else:
            if hasattr(self.gui, "refresh_profile_selector"):
                self.gui.refresh_profile_selector()
            self.gui.update_domain_configs()
            self.gui.status_bar.showMessage("No changes made to preferences.", 5000)
            logging.debug("No changes made to preferences.")

    # ------------------------------------------------------------------
    def save_experiment(self):
        if self.gui.current_experiment:
            self.gui.current_experiment.save()
            self.gui.status_bar.showMessage("Experiment saved successfully.", 5000)
            logging.debug("Experiment saved successfully.")
            self.gui.has_unsaved_changes = False
            return True
        return False

    def load_experiment(self, index):
        if index < 0 or index >= len(self.gui.expts_dict_keys):
            logging.warning(f"Invalid experiment index: {index}. Available experiments: {len(self.gui.expts_dict_keys)}")
            return

        # Cancel any existing loading operation
        if hasattr(self, "loading_thread") and self.loading_thread.isRunning():
            logging.info("Cancelling previous experiment loading operation")
            # Use safe cancellation instead of terminate()
            self.loading_thread.request_cancel()
            # Wait with timeout (max 5 seconds for graceful shutdown)
            if not self.loading_thread.wait(5000):
                logging.warning("Loading thread did not stop gracefully, forcing termination")
                self.loading_thread.terminate()
                self.loading_thread.wait()
            self.loading_thread.deleteLater()
            if hasattr(self, "current_progress_dialog"):
                self.current_progress_dialog.close()
                del self.current_progress_dialog

        # Reset loading completion flag
        self.loading_completed_successfully = False

        # Get the actual experiment ID from the combo box UserRole data
        # The combo box index needs to be adjusted for the placeholder at index 0
        combo_index = index + 1  # Add 1 to account for placeholder
        stored_experiment_id = self.gui.data_selection_widget.experiment_combo.itemData(combo_index, Qt.ItemDataRole.UserRole)

        if stored_experiment_id:
            experiment_name = stored_experiment_id
        else:
            # Fallback to the old method if UserRole data is not available
            experiment_name = self.gui.expts_dict_keys[index]
        exp_path = os.path.join(self.gui.output_path, experiment_name)
        logging.info(f"Loading experiment: '{experiment_name}'.")

        # Close any open files/handles from the currently loaded experiment before switching
        try:
            if self.gui.current_experiment and getattr(self.gui.current_experiment, "id", None) != experiment_name:
                logging.debug(
                    f"Closing currently loaded experiment '{self.gui.current_experiment.id}' before switching to '{experiment_name}'."
                )
                # Notify user via status bar
                try:
                    self.gui.status_bar.showMessage(
                        f"Closing current experiment '{self.gui.current_experiment.id}'...",
                        3000,
                    )
                except Exception:
                    pass
                try:
                    self.gui.current_experiment.close()
                    logging.debug("Previous experiment closed successfully.")
                except Exception as e:
                    logging.exception(f"Error closing previous experiment: {e}")
        except Exception:
            # Never block switching due to cleanup errors
            pass

        # Proactively clear dataset and session to avoid UI operating on stale selections
        # while the experiment is loading (especially for long loads).
        self.gui.set_current_dataset(None)
        self.gui.set_current_session(None)
        self.gui.set_current_experiment(None)

        # Create and show progress dialog
        progress_dialog = QProgressDialog("Loading experiment...", "Cancel", 0, 100, self.gui)
        progress_dialog.setWindowTitle(f"Loading: {experiment_name}")
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.setAutoClose(False)
        progress_dialog.setAutoReset(False)
        progress_dialog.setMinimumDuration(0)  # Show immediately
        progress_dialog.setMinimumWidth(450)  # Make it wider for better text visibility
        progress_dialog.resize(450, 120)

        # Show dialog - don't set fixed size to allow dynamic resizing
        progress_dialog.show()
        QApplication.processEvents()

        # Simple initial message - let the loader provide specific time estimates
        initial_text = f"Initializing {experiment_name}..."
        progress_dialog.setLabelText(initial_text)

        # Force immediate display
        QApplication.processEvents()

        # Create and start loading thread
        config = self.gui.config_repo.read_config()
        self.loading_thread = ExperimentLoadingThread(exp_path, config)

        # Create a lambda to handle dynamic resizing when status updates
        def update_status_with_resize(text):
            progress_dialog.setLabelText(text)
            # Force dialog to recalculate size based on content
            progress_dialog.adjustSize()
            # Ensure width stays consistent
            current_size = progress_dialog.size()
            progress_dialog.resize(450, current_size.height())
            QApplication.processEvents()

        # Connect signals
        self.loading_thread.progress.connect(progress_dialog.setValue)
        self.loading_thread.status_update.connect(update_status_with_resize)
        self.loading_thread.finished.connect(self._on_experiment_loaded)
        self.loading_thread.error.connect(self._on_experiment_load_error)
        self.loading_thread.datasets_skipped.connect(self._on_datasets_skipped)
        self.loading_thread.finished.connect(progress_dialog.close)
        self.loading_thread.error.connect(progress_dialog.close)

        # Handle cancellation
        progress_dialog.canceled.connect(self._on_experiment_load_canceled)

        # Store dialog and state for cleanup
        self.current_progress_dialog = progress_dialog
        self.loading_completed_successfully = False

        # Start loading
        self.loading_thread.start()

        # Reflect loading state in dependent combos immediately
        try:
            self.gui.data_selection_widget.update(levels=("dataset", "session"), preserve_selection=False)
            self.gui.data_selection_widget.dataset_combo.setEnabled(False)
            self.gui.data_selection_widget.session_combo.setEnabled(False)
            # Notify downstream UI to clear any views bound to the prior selection
            self.gui.plot_widget.on_data_selection_changed()
        except Exception:
            pass

    def _on_experiment_loaded(self, experiment: "Experiment"):
        """Handle successful experiment loading."""
        try:
            # Mark loading as completed successfully
            self.loading_completed_successfully = True

            # Clear current dataset and session when loading a new experiment
            self.gui.set_current_dataset(None)
            self.gui.set_current_session(None)

            self.gui.set_current_experiment(experiment)

            # Track this experiment as recently used
            app_state.save_recent_experiment(experiment.id)

            # Save session state for restoration
            profile_name = (
                self.gui.profile_selector_combo.currentText() if hasattr(self.gui, "profile_selector_combo") else None
            )
            app_state.save_current_session_state(experiment_id=experiment.id, profile_name=profile_name)

            self.gui.data_selection_widget.update(levels=("dataset", "session"))
            # Enable/disable dataset combo based on whether experiment has datasets
            self.gui.data_selection_widget.dataset_combo.setEnabled(len(experiment.datasets) > 0)

            # Check if we're in the middle of session restoration
            if app_state._is_restoring_session:
                # Complete the restoration by loading dataset/session
                logging.debug("Completing session restoration after experiment load")
                app_state.complete_session_restoration(self.gui)
            else:
                # Normal load: automatically load the first dataset and session if available
                if experiment.datasets:
                    logging.debug(f"Auto-loading first dataset from experiment '{experiment.id}'")
                    self.load_dataset(0, auto_load_first_session=True)

            self.gui.plot_widget.on_data_selection_changed()

            # Provide informative status message based on experiment content
            if experiment.datasets:
                status_msg = f"Experiment '{experiment.id}' loaded successfully ({len(experiment.datasets)} datasets)."
            else:
                status_msg = f"Empty experiment '{experiment.id}' loaded (no datasets found)."

            self.gui.status_bar.showMessage(status_msg, 5000)
            logging.info(status_msg)

            # Ensure index is fresh after load
            try:
                if hasattr(self.gui, "current_experiment") and self.gui.current_experiment is not None:
                    exp_path = self.gui.current_experiment.repo.folder if self.gui.current_experiment.repo else None
                    if exp_path:
                        from monstim_signals.io.experiment_index import ensure_fresh_index

                        ensure_fresh_index(self.gui.current_experiment.id, exp_path)
            except Exception:
                logging.debug("Non-fatal: index refresh after load failed.", exc_info=True)
        except Exception as e:
            logging.error(f"Error setting loaded experiment: {e}")
            QMessageBox.critical(self.gui, "Error", f"Error setting loaded experiment: {e}")
        finally:
            # Cleanup
            if hasattr(self, "loading_thread"):
                self.loading_thread.deleteLater()
                del self.loading_thread
            if hasattr(self, "current_progress_dialog"):
                del self.current_progress_dialog

    def _on_experiment_load_error(self, error_message):
        """Handle experiment loading error."""
        QMessageBox.critical(self.gui, "Error", error_message)
        logging.error(f"Experiment loading error: {error_message}")

        # Cleanup
        if hasattr(self, "loading_thread"):
            self.loading_thread.deleteLater()
            del self.loading_thread
        if hasattr(self, "current_progress_dialog"):
            del self.current_progress_dialog

    def _on_datasets_skipped(self, skipped_list):
        """Handle warning about skipped datasets during load."""
        from PySide6.QtWidgets import QMessageBox

        if not skipped_list:
            return

        # Build warning message
        num_skipped = len(skipped_list)
        message = f"{num_skipped} dataset(s) could not be loaded due to data validation errors:\n\n"

        # Show up to 5 datasets in the message
        for i, (dataset_name, error) in enumerate(skipped_list[:5]):
            message += f"• {dataset_name}\n  Error: {error}\n\n"

        if num_skipped > 5:
            message += f"... and {num_skipped - 5} more dataset(s).\n\n"

        message += "Please review these datasets and fix the data inconsistencies.\n"
        message += "Check the application log for complete details."

        # Show warning dialog
        msg_box = QMessageBox(self.gui)
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle("Dataset Load Warnings")
        msg_box.setText(message)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec()

    def _on_experiment_load_canceled(self):
        """Handle experiment loading cancellation."""
        # Check if loading actually completed successfully - if so, ignore this cancel signal
        if hasattr(self, "loading_completed_successfully") and self.loading_completed_successfully:
            logging.debug("Ignoring cancel signal - experiment loading completed successfully")
            if hasattr(self, "current_progress_dialog"):
                del self.current_progress_dialog
            return

        if hasattr(self, "loading_thread") and self.loading_thread.isRunning():
            logging.info("User canceled experiment loading")
            # Use safe cancellation instead of terminate()
            self.loading_thread.request_cancel()
            # Wait with timeout (max 5 seconds for graceful shutdown)
            if not self.loading_thread.wait(5000):
                logging.warning("Loading thread did not stop gracefully after cancel, forcing termination")
                self.loading_thread.terminate()
                self.loading_thread.wait()
            self.loading_thread.deleteLater()
            del self.loading_thread

        self.gui.status_bar.showMessage("Experiment loading canceled.", 3000)
        logging.info("Experiment loading canceled by user.")

        # Reset experiment combo to no selection
        self.gui.data_selection_widget.experiment_combo.blockSignals(True)
        self.gui.data_selection_widget.experiment_combo.setCurrentIndex(0)  # Select placeholder
        self.gui.data_selection_widget.experiment_combo.blockSignals(False)

        if hasattr(self, "current_progress_dialog"):
            del self.current_progress_dialog

    def load_dataset(self, index, auto_load_first_session=True):
        # Set busy cursor if QApplication is available
        if QApplication.instance() is not None:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        if not self.gui.current_experiment:
            logging.debug("No current experiment to load dataset from.")
            return

        if index < 0 or index >= len(self.gui.current_experiment.datasets):
            logging.warning(
                f"Invalid dataset index: {index}. Available datasets: {len(self.gui.current_experiment.datasets) if self.gui.current_experiment else 0}"
            )
            return

        logging.debug(f"Loading dataset [{index}] from experiment '{self.gui.current_experiment.id}'.")
        dataset = self.gui.current_experiment.datasets[index]

        # Clear any previously selected session when switching datasets to avoid stale references
        # and ensure auto-load of the first session (if requested) can occur.
        if self.gui.current_session is not None and self.gui.current_dataset is not dataset:
            try:
                # Close open file handles within the current session
                self.gui.current_session.close()
            except Exception:
                logging.debug("Non-fatal: error while closing previous session during dataset switch.")
            self.gui.set_current_session(None)

        # If truly switching to a different dataset, close the old dataset to release handles
        if self.gui.current_dataset is not None and self.gui.current_dataset is not dataset:
            try:
                self.gui.current_dataset.close()
            except Exception:
                logging.debug("Non-fatal: error while closing previous dataset during dataset switch.")

        # Now set the new dataset
        self.gui.set_current_dataset(dataset)

        # Save session state for restoration
        profile_name = self.gui.profile_selector_combo.currentText() if hasattr(self.gui, "profile_selector_combo") else None
        app_state.save_current_session_state(
            experiment_id=self.gui.current_experiment.id,
            dataset_id=dataset.id,
            profile_name=profile_name,
        )

        if self.gui.current_dataset is not None:
            self.gui.channel_names = self.gui.current_dataset.channel_names
        else:
            self.gui.channel_names = []
            logging.warning("No dataset selected. Channel names will not be updated.")

        # Update the session list for the newly selected dataset and enable the combo
        # Do not preserve previous session selection since it belonged to a different dataset
        self.gui.data_selection_widget.update(levels=("session",), preserve_selection=False)
        self.gui.data_selection_widget.session_combo.setEnabled(True)

        # Automatically load the first session if requested and conditions are met
        if auto_load_first_session and not self.gui.current_session and dataset.sessions:
            logging.debug(f"Auto-loading first session from dataset '{dataset.id}'")
            self.load_session(0)  # Load the first session
        else:
            self.gui.plot_widget.on_data_selection_changed()

        # Restore normal cursor
        if QApplication.instance() is not None:
            QApplication.restoreOverrideCursor()

    def load_session(self, index):
        # Set busy cursor if QApplication is available
        if QApplication.instance() is not None:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        if not self.gui.current_dataset:
            logging.debug("No current dataset to load session from.")
            return

        if index < 0 or index >= len(self.gui.current_dataset.sessions):
            logging.warning(
                f"Invalid session index: {index}. Available sessions: {len(self.gui.current_dataset.sessions) if self.gui.current_dataset else 0}"
            )
            return

        logging.debug(f"Loading session [{index}] from dataset '{self.gui.current_dataset.id}'.")
        session = self.gui.current_dataset.sessions[index]
        # On-demand materialization of recordings for lazy-loaded sessions
        try:
            if session and session.repo and not session.recordings:
                session = session.repo.materialize_recordings(
                    session, config=self.gui.config_repo.read_config(), allow_write=False
                )
        except Exception:
            logging.debug("Non-fatal: error materializing recordings on session selection.", exc_info=True)
        self.gui.set_current_session(session)

        # Save complete session state for restoration
        profile_name = self.gui.profile_selector_combo.currentText() if hasattr(self.gui, "profile_selector_combo") else None
        app_state.save_current_session_state(
            experiment_id=(self.gui.current_experiment.id if self.gui.current_experiment else None),
            dataset_id=self.gui.current_dataset.id,
            session_id=session.id,
            profile_name=profile_name,
        )

        if hasattr(self.gui.plot_widget.current_option_widget, "recording_cycler"):
            self.gui.plot_widget.current_option_widget.recording_cycler.reset_max_recordings()  # type: ignore
        self.gui.plot_widget.on_data_selection_changed()

        # Restore normal cursor
        if QApplication.instance() is not None:
            QApplication.restoreOverrideCursor()

    # ------------------------------------------------------------------
    @staticmethod
    def validate_dataset_name(dataset_path):
        """
        Validate and optionally correct dataset names, with flexible handling for non-standard formats.
        Returns the validated dataset path and extracted metadata.
        """
        original_dataset_name = os.path.basename(dataset_path)
        dataset_basepath = os.path.dirname(dataset_path)

        def extract_metadata_from_name(dataset_name):
            """Extract date, animal_id, and condition from dataset name if possible."""
            metadata = {"date": None, "animal_id": None, "condition": None}

            # Try standard format: "YYMMDD AnimalID Condition" or "YYYYMMDD AnimalID Condition"
            # Note: requires exactly one space between components
            pattern = r"^(\d{6,8})\s([a-zA-Z0-9.]+)\s(.+)$"
            match = re.match(pattern, dataset_name)
            if match:
                metadata["date"] = match.group(1)
                metadata["animal_id"] = match.group(2)
                metadata["condition"] = match.group(3)

            return metadata

        def get_user_choice_for_invalid_name(dataset_name, metadata):
            """Show dialog allowing user to rename, keep as-is, or cancel."""
            if "PySide6" in sys.modules:
                from PySide6.QtWidgets import (
                    QApplication,
                    QDialog,
                    QHBoxLayout,
                    QLabel,
                    QLineEdit,
                    QMessageBox,
                    QPushButton,
                    QVBoxLayout,
                )

                app = QApplication.instance()
                if app is None:
                    app = QApplication([])

                dialog = QDialog()
                dialog.setWindowTitle("Dataset Name Format")
                dialog.setModal(True)
                dialog.resize(600, 350)
                layout = QVBoxLayout()

                # Main explanation
                main_text = QLabel(
                    f'The dataset name "{dataset_name}" does not follow the recommended format:\n'
                    f'"[YYMMDD or YYYYMMDD] [Animal ID] [Experimental Condition]"\n\n'
                    f"Missing or invalid components:"
                )
                main_text.setWordWrap(True)
                layout.addWidget(main_text)

                # Show what's missing
                missing_items = []
                if not metadata["date"]:
                    missing_items.append("• Date (YYMMDD or YYYYMMDD)")
                if not metadata["animal_id"]:
                    missing_items.append("• Animal ID (e.g., XX000.0)")
                if not metadata["condition"]:
                    missing_items.append("• Experimental Condition")

                if missing_items:
                    missing_label = QLabel("\n".join(missing_items))
                    layout.addWidget(missing_label)

                # Explanation of options
                options_text = QLabel(
                    "\nYou can:\n"
                    "• Rename the dataset to follow the standard format\n"
                    "• Keep the current name (will be displayed as-is in the interface)\n"
                    "• Cancel the import"
                )
                options_text.setWordWrap(True)
                layout.addWidget(options_text)

                # Rename section
                layout.addWidget(QLabel("\nRename dataset (optional):"))
                line_edit = QLineEdit(dataset_name)
                line_edit.setPlaceholderText("Example: 240815 BEM3 test condition")
                line_edit.setToolTip(
                    "Format: [YYMMDD or YYYYMMDD] [Animal ID] [Experimental Condition]\nUse single spaces between components"
                )
                layout.addWidget(line_edit)

                # Buttons
                button_layout = QHBoxLayout()
                rename_button = QPushButton("Rename and Continue")
                keep_button = QPushButton("Keep Current Name")
                cancel_button = QPushButton("Cancel Import")

                button_layout.addWidget(rename_button)
                button_layout.addWidget(keep_button)
                button_layout.addWidget(cancel_button)
                layout.addLayout(button_layout)

                dialog.setLayout(layout)

                # Store the result
                result = {"action": None, "new_name": None}

                def on_rename():
                    result["action"] = "rename"
                    result["new_name"] = line_edit.text().strip()
                    if not result["new_name"]:
                        QMessageBox.warning(dialog, "Warning", "Please enter a valid dataset name.")
                        return

                    # Validate the new name follows the correct format
                    new_metadata = extract_metadata_from_name(result["new_name"])
                    is_valid_format = all(new_metadata.values())

                    if not is_valid_format:
                        # Show which components are still missing
                        missing_items = []
                        if not new_metadata["date"]:
                            missing_items.append("• Date (YYMMDD or YYYYMMDD)")
                        if not new_metadata["animal_id"]:
                            missing_items.append("• Animal ID (e.g., XX000.0)")
                        if not new_metadata["condition"]:
                            missing_items.append("• Experimental Condition")

                        missing_text = "\n".join(missing_items)
                        QMessageBox.warning(
                            dialog,
                            "Invalid Format",
                            f'The name "{result["new_name"]}" still does not follow the required format.\n\n'
                            f"Missing components:\n{missing_text}\n\n"
                            f'Required format: "[YYMMDD or YYYYMMDD] [Animal ID] [Experimental Condition]"\n'
                            f'Example: "240815 BEM3 test condition"\n\n'
                            f"Please use single spaces between components.",
                        )
                        return

                    dialog.accept()

                def on_keep():
                    result["action"] = "keep"
                    dialog.accept()

                def on_cancel():
                    result["action"] = "cancel"
                    dialog.reject()

                rename_button.clicked.connect(on_rename)
                keep_button.clicked.connect(on_keep)
                cancel_button.clicked.connect(on_cancel)

                dialog_result = dialog.exec()
                if dialog_result == QDialog.DialogCode.Rejected or result["action"] == "cancel":
                    raise ValueError("User canceled dataset import.")

                return result
            else:
                # Command line interface
                print(f'The dataset name "{dataset_name}" does not follow the recommended format.')
                print("Missing or invalid components:")
                if not metadata["date"]:
                    print("\t- Date (YYMMDD or YYYYMMDD)")
                if not metadata["animal_id"]:
                    print("\t- Animal ID (e.g., XX000.0)")
                if not metadata["condition"]:
                    print("\t- Experimental Condition")

                print("\nOptions:")
                print("1. Rename the dataset")
                print("2. Keep current name (will be displayed as-is)")
                print("3. Cancel import")

                choice = input("Enter choice (1/2/3): ").strip()
                if choice == "1":
                    while True:
                        new_name = input(f"Enter new name (current: {dataset_name}): ").strip()
                        if not new_name:
                            raise ValueError("Invalid dataset name provided.")

                        # Validate the new name follows the correct format
                        new_metadata = extract_metadata_from_name(new_name)
                        is_valid_format = all(new_metadata.values())

                        if is_valid_format:
                            return {"action": "rename", "new_name": new_name}
                        else:
                            print(f"\nThe name '{new_name}' still does not follow the required format.")
                            print("Missing components:")
                            if not new_metadata["date"]:
                                print("\t- Date (YYMMDD or YYYYMMDD)")
                            if not new_metadata["animal_id"]:
                                print("\t- Animal ID (e.g., XX000.0)")
                            if not new_metadata["condition"]:
                                print("\t- Experimental Condition")
                            print("\nPlease try again or press Ctrl+C to cancel.")
                elif choice == "2":
                    return {"action": "keep"}
                else:
                    raise ValueError("User canceled dataset import.")

        # Extract metadata from the original name
        metadata = extract_metadata_from_name(original_dataset_name)

        # Check if the name follows the standard format
        is_valid_format = all(metadata.values())

        if is_valid_format:
            # Name is already in correct format
            logging.debug(f"Dataset name '{original_dataset_name}' follows standard format.")
            return dataset_path, metadata
        else:
            # Name doesn't follow standard format - give user options
            logging.info(f"Dataset name '{original_dataset_name}' does not follow standard format.")
            user_choice = get_user_choice_for_invalid_name(original_dataset_name, metadata)

            if user_choice["action"] == "rename":
                new_name = user_choice["new_name"]
                new_metadata = extract_metadata_from_name(new_name)
                validated_dataset_path = os.path.join(dataset_basepath, new_name)

                # Rename the folder using repository API if possible
                try:
                    # Try to map the dataset_path to a DatasetRepository and use its rename method
                    from monstim_signals.io.repositories import DatasetRepository

                    ds_repo = DatasetRepository(Path(dataset_path))
                    ds_repo.rename(Path(validated_dataset_path))
                    logging.info(
                        f'Dataset folder renamed from "{dataset_path}" to "{validated_dataset_path}" via DatasetRepository.rename().'
                    )
                except Exception:
                    # Fallback to os.rename if repository rename failed or repo not available
                    os.rename(dataset_path, validated_dataset_path)
                    logging.info(
                        f'Dataset folder renamed from "{dataset_path}" to "{validated_dataset_path}" using os.rename().'
                    )

                # TODO: Smart rename UX
                # - Consider adding a "Suggest Name" button that attempts to
                #   auto-format common non-standard names (e.g., swapping delimiters,
                #   filling missing year digits) and presents suggested renames
                #   in batch for multi-dataset imports.
                # - Add an option to apply the same rename rule across multiple
                #   datasets (preview & confirm) during MultiExpt import.
                return validated_dataset_path, new_metadata

            elif user_choice["action"] == "keep":
                # Keep original name, use extracted metadata (may have None values)
                logging.info(f"Keeping original dataset name '{original_dataset_name}'. Display name will use folder name.")
                return dataset_path, metadata

            else:
                raise ValueError("Unexpected user choice result.")

    # ------------------------------------------------------------------
    def open_log_directory(self):
        log_dir = get_log_dir()
        QDesktopServices.openUrl(QUrl.fromLocalFile(log_dir))
        self.gui.status_bar.showMessage(f"Opened log folder: {log_dir}", 5000)
        logging.info(f"Opened log folder: {log_dir}")

    def save_error_report(self):
        log_dir = get_log_dir()

        # Get the last used export directory, fallback to user home
        last_export_path = app_state.get_last_export_path()
        if not last_export_path or not os.path.isdir(last_export_path):
            last_export_path = os.path.expanduser("~")

        default_name = os.path.join(last_export_path, "monstim_logs.zip")
        file_path, _ = QFileDialog.getSaveFileName(self.gui, "Save Error Report", default_name, "Zip Files (*.zip)")
        if not file_path:
            return

        # Save the directory for next time
        app_state.save_last_export_path(os.path.dirname(file_path))

        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            with ZipFile(file_path, "w", ZIP_DEFLATED) as zf:
                for f in glob.glob(os.path.join(log_dir, "*")):
                    if os.path.isfile(f):
                        zf.write(f, arcname=os.path.basename(f))
            self.gui.status_bar.showMessage("Error report saved.", 5000)
            QMessageBox.information(self.gui, "Saved", f"Error report saved to:\n{file_path}")
            logging.info(f"Saved error report to {file_path}")
        except Exception as e:
            logging.exception("Failed to save error report")
            QMessageBox.critical(self.gui, "Error", f"Could not save error report:\n{e}")
        finally:
            QApplication.restoreOverrideCursor()

    # ------------------------------------------------------------------
    def edit_dataset_metadata(self):
        """Show the dataset metadata editor dialog."""
        if not self.gui.current_dataset:
            QMessageBox.warning(self.gui, "No Dataset Selected", "Please select a dataset first to edit its metadata.")
            return

        try:
            from monstim_gui.dialogs.dataset_metadata_editor import DatasetMetadataEditor

            # Show editor dialog with the current dataset
            dialog = DatasetMetadataEditor(self.gui.current_dataset, parent=self.gui)

            if dialog.exec() == QDialog.DialogCode.Accepted:
                # Refresh the dataset and session combos to show updated display names
                # Will automatically restore selection based on current_dataset reference
                self.gui.data_selection_widget.update(levels=("dataset", "session"))

                self.gui.status_bar.showMessage("Dataset metadata updated successfully.", 5000)
                logging.info(f"Dataset metadata updated for '{self.gui.current_dataset.id}'")

        except Exception as e:
            logging.error(f"Error opening dataset metadata editor: {e}")
            QMessageBox.critical(self.gui, "Error", f"Failed to open dataset metadata editor:\n{e}")

    # ------------------------------------------------------------------
    def close_all_data(self):
        """Close all currently open data (experiment, dataset, session)."""
        import gc

        logging.info("Closing all data.")

        # Close current data hierarchy
        if self.gui.current_session:
            self.gui.current_session.close()
            self.gui.set_current_session(None)
        if self.gui.current_dataset:
            self.gui.current_dataset.close()
            self.gui.set_current_dataset(None)
        if self.gui.current_experiment:
            self.gui.current_experiment.close()
            self.gui.set_current_experiment(None)

        # Force garbage collection to release any lingering file handles
        gc.collect()

        # Small delay to allow file handles to be released on Windows
        import time

        time.sleep(0.1)

        logging.info("All data closed successfully.")

    # ------------------------------------------------------------------
    # Data Curation I/O Operations
    # ------------------------------------------------------------------

    def create_experiment(self, exp_name: str):
        """Create a new empty experiment directory with annotation file."""
        import json
        from dataclasses import asdict
        from pathlib import Path

        from monstim_signals.core import ExperimentAnnot, get_output_path

        try:
            # Create experiment directory
            output_path = Path(get_output_path())
            exp_path = output_path / exp_name
            exp_path.mkdir(parents=True, exist_ok=True)

            # Create empty experiment annotation
            annot = ExperimentAnnot.create_empty()
            annot_file = exp_path / "experiment.annot.json"
            annot_file.write_text(json.dumps(asdict(annot), indent=2))

            # Add to GUI's experiment dictionary
            self.gui.expts_dict[exp_name] = str(exp_path)
            self.gui.expts_dict_keys = sorted(self.gui.expts_dict.keys())

            logging.info(f"Created empty experiment: {exp_name}")

        except Exception as e:
            logging.error(f"Failed to create experiment {exp_name}: {e}")
            raise Exception(f"Failed to create experiment '{exp_name}': {str(e)}")

    def delete_experiment_by_id(self, exp_id: str):
        """Delete an experiment by ID (used by data curation manager)."""
        import gc
        import time

        try:
            # Close all open data to prevent file handle conflicts
            logging.info(f"Closing all open data before deleting experiment '{exp_id}'")
            self.close_all_data()

            # Check if this is the current experiment
            if self.gui.current_experiment and self.gui.current_experiment.id == exp_id:
                # Close the current experiment (redundant now, but kept for clarity)
                self.gui.current_experiment.close()
                self.gui.current_experiment = None
                self.gui.current_dataset = None
                self.gui.current_session = None

            # Get experiment path and delete
            exp_path = os.path.join(self.gui.output_path, exp_id)
            if os.path.exists(exp_path):
                gc.collect()

                max_retries = 3
                for retry in range(max_retries):
                    try:
                        shutil.rmtree(exp_path)
                        logging.info(f"Deleted experiment folder: {exp_path}")
                        break
                    except (OSError, PermissionError) as e:
                        if retry < max_retries - 1:
                            logging.warning(f"Failed to delete '{exp_id}' on attempt {retry + 1}: {e}. Retrying...")
                            time.sleep(0.5)
                        else:
                            raise e

            # Remove from GUI's experiment dictionary
            if exp_id in self.gui.expts_dict:
                del self.gui.expts_dict[exp_id]
                self.gui.expts_dict_keys = sorted(self.gui.expts_dict.keys())

            logging.info(f"Deleted experiment: {exp_id}")

        except Exception as e:
            logging.error(f"Failed to delete experiment {exp_id}: {e}")
            raise Exception(f"Failed to delete experiment '{exp_id}': {str(e)}")

    def rename_experiment_by_id(self, old_name: str, new_name: str):
        """Rename an experiment by ID, regardless of what's currently selected."""
        import shutil
        from pathlib import Path

        try:
            # Validate new name
            if not new_name or any(c in r'<>:"/\\|?*' for c in new_name):
                raise ValueError("Experiment name contains invalid characters for a directory name.")

            if old_name == new_name:
                raise ValueError("The new experiment name is the same as the current one.")

            # Check if new name already exists
            if new_name in self.gui.expts_dict:
                raise FileExistsError(f"An experiment with the name '{new_name}' already exists.")

            # Close all open data to prevent file handle conflicts
            logging.info(f"Closing all open data before renaming experiment '{old_name}' to '{new_name}'")
            self.close_all_data()

            # Get old and new paths
            old_exp_path = Path(self.gui.expts_dict[old_name])
            new_exp_path = old_exp_path.parent / new_name

            # Check if the current experiment is being renamed
            current_exp_being_renamed = self.gui.current_experiment and self.gui.current_experiment.id == old_name

            # Close current experiment if it's the one being renamed (redundant now, but kept for clarity)
            if current_exp_being_renamed:
                self.gui.current_experiment.close()

            # Rename the directory
            shutil.move(str(old_exp_path), str(new_exp_path))

            # Update GUI experiment dictionary
            del self.gui.expts_dict[old_name]
            self.gui.expts_dict[new_name] = str(new_exp_path)
            self.gui.expts_dict_keys = sorted(self.gui.expts_dict.keys())

            # If we renamed the current experiment, update the current experiment reference
            if current_exp_being_renamed:
                self.gui.current_experiment.id = new_name
                if self.gui.current_experiment.repo is not None:
                    self.gui.current_experiment.repo.update_path(new_exp_path)
                for ds in self.gui.current_experiment._all_datasets:
                    if ds.repo is not None:
                        ds.repo.update_path(new_exp_path / ds.id)
                    for sess in ds.get_all_sessions(include_excluded=True):
                        if sess.repo is not None:
                            sess.repo.update_path(new_exp_path / ds.id / sess.id)

            # Refresh UI to reflect the changes
            self.unpack_existing_experiments()
            if hasattr(self.gui, "data_selection_widget"):
                self.gui.data_selection_widget.update(levels=("experiment",))

            logging.info(f"Renamed experiment '{old_name}' to '{new_name}'")

        except ValueError as ve:
            logging.error(f"Invalid experiment name: {ve}")
            raise Exception(f"Invalid experiment name: {str(ve)}")
        except FileExistsError as fe:
            logging.error(f"Experiment name conflict: {fe}")
            raise Exception(str(fe))
        except Exception as e:
            logging.error(f"Failed to rename experiment {old_name} to {new_name}: {e}")
            raise Exception(f"Failed to rename experiment: {str(e)}")

    def move_dataset(self, dataset_id: str, dataset_name: str, from_exp: str, to_exp: str):
        """Move a dataset from one experiment to another."""
        from pathlib import Path

        try:
            # Close all open data to prevent file handle conflicts
            logging.info(f"Closing all open data before moving dataset '{dataset_name}'")
            self.close_all_data()

            # Get source and destination paths
            from_exp_path = Path(self.gui.expts_dict[from_exp])
            to_exp_path = Path(self.gui.expts_dict[to_exp])

            # Find the dataset folder name - try exact match first
            dataset_folder_name = self._find_dataset_folder(from_exp_path, dataset_id, dataset_name)
            source_path = from_exp_path / dataset_folder_name
            dest_path = to_exp_path / dataset_folder_name

            if not source_path.exists():
                raise Exception(f"Source dataset folder not found: {source_path}")

            # Check for naming conflicts
            if dest_path.exists():
                raise Exception(f"Dataset '{dataset_folder_name}' already exists in experiment '{to_exp}'")

            # Move the dataset folder
            shutil.move(str(source_path), str(dest_path))

            logging.info(f"Moved dataset {dataset_name} from {from_exp} to {to_exp}")

        except Exception as e:
            logging.error(f"Failed to move dataset {dataset_name}: {e}")
            raise Exception(f"Failed to move dataset '{dataset_name}': {str(e)}")

    def copy_dataset(self, dataset_id: str, dataset_name: str, from_exp: str, to_exp: str, new_name: str = None):
        """Copy a dataset from one experiment to another, optionally with a new name."""
        from pathlib import Path

        try:
            # Close all open data to prevent file handle conflicts
            logging.info(f"Closing all open data before copying dataset '{dataset_name}'")
            self.close_all_data()

            # Get source and destination paths
            from_exp_path = Path(self.gui.expts_dict[from_exp])
            to_exp_path = Path(self.gui.expts_dict[to_exp])

            # Find the dataset folder name - try exact match first
            dataset_folder_name = self._find_dataset_folder(from_exp_path, dataset_id, dataset_name)
            source_path = from_exp_path / dataset_folder_name

            # Use new_name if provided, otherwise use original folder name
            if new_name:
                dest_folder_name = new_name
            else:
                dest_folder_name = dataset_folder_name

            dest_path = to_exp_path / dest_folder_name

            if not source_path.exists():
                raise Exception(f"Source dataset folder not found: {source_path}")

            # Handle naming conflicts for copies (append number if needed)
            if dest_path.exists():
                counter = 1
                base_name = dest_folder_name
                while dest_path.exists():
                    dest_path = to_exp_path / f"{base_name}_copy{counter}"
                    counter += 1

            # Copy the dataset folder
            shutil.copytree(str(source_path), str(dest_path))

            logging.info(f"Copied dataset {dataset_name} from {from_exp} to {to_exp} as {dest_path.name}")

        except Exception as e:
            logging.error(f"Failed to copy dataset {dataset_name}: {e}")
            raise Exception(f"Failed to copy dataset '{dataset_name}': {str(e)}")

    def delete_dataset(self, dataset_id: str, dataset_name: str, exp_id: str):
        """Delete a dataset from an experiment."""
        from pathlib import Path

        try:
            # Close all open data to prevent file handle conflicts
            logging.info(f"Closing all open data before deleting dataset '{dataset_name}'")
            self.close_all_data()

            # Get experiment path
            exp_path = Path(self.gui.expts_dict[exp_id])

            # Find the dataset folder - try exact match first
            dataset_folder_name = self._find_dataset_folder(exp_path, dataset_id, dataset_name)
            dataset_path = exp_path / dataset_folder_name

            if dataset_path.exists():
                # Delete the dataset folder
                shutil.rmtree(dataset_path)
                logging.info(f"Deleted dataset folder: {dataset_path}")
            else:
                logging.warning(f"Dataset folder not found for deletion: {dataset_path}")

            logging.info(f"Deleted dataset {dataset_name} from {exp_id}")

        except Exception as e:
            logging.error(f"Failed to delete dataset {dataset_name}: {e}")
            raise Exception(f"Failed to delete dataset '{dataset_name}': {str(e)}")

    def _find_dataset_folder(self, exp_path: Path, dataset_id: str, dataset_name: str) -> str:
        """Find the actual dataset folder name in the experiment directory."""
        # Try exact match first
        if (exp_path / dataset_id).exists():
            return dataset_id

        # Search through folders for a match
        for folder_name in os.listdir(exp_path):
            folder_path = exp_path / folder_name
            if folder_path.is_dir():
                # Check if this folder contains the dataset we're looking for
                if folder_name in dataset_name or dataset_name in folder_name:
                    return folder_name

        # Fallback: use the dataset_id directly
        return dataset_id
