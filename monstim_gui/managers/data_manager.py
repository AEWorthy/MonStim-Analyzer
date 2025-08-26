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

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
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
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                self.gui.expts_dict = {
                    name: os.path.join(self.gui.output_path, name)
                    for name in os.listdir(self.gui.output_path)
                    if os.path.isdir(os.path.join(self.gui.output_path, name))
                }
                self.gui.expts_dict_keys = sorted(self.gui.expts_dict.keys())
                logging.debug("Existing experiments unpacked successfully.")
            except Exception as e:
                QApplication.restoreOverrideCursor()
                QMessageBox.critical(
                    self.gui,
                    "Error",
                    f"An error occurred while unpacking existing experiments: {e}",
                )
                logging.error(f"An error occurred while unpacking existing experiments: {e}")
                logging.error(traceback.format_exc())
            finally:
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
            self.thread.finished.connect(self.refresh_existing_experiments)
            self.thread.finished.connect(lambda: self.gui.data_selection_widget.experiment_combo.setCurrentIndex(0))
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
            self.thread.canceled.connect(self.refresh_existing_experiments)

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
        from PyQt6.QtWidgets import QFileDialog

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
        from PyQt6.QtWidgets import (
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
        from PyQt6.QtWidgets import QHBoxLayout

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
        self.multi_thread.finished.connect(self.refresh_existing_experiments)
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
        self.multi_thread.canceled.connect(self.refresh_existing_experiments)

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
        logging.info("Renaming experiment.")
        if self.gui.current_experiment:
            new_name, ok = QInputDialog.getText(
                self.gui,
                "Rename Experiment",
                "Enter new experiment name:",
                text=self.gui.current_experiment.id,
            )

            if ok and new_name:
                try:
                    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                    self.gui.current_experiment.close()
                    old_expt_path = os.path.join(self.gui.output_path, self.gui.current_experiment.id)
                    new_expt_path = os.path.join(self.gui.output_path, new_name)

                    if not new_name or any(c in r'<>:"/\\|?*' for c in new_name):
                        raise ValueError("Experiment name contains invalid characters for a directory name.")
                    if old_expt_path == new_expt_path:
                        QMessageBox.warning(
                            self.gui,
                            "Warning",
                            "The new experiment name is the same as the current one. No changes made.",
                        )
                        logging.info("No changes made to experiment name as it is the same as the current one.")
                        return
                    if os.path.exists(new_expt_path):
                        raise FileExistsError(f"An experiment with the name '{new_name}' already exists.")

                    shutil.move(old_expt_path, new_expt_path)

                    self.gui.current_experiment.id = new_name
                    if self.gui.current_experiment.repo is not None:
                        self.gui.current_experiment.repo.update_path(Path(new_expt_path))
                    for ds in self.gui.current_experiment._all_datasets:
                        if ds.repo is not None:
                            ds.repo.update_path(Path(new_expt_path) / ds.id)
                        for sess in ds._all_sessions:
                            if sess.repo is not None:
                                sess.repo.update_path(Path(new_expt_path) / ds.id / sess.id)

                except ValueError as ve:
                    QMessageBox.critical(self.gui, "Error", f"Invalid experiment name: {ve}")
                    logging.error(f"Invalid experiment name: {ve}")
                    return
                except FileExistsError:
                    QMessageBox.critical(
                        self.gui,
                        "Error",
                        f"An experiment with the name '{new_name}' already exists. Please choose a different name.",
                    )
                    logging.warning(
                        f"An experiment with the name '{new_name}' already exists. Could not rename current experiment."
                    )
                    return
                finally:
                    QApplication.restoreOverrideCursor()

                self.refresh_existing_experiments(select_expt_id=new_name)
                self.gui.status_bar.showMessage("Experiment renamed successfully.", 5000)

    # ------------------------------------------------------------------
    def delete_experiment(self):
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

                self.refresh_existing_experiments()
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
                idx = self.gui.current_dataset._all_sessions.index(self.gui.current_session)
                self.gui.current_dataset._all_sessions[idx] = new_sess
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

                for sess in self.gui.current_dataset._all_sessions:
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

            self.gui.data_selection_widget.update_session_combo()
            self.gui.plot_widget.on_data_selection_changed()
            self.gui.status_bar.showMessage("Dataset reloaded successfully.", 5000)
            logging.info("Dataset reloaded successfully.")
        else:
            QMessageBox.warning(self.gui, "Warning", "Please select a dataset first.")

    def reload_current_experiment(self):
        # TODO: Fix pathing issues with reloading experiments.
        current_experiment_combo_index = self.gui.data_selection_widget.experiment_combo.currentIndex()
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

                    for sess in ds._all_sessions:
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

            self.refresh_existing_experiments()
            # Adjust for placeholder: if we had a valid experiment selected (index > 0), maintain it
            if current_experiment_combo_index > 0:
                self.gui.data_selection_widget.experiment_combo.setCurrentIndex(current_experiment_combo_index)
            else:
                # If we were on placeholder, stay on placeholder
                self.gui.data_selection_widget.experiment_combo.setCurrentIndex(0)

            if self.gui.current_experiment:
                self.gui.current_experiment.reset_all_caches()
            self.gui.plot_widget.on_data_selection_changed()

            logging.debug("Experiment reloaded successfully.")
            self.gui.status_bar.showMessage("Experiment reloaded successfully.", 5000)

    # ------------------------------------------------------------------
    def refresh_existing_experiments(self, select_expt_id: str | None = None) -> None:
        logging.debug("Refreshing existing experiments.")
        self.unpack_existing_experiments()
        self.gui.data_selection_widget.update_experiment_combo()
        self.gui.plot_widget.on_data_selection_changed()

        if select_expt_id is not None:
            # Find the experiment in the list and select it (accounting for placeholder at index 0)
            if select_expt_id in self.gui.expts_dict_keys:
                index = self.gui.expts_dict_keys.index(select_expt_id) + 1  # +1 for placeholder
            else:
                index = 0  # Select placeholder if not found
        else:
            # Check if we currently have an experiment loaded and try to maintain that selection
            if self.gui.current_experiment and self.gui.current_experiment.id in self.gui.expts_dict_keys:
                index = self.gui.expts_dict_keys.index(self.gui.current_experiment.id) + 1  # +1 for placeholder
            else:
                index = 0  # Default to placeholder (no selection)

        self.gui.data_selection_widget.experiment_combo.setCurrentIndex(index)
        logging.debug("Existing experiments refreshed successfully.")

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
            self.loading_thread.terminate()
            self.loading_thread.wait()
            self.loading_thread.deleteLater()
            if hasattr(self, "current_progress_dialog"):
                self.current_progress_dialog.close()
                del self.current_progress_dialog

        # Reset loading completion flag
        self.loading_completed_successfully = False

        experiment_name = self.gui.expts_dict_keys[index]
        exp_path = os.path.join(self.gui.output_path, experiment_name)
        logging.info(f"Loading experiment: '{experiment_name}'.")

        # Create and show progress dialog
        progress_dialog = QProgressDialog("Loading experiment...", "Cancel", 0, 100, self.gui)
        progress_dialog.setWindowTitle(f"Loading: {experiment_name}")
        progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        progress_dialog.setAutoClose(False)
        progress_dialog.setAutoReset(False)
        progress_dialog.setMinimumDuration(0)  # Show immediately
        progress_dialog.setMinimumWidth(450)  # Make it wider for better text visibility
        progress_dialog.resize(450, 120)  # Set a specific size

        # Simple initial message - let the loader provide specific time estimates
        initial_text = f"Initializing {experiment_name}..."
        progress_dialog.setLabelText(initial_text)
        progress_dialog.show()

        # Force immediate display
        QApplication.processEvents()

        # Create and start loading thread
        config = self.gui.config_repo.read_config()
        self.loading_thread = ExperimentLoadingThread(exp_path, config)

        # Connect signals
        self.loading_thread.progress.connect(progress_dialog.setValue)
        self.loading_thread.status_update.connect(progress_dialog.setLabelText)
        self.loading_thread.finished.connect(self._on_experiment_loaded)
        self.loading_thread.error.connect(self._on_experiment_load_error)
        self.loading_thread.finished.connect(progress_dialog.close)
        self.loading_thread.error.connect(progress_dialog.close)

        # Handle cancellation
        progress_dialog.canceled.connect(self._on_experiment_load_canceled)

        # Store dialog and state for cleanup
        self.current_progress_dialog = progress_dialog
        self.loading_completed_successfully = False

        # Start loading
        self.loading_thread.start()

    def _on_experiment_loaded(self, experiment):
        """Handle successful experiment loading."""
        try:
            # Mark loading as completed successfully
            self.loading_completed_successfully = True

            self.gui.set_current_experiment(experiment)

            # Track this experiment as recently used
            app_state.save_recent_experiment(experiment.id)

            # Save session state for restoration
            profile_name = (
                self.gui.profile_selector_combo.currentText() if hasattr(self.gui, "profile_selector_combo") else None
            )
            app_state.save_current_session_state(experiment_id=experiment.id, profile_name=profile_name)

            self.gui.data_selection_widget.update_dataset_combo()
            # Re-enable dataset combo since an experiment is now loaded
            self.gui.data_selection_widget.dataset_combo.setEnabled(True)
            self.gui.plot_widget.on_data_selection_changed()
            self.gui.status_bar.showMessage(f"Experiment '{experiment.id}' loaded successfully.", 5000)
            logging.info(f"Experiment '{experiment.id}' loaded successfully.")
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

    def _on_experiment_load_canceled(self):
        """Handle experiment loading cancellation."""
        # Check if loading actually completed successfully - if so, ignore this cancel signal
        if hasattr(self, "loading_completed_successfully") and self.loading_completed_successfully:
            logging.debug("Ignoring cancel signal - experiment loading completed successfully")
            if hasattr(self, "current_progress_dialog"):
                del self.current_progress_dialog
            return

        if hasattr(self, "loading_thread") and self.loading_thread.isRunning():
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

    def load_dataset(self, index):
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

        self.gui.data_selection_widget.update_session_combo()
        self.gui.data_selection_widget.session_combo.setEnabled(True)  # Enable session combo since a dataset is now loaded
        self.gui.plot_widget.on_data_selection_changed()

    def load_session(self, index):
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
            if "PyQt6" in sys.modules:
                from PyQt6.QtWidgets import (
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
            logging.info(f"Dataset name '{original_dataset_name}' follows standard format.")
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
                # The dialog handles folder renaming and updates dataset ID if needed
                updated_dataset_id = self.gui.current_dataset.id

                # Refresh the dataset combo to show updated display names
                self.gui.data_selection_widget.update_dataset_combo()

                # Find and select the updated dataset in the combo box
                if self.gui.current_experiment:
                    for index, dataset in enumerate(self.gui.current_experiment.datasets):
                        if dataset.id == updated_dataset_id:
                            self.gui.data_selection_widget.dataset_combo.setCurrentIndex(index)
                            logging.info(f"Reselected dataset '{updated_dataset_id}' at index {index} after metadata update")
                            break
                    else:
                        # Fallback: if we can't find the dataset by ID, try to find it by reference
                        for index, dataset in enumerate(self.gui.current_experiment.datasets):
                            if dataset is self.gui.current_dataset:
                                self.gui.data_selection_widget.dataset_combo.setCurrentIndex(index)
                                logging.info(f"Reselected dataset by reference at index {index} after metadata update")
                                break
                        else:
                            logging.warning(
                                f"Could not find dataset '{updated_dataset_id}' in experiment after metadata update"
                            )

                # Update session combo box as well to ensure consistency
                self.gui.data_selection_widget.update_session_combo()

                self.gui.status_bar.showMessage("Dataset metadata updated successfully.", 5000)
                logging.info(f"Dataset metadata updated for '{self.gui.current_dataset.id}'")

        except Exception as e:
            logging.error(f"Error opening dataset metadata editor: {e}")
            QMessageBox.critical(self.gui, "Error", f"Failed to open dataset metadata editor:\n{e}")

    # ------------------------------------------------------------------
    def close_all_data(self):
        """Close all currently open data (experiment, dataset, session)."""
        logging.info("Closing all data.")
        if self.gui.current_session:
            self.gui.current_session.close()
            self.gui.set_current_session(None)
        if self.gui.current_dataset:
            self.gui.current_dataset.close()
            self.gui.set_current_dataset(None)
        if self.gui.current_experiment:
            self.gui.current_experiment.close()
            self.gui.set_current_experiment(None)
        logging.info("All data closed successfully.")
