"""Asynchronous experiment loading functionality."""

import logging
import traceback
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from monstim_signals.io.repositories import ExperimentRepository


class ExperimentLoadingThread(QThread):
    """Thread for loading experiments asynchronously."""

    # Signals
    finished = pyqtSignal(object)  # Emits the loaded experiment
    error = pyqtSignal(str)  # Emits error message
    progress = pyqtSignal(int)  # Emits progress percentage
    status_update = pyqtSignal(str)  # Emits status message

    def __init__(self, experiment_path: str, config: dict):
        super().__init__()
        self.experiment_path = experiment_path
        self.config = config
        self.experiment_name = Path(experiment_path).name
        self._is_first_load = None  # Will be determined during analysis
        self._estimated_time = None

    def _analyze_load_requirements(self, exp_path: Path) -> tuple[bool, int, int]:
        """
        Analyze the experiment to determine if this is a first-time load.
        Returns (is_first_load, total_files, missing_annotations).
        """
        try:
            expected_annotations = 1  # Count the experiment itself
            missing_annotations = 0

            # Check experiment level annotation
            exp_annot = exp_path / "experiment.annot.json"
            if not exp_annot.exists():
                missing_annotations += 1

            # Check each dataset
            for dataset_dir in exp_path.iterdir():
                expected_annotations += 1  # Count each dataset

                if not dataset_dir.is_dir():
                    continue

                # Check dataset annotation
                ds_annot = dataset_dir / "dataset.annot.json"
                if not ds_annot.exists():
                    missing_annotations += 1

                # Check each session in the dataset
                for session_dir in dataset_dir.iterdir():
                    expected_annotations += 1  # Count each session
                    if not session_dir.is_dir():
                        continue

                    # Check session annotation
                    sess_annot = session_dir / "session.annot.json"
                    if not sess_annot.exists():
                        missing_annotations += 1

                    # Do not count recording annotations since they are not required for first load.

            # Consider it a first load if more than 25% of dataset/session/experiment annotations are missing
            is_first_load = missing_annotations > (expected_annotations * 0.25)
            return is_first_load, expected_annotations, missing_annotations

        except Exception as e:
            logging.warning(f"Error analyzing load requirements: {e}")
            return True, 0, 0  # Assume first load if analysis fails

    def _count_files_to_load(self, exp_path: Path) -> int:
        """Count the approximate number of files that will need to be loaded."""
        try:
            file_count = 0
            # Count meta.json files (recordings)
            for meta_file in exp_path.rglob("*.meta.json"):
                file_count += 1
            # Count annotation files
            for annot_file in exp_path.rglob("*.annot.json"):
                file_count += 1

            return file_count

        except Exception as e:
            logging.debug(f"Error counting files: {e}")
            return 0

    def run(self):
        """Load the experiment in a separate thread."""
        try:
            logging.debug(f"Starting async load of experiment: '{self.experiment_name}'")
            self.status_update.emit(f"Loading experiment: '{self.experiment_name}'")
            self.progress.emit(10)

            # Check if path exists
            exp_path = Path(self.experiment_path)
            if not exp_path.exists():
                self.error.emit(f"Experiment folder '{self.experiment_path}' not found.")
                return

            self.progress.emit(15)
            self.status_update.emit("Analyzing experiment structure...")

            # Analyze if this is a first-time load
            is_first_load, annotations_required, missing_annotations = self._analyze_load_requirements(exp_path)
            self._is_first_load = is_first_load

            files_to_load = self._count_files_to_load(exp_path)

            logging.debug(
                f"First load: {is_first_load}, Total files: {files_to_load}, Total Annotations Required: {annotations_required}, Missing annotations: {missing_annotations}"
            )

            # Provide appropriate time estimates in logging
            if is_first_load:
                estimated_time = int(missing_annotations / 100 * 60)  # Rough estimate: 100 files per minute
                self._estimated_time = estimated_time
                time_msg = f"First-time load detected: {missing_annotations} annotation files need to be created.\nEstimated time: {estimated_time} seconds for {annotations_required} annotations."
                logging.info(time_msg)
            elif files_to_load > 5000:
                time_msg = f"Large experiment detected: {files_to_load} recordings. Loading may take several seconds."
                logging.info(time_msg)

            self.progress.emit(25)
            self.status_update.emit("Loading experiment repository...")

            # Create repository
            repo = ExperimentRepository(exp_path)
            self.progress.emit(35)

            # This is the slow part - the actual repo.load() call
            if is_first_load:
                self.status_update.emit(
                    f"First-time loading '{self.experiment_name}'...\n\nCreating {missing_annotations} annotation files.\nEstimated time: {estimated_time} seconds."
                )
            elif files_to_load > 5000:
                self.status_update.emit(
                    f"Loading '{self.experiment_name}'...\n\nLarge experiment with {files_to_load} recordings.\nThis may take several seconds."
                )
            else:
                self.status_update.emit("Reading experiment metadata...")
            self.progress.emit(40)

            # Load experiment - this can take a long time for large experiments
            experiment = repo.load(config=self.config)

            self.progress.emit(90)
            self.status_update.emit("Finalizing experiment structure...")

            self.progress.emit(100)
            logging.debug(f"Experiment '{self.experiment_name}' loaded successfully in thread.")
            self.finished.emit(experiment)

        except OSError as e:
            if "Too many open files" in str(e):
                error_msg = f"Too many files open while loading experiment '{self.experiment_name}'. This experiment may be too large or have corrupted files. Try closing other applications and retry."
            else:
                error_msg = f"File system error while loading experiment '{self.experiment_name}': {e}"
            logging.error(error_msg)
            self.error.emit(error_msg)
        except FileNotFoundError as e:
            error_msg = f"Experiment file not found or corrupted: {e}"
            logging.error(error_msg)
            self.error.emit(error_msg)
        except Exception as e:
            error_msg = f"An error occurred while loading experiment '{self.experiment_name}': {e}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            self.error.emit(error_msg)
