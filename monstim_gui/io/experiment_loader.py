"""Asynchronous experiment loading functionality."""

import logging
import traceback
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from monstim_gui.core.application_state import app_state

# Note: skip preflight migration scans during load for performance.
# Post-load migrations can be initiated separately in a background task.
from monstim_signals.io.repositories import ExperimentRepository


class ExperimentLoadingThread(QThread):
    """Thread for loading experiments asynchronously."""

    # Signals
    finished = Signal(object)  # Emits the loaded experiment
    error = Signal(str)  # Emits error message
    progress = Signal(int)  # Emits progress percentage
    status_update = Signal(str)  # Emits status message

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

            # Check if path exists
            exp_path = Path(self.experiment_path)
            if not exp_path.exists():
                self.error.emit(f"Experiment folder '{self.experiment_path}' not found.")
                # Wait so user can read the message
                QThread.sleep(3)
                return

            self.progress.emit(10)
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

            # Create repository
            repo = ExperimentRepository(exp_path)
            self.progress.emit(15)

            # This is the slow part - the actual repo.load() call
            if is_first_load:
                self.status_update.emit(
                    f"First-time load detected for '{self.experiment_name}'.\n\nCreating {missing_annotations} annotation files and building indexes.\nEstimated time: {estimated_time} seconds."
                )
            elif files_to_load > 5000:
                self.status_update.emit(
                    f"Loading '{self.experiment_name}'...\n\nLarge experiment detected ({files_to_load} files). Building indexes and applying migrations may take longer."
                )
            else:
                self.status_update.emit("Reading experiment metadata...")
            self.progress.emit(20)

            # Skipped: Preflight migration scan moved to post-load background task.

            self.progress.emit(30)
            # Show a distinct "Indexing..." step only when needed
            try:
                from monstim_signals.io.experiment_index import is_index_stale, load_experiment_index

                idx = load_experiment_index(exp_path)
                needs_index = idx is None or is_index_stale(idx)
            except Exception:
                needs_index = False

            from monstim_gui.core.application_state import app_state as _app

            if needs_index and _app.should_build_index_on_load():
                self.status_update.emit("Indexing experiment folders...")
            else:
                self.status_update.emit("Loading experiment repository...")

            # Load experiment - this can take a long time for large experiments
            # Map dataset iteration progress (callback driven) into progress range 30-85.
            # Rate-limit progress updates to ~10/sec to reduce GUI churn.
            _last_emit_ts = 0.0

            def _progress_cb(level: str, index: int, total: int, name: str):
                nonlocal _last_emit_ts
                import time as _t

                if level == "dataset" and total > 0:
                    now = _t.monotonic()
                    if now - _last_emit_ts < 0.1:
                        return
                    # Reserve 55% of the bar (30 -> 85) for dataset loading.
                    base = 30
                    span = 55
                    frac = index / total
                    pct = base + int(span * frac)
                    # Truncate very long dataset names to keep dialog width stable
                    if len(name) > 48:
                        name_display = f"{name[:22]}…{name[-22:]}"
                    else:
                        name_display = name
                    self.progress.emit(pct)
                    self.status_update.emit(f"Loading dataset {index}/{total}: '{name_display}' ...")
                    _last_emit_ts = now
                elif level == "index" and total > 0:
                    # Map index progress into 20-30% range before dataset load
                    try:
                        pct = 20 + int(10 * (index / total))
                        self.progress.emit(pct)
                        self.status_update.emit(f"Building index {index}/{total}: '{name}' ...")
                    except Exception:
                        pass

            # Overlay application preferences (QSettings) for loading:
            cfg = dict(self.config or {})
            # If config doesn't explicitly set lazy_open_h5, use QSettings default
            if "lazy_open_h5" not in cfg:
                cfg["lazy_open_h5"] = app_state.should_use_lazy_open_h5()

            # Determine load_workers: prefer explicit config value, else use QSettings auto behavior
            if "load_workers" not in cfg:
                if app_state.should_use_parallel_loading():
                    cfg["load_workers"] = app_state.get_parallel_load_workers()
                else:
                    cfg["load_workers"] = 1

            # Enforce read-only load: do not write annotations or migrate on load.
            # For first-time loads (missing annotations), we must load recordings eagerly
            # so sessions have valid data; otherwise lazy loading is fine.
            # Ensure sessions materialize recordings during load to avoid
            # "no recordings" errors from strict domain checks. Lazy
            # access can still be applied at the HDF5 level via
            # `lazy_open_h5` in config.
            experiment = repo.load(
                config=cfg,
                progress_callback=_progress_cb,
                allow_write=False,
                load_recordings=False,
            )

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
