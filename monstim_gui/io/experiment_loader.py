"""Asynchronous experiment loading functionality."""

import logging
import re
import traceback
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from monstim_gui.core.application_state import app_state

# Note: skip preflight migration scans during load for performance.
# Post-load migrations can be initiated separately in a background task.
from monstim_signals.io.repositories import ExperimentRepository

logger = logging.getLogger(__name__)


class DatasetSkipLogHandler(logging.Handler):
    """Logging handler to capture dataset skip warnings during load."""

    def __init__(self):
        super().__init__()
        self.skipped_datasets = []

    def emit(self, record):
        if record.levelno == logging.WARNING and "skipped due to validation error" in record.getMessage():
            # Parse dataset name and error from log message
            msg = record.getMessage()
            match = re.search(r"Dataset '([^']+)' skipped due to validation error: (.+)", msg)
            if match:
                dataset_name = match.group(1)
                error_detail = match.group(2).split("\n")[0]  # Get first line only
                self.skipped_datasets.append((dataset_name, error_detail))


class ExperimentLoadingThread(QThread):
    """Thread for loading experiments asynchronously."""

    # Signals
    finished = Signal(object)  # Emits the loaded experiment
    error = Signal(str)  # Emits error message
    progress = Signal(int)  # Emits progress percentage
    status_update = Signal(str)  # Emits status message
    datasets_skipped = Signal(list)  # Emits list of (dataset_name, error_msg) tuples for skipped datasets

    def __init__(self, experiment_path: str, config: dict):
        super().__init__()
        self.experiment_path = experiment_path
        self.config = config
        self.experiment_name = Path(experiment_path).name
        self._skipped_datasets = []  # Track skipped datasets
        self._cancel_requested = False  # Flag for safe cancellation

    def request_cancel(self):
        """Request graceful cancellation of the loading operation."""
        logger.info("Cancellation requested for experiment loading thread")
        self._cancel_requested = True

    def run(self):
        """Load the experiment in a separate thread."""
        # Set up logging handler to capture dataset skip warnings
        skip_handler = DatasetSkipLogHandler()
        skip_handler.setLevel(logging.WARNING)
        root_logger = logging.getLogger()
        root_logger.addHandler(skip_handler)

        try:
            logger.debug(f"Starting async load of experiment: '{self.experiment_name}'")
            self.status_update.emit(f"Loading experiment: '{self.experiment_name}'")

            # Check if path exists
            exp_path = Path(self.experiment_path)
            if not exp_path.exists():
                self.error.emit(f"Experiment folder '{self.experiment_path}' not found.")
                # Wait so user can read the message
                QThread.sleep(3)
                return

            # Check for cancellation early
            if self._cancel_requested:
                logger.info("Loading canceled before analysis")
                return

            self.progress.emit(10)
            self.status_update.emit("Opening experiment catalog...")

            # Check for cancellation before repository creation
            if self._cancel_requested:
                logger.info("Loading canceled before repository creation")
                return

            # Create repository
            repo = ExperimentRepository(exp_path)
            self.progress.emit(15)

            self.status_update.emit(f"Loading '{self.experiment_name}'...")
            self.progress.emit(20)

            # Load experiment - this can take a long time for large experiments
            # Map dataset iteration progress (callback driven) into progress range 30-85.
            # Rate-limit progress updates to ~10/sec to reduce GUI churn.
            _last_emit_ts = 0.0

            def _progress_cb(level: str, index: int, total: int, name: str, *extra):
                nonlocal _last_emit_ts
                import time as _t

                # Check cancellation in callback - do this FIRST before any processing
                if self._cancel_requested:
                    logger.info("Progress callback detected cancellation flag")
                    raise InterruptedError("Loading canceled by user")

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
                    name_display = f"{name[:22]}…{name[-22:]}" if len(name) > 48 else name
                    self.progress.emit(pct)
                    self.status_update.emit(f"Loading dataset {index}/{total}:\n'{name_display}'")
                    _last_emit_ts = now
                elif level == "catalog" and total > 0:
                    # Map catalog construction into 20-30% before domain loading.
                    try:
                        pct = 20 + int(10 * (index / total))
                        self.progress.emit(pct)
                        self.status_update.emit(f"Building experiment catalog {index}/{total}:\n'{name}'")
                    except Exception as exc:
                        # Progress UI failures must not abort experiment loading; log for diagnostics.
                        logger.debug("Non-fatal error while updating catalog progress: %s", exc)

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

            # Initial loads may need to create missing annotations. The catalog is
            # non-authoritative and is rebuilt from those source files when needed.

            # Check for cancellation before starting the actual load
            if self._cancel_requested:
                logger.info("Loading canceled before repo.load()")
                return

            try:
                experiment = repo.load(
                    config=cfg,
                    progress_callback=_progress_cb,
                    allow_write=True,
                    lazy_open_h5=cfg.get("lazy_open_h5"),
                    load_workers=cfg.get("load_workers", 1),
                )
            except InterruptedError as e:
                # Graceful cancellation from progress callback
                logger.info(f"Loading interrupted: {e}")
                self.status_update.emit("Loading canceled by user...")
                return

            # Check for cancellation after load completes
            if self._cancel_requested:
                logger.info("Loading canceled after repo.load()")
                try:
                    experiment.close()
                except Exception:
                    logger.debug("Failed to close experiment after canceled load", exc_info=True)
                return

            self.progress.emit(90)
            self.status_update.emit("Finalizing experiment structure...")

            # Check if any datasets were skipped and inform user
            expected_datasets = len([p for p in exp_path.iterdir() if p.is_dir()])
            loaded_datasets = len(experiment.datasets)

            if loaded_datasets < expected_datasets:
                skipped = expected_datasets - loaded_datasets
                logger.warning(
                    f"{skipped} dataset(s) could not be loaded from '{self.experiment_name}'. "
                    f"Check the log for details about validation errors or data inconsistencies."
                )
                # Emit the skipped datasets information for GUI warning
                if skip_handler.skipped_datasets:
                    self._skipped_datasets = skip_handler.skipped_datasets
                    self.datasets_skipped.emit(self._skipped_datasets)

            self.progress.emit(100)
            logger.debug(f"Experiment '{self.experiment_name}' loaded successfully in thread.")
            self.finished.emit(experiment)

        except OSError as e:
            if "Too many open files" in str(e):
                error_msg = f"Too many files open while loading experiment '{self.experiment_name}'. "
                error_msg += "This experiment may be too large or have corrupted files. Try closing other applications and retry."
            else:
                error_msg = f"File system error while loading experiment '{self.experiment_name}': {e}"
            logger.error(error_msg)
            self.error.emit(error_msg)
        except FileNotFoundError as e:
            error_msg = f"Experiment file not found or corrupted: {e}"
            logger.error(error_msg)
            self.error.emit(error_msg)
        except Exception as e:
            error_msg = f"An error occurred while loading experiment '{self.experiment_name}': {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.error.emit(error_msg)
        finally:
            # Clean up logging handler
            root_logger.removeHandler(skip_handler)

            # If cancellation was requested, ensure cleanup
            if self._cancel_requested:
                logger.debug("Performing cleanup after canceled load")
                try:
                    # Force garbage collection to clean up any partially loaded objects
                    import gc

                    collected = gc.collect()
                    logger.debug(f"Post-cancellation cleanup: collected {collected} objects")
                except Exception as cleanup_error:
                    logger.warning(f"Error during post-cancellation cleanup: {cleanup_error}")
