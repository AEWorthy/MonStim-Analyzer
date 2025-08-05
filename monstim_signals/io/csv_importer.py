import json
import h5py
from pathlib import Path
from dataclasses import asdict
from typing import Any, Callable
import logging
import numpy as np
import traceback


from monstim_signals.io.csv_parser import parse
from monstim_signals.core import RecordingAnnot
from monstim_signals.version import DATA_VERSION

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

lock = Lock()


def discover_by_ext(base: Path, pattern="*.csv") -> list[Path]:
    """
    Discover all files in the given base directory and its subdirectories that match the given pattern.
    The pattern is typically '*.csv' to find all CSV files: returns a list of Paths to CSV files that are non-empty.
    """
    all_csv = list(base.rglob(pattern))
    return [csv for csv in all_csv if csv.is_file() and csv.stat().st_size > 0]


def parse_session_rec(csv_path: Path):
    """
    Expects name like 'AB12-0034.csv':
      - first part (4 letters/numbers) = session ID
      - second part (4 digits)     = recording ID
    """
    stem = csv_path.stem  # e.g. "AB12-0034"
    if "-" in stem:
        session_id, recording_id = stem.split("-", 1)
        if len(session_id) == 4 and recording_id.isdigit() and len(recording_id) == 4:
            return session_id, recording_id
    return None, None


def infer_ds_ex(csv_path: Path, base_dir: Path):
    """
    dataset_name = immediate parent folder (if not base_dir)
    experiment_name = grandparent folder (if not base_dir)
    """
    parent = csv_path.parent
    dataset_name = parent.name if parent != base_dir else None
    grandparent = parent.parent
    experiment_name = grandparent.name if grandparent != base_dir else None
    return dataset_name, experiment_name


def detect_format(path: Path) -> str:
    """
    Inspect the first handful of lines to pick a format tag.
    Returns 'v3h' if it sees the new [Parameters]/[DATA] markers, else 'v3d'.
    """
    with path.open() as f:
        for _ in range(5):
            line = f.readline()
            if not line:
                break
            if line.strip().startswith("[Parameters]"):
                return "v3h"
            if line.lower().startswith("file version"):
                return "v3d"
    raise ValueError(
        f"Could not detect MonStim version for {path}. "
        "Please ensure it is a valid MonStim CSV file."
    )


def csv_to_store(
    csv_path: Path,
    output_fp: Path,
    overwrite_h5: bool = False,
    overwrite_meta: bool = False,
    overwrite_annot: bool = False,
):
    """Convert a CSV file to an HDF5 file with metadata and data.
    This verion is compatible for MonStim V3D and later."""
    meta_dict: dict[str, Any]
    arr: np.ndarray
    meta_dict, arr = parse(csv_path)
    meta_dict["session_id"] = output_fp.stem.split("-")[
        0
    ]  # Use the first part of the filename as session ID
    meta_dict["recording_id"] = (
        output_fp.stem.split("-")[1] if "-" in output_fp.stem else None
    )

    # Add meta's data_version key
    meta_dict["data_version"] = DATA_VERSION  # Use the global DATA_VERSION

    h5_path = output_fp.with_suffix(".raw.h5")
    if h5_path.exists() and not overwrite_h5:
        logging.warning(
            f"HDF5 file {h5_path} already exists. Use 'overwrite=True' to replace it."
        )
    else:
        if h5_path.exists() and overwrite_h5:
            logging.warning(f"HDF5 file {h5_path} already exists. Overwriting it.")
        elif not h5_path.parent.exists():
            logging.info(f"Creating directory {h5_path.parent} for HDF5 file.")
            h5_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(h5_path, "w") as h5:
            h5.create_dataset(
                "raw",
                data=arr,
                chunks=(min(30000, arr.shape[0]), arr.shape[1]),
                compression="gzip",
            )
            h5.attrs["scan_rate"] = meta_dict.get("scan_rate")
            h5.attrs["num_channels"] = meta_dict.get("num_channels")
            h5.attrs["channel_types"] = meta_dict.get("channel_types")
            h5.attrs["num_samples"] = arr.shape[0]  # (#samples × #channels)

    # Write meta JSON
    meta_path = output_fp.with_suffix(".meta.json")
    if meta_path.exists() and not overwrite_meta:
        logging.warning(
            f"Meta file {meta_path} already exists. Use 'overwrite_meta=True' to replace it."
        )
    else:
        if meta_path.exists() and overwrite_meta:
            logging.warning(f"Meta file {meta_path} already exists. Overwriting it.")
        with meta_path.open("w") as f:
            json.dump(meta_dict, f, indent=4)

    # Write annotation JSON
    annot_path = output_fp.with_suffix(".annot.json")
    if annot_path.exists() and not overwrite_annot:
        logging.warning(
            f"Annotation file {annot_path} already exists. Use 'overwrite_annot=True' to replace it."
        )
    else:
        if annot_path.exists() and overwrite_annot:
            logging.warning(
                f"Annotation file {annot_path} already exists. Overwriting it."
            )
        with annot_path.open("w") as f:
            annot = RecordingAnnot.create_empty()
            json.dump(asdict(annot), f, indent=2)


def get_dataset_session_dict(dataset_path: Path) -> dict[str, list[Path]]:
    """Return a mapping of session IDs to CSV file paths for one dataset."""
    csv_paths = [p for p in dataset_path.iterdir() if p.suffix.lower() == ".csv"]
    mapping: dict[str, list[Path]] = {}
    for csv_path in csv_paths:
        session_id, _ = parse_session_rec(csv_path)
        if session_id:
            mapping.setdefault(session_id, []).append(csv_path)
    return mapping


def import_experiment(
    expt_path: Path,
    output_path: Path,
    progress_callback: Callable[[int], None] = lambda v: None,
    is_canceled: Callable[[], bool] = lambda: False,
    overwrite: bool = True,
    max_workers: int | None = None,
) -> None:
    """Convert all CSV files under *expt_path* into the store at *output_path*."""
    datasets = [d for d in expt_path.iterdir() if d.is_dir()]
    all_csv = []
    dataset_maps = {}
    for ds_dir in datasets:
        mapping = get_dataset_session_dict(ds_dir)
        dataset_maps[ds_dir.name] = mapping
        for files in mapping.values():
            all_csv.extend(files)

    total_files = len(all_csv)
    processed = 0

    def process_csv(csv_path: Path, ds_name: str, sess_name: str):
        nonlocal processed
        if is_canceled():
            return
        out_dir = output_path / ds_name / sess_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_fp = out_dir / csv_path.stem
        csv_to_store(
            csv_path,
            out_fp,
            overwrite_h5=overwrite,
            overwrite_meta=overwrite,
            overwrite_annot=overwrite,
        )
        with lock:
            processed += 1
        progress_callback(int((processed / total_files) * 100))

    for ds_name, sess_map in dataset_maps.items():
        if max_workers and max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                future_to_args = {
                    ex.submit(process_csv, csv_path, ds_name, sess_name): (
                        csv_path,
                        ds_name,
                        sess_name,
                    )
                    for sess_name, paths in sess_map.items()
                    for csv_path in paths
                }
                for f in as_completed(future_to_args):
                    if is_canceled():
                        return
                    try:
                        f.result()
                    except Exception:
                        csv_path, ds_name, sess_name = future_to_args[f]
                        logging.error(
                            f"Error processing CSV: {csv_path} in dataset {ds_name}, session {sess_name}"
                        )
                        logging.error(traceback.format_exc())
        else:
            for sess_name, paths in sess_map.items():
                for csv_path in paths:
                    process_csv(csv_path, ds_name, sess_name)

    logging.info("Processing complete.")


from PyQt6.QtCore import QThread, pyqtSignal  # noqa: E402


class GUIExptImportingThread(QThread):
    """Threaded wrapper to import an experiment from the GUI."""

    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(Exception)
    canceled = pyqtSignal()

    def __init__(
        self,
        expt_name: str,
        expt_path: str,
        output_dir_path: str,
        max_workers: int | None = None,
        overwrite: bool = True,
    ) -> None:
        super().__init__()
        self.expt_path = Path(expt_path)
        self.output_path = Path(output_dir_path) / expt_name
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.overwrite = overwrite
        self._is_canceled = False
        self._is_finished = False
        self._last_logged_progress = -5  # For throttling log output

    def run(self) -> None:
        try:
            import_experiment(
                self.expt_path,
                self.output_path,
                self.report_progress,
                self.is_canceled,
                overwrite=self.overwrite,
                max_workers=self.max_workers,
            )
            if not self._is_canceled:
                self.finished.emit()
                self._is_finished = True
        except Exception as e:
            if not self._is_canceled:
                self.error.emit(e)
                logging.error(f"Error in GUIExptImportingThread: {e}")
                logging.error(traceback.format_exc())

    def report_progress(self, value: int) -> None:
        if not self._is_canceled:
            self.progress.emit(value)
        # Only log every 5% increment
        if value >= self._last_logged_progress + 5 or value == 100:
            logging.info(f"CSV conversion progress: {value}%")
            self._last_logged_progress = value

    def cancel(self) -> None:
        if not self._is_canceled and not self._is_finished:
            self._is_canceled = True
            self.canceled.emit()

    def is_canceled(self) -> bool:
        return self._is_canceled

    def is_finished(self) -> bool:
        return self._is_finished


class MultiExptImportingThread(QThread):
    """Threaded wrapper to import multiple experiments from the GUI."""

    progress = pyqtSignal(int)
    status_update = pyqtSignal(str)
    finished = pyqtSignal(int)  # Emits count of successfully imported experiments
    error = pyqtSignal(Exception)
    canceled = pyqtSignal()

    def __init__(
        self,
        experiment_paths: list[str],
        output_dir_path: str,
        max_workers: int | None = None,
        overwrite: bool = True,
    ) -> None:
        super().__init__()
        self.experiment_paths = [Path(p) for p in experiment_paths]
        self.output_dir_path = Path(output_dir_path)
        self.max_workers = max_workers
        self.overwrite = overwrite
        self._is_canceled = False
        self._is_finished = False
        self.successful_imports = 0
        self.total_experiments = len(experiment_paths)

    def run(self) -> None:
        try:
            for i, expt_path in enumerate(self.experiment_paths):
                if self._is_canceled:
                    break

                expt_name = expt_path.name
                output_path = self.output_dir_path / expt_name
                output_path.mkdir(parents=True, exist_ok=True)

                # Update status
                self.status_update.emit(
                    f"Importing experiment {i+1}/{self.total_experiments}: '{expt_name}'"
                )

                try:
                    import_experiment(
                        expt_path,
                        output_path,
                        lambda value: self.report_progress(i, value),
                        self.is_canceled,
                        overwrite=self.overwrite,
                        max_workers=self.max_workers,
                    )

                    if not self._is_canceled:
                        self.successful_imports += 1
                        logging.info(f"Successfully imported experiment: '{expt_name}'")

                except Exception as e:
                    logging.error(f"Failed to import experiment '{expt_name}': {e}")
                    # Continue with other experiments instead of stopping
                    continue

            if not self._is_canceled:
                self.finished.emit(self.successful_imports)
                self._is_finished = True

        except Exception as e:
            if not self._is_canceled:
                self.error.emit(e)
                logging.error(f"Error in MultiExptImportingThread: {e}")
                logging.error(traceback.format_exc())

    def report_progress(self, experiment_index: int, experiment_progress: int) -> None:
        if not self._is_canceled:
            # Calculate overall progress across all experiments
            overall_progress = int(
                ((experiment_index * 100) + experiment_progress)
                / self.total_experiments
            )
            self.progress.emit(overall_progress)

    def cancel(self) -> None:
        if not self._is_canceled and not self._is_finished:
            self._is_canceled = True
            self.canceled.emit()

    def is_canceled(self) -> bool:
        return self._is_canceled

    def is_finished(self) -> bool:
        return self._is_finished
