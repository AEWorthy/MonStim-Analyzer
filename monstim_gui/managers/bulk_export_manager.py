# monstim_gui/managers/bulk_export_manager.py
"""
BulkExportManager – orchestrates the Bulk Data Export feature.

Responsibilities
----------------
- Show the BulkExportDialog and collect user configuration.
- Launch a background worker thread that loads experiments sequentially and
  writes one xlsx file per selected object (dataset or experiment level).
- Keep all computation / file I/O off the main thread.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from monstim_gui.gui_main import MonstimGUI

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

#: Human-readable data-type keys → Excel sheet names
DATA_TYPE_LABELS: dict[str, str] = {
    "avg_reflex_curves": "Avg Reflex Curves",
    "mmax": "M-max Summary",
    "max_h": "Max H-Reflex",
}

#: Calculation method keys → display labels
METHOD_LABELS: dict[str, str] = {
    "rms": "RMS",
    "average_rectified": "Avg Rectified",
    "peak_to_trough": "Peak-to-Trough",
    "average_unrectified": "Avg Unrectified",
    "auc": "AUC",
}


# ─────────────────────────────────────────────────────────────────────────────
# Config dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class BulkExportConfig:
    """All parameters collected from the BulkExportDialog."""

    #: "dataset" or "experiment"
    data_level: str

    #: {expt_name: [dataset_id, ...]}  (empty list for experiment-level exports)
    selected_objects: dict[str, list[str]]

    #: e.g. ["avg_reflex_curves", "mmax", "max_h"]
    data_types: list[str]

    #: e.g. ["rms", "auc"]
    methods: list[str]

    #: Zero-based channel indices to include
    channel_indices: list[int]

    #: Root folder for written output files
    output_path: str

    #: When True, add M-max-normalized amplitude columns alongside raw columns
    normalize_to_mmax: bool = False

    #: {expt_name: str(folder_path)} – sourced from gui.expts_dict
    experiment_paths: dict[str, str] = field(default_factory=dict)

    #: Number of parallel worker threads for dataset-level exports (1 = serial).
    max_workers: int = 1


# ─────────────────────────────────────────────────────────────────────────────
# Pure-function export engine (no Qt – safe to run in a worker thread)
# ─────────────────────────────────────────────────────────────────────────────


def _sanitize_path_component(name: str) -> str:
    """Return a string safe to use as a file/folder name component."""
    if not name:
        return "unnamed"
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", name)
    sanitized = re.sub(r"[.\s]+$", "", sanitized)  # trailing dots/spaces
    return sanitized or "unnamed"


def _safe_channel_name(obj, channel_idx: int) -> str:
    try:
        names = obj.channel_names
        if channel_idx < len(names):
            return names[channel_idx]
    except Exception:
        pass
    return f"Ch{channel_idx}"


def _n_col_label(config: BulkExportConfig) -> str:
    """Return the contributor-count column prefix appropriate for the data level.

    At dataset level averages are computed across *sessions*; at experiment
    level they are computed across *datasets*.
    """
    return "n_datasets" if config.data_level == "experiment" else "n_sessions"


def _get_mmax_cache(obj, config: BulkExportConfig) -> dict[tuple[int, str], Optional[float]]:
    """Pre-compute M-max per (channel_index, method) to avoid repeated calls."""
    cache: dict[tuple[int, str], Optional[float]] = {}
    for ch_idx in config.channel_indices:
        for method in config.methods:
            try:
                val = obj.get_avg_m_max(method, ch_idx)
                cache[(ch_idx, method)] = float(val) if val is not None else None
            except Exception as exc:
                logging.debug("M-max unavailable for ch=%d method=%s: %s", ch_idx, method, exc)
                cache[(ch_idx, method)] = None
    return cache


def _compute_avg_reflex_curves(obj, config: BulkExportConfig) -> pd.DataFrame:
    """Build a DataFrame of averaged reflex:stimulus curve data.

    Columns: voltage, channel, window, mean_amplitude_{m}, stdev_amplitude_{m},
    n_contributions_{m}  for each method m.

    When ``config.normalize_to_mmax`` is True, also adds:
      mean_amplitude_norm_mmax_{m}, stdev_amplitude_norm_mmax_{m}
    """
    try:
        windows = obj.unique_latency_window_names()
    except Exception:
        try:
            windows = [lw.name for lw in obj.latency_windows]
        except Exception:
            windows = []

    if not windows:
        logging.warning("No latency windows found – skipping avg_reflex_curves.")
        return pd.DataFrame()

    mmax_cache = _get_mmax_cache(obj, config) if config.normalize_to_mmax else {}

    rows: list[dict] = []
    for ch_idx in config.channel_indices:
        ch_name = _safe_channel_name(obj, ch_idx)
        for window_name in windows:
            # Gather per-method results aligned on a common voltage axis
            voltage_array: Optional[np.ndarray] = None
            method_cols: dict[str, np.ndarray] = {}

            for method in config.methods:
                try:
                    result = obj.get_average_lw_reflex_curve(method, ch_idx, window_name)
                    volts = result.get("voltages", np.array([]))
                    if len(volts) == 0:
                        continue
                    if voltage_array is None:
                        voltage_array = volts
                    means = result.get("means", np.full(len(volts), np.nan))
                    stdevs = result.get("stdevs", np.full(len(volts), np.nan))
                    method_cols[f"mean_amplitude_{method}"] = means
                    method_cols[f"stdev_amplitude_{method}"] = stdevs
                    method_cols[f"{_n_col_label(config)}_{method}"] = result.get("n_sessions", np.full(len(volts), np.nan))
                    # M-max normalization columns
                    if config.normalize_to_mmax:
                        mmax = mmax_cache.get((ch_idx, method))
                        if mmax and mmax != 0.0:
                            method_cols[f"mean_amplitude_norm_mmax_{method}"] = means / mmax
                            method_cols[f"stdev_amplitude_norm_mmax_{method}"] = stdevs / mmax
                        else:
                            logging.warning(
                                "M-max unavailable or zero for ch=%s method=%s – normalized columns skipped.",
                                ch_name,
                                method,
                            )
                except Exception as exc:
                    logging.warning(
                        "avg_reflex_curves error ch=%s window=%s method=%s: %s",
                        ch_name,
                        window_name,
                        method,
                        exc,
                    )

            if voltage_array is None or len(method_cols) == 0:
                continue

            n = len(voltage_array)
            for i in range(n):
                row: dict = {"voltage": voltage_array[i], "channel": ch_name, "window": window_name}
                for col, arr in method_cols.items():
                    row[col] = arr[i] if i < len(arr) else np.nan
                rows.append(row)

    return pd.DataFrame(rows)


def _compute_mmax(obj, config: BulkExportConfig) -> pd.DataFrame:
    """Build a DataFrame with one row per channel showing M-max per method."""
    try:
        n_channels = len(obj.channel_names)
    except Exception:
        n_channels = 0

    rows: list[dict] = []
    for ch_idx in config.channel_indices:
        if n_channels > 0 and ch_idx >= n_channels:
            logging.debug("_compute_mmax: channel index %d out of range (%d) – skipped.", ch_idx, n_channels)
            continue
        ch_name = _safe_channel_name(obj, ch_idx)
        row: dict = {"channel": ch_name, "channel_index": ch_idx}
        for method in config.methods:
            try:
                mmax, mthresh = obj.get_avg_m_max(method, ch_idx, return_avg_mmax_thresholds=True)
                row[f"mmax_{method}"] = mmax
                row[f"mmax_threshold_{method}"] = mthresh
            except Exception as exc:
                logging.warning("mmax error ch=%s method=%s: %s", ch_name, method, exc)
                row[f"mmax_{method}"] = None
                row[f"mmax_threshold_{method}"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def _compute_max_h(obj, config: BulkExportConfig) -> pd.DataFrame:
    """Build a DataFrame of average H-reflex amplitudes across stimulus voltages."""
    try:
        voltages = obj.stimulus_voltages
    except Exception:
        logging.warning("Could not obtain stimulus_voltages – skipping max_h.")
        return pd.DataFrame()

    if voltages is None or len(voltages) == 0:
        return pd.DataFrame()

    try:
        n_channels = len(obj.channel_names)
    except Exception:
        n_channels = 0

    rows: list[dict] = []
    for ch_idx in config.channel_indices:
        if n_channels > 0 and ch_idx >= n_channels:
            logging.debug("_compute_max_h: channel index %d out of range (%d) – skipped.", ch_idx, n_channels)
            continue
        ch_name = _safe_channel_name(obj, ch_idx)
        method_data: dict[str, np.ndarray] = {}
        for method in config.methods:
            try:
                avg, std = obj.get_avg_h_wave_amplitudes(method, ch_idx)
                avg_arr = np.asarray(avg)
                std_arr = np.asarray(std)
                method_data[f"avg_h_amplitude_{method}"] = avg_arr
                method_data[f"std_h_amplitude_{method}"] = std_arr
                # M-max normalization
                if config.normalize_to_mmax:
                    try:
                        mmax = obj.get_avg_m_max(method, ch_idx)
                        if mmax and float(mmax) != 0.0:
                            method_data[f"avg_h_amplitude_norm_mmax_{method}"] = avg_arr / float(mmax)
                            method_data[f"std_h_amplitude_norm_mmax_{method}"] = std_arr / float(mmax)
                        else:
                            logging.warning(
                                "M-max unavailable or zero for ch=%s method=%s – normalized columns skipped.",
                                ch_name,
                                method,
                            )
                    except Exception as exc:
                        logging.warning("M-max lookup failed for ch=%s method=%s: %s", ch_name, method, exc)
            except Exception as exc:
                logging.warning("max_h error ch=%s method=%s: %s", ch_name, method, exc)

        if not method_data:
            continue

        for i, v in enumerate(voltages):
            row: dict = {"voltage": float(v), "channel": ch_name}
            for col, arr in method_data.items():
                row[col] = arr[i] if i < len(arr) else np.nan
            rows.append(row)

    return pd.DataFrame(rows)


_DATA_TYPE_HANDLERS: dict[str, Callable] = {
    "avg_reflex_curves": _compute_avg_reflex_curves,
    "mmax": _compute_mmax,
    "max_h": _compute_max_h,
}


def _write_object_export(
    obj,
    expt_name: str,
    obj_id: str,
    config: BulkExportConfig,
) -> Path:
    """Compute all requested data types and write a single xlsx file.

    Returns the path to the written file.
    """
    out_dir = Path(config.output_path) / _sanitize_path_component(expt_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_id = _sanitize_path_component(obj_id)
    out_file = out_dir / f"{safe_id}_bulk_export.xlsx"

    # Compute all DataFrames before opening the writer to avoid creating empty xlsx files
    sheets: list[tuple[str, pd.DataFrame]] = []
    for data_type in config.data_types:
        handler = _DATA_TYPE_HANDLERS.get(data_type)
        if handler is None:
            logging.warning("Unknown data type '%s' – skipped.", data_type)
            continue
        try:
            df = handler(obj, config)
        except Exception as exc:
            logging.error("Error computing '%s' for '%s/%s': %s", data_type, expt_name, obj_id, exc)
            df = pd.DataFrame()
        if df is not None and not df.empty:
            sheet_name = DATA_TYPE_LABELS.get(data_type, data_type)[:31]  # Excel sheet name limit
            sheets.append((sheet_name, df))
        else:
            logging.debug("No data for type '%s' in '%s/%s' – sheet skipped.", data_type, expt_name, obj_id)

    if not sheets:
        logging.warning("No data written for '%s/%s' – file not created.", expt_name, obj_id)
        return out_file  # file was never created; caller can check existence

    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        for sheet_name, df in sheets:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    return out_file


def _load_and_export_dataset_task(
    expt_name: str,
    ds_id: str,
    expt_folder: Path,
    config: "BulkExportConfig",
    is_canceled: Optional[Callable[[], bool]] = None,
) -> tuple[Optional[str], str]:
    """Load one dataset, write its xlsx, release all file handles, and return.

    Returns ``(output_path_or_None, display_message)``.
    Thread-safe: each invocation opens its own independent set of file handles.
    Checks *is_canceled* after the load step so a cancellation request issued
    while loading is honoured as soon as possible (the load itself cannot be
    interrupted mid-flight).
    """
    import gc

    from monstim_signals.io.repositories import DatasetRepository

    ds_folder = expt_folder / ds_id
    if not ds_folder.is_dir():
        logging.error("Dataset folder not found for '%s/%s' at '%s' – skipping.", expt_name, ds_id, ds_folder)
        return None, f"Not found: {ds_id}"

    # Check before starting the (potentially slow) load
    if is_canceled and is_canceled():
        logging.info("Bulk export: skipping dataset '%s/%s' – canceled.", expt_name, ds_id)
        return None, f"Canceled: {ds_id}"

    try:
        logging.info("Bulk export: loading dataset '%s/%s'", expt_name, ds_id)
        dataset = DatasetRepository(ds_folder).load(allow_write=False)
    except Exception as exc:
        logging.error("Failed to load dataset '%s/%s': %s", expt_name, ds_id, exc)
        return None, f"Error loading: {ds_id}"

    # Check again after loading (load may have taken seconds/minutes)
    if is_canceled and is_canceled():
        logging.info("Bulk export: skipping write for '%s/%s' – canceled after load.", expt_name, ds_id)
        del dataset
        gc.collect()
        return None, f"Canceled: {ds_id}"

    out_path: Optional[str] = None
    try:
        out_file = _write_object_export(dataset, expt_name, ds_id, config)
        out_path = str(out_file)
        logging.info("Wrote: %s", out_file)
    except Exception as exc:
        logging.error("Export error for dataset '%s/%s': %s", expt_name, ds_id, exc)
    finally:
        del dataset
        gc.collect()

    return out_path, f"{expt_name} / {ds_id}"


def run_bulk_export(
    config: BulkExportConfig,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    is_canceled: Optional[Callable[[], bool]] = None,
) -> list[str]:
    """Load each selected object and write export xlsx files.

    For **dataset-level** exports each dataset is loaded individually via
    :func:`_load_and_export_dataset_task` so only one dataset's file handles
    are open at a time (preventing *Too many open files* errors).

    When ``config.max_workers > 1``, dataset-level exports are processed in
    parallel using a :class:`~concurrent.futures.ThreadPoolExecutor`.  Each
    worker thread loads and writes one dataset independently, which can
    dramatically reduce total wall-clock time for large experiments.

    For **experiment-level** exports the full experiment is loaded, written,
    then freed before processing the next one (always serial).

    Parameters
    ----------
    config:
        Fully populated :class:`BulkExportConfig`.
    progress_callback:
        Called as ``(current, total, message)`` before loading (so the UI
        label updates immediately) and after writing (to advance the bar).
    is_canceled:
        Callable returning ``True`` if the user requested cancellation.

    Returns
    -------
    list[str]
        Paths of successfully written output files.
    """
    import concurrent.futures
    import gc
    import threading

    from monstim_signals.io.repositories import ExperimentRepository

    written_files: list[str] = []
    total_objects = sum(max(len(ds_ids), 1) for ds_ids in config.selected_objects.values())
    max_workers = max(1, getattr(config, "max_workers", 1))

    # ── Dataset level ─────────────────────────────────────────────────────────
    if config.data_level == "dataset":
        # Flatten to a list of (expt_name, ds_id, expt_folder) tasks,
        # skipping experiments that have no resolved path.
        tasks: list[tuple[str, str, Path]] = []
        for expt_name, ds_ids in config.selected_objects.items():
            expt_path_str = config.experiment_paths.get(expt_name)
            if not expt_path_str:
                logging.error("No path found for experiment '%s' – skipping.", expt_name)
                if progress_callback:
                    progress_callback(len(tasks), total_objects, f"Skipped: {expt_name}")
                continue
            expt_folder = Path(expt_path_str)
            for ds_id in ds_ids:
                tasks.append((expt_name, ds_id, expt_folder))

        if max_workers > 1:
            # ── Parallel ──────────────────────────────────────────────────
            counter_lock = threading.Lock()
            current_ref = [0]

            def _parallel_task(task: tuple[str, str, Path]) -> Optional[str]:
                expt_n, ds, folder = task
                # Bail out immediately if already canceled before we even start
                if is_canceled and is_canceled():
                    return None
                if progress_callback:
                    with counter_lock:
                        pre = current_ref[0]
                    progress_callback(pre, total_objects, f"Loading: {expt_n} / {ds}…")
                out_path, msg = _load_and_export_dataset_task(expt_n, ds, folder, config, is_canceled=is_canceled)
                with counter_lock:
                    current_ref[0] += 1
                    cur = current_ref[0]
                if progress_callback:
                    progress_callback(cur, total_objects, msg)
                return out_path

            # Manage the executor manually so we can cancel_futures on shutdown
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
            future_map: dict = {}
            try:
                for task in tasks:
                    if is_canceled and is_canceled():
                        break
                    future_map[executor.submit(_parallel_task, task)] = task

                for fut in concurrent.futures.as_completed(future_map):
                    if is_canceled and is_canceled():
                        # cancel_futures=True drops queued (not-yet-started) futures;
                        # already-running tasks will complete their current step then
                        # see is_canceled() and exit early.
                        executor.shutdown(wait=False, cancel_futures=True)
                        logging.info("Bulk export: parallel export canceled by user.")
                        break
                    try:
                        result = fut.result()
                        if result:
                            written_files.append(result)
                    except Exception as exc:
                        expt_n, ds, _ = future_map[fut]
                        logging.error("Unhandled error for dataset '%s/%s': %s", expt_n, ds, exc)
            finally:
                # Ensure threads are cleaned up whether we finished, broke, or raised
                executor.shutdown(wait=True)

        else:
            # ── Serial ────────────────────────────────────────────────────
            current = 0
            for expt_name, ds_id, expt_folder in tasks:
                if is_canceled and is_canceled():
                    logging.info("Bulk export canceled by user.")
                    break
                if progress_callback:
                    progress_callback(current, total_objects, f"Loading: {expt_name} / {ds_id}…")
                out_path, msg = _load_and_export_dataset_task(expt_name, ds_id, expt_folder, config, is_canceled=is_canceled)
                if out_path:
                    written_files.append(out_path)
                current += 1
                if progress_callback:
                    progress_callback(current, total_objects, msg)

    # ── Experiment level (always serial) ──────────────────────────────────────
    else:
        current = 0
        for expt_name, ds_ids in config.selected_objects.items():
            if is_canceled and is_canceled():
                logging.info("Bulk export canceled by user.")
                break

            expt_path_str = config.experiment_paths.get(expt_name)
            if not expt_path_str:
                logging.error("No path found for experiment '%s' – skipping.", expt_name)
                current += 1
                if progress_callback:
                    progress_callback(current, total_objects, f"Skipped: {expt_name}")
                continue

            expt_folder = Path(expt_path_str)
            if progress_callback:
                progress_callback(current, total_objects, f"Loading: {expt_name}…")
            try:
                logging.info("Bulk export: loading experiment '%s' from '%s'", expt_name, expt_folder)
                experiment = ExperimentRepository(expt_folder).load(allow_write=False)
            except Exception as exc:
                logging.error("Failed to load experiment '%s': %s", expt_name, exc)
                current += 1
                if progress_callback:
                    progress_callback(current, total_objects, f"Error loading: {expt_name}")
                continue
            try:
                out_file = _write_object_export(experiment, expt_name, expt_name, config)
                written_files.append(str(out_file))
                logging.info("Wrote: %s", out_file)
            except Exception as exc:
                logging.error("Export error for experiment '%s': %s", expt_name, exc)
            finally:
                del experiment
                gc.collect()
            current += 1
            if progress_callback:
                progress_callback(current, total_objects, expt_name)

    return written_files


# ─────────────────────────────────────────────────────────────────────────────
# Manager class (GUI-side orchestration)
# ─────────────────────────────────────────────────────────────────────────────


class BulkExportManager:
    """Owned by :class:`MonstimGUI`; surfaces the bulk-export workflow."""

    def __init__(self, gui: "MonstimGUI"):
        self.gui = gui

    def show_bulk_export_dialog(self) -> None:
        """Open the BulkExportDialog and launch the worker on acceptance."""
        from monstim_gui.dialogs.bulk_export_dialog import BulkExportDialog

        dialog = BulkExportDialog(self.gui)
        dialog.exec()
