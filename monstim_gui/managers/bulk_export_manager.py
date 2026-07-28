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

logger = logging.getLogger(__name__)
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

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
    "longform_reflex_amplitudes": "Longform Reflex Amplitudes",
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

BULK_EXPORT_OPEN_FILE_BUDGET = 128
BULK_EXPORT_OPEN_FILE_RESERVE = 32
BULK_EXPORT_MIN_DATASET_FILE_COST = 8


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


def _safe_channel_gain(session, recording, channel_idx: int) -> float:
    """Return the EMG amplifier gain for a channel from row-level metadata."""
    gain_sources = []
    if recording is not None:
        meta = getattr(recording, "meta", None)
        gain_sources.append(getattr(meta, "emg_amp_gains", None))
    gain_sources.append(getattr(session, "emg_amp_gains", None))

    for gains in gain_sources:
        if gains is None:
            continue
        try:
            if channel_idx < len(gains):
                gain = gains[channel_idx]
                return float(gain) if gain is not None else np.nan
        except Exception:
            continue
    return np.nan


def _auto_max_workers(task_count: int, max_dataset_recordings: int = 0) -> int:
    """Return a worker count that leaves GUI and file-handle headroom."""
    if task_count <= 1:
        return 1
    available_cpus = os.cpu_count() or 2
    cpu_workers = max(1, min(task_count, available_cpus - 1))

    file_budget = max(1, BULK_EXPORT_OPEN_FILE_BUDGET - BULK_EXPORT_OPEN_FILE_RESERVE)
    file_workers = max(1, file_budget // BULK_EXPORT_MIN_DATASET_FILE_COST)
    return max(1, min(cpu_workers, file_workers))


def _count_dataset_recording_files(ds_folder: Path) -> int:
    """Estimate dataset size without opening recording files."""
    try:
        return sum(1 for _ in ds_folder.rglob("*.meta.json"))
    except OSError as exc:
        logger.warning("Could not estimate recording count for '%s': %s", ds_folder, exc)
        return 0


def _bulk_export_load_config() -> dict:
    """Repository load options used only by bulk export."""
    from monstim_signals.core import load_config

    config = dict(load_config())
    config["lazy_open_h5"] = True
    config["signal_processing_workers"] = 1
    config["close_raw_after_filter"] = True
    return config


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
                logger.debug("M-max unavailable for ch=%d method=%s: %s", ch_idx, method, exc)
                cache[(ch_idx, method)] = None
    return cache


def _iter_object_datasets(obj):
    """Yield dataset-like objects from either a Dataset or Experiment."""
    datasets = getattr(obj, "datasets", None)
    if datasets is not None:
        for dataset in datasets:
            yield dataset
    else:
        yield obj


def _compute_longform_reflex_amplitudes(obj, config: BulkExportConfig) -> pd.DataFrame:
    """Build one row per active recording/channel/window/method amplitude.

    This export preserves recording-level values for downstream mixed-effects
    models. It intentionally does not bin or average amplitudes; the binned
    stimulus column is included only as an optional modeling/grouping helper.
    """
    mmax_cache = _get_mmax_cache(obj, config) if config.normalize_to_mmax else {}
    rows: list[dict] = []

    for dataset in _iter_object_datasets(obj):
        dataset_id = getattr(dataset, "id", "")
        dataset_date = getattr(dataset, "date", "")
        animal_id = getattr(dataset, "animal_id", "")
        condition = getattr(dataset, "condition", "")
        bin_size = getattr(dataset, "bin_size", getattr(obj, "bin_size", np.nan))
        try:
            window_names = dataset.unique_latency_window_names()
        except Exception:
            window_names = [getattr(window, "name", "") for window in getattr(dataset, "latency_windows", [])]

        for session in getattr(dataset, "sessions", []):
            session_id = getattr(session, "id", "")
            active_recordings = list(getattr(session, "recordings", []))
            stimulus_values = np.asarray(getattr(session, "stimulus_voltages", []), dtype=float)
            if bin_size and not pd.isna(bin_size):
                binned_values = np.round(stimulus_values / bin_size) * bin_size
            else:
                binned_values = np.full(len(stimulus_values), np.nan)

            for ch_idx in config.channel_indices:
                if ch_idx >= getattr(session, "num_channels", 0):
                    logger.debug(
                        "_compute_longform_reflex_amplitudes: channel index %d out of range for session %s - skipped.",
                        ch_idx,
                        session_id,
                    )
                    continue
                ch_name = _safe_channel_name(session, ch_idx)

                for window_name in window_names:
                    try:
                        latency_window = dataset.get_session_latency_window(session, window_name)
                    except Exception:
                        try:
                            latency_window = session.get_latency_window(window_name)
                        except Exception:
                            latency_window = None
                    if latency_window is None:
                        continue

                    try:
                        window_start_ms = latency_window.start_times[ch_idx]
                        window_end_ms = latency_window.end_times[ch_idx]
                        window_duration_ms = latency_window.durations[ch_idx]
                    except Exception:
                        window_start_ms = np.nan
                        window_end_ms = np.nan
                        window_duration_ms = np.nan

                    for method in config.methods:
                        try:
                            amplitudes = np.asarray(
                                session.get_lw_reflex_amplitudes(method, ch_idx, window_name),
                                dtype=float,
                            )
                        except Exception as exc:
                            logger.warning(
                                "longform_reflex_amplitudes error dataset=%s session=%s ch=%s window=%s method=%s: %s",
                                dataset_id,
                                session_id,
                                ch_name,
                                window_name,
                                method,
                                exc,
                            )
                            continue

                        mmax = mmax_cache.get((ch_idx, method)) if config.normalize_to_mmax else None
                        for rec_idx, amplitude in enumerate(amplitudes):
                            recording = active_recordings[rec_idx] if rec_idx < len(active_recordings) else None
                            stimulus_value = stimulus_values[rec_idx] if rec_idx < len(stimulus_values) else np.nan
                            binned_value = binned_values[rec_idx] if rec_idx < len(binned_values) else np.nan
                            emg_amp_gain = _safe_channel_gain(session, recording, ch_idx)
                            row = {
                                "dataset_id": dataset_id,
                                "dataset_date": dataset_date,
                                "animal_id": animal_id,
                                "condition": condition,
                                "session_id": session_id,
                                "recording_id": getattr(recording, "id", "") if recording is not None else "",
                                "recording_index": rec_idx,
                                "stimulus_value": stimulus_value,
                                "stimulus_binned": binned_value,
                                "channel_index": ch_idx,
                                "channel": ch_name,
                                "emg_amp_gain": emg_amp_gain,
                                "window": window_name,
                                "window_start_ms": window_start_ms,
                                "window_end_ms": window_end_ms,
                                "window_duration_ms": window_duration_ms,
                                "method": method,
                                "amplitude": amplitude,
                            }
                            if config.normalize_to_mmax:
                                row["mmax_for_normalization"] = mmax
                                row["amplitude_norm_mmax"] = amplitude / mmax if mmax is not None and mmax != 0.0 else np.nan
                            rows.append(row)

    return pd.DataFrame(rows)


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
        logger.warning("No latency windows found – skipping avg_reflex_curves.")
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
                            logger.warning(
                                "M-max unavailable or zero for ch=%s method=%s – normalized columns skipped.",
                                ch_name,
                                method,
                            )
                except Exception as exc:
                    logger.warning(
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
            logger.debug("_compute_mmax: channel index %d out of range (%d) – skipped.", ch_idx, n_channels)
            continue
        ch_name = _safe_channel_name(obj, ch_idx)
        row: dict = {"channel": ch_name, "channel_index": ch_idx}
        for method in config.methods:
            try:
                mmax, mthresh = obj.get_avg_m_max(method, ch_idx, return_avg_mmax_thresholds=True)
                row[f"mmax_{method}"] = mmax
                row[f"mmax_threshold_{method}"] = mthresh
            except Exception as exc:
                logger.warning("mmax error ch=%s method=%s: %s", ch_name, method, exc)
                row[f"mmax_{method}"] = None
                row[f"mmax_threshold_{method}"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def _compute_max_h(obj, config: BulkExportConfig) -> pd.DataFrame:
    """Build a DataFrame of average H-reflex amplitudes across stimulus voltages."""
    try:
        voltages = obj.stimulus_voltages
    except Exception:
        logger.warning("Could not obtain stimulus_voltages – skipping max_h.")
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
            logger.debug("_compute_max_h: channel index %d out of range (%d) – skipped.", ch_idx, n_channels)
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
                            logger.warning(
                                "M-max unavailable or zero for ch=%s method=%s – normalized columns skipped.",
                                ch_name,
                                method,
                            )
                    except Exception as exc:
                        logger.warning("M-max lookup failed for ch=%s method=%s: %s", ch_name, method, exc)
            except Exception as exc:
                logger.warning("max_h error ch=%s method=%s: %s", ch_name, method, exc)

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
    "longform_reflex_amplitudes": _compute_longform_reflex_amplitudes,
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
            logger.warning("Unknown data type '%s' – skipped.", data_type)
            continue
        try:
            df = handler(obj, config)
        except Exception as exc:
            logger.error("Error computing '%s' for '%s/%s': %s", data_type, expt_name, obj_id, exc)
            df = pd.DataFrame()
        if df is not None and not df.empty:
            sheet_name = DATA_TYPE_LABELS.get(data_type, data_type)[:31]  # Excel sheet name limit
            sheets.append((sheet_name, df))
        else:
            logger.debug("No data for type '%s' in '%s/%s' – sheet skipped.", data_type, expt_name, obj_id)

    if not sheets:
        logger.warning("No data written for '%s/%s' – file not created.", expt_name, obj_id)
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
        logger.error("Dataset folder not found for '%s/%s' at '%s' – skipping.", expt_name, ds_id, ds_folder)
        return None, f"Not found: {ds_id}"

    # Check before starting the (potentially slow) load
    if is_canceled and is_canceled():
        logger.info("Bulk export: skipping dataset '%s/%s' – canceled.", expt_name, ds_id)
        return None, f"Canceled: {ds_id}"

    try:
        logger.info("Bulk export: loading dataset '%s/%s'", expt_name, ds_id)
        dataset = DatasetRepository(ds_folder).load(
            config=_bulk_export_load_config(),
            lazy_open_h5=True,
            allow_write=False,
        )
    except Exception as exc:
        logger.error("Failed to load dataset '%s/%s': %s", expt_name, ds_id, exc)
        return None, f"Error loading: {ds_id}"

    # Check again after loading (load may have taken seconds/minutes)
    if is_canceled and is_canceled():
        logger.info("Bulk export: skipping write for '%s/%s' – canceled after load.", expt_name, ds_id)
        try:
            close = getattr(dataset, "close", None)
            if callable(close):
                close()
        except Exception as exc:
            logger.debug("Bulk export: failed to close canceled dataset '%s/%s': %s", expt_name, ds_id, exc)
        del dataset
        gc.collect()
        return None, f"Canceled: {ds_id}"

    out_path: Optional[str] = None
    try:
        out_file = _write_object_export(dataset, expt_name, ds_id, config)
        out_path = str(out_file)
        logger.info("Wrote: %s", out_file)
    except Exception as exc:
        logger.error("Export error for dataset '%s/%s': %s", expt_name, ds_id, exc)
    finally:
        try:
            close = getattr(dataset, "close", None)
            if callable(close):
                close()
        except Exception as exc:
            logger.debug("Bulk export: failed to close dataset '%s/%s': %s", expt_name, ds_id, exc)
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
    :func:`_load_and_export_dataset_task` and worker count is capped by an
    estimated open-file budget.

    Dataset-level exports are processed in parallel using a
    :class:`~concurrent.futures.ThreadPoolExecutor` sized from the available CPU
    count while leaving one CPU for the GUI. Each worker thread loads and writes
    one dataset independently, which can dramatically reduce total wall-clock
    time for large experiments.

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

    # ── Dataset level ─────────────────────────────────────────────────────────
    if config.data_level == "dataset":
        # Flatten to a list of (expt_name, ds_id, expt_folder) tasks,
        # skipping experiments that have no resolved path.
        tasks: list[tuple[str, str, Path]] = []
        max_dataset_recordings = 0
        for expt_name, ds_ids in config.selected_objects.items():
            expt_path_str = config.experiment_paths.get(expt_name)
            if not expt_path_str:
                logger.error("No path found for experiment '%s' – skipping.", expt_name)
                if progress_callback:
                    progress_callback(len(tasks), total_objects, f"Skipped: {expt_name}")
                continue
            expt_folder = Path(expt_path_str)
            for ds_id in ds_ids:
                ds_folder = expt_folder / ds_id
                max_dataset_recordings = max(max_dataset_recordings, _count_dataset_recording_files(ds_folder))
                tasks.append((expt_name, ds_id, expt_folder))

        max_workers = _auto_max_workers(len(tasks), max_dataset_recordings=max_dataset_recordings)
        logger.info(
            "Bulk export: using %d dataset worker(s) for %d task(s); largest selected dataset has %d recording(s).",
            max_workers,
            len(tasks),
            max_dataset_recordings,
        )

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
                        logger.info("Bulk export: parallel export canceled by user.")
                        break
                    try:
                        result = fut.result()
                        if result:
                            written_files.append(result)
                    except Exception as exc:
                        expt_n, ds, _ = future_map[fut]
                        logger.error("Unhandled error for dataset '%s/%s': %s", expt_n, ds, exc)
            finally:
                # Ensure threads are cleaned up whether we finished, broke, or raised
                executor.shutdown(wait=True)

        else:
            # ── Serial ────────────────────────────────────────────────────
            current = 0
            for expt_name, ds_id, expt_folder in tasks:
                if is_canceled and is_canceled():
                    logger.info("Bulk export canceled by user.")
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
                logger.info("Bulk export canceled by user.")
                break

            expt_path_str = config.experiment_paths.get(expt_name)
            if not expt_path_str:
                logger.error("No path found for experiment '%s' – skipping.", expt_name)
                current += 1
                if progress_callback:
                    progress_callback(current, total_objects, f"Skipped: {expt_name}")
                continue

            expt_folder = Path(expt_path_str)
            if progress_callback:
                progress_callback(current, total_objects, f"Loading: {expt_name}…")
            try:
                logger.info("Bulk export: loading experiment '%s' from '%s'", expt_name, expt_folder)
                experiment = ExperimentRepository(expt_folder).load(
                    config=_bulk_export_load_config(),
                    lazy_open_h5=True,
                    allow_write=False,
                )
            except Exception as exc:
                logger.error("Failed to load experiment '%s': %s", expt_name, exc)
                current += 1
                if progress_callback:
                    progress_callback(current, total_objects, f"Error loading: {expt_name}")
                continue
            try:
                out_file = _write_object_export(experiment, expt_name, expt_name, config)
                written_files.append(str(out_file))
                logger.info("Wrote: %s", out_file)
            except Exception as exc:
                logger.error("Export error for experiment '%s': %s", expt_name, exc)
            finally:
                try:
                    close = getattr(experiment, "close", None)
                    if callable(close):
                        close()
                except Exception as exc:
                    logger.debug("Bulk export: failed to close experiment '%s': %s", expt_name, exc)
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
