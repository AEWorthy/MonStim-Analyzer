# monstim_signals/domain/session.py
import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from itertools import pairwise
from threading import RLock
from typing import TYPE_CHECKING, Any

import numpy as np

from monstim_signals.core import ConfigChange, LatencyWindow, LatencyWindowNotFoundError, ResolvedConfig, SessionAnnot, StimCluster, load_config
from monstim_signals.core.configuration import deep_merge
from monstim_signals.domain.recording import Recording
from monstim_signals.plotting import SessionPlotterPyQtGraph
from monstim_signals.transform import (
    NoCalculableMmaxError,
    butter_bandpass_filter,
    calculate_emg_amplitude,  # noqa: F401 - re-exported for compatibility with callers/tests
    calculate_window_amplitude_results,
    correct_emg_to_baseline,
    get_avg_mmax,
)

if TYPE_CHECKING:
    from monstim_signals.domain.dataset import Dataset
    from monstim_signals.io.repositories import SessionRepository


logger = logging.getLogger(__name__)

_M_WAVE_WINDOW_NAMES = frozenset({"m-wave", "m_wave", "m wave", "mwave", "m-response", "m_response", "m response"})
_H_REFLEX_WINDOW_NAMES = frozenset(
    {"h-wave", "h_wave", "h wave", "hwave", "h-reflex", "h_reflex", "h reflex", "hresponse", "h_response", "h response"}
)


@dataclass(frozen=True)
class WindowAmplitudeSeries:
    """Results for one configured latency window, aligned to recording_ids."""

    window_index: int
    window: LatencyWindow
    priority_rank: int | None
    recording_ids: tuple[str, ...]
    results: tuple[object, ...]


class Session:
    """
    A collection of multiple Recordings, each at a different stimulus amplitude,
    all belonging to one “session” (animal & date).
    """

    def __init__(
        self,
        session_id: str,
        recordings: list[Recording],
        annot: SessionAnnot,
        repo: SessionRepository | None = None,
        config: ResolvedConfig | dict | None = None,
    ):
        self.id: str = session_id
        self._all_recordings: list[Recording] = recordings
        self.annot: SessionAnnot = annot
        self.repo: SessionRepository = repo
        self.parent_dataset: Dataset | None = None
        self._config = config if isinstance(config, ResolvedConfig) else ResolvedConfig(deep_merge(load_config(), config or {}))
        self._cache_lock = RLock()
        self._signal_revision = 0
        self._window_revision = 0
        self._selection_revision = 0
        self._analysis_revision = 0
        self._signal_caches: dict[str, dict[str, np.ndarray]] = {
            "raw": {},
            "filtered": {},
            "rectified_raw": {},
            "rectified_filtered": {},
        }
        self._signal_list_cache: dict[str, tuple[np.ndarray, ...]] = {}
        self._signal_inflight: dict[tuple[int, str, str], Future[np.ndarray]] = {}
        self._window_result_cache: dict[tuple[str, int], tuple[object, tuple[WindowAmplitudeSeries, ...]]] = {}
        self._latency_window_amplitude_cache: dict[tuple[str, int], tuple[object, tuple[np.ndarray, ...]]] = {}
        self._mmax_cache: dict[tuple[str, int], tuple[object, tuple[Any, Any, Any]]] = {}
        self._distribution_cache: dict[object, tuple[object, object]] = {}
        self._load_config_settings()

        # Load session parameters from recordings
        self._load_session_parameters()
        self._initialize_annotations()

        self.plotter = SessionPlotterPyQtGraph(self)
        self.update_latency_window_parameters()
        self.__check_recording_consistency()

    @property
    def is_completed(self) -> bool:
        return getattr(self.annot, "is_completed", False)

    @is_completed.setter
    def is_completed(self, value: bool) -> None:
        self.annot.is_completed = bool(value)
        if self.repo is not None:
            self.repo.save(self)

    def _load_config_settings(self):
        _config = self._config
        self.time_window_ms: float = _config["time_window"]
        self.pre_stim_time_ms: float = _config["pre_stim_time"]
        self.bin_size: float = _config["bin_size"]
        self.latency_window_style: str = _config["latency_window_style"]
        self.m_color: str = _config["m_color"]
        self.h_color: str = _config["h_color"]
        self.title_font_size: int = _config["title_font_size"]
        self.axis_label_font_size: int = _config["axis_label_font_size"]
        self.tick_font_size: int = _config["tick_font_size"]
        self.subplot_adjust_args = _config["subplot_adjust_args"]
        self.m_max_args = _config["m_max_args"]
        self.butter_filter_args = _config["butter_filter_args"]
        self.default_method: str = _config["default_method"]
        self.default_channel_names: list[str] = _config.get("default_channel_names", [])

    def _load_session_parameters(self):
        # ---------- Pull session-wide parameters from the first recording's meta ----------
        # Use all_recordings (including excluded) so session can load even with all recordings excluded
        if self.all_recordings:
            first_meta = self.all_recordings[0].meta
            self.formatted_name = self.id
            self.scan_rate = first_meta.scan_rate  # Hz
            self.num_samples = first_meta.num_samples  # samples/channel
            self.num_channels = first_meta.num_channels  # number of channels
            self._channel_types: list[str] = first_meta.channel_types.copy()  # list of channel types

            # Stimulus parameters
            self.stim_clusters: list[StimCluster] = first_meta.stim_clusters.copy()  # list of StimCluster objects
            self.primary_stim: StimCluster = getattr(first_meta, "primary_stim", None)  # the primary StimCluster for this session
            if self.primary_stim is None:
                logger.warning(f"Session {self.id} does not have a primary stimulus defined. Defaulting to the first StimCluster.")
                self.primary_stim = self.stim_clusters[0] if self.stim_clusters else None
                if self.primary_stim is None:
                    logger.error(f"Session {self.id} has no StimClusters defined. Cannot determine primary stimulus.")
                    raise ValueError(f"Session {self.id} has no StimClusters defined. Cannot determine primary stimulus.")
            self.pre_stim_acquired = first_meta.pre_stim_acquired
            self.post_stim_acquired = first_meta.post_stim_acquired
            self.stim_delay = self.primary_stim.stim_delay  # in ms, delay
            self.stim_duration = self.primary_stim.stim_duration
            self.stim_start: float = self.stim_delay + self.pre_stim_acquired

            # Parameters that may sometimes be None
            self.recording_interval: float = getattr(first_meta, "recording_interval", None)  # in seconds, time between recordings (if applicable)
            self.emg_amp_gains: list[int] = getattr(first_meta, "emg_amp_gains", None)  # default to 1000 if not specified
        else:
            raise ValueError(f"Session {self.id} has no recordings associated with it.")

    def _initialize_annotations(self):
        # Check in case of empty list annot
        if len(self.annot.channels) != self.num_channels:
            from monstim_signals.core import SignalChannel

            logger.warning(
                f"Session {self.id} has {len(self.annot.channels)} channels in annot, but {self.num_channels} channels in recordings."
                " Reinitializing channel annotations."
            )
            self.annot.channels = [
                SignalChannel(
                    name=(self.default_channel_names[i] if i < len(self.default_channel_names) else f"Channel {i + 1}"),
                    invert=False,
                    type_override=None,
                )
                for i in range(self.num_channels)
            ]
        self.channel_names = [self.annot.channels[i].name for i in range(self.num_channels)]
        self.channel_units = [self.annot.channels[i].unit for i in range(self.num_channels)]
        self.channel_types = [
            (
                self.annot.channels[i].type_override
                if self.annot.channels[i].type_override is not None
                else (self._channel_types[i] if i < len(self._channel_types) else "SIGNAL")
            )
            for i in range(self.num_channels)
        ]

    # TODO: Latency window UX
    # - Consider adding an automated latency-window suggestion routine that
    #   detects candidate M-wave/H-reflex windows from averaged or median
    #   traces and prompts the user to accept/modify them.
    # - The current Jupyter-only `update_window_settings` helper should be
    #   integrated into the main GUI latency editor so editing is consistent
    #   across environments.

    def apply_config(self, changes: ConfigChange = ConfigChange.PLOT) -> None:
        """
        Apply the loaded configuration settings to the session.
        This is called after loading the session or when preferences are changed.

        Note: This method does NOT automatically persist changes to disk.
        Callers must explicitly call save() if persistence is required.
        """
        self._load_config_settings()  # Reload config settings to ensure they are up-to-date

        # Plotters read ordinary style values from the session at call time.  Keep
        # the existing plotter so reselecting an equal profile is a true no-op.
        for window in self.latency_windows:
            window.linestyle = self.latency_window_style
            window.color = self.m_color if window.name == "M-wave" else window.color
            window.color = self.h_color if window.name == "H-reflex" else window.color

        if changes & ConfigChange.SIGNAL:
            self.invalidate_signal_data()
        elif changes & ConfigChange.ANALYSIS:
            self.invalidate_analysis_results()

    @property
    def num_recordings(self) -> int:
        return len(self.recordings)

    @property
    def num_all_recordings(self) -> int:
        return len(self.all_recordings)

    @property
    def latency_windows(self) -> list[LatencyWindow]:
        """
        Return the list of latency windows defined in the session annotations.
        """
        return self.annot.latency_windows

    # ------------------------------------------------------------------
    # Low-level consistency checks
    # ------------------------------------------------------------------
    def __check_recording_consistency(self) -> None:
        """Check that recordings within the session share key acquisition parameters.

        Populates internal warning list (for GUI notices) rather than raising.
        Currently checks:
          - scan_rate uniformity
          - num_channels uniformity
          - stim_start / stim_delay consistency
          - stimulus voltage monotonicity & duplicates
        """
        warnings: list[str] = []
        try:
            if not self.recordings:
                return
            first = self.recordings[0].meta
            for rec in self.recordings[1:]:
                m = rec.meta
                if m.scan_rate != first.scan_rate:
                    warnings.append(f"Recording {rec.id} scan_rate {m.scan_rate} != {first.scan_rate}.")
                if m.num_channels != first.num_channels:
                    warnings.append(f"Recording {rec.id} num_channels {m.num_channels} != {first.num_channels}.")
                # Stim delay / start relative metrics
                if hasattr(m, "primary_stim") and hasattr(first, "primary_stim") and m.primary_stim.stim_delay != first.primary_stim.stim_delay:
                    warnings.append(f"Recording {rec.id} stim_delay {m.primary_stim.stim_delay} != {first.primary_stim.stim_delay}.")
            # Stimulus voltage issues
            volts = [r.meta.primary_stim.stim_v for r in self.recordings if getattr(r.meta, "primary_stim", None)]
            if volts:
                # Duplicates
                from collections import Counter

                dupes = [v for v, c in Counter(volts).items() if c > 1]
                if dupes:
                    # warnings.append(f"Duplicate stimulus voltages detected: {sorted(set(dupes))}.")
                    pass  # do nothing; duplicates are allowed
                # Monotonicity expectation (optional): should be non-decreasing sequence
                if any(b < a for a, b in pairwise(volts)):
                    warnings.append("Stimulus voltages are not sorted non-decreasing.")
        finally:
            # Store for notice system
            self._consistency_warnings = warnings
            for w in warnings:
                logger.warning(f"Session {self.id} consistency: {w}")

    # ------------------------------------------------------------------
    # Latency window helper methods
    # ------------------------------------------------------------------
    def add_latency_window(
        self,
        name: str,
        start_times: list[float],
        durations: list[float],
        color: str | None = None,
        linestyle: str | None = None,
        *,
        persist: bool = True,
    ) -> None:
        """Add a new :class:`LatencyWindow` to the session.

        Args:
            name (str): The name of the latency window.
            start_times (list[float]): A list of start times for the latency window.
            durations (list[float]): A list of durations for the latency window.
            color (str | None): The color of the latency window. If None, uses the default color.
            linestyle (str | None): The line style of the latency window. If None, uses the default line style.
            persist (bool): If True, save the session annotation file after adding the latency window.
                You may want to delay persistence if adding multiple latency windows at once.
                Use from SessionRepository.save_many() to persist all sessions at once for efficiency.
        """
        window = LatencyWindow(
            name=name,
            start_times=start_times,
            durations=durations,
            color=color or self.m_color,
            linestyle=linestyle or self.latency_window_style,
        )
        self.annot.latency_windows.append(window)
        self.invalidate_window_results()
        if persist and self.repo is not None:
            self.repo.save(self)

    def apply_latency_window_preset(self, preset_name: str, *, persist: bool = True) -> None:
        """Replace latency windows using a preset defined in the config file.

        Optional Args:
            preset_name (str): The name of the preset to apply.
            persist (bool): If True, save the session annotation file after applying the preset.
                You may want to delay persistence if applying the preset to multiple sessions at once.
                Use from SessionRepository.save_many() to persist all sessions at once for efficiency.
        """
        presets = self._config.presets.latency_window_presets
        if preset_name not in presets:
            logger.warning(f"Preset '{preset_name}' not found in config.")
            return

        self.annot.latency_windows = []
        num_channels = self.num_channels
        for win in presets[preset_name]:
            window = LatencyWindow(
                name=win.get("name", "Window"),
                start_times=[float(win.get("start", 0.0))] * num_channels,
                durations=[float(win.get("duration", 1.0))] * num_channels,
                color=win.get("color", self.m_color),
                linestyle=win.get("linestyle", self.latency_window_style),
            )
            self.annot.latency_windows.append(window)

        self.invalidate_window_results()
        if persist and self.repo is not None:
            self.repo.save(self)

    def remove_latency_window(self, name: str, *, persist: bool = True) -> None:
        """Remove a latency window by name."""
        self.annot.latency_windows = [w for w in self.annot.latency_windows if w.name != name]
        self.invalidate_window_results()
        if persist and self.repo is not None:
            self.repo.save(self)

    def get_latency_window(self, name: str) -> LatencyWindow | None:
        for w in self.latency_windows:
            if w.name == name:
                return w
        return None

    @property
    def excluded_recordings(self):
        return set(self.annot.excluded_recordings)

    @property
    def stimulus_voltages(self) -> np.ndarray:
        """
        Return a list of stimulus voltages for each active recording in the session.
        This assumes that each recording's primary cluster stim_v is the amplitude for that recording.
        """
        return np.array([rec.meta.primary_stim.stim_v for rec in self.recordings])

    @property
    def all_stimulus_voltages(self) -> np.ndarray:
        """
        Return a list of stimulus voltages for all recordings in the session (including excluded).
        This assumes that each recording's primary cluster stim_v is the amplitude for that recording.
        """
        return np.array([rec.meta.primary_stim.stim_v for rec in self.all_recordings])

    @property
    def recordings(self) -> list[Recording]:
        """
        Return a list of active recordings in the session.
        This filters out any recordings that are marked as excluded in the session annotations.
        """
        return self.get_all_recordings(include_excluded=False)

    @property
    def all_recordings(self) -> list[Recording]:
        """
        Return a list of all recordings in the session, including excluded ones.
        """
        return self.get_all_recordings(include_excluded=True)

    def get_all_recordings(self, include_excluded: bool = False) -> list[Recording]:
        """
        Return a list of recordings in the session.

        Args:
            include_excluded (bool): If True, returns all recordings including excluded ones.
                                   If False, returns only active (non-excluded) recordings.

        Returns:
            list[Recording]: The list of recordings based on the include_excluded parameter.
        """
        if include_excluded:
            return self._all_recordings
        else:
            return [rec for rec in self._all_recordings if rec.id not in self.excluded_recordings]

    # ------------------------------------------------------------------
    # Diagnostic / notice helpers (queried by GUI)
    # ------------------------------------------------------------------
    def collect_notices(self) -> list[dict[str, str]]:
        """Return structured session-level notices for GUI warning/info icons.

        Codes:
          - inconsistent_scan_rate
          - inconsistent_num_channels
          - inconsistent_stim_delay
          - duplicate_stim_voltages
          - unsorted_stim_voltages
          - heterogeneous_latency_windows (session-specific: overlapping or zero-duration windows)
        """
        notices: list[dict[str, str]] = []
        try:
            # Consistency warnings captured earlier
            for msg in getattr(self, "_consistency_warnings", []):
                code = "generic_consistency"
                if "scan_rate" in msg:
                    code = "inconsistent_scan_rate"
                elif "num_channels" in msg:
                    code = "inconsistent_num_channels"
                elif "stim_delay" in msg:
                    code = "inconsistent_stim_delay"
                elif "Duplicate stimulus voltages" in msg:
                    code = "duplicate_stim_voltages"
                elif "not sorted" in msg:
                    code = "unsorted_stim_voltages"
                notices.append({"level": "warning", "code": code, "message": msg})

            # Latency window sanity checks (per-session)
            for w in self.latency_windows:
                for ch, (_start, dur) in enumerate(zip(w.start_times, w.durations, strict=True)):
                    if dur <= 0:
                        notices.append(
                            {
                                "level": "warning",
                                "code": "zero_or_negative_window",
                                "message": f"Latency window '{w.name}' channel {ch} has non-positive duration {dur}.",
                            }
                        )
            # Missing canonical M-wave window
            if not self.has_m_wave_window():
                notices.append(
                    {
                        "level": "info",
                        "code": "missing_m_wave_window",
                        "message": "Session is missing an M-wave latency window.",
                    }
                )

            # No active recordings
            if len(self.recordings) == 0:
                notices.append(
                    {
                        "level": "warning",
                        "code": "no_active_recordings",
                        "message": "Session has no active recordings.",
                    }
                )

            # Window bounds validation
            total_window_ms = self.time_window_ms  # configured acquisition window
            for w in self.latency_windows:
                for ch, (start, dur) in enumerate(zip(w.start_times, w.durations, strict=True)):
                    if start < 0 or (start + dur) > total_window_ms:
                        notices.append(
                            {
                                "level": "warning",
                                "code": "window_out_of_bounds",
                                "message": f"Window '{w.name}' channel {ch} exceeds acquisition bounds (start={start}, dur={dur}).",
                            }
                        )

            # Excessive overlap detection (replace previous simple overlap notice)
            overlap_threshold = 0.5  # 50% of the shorter window
            for ch in range(self.num_channels):
                spans = []
                for w in self.latency_windows:
                    if ch < len(w.start_times):
                        spans.append((w.name, w.start_times[ch], w.start_times[ch] + w.durations[ch]))
                spans.sort(key=lambda x: x[1])
                for i in range(len(spans)):
                    n1, s1, e1 = spans[i]
                    for j in range(i + 1, len(spans)):
                        n2, s2, e2 = spans[j]
                        if s2 >= e1:
                            break  # since sorted by start
                        overlap = min(e1, e2) - max(s1, s2)
                        if overlap > 0:
                            len1 = e1 - s1
                            len2 = e2 - s2
                            shorter = min(len1, len2) or 1.0
                            if (overlap / shorter) >= overlap_threshold:
                                notices.append(
                                    {
                                        "level": "info",
                                        "code": "excessive_window_overlap",
                                        "message": f"Windows '{n1}' and '{n2}' overlap >50% on channel {ch}.",
                                    }
                                )
        except Exception as e:
            logger.debug(f"Notice collection error (session {self.id}): {e}")
        return notices

    def _filter_active(self, source_list: list[Any]) -> list[Any]:
        """
        Helper to filter a list of data corresponding to all_recordings,
        returning only items corresponding to active (non-excluded) recordings.
        """
        excluded = self.excluded_recordings
        return [item for i, item in enumerate(source_list) if self.all_recordings[i].id not in excluded]

    def has_m_wave_window(self) -> bool:
        """Check if the session has a defined M-wave latency window."""
        return self._find_reflex_latency_window(_M_WAVE_WINDOW_NAMES) is not None

    def has_h_reflex_window(self) -> bool:
        """Check if the session has a defined H-reflex latency window."""
        return self._find_reflex_latency_window(_H_REFLEX_WINDOW_NAMES) is not None

    def _find_reflex_latency_window(self, names: frozenset[str]) -> LatencyWindow | None:
        """Return the configured window matching one of a reflex's canonical aliases."""
        return next((window for window in self.latency_windows if (window.name or "").casefold() in names), None)

    # ──────────────────────────────────────────────────────────────────
    # 0) Dependency-aware signal and result caches
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _readonly(array: np.ndarray) -> np.ndarray:
        value = np.asarray(array)
        value.setflags(write=False)
        return value

    def _ensure_cache_state(self) -> None:
        """Initialize cache state for lightweight test/domain objects made via ``__new__``."""
        if hasattr(self, "_cache_lock"):
            return
        self._cache_lock = RLock()
        self._signal_revision = self._window_revision = self._selection_revision = self._analysis_revision = 0
        self._signal_caches = {"raw": {}, "filtered": {}, "rectified_raw": {}, "rectified_filtered": {}}
        self._signal_list_cache = {}
        self._signal_inflight = {}
        self._window_result_cache = {}
        self._latency_window_amplitude_cache = {}
        self._mmax_cache = {}
        self._distribution_cache = {}

    @property
    def cache_token(self) -> tuple[int, int, int, int]:
        """Revision token used by parent aggregate caches."""
        self._ensure_cache_state()
        return (self._signal_revision, self._window_revision, self._selection_revision, self._analysis_revision)

    def _compute_signal_recording(self, rec: Recording, kind: str) -> np.ndarray:
        if kind in {"rectified_raw", "rectified_filtered"}:
            base_kind = "raw" if kind == "rectified_raw" else "filtered"
            return self._readonly(np.abs(self._get_signal_recording(rec, base_kind)))

        if kind == "raw":
            try:
                raw_data = np.asarray(rec.raw_view())
                result = np.array(raw_data, copy=True)
                for channel in range(rec.meta.num_channels):
                    if self.annot.channels[channel].invert:
                        result[:, channel] *= -1.0
                return self._readonly(result)
            finally:
                rec.close()

        if kind != "filtered":
            raise ValueError(f"Unknown signal cache kind: {kind}")
        # Filtering depends on the cached, polarity-corrected raw copy. This
        # guarantees that a later raw plot never reopens a recording already
        # read for filtering, and raw/filtered concurrent requests share I/O.
        raw_data = self._get_signal_recording(rec, "raw")
        filtered_channels = []
        for channel in range(rec.meta.num_channels):
            channel_data = raw_data[:, channel]
            channel_type = self.channel_types[channel].lower()
            if channel_type in ("force", "length"):
                filtered = correct_emg_to_baseline(channel_data, self.scan_rate, self.stim_delay)
            elif channel_type == "emg":
                filtered = butter_bandpass_filter(
                    channel_data,
                    fs=self.scan_rate,
                    lowcut=self.butter_filter_args["lowcut"],
                    highcut=self.butter_filter_args["highcut"],
                    order=self.butter_filter_args["order"],
                )
            else:
                logger.warning(f"No specific processing for channel type: {channel_type}")
                filtered = np.array(channel_data, copy=True)
            filtered_channels.append(filtered)
        return self._readonly(np.column_stack(filtered_channels))

    def _get_signal_recording(self, rec: Recording, kind: str) -> np.ndarray:
        """Return one cached recording, sharing concurrent calculations."""
        self._ensure_cache_state()
        with self._cache_lock:
            cached = self._signal_caches[kind].get(rec.id)
            if cached is not None:
                return cached
            revision = self._signal_revision
            inflight_key = (revision, kind, rec.id)
            future = self._signal_inflight.get(inflight_key)
            owner = future is None
            if owner:
                future = Future()
                self._signal_inflight[inflight_key] = future

        if not owner:
            return future.result()

        try:
            value = self._compute_signal_recording(rec, kind)
            with self._cache_lock:
                if revision == self._signal_revision:
                    self._signal_caches[kind][rec.id] = value
                future.set_result(value)
            return value
        except BaseException as exc:
            with self._cache_lock:
                future.set_exception(exc)
            raise
        finally:
            with self._cache_lock:
                self._signal_inflight.pop(inflight_key, None)

    def _all_signal_recordings(self, kind: str) -> list[np.ndarray]:
        self._ensure_cache_state()
        with self._cache_lock:
            cached = self._signal_list_cache.get(kind)
            if cached is not None:
                return list(cached)
        configured_workers = self._config.get("signal_processing_workers")
        max_workers = max(1, int(configured_workers)) if configured_workers is not None else max(1, (os.cpu_count() or 2) - 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            values = list(executor.map(lambda recording: self._get_signal_recording(recording, kind), self.all_recordings))
        with self._cache_lock:
            if all(recording.id in self._signal_caches[kind] for recording in self.all_recordings):
                self._signal_list_cache[kind] = tuple(values)
        return list(values)

    @property
    def all_recordings_raw(self) -> list[np.ndarray]:
        """
        Return a list of raw data arrays for all recordings (including excluded).
        Each array is of shape (num_samples, num_channels).
        """
        return self._all_signal_recordings("raw")

    @property
    def recordings_raw(self) -> list[np.ndarray]:
        """Return a list of raw data arrays for active recordings only."""
        return self._filter_active(self.all_recordings_raw)

    @property
    def all_recordings_filtered(self) -> list[np.ndarray]:
        """
        Return a list of processed data arrays for all recordings (including excluded).
        Each array is of shape (num_samples, num_channels).
        This applies a butter bandpass filter to the raw data and inverts if
        indicated in the channel annotations in the session annot.json file.
        """

        return self._all_signal_recordings("filtered")

    @property
    def recordings_filtered(self) -> list[np.ndarray]:
        """Return a list of processed data arrays for active recordings only."""
        return self._filter_active(self.all_recordings_filtered)

    @property
    def all_recordings_rectified_raw(self) -> list[np.ndarray]:
        """
        Return a list of rectified raw data arrays for all recordings.
        """
        return self._all_signal_recordings("rectified_raw")

    @property
    def recordings_rectified_raw(self) -> list[np.ndarray]:
        """Return a list of rectified raw data arrays for active recordings only."""
        return self._filter_active(self.all_recordings_rectified_raw)

    @property
    def all_recordings_rectified_filtered(self) -> list[np.ndarray]:
        """
        Return a list of rectified filtered data arrays for all recordings.
        """
        return self._all_signal_recordings("rectified_filtered")

    @property
    def recordings_rectified_filtered(self) -> list[np.ndarray]:
        """Return a list of rectified filtered data arrays for active recordings only."""
        return self._filter_active(self.all_recordings_rectified_filtered)

    def _clear_derived_results(self, *, windows: bool) -> None:
        with self._cache_lock:
            if windows:
                self._window_result_cache.clear()
                self._latency_window_amplitude_cache.clear()
                self._distribution_cache.clear()
            self._mmax_cache.clear()

    def invalidate_signal_data(self) -> None:
        """Invalidate source-dependent arrays and every derived result."""
        self._ensure_cache_state()
        with self._cache_lock:
            self._signal_revision += 1
            for cache in self._signal_caches.values():
                cache.clear()
            self._signal_list_cache.clear()
        self._clear_derived_results(windows=True)

    def invalidate_window_results(self) -> None:
        """Retain signal arrays while invalidating latency-window calculations."""
        self._ensure_cache_state()
        with self._cache_lock:
            self._window_revision += 1
        self._clear_derived_results(windows=True)
        self.update_latency_window_parameters()

    def invalidate_selection_results(self) -> None:
        """Retain all-recording work while invalidating active-selection results."""
        self._ensure_cache_state()
        with self._cache_lock:
            self._selection_revision += 1
            self._latency_window_amplitude_cache.clear()
            self._mmax_cache.clear()
            self._distribution_cache.clear()

    def invalidate_analysis_results(self) -> None:
        """Retain signals/window batches while invalidating analysis aggregates."""
        self._ensure_cache_state()
        with self._cache_lock:
            self._analysis_revision += 1
            self._mmax_cache.clear()
            self._distribution_cache.clear()

    def release_cached_data(self) -> None:
        """Release all in-memory cache entries during close/reload/unload."""
        self._ensure_cache_state()
        self.wait_for_cache_work()
        with self._cache_lock:
            self._signal_revision += 1
            self._window_revision += 1
            self._selection_revision += 1
            self._analysis_revision += 1
            for cache in self._signal_caches.values():
                cache.clear()
            self._signal_list_cache.clear()
            self._window_result_cache.clear()
            self._latency_window_amplitude_cache.clear()
            self._mmax_cache.clear()
            self._distribution_cache.clear()

    def wait_for_cache_work(self) -> None:
        """Join currently published per-recording calculations."""
        self._ensure_cache_state()
        with self._cache_lock:
            pending = tuple(self._signal_inflight.values())
        for future in pending:
            # The original requester receives the calculation error. Release
            # still proceeds so close/reload cannot retain resources.
            with suppress(Exception):
                future.result()

    def update_latency_window_parameters(self):
        """
        Update cached M/H-response parameters from latency windows.
        This remains for backwards compatibility. If no M-wave or H-reflex
        windows exist, the corresponding attributes will be set to empty lists.
        """
        for window in self.latency_windows:
            lname = (window.name or "").casefold()
            if lname in _M_WAVE_WINDOW_NAMES:
                self.m_start = window.start_times
                self.m_duration = window.durations
            elif lname in _H_REFLEX_WINDOW_NAMES:
                self.h_start = window.start_times
                self.h_duration = window.durations

        if not self.has_m_wave_window():
            self.m_start = [0.0] * self.num_channels
            self.m_duration = [0.0] * self.num_channels
        if not self.has_h_reflex_window():
            self.h_start = [0.0] * self.num_channels
            self.h_duration = [0.0] * self.num_channels

    # ──────────────────────────────────────────────────────────────────
    # 1) Properties for GUI & analysis code
    # ──────────────────────────────────────────────────────────────────
    def plot(self, plot_type: str | None = None, **kwargs):
        """
        Plots EMG data from a single session using the specified plot_type.

        Args:
            - plot_type (str): The type of plot to generate. Options include 'emg', 'suspectedH',
                'reflexCurves', 'reflexAverages', and 'mCurvesSmoothened'.
                Plot types are defined in the EMGSessionPlotter class in Plot_EMG.py.
            - channel_names (list): A list of channel names to plot. If None, all channels will be plotted.
            - **kwargs: Additional keyword arguments to pass to the plotting function.

                The most common keyword arguments include:
                - 'data_type' (str): The type of data to plot. Options are 'filtered', 'raw', 'rectified_raw', or 'rectified_filtered'.
                    Default is 'filtered'.
                - 'method' (str): The method to use for calculating the M-wave/reflex amplitudes. Options include
                    'average_rectified', 'rms', 'peak_to_trough', and 'average_unrectified'.
                    Default method is set in config.yml under 'default_method'.
                - 'relative_to_mmax' (bool): Whether to plot the data proportional to the M-wave amplitude (True) or as the
                    actual recorded amplitude (False). Default is False.
                - 'all_flags' (bool): Whether to plot flags at all windows (True) or not (False). Default is False.

                Less common keyword arguments include:
                - 'm_flags' (bool): Whether to plot flags at the M-wave window (True) or not (False). Default is False.
                - 'h_flags' (bool): Whether to plot flags at the H-reflex window (True) or not (False). Default is False.
                - 'h_threshold' (float): The threshold for detecting the H-reflex in the suspectedH plot. Default is 0.3.
                - 'mmax_report' (bool): Whether to print the details of the M-max calculations (True) or not (False). Default is False.
                - 'manual_mmax' (float): The manually set M-wave amplitude to use for plotting the reflex curves. Default is None.

            The Session-level plot menu does not include 'mmax' because there can be only one M-max value per session.
            Use Dataset or Experiment plots for M-max summaries.

        Example Usages:
            # Plot filtered EMG data
                session.plot()

            # Plot raw EMG data with flags at the M-wave and H-reflex windows
            session.plot(data_type='raw', all_flags=True)

            # Plot all EMG data with the M-wave and H-reflex windows highlighted
            session.plot(plot_type='suspectedH')

            # Plot the reflex curves for each channel
            session.plot(plot_type='reflexCurves')
        """
        if plot_type == "mmax":
            raise ValueError("Session-level M-max plots are not supported for Session-level analysis.")

        # Call the appropriate plotting method from the plotter object
        raw_data = getattr(self.plotter, f"plot_{plot_type if plot_type else 'emg'}")(**kwargs)
        return raw_data

    def get_m_max(self, method, channel_index, return_mmax_stim_range=False):
        """
        Calculates the M-max amplitude for a specific channel in the session.

        Args:
            method (str): The method to use for calculating M-wave amplitudes from EMG data.
                Options include 'average_rectified', 'rms', 'peak_to_trough', and 'average_unrectified'.
            channel_index (int): The index of the channel to calculate the M-max amplitude for.

        Returns:
            float: The M-max amplitude for the specified channel.
        """
        self._ensure_cache_state()
        key = (method, channel_index)
        token = (
            self.cache_token,
            tuple(recording.id for recording in self.recordings),
            tuple(sorted(self.m_max_args.items())),
        )
        with self._cache_lock:
            entry = self._mmax_cache.get(key)
        if entry is None or entry[0] != token:
            values = get_avg_mmax(
                self.stimulus_voltages,
                self.get_m_wave_amplitudes(method, channel_index),
                **self.m_max_args,
                return_mmax_stim_range=True,
            )
            cached_values = values if isinstance(values, tuple) else (values, None, None)
            with self._cache_lock:
                self._mmax_cache[key] = (token, cached_values)
        else:
            cached_values = entry[1]
        return cached_values if return_mmax_stim_range else cached_values[0]

    def get_m_wave_amplitudes(self, method, channel_index):
        """Return a list of M-wave amplitudes for each recording."""

        m_window = self._find_reflex_latency_window(_M_WAVE_WINDOW_NAMES)
        if m_window is None:
            raise LatencyWindowNotFoundError(window_name="M-wave", object_type="Session", object_id=self.id)

        return self.get_lw_reflex_amplitudes(method, channel_index, m_window)

    def get_h_wave_amplitudes(self, method, channel_index):
        """Return a list of H-reflex amplitudes for each recording."""
        h_window = self._find_reflex_latency_window(_H_REFLEX_WINDOW_NAMES)
        if h_window is None:
            raise LatencyWindowNotFoundError(window_name="H-reflex", object_type="Session", object_id=self.id)

        return self.get_lw_reflex_amplitudes(method, channel_index, h_window)

    def _window_spans(self, channel_index: int):
        """Build channel-specific absolute spans in configured window order."""
        if not 0 <= channel_index < self.num_channels:
            raise ValueError(f"Invalid channel index {channel_index} for session {self.id}")
        from monstim_signals.transform.extrema import WindowSpan, make_window_span

        raw = []
        for index, window in enumerate(self.latency_windows):
            try:
                start = float(window.start_times[channel_index]) + self.stim_start
                end = float(window.end_times[channel_index]) + self.stim_start
            except IndexError, TypeError, ValueError:
                start = end = float("nan")
            raw.append((index, window, start, end))
        ordered = sorted(
            (item for item in raw if np.isfinite(item[2]) and np.isfinite(item[3])), key=lambda item: (int(item[2] * self.scan_rate / 1000), item[0])
        )
        ranks = {index: rank for rank, (index, _window, _start, _end) in enumerate(ordered)}
        return tuple(
            make_window_span(index, window.name, start, end, self.scan_rate, ranks.get(index))
            if np.isfinite(start) and np.isfinite(end)
            else WindowSpan(index, window.name, None, start, end, -1, -1)
            for index, window, start, end in raw
        )

    def get_all_lw_reflex_amplitude_results(
        self, method: str, channel_index: int, *, include_excluded: bool = False
    ) -> tuple[WindowAmplitudeSeries, ...]:
        """Return every latency-window result, retaining recording/window identity."""
        self._ensure_cache_state()
        key = (method, channel_index)
        window_fingerprint = tuple((window.name, tuple(window.start_times), tuple(window.durations)) for window in self.latency_windows)
        all_ids = tuple(recording.id for recording in self.all_recordings)
        token = (self._signal_revision, self._window_revision, method, channel_index, all_ids, window_fingerprint)
        with self._cache_lock:
            entry = self._window_result_cache.get(key)
        if entry is None or entry[0] != token:
            spans = self._window_spans(channel_index)
            per_window: list[list[object]] = [[] for _ in spans]
            for filtered in self.all_recordings_filtered:
                results = calculate_window_amplitude_results(filtered[:, channel_index], spans, self.scan_rate, method)
                if len(results) != len(spans):
                    raise RuntimeError("Latency-window result batch length mismatch")
                for index, result in enumerate(results):
                    per_window[index].append(result)
            series = tuple(
                WindowAmplitudeSeries(
                    span.window_index,
                    self.latency_windows[span.window_index],
                    span.priority_rank,
                    all_ids,
                    tuple(per_window[index]),
                )
                for index, span in enumerate(spans)
            )
            with self._cache_lock:
                self._window_result_cache[key] = (token, series)
        else:
            series = entry[1]

        # Presentation-only window replacements keep the calculated results but
        # expose the current window objects and styles.
        series = tuple(
            WindowAmplitudeSeries(
                item.window_index,
                self.latency_windows[item.window_index],
                item.priority_rank,
                item.recording_ids,
                item.results,
            )
            for item in series
        )

        if include_excluded:
            return series
        active_ids = tuple(recording.id for recording in self.recordings)
        active = set(active_ids)
        return tuple(
            WindowAmplitudeSeries(
                item.window_index,
                item.window,
                item.priority_rank,
                active_ids,
                tuple(result for recording_id, result in zip(item.recording_ids, item.results, strict=True) if recording_id in active),
            )
            for item in series
        )

    def get_recording_lw_amplitude_results(self, method: str, channel_index: int, recording_id: str) -> tuple[object, ...]:
        """Return all window results for one recording, including excluded recordings."""
        series = self.get_all_lw_reflex_amplitude_results(method, channel_index, include_excluded=True)
        if not series:
            if recording_id in {recording.id for recording in self.all_recordings}:
                return ()
            raise KeyError(f"Recording '{recording_id}' not found in session {self.id}")
        try:
            index = series[0].recording_ids.index(recording_id)
        except ValueError as exc:
            raise KeyError(f"Recording '{recording_id}' not found in session {self.id}") from exc
        return tuple(item.results[index] for item in series)

    def get_lw_reflex_amplitudes(self, method: str, channel_index: int, window: str | LatencyWindow) -> np.ndarray:
        """
        Returns the reflex amplitudes for a specific latency window across all sessions in the dataset.

        The array in the same order as the stimulus voltage of each recording.
        """
        self._ensure_cache_state()
        if isinstance(window, LatencyWindow):
            try:
                window_index = next(index for index, item in enumerate(self.latency_windows) if item is window)
            except StopIteration as exc:
                raise LatencyWindowNotFoundError(window_name=window.name, object_type="Session", object_id=self.id) from exc
        else:
            window_index = next((index for index, item in enumerate(self.latency_windows) if item.name == window), None)
        if window_index is None:
            logger.warning(f"Latency window '{window}' not found.")
            return np.array([])
        # Aggregate Dataset/Experiment plots request one window at a time.  The
        # extrema methods, however, deliberately evaluate all windows together
        # so their overlap/priority semantics remain correct.  Keep just the
        # resulting numeric arrays, rather than the detailed extrema objects,
        # so subsequent window requests reuse that single calculation without
        # a large long-lived memory cost.
        cache_key = (method, channel_index)
        token = (
            self._signal_revision,
            self._window_revision,
            self._selection_revision,
            tuple(recording.id for recording in getattr(self, "_all_recordings", ()) if recording.id not in self.excluded_recordings),
            tuple((item.name, tuple(item.start_times), tuple(item.durations)) for item in self.latency_windows),
        )
        with self._cache_lock:
            entry = self._latency_window_amplitude_cache.get(cache_key)
        if entry is None or entry[0] != token:
            series = self.get_all_lw_reflex_amplitude_results(method, channel_index)
            amplitudes = tuple(self._readonly(np.asarray([result.amplitude for result in item.results], dtype=float)) for item in series)
            with self._cache_lock:
                self._latency_window_amplitude_cache[cache_key] = (token, amplitudes)
        else:
            amplitudes = entry[1]
        # Preserve the previous API's independent array result: callers may
        # safely modify it without corrupting the derived-data cache.
        return amplitudes[window_index].copy()

    def get_lw_distribution(self, method: str, channel_index: int, bins=30, density: bool = False) -> dict[str, object]:
        """Return cached common-bin histogram inputs/results for all windows."""
        self._ensure_cache_state()
        bins_key = int(bins) if isinstance(bins, int) else tuple(np.asarray(bins, dtype=float))
        key = (method, channel_index, bins_key, bool(density))
        token = (
            self.cache_token,
            tuple(recording.id for recording in getattr(self, "_all_recordings", ()) if recording.id not in self.excluded_recordings),
            tuple((window.name, tuple(window.start_times), tuple(window.durations)) for window in self.latency_windows),
        )
        with self._cache_lock:
            entry = self._distribution_cache.get(key)
        if entry is not None and entry[0] == token:
            return {
                "bin_edges": entry[1]["bin_edges"].copy(),
                "bin_centers": entry[1]["bin_centers"].copy(),
                "values": {label: values.copy() for label, values in entry[1]["values"].items()},
            }

        amplitudes: dict[str, np.ndarray] = {}
        for window in self.latency_windows:
            values = self.get_lw_reflex_amplitudes(method, channel_index, window)
            amplitudes[getattr(window, "label", window.name)] = values[np.isfinite(values)]
        nonempty = [values for values in amplitudes.values() if values.size]
        if nonempty:
            all_values = np.concatenate(nonempty)
            edges = np.histogram_bin_edges(all_values, bins=bins) if isinstance(bins, int) else np.asarray(bins, dtype=float)
            centers = (edges[:-1] + edges[1:]) / 2.0
            histograms = {}
            for label, values in amplitudes.items():
                counts, _ = np.histogram(values, bins=edges)
                if density and values.size:
                    counts = counts.astype(float, copy=False) / (values.size * np.diff(edges))
                histograms[label] = self._readonly(counts)
        else:
            edges = np.array([], dtype=float)
            centers = np.array([], dtype=float)
            histograms = {label: np.array([], dtype=float) for label in amplitudes}
        stored = {
            "bin_edges": self._readonly(edges),
            "bin_centers": self._readonly(centers),
            "values": {label: self._readonly(values) for label, values in histograms.items()},
        }
        with self._cache_lock:
            self._distribution_cache[key] = (token, stored)
        return {
            "bin_edges": stored["bin_edges"].copy(),
            "bin_centers": stored["bin_centers"].copy(),
            "values": {label: values.copy() for label, values in stored["values"].items()},
        }

    # ──────────────────────────────────────────────────────────────────
    # 2) User actions that update annot files
    # ──────────────────────────────────────────────────────────────────
    def rename_channels(self, new_names: dict[str, str]):
        # Compute a prospective list of channel names after applying the mapping,
        # and validate uniqueness to avoid duplicate channel names.
        proposed_names = []
        for ch in self.annot.channels:
            proposed_names.append(new_names.get(ch.name, ch.name))

        if len(set(proposed_names)) != len(proposed_names):
            # Find duplicates to report a helpful error
            from collections import Counter

            dupes = [name for name, cnt in Counter(proposed_names).items() if cnt > 1]
            raise ValueError(f"Channel renaming would create duplicate names {dupes} in session '{self.id}'. Aborting rename.")

        # Support renaming when multiple channels share the same name by updating all matches
        for old_name, new_name in new_names.items():
            matched = False
            for ch in self.annot.channels:
                if ch.name == old_name:
                    ch.name = new_name
                    matched = True
            if matched:
                logger.info(f"Renamed channel '{old_name}' to '{new_name}' in session {self.id}.")
            else:
                logger.warning(f"Channel '{old_name}' not found in session {self.id}. No action taken.")
        # Optionally update cached names and save
        self.channel_names = [ch.name for ch in self.annot.channels]
        if self.repo is not None:
            self.repo.save(self)

    def invert_channel_polarity(self, channel: int):
        """
        Invert the signal for a specific channel across all recordings in the session.
        This is a user action that modifies the channel's invert flag.
        """
        self.annot.channels[channel].invert = not self.annot.channels[channel].invert
        if self.repo is not None:
            self.repo.save(self)
        self.invalidate_signal_data()

    def change_reflex_latency_windows(self, m_start, m_duration, h_start, h_duration):
        m_window = self.get_latency_window("M-wave")
        if m_window:
            m_window.start_times = m_start
            m_window.durations = m_duration
        h_window = self.get_latency_window("H-reflex")
        if h_window:
            h_window.start_times = h_start
            h_window.durations = h_duration
        self.invalidate_window_results()
        if self.repo is not None:
            self.repo.save(self)

    def include_recording(self, recording_id: str):
        """
        Include a previously excluded recording by its ID.
        If the recording is not found, log a warning.
        """
        if recording_id in self.excluded_recordings:
            for rec in self.get_all_recordings(include_excluded=True):
                if rec.id == recording_id:
                    self.annot.excluded_recordings.remove(recording_id)
                    break
            else:
                logger.warning(f"Recording {recording_id} not found in session {self.id}.")
                return
            self.invalidate_selection_results()
            if self.repo is not None:
                self.repo.save(self)
        else:
            logger.warning(f"Recording {recording_id} is not excluded from session {self.id}. No action taken.")

    def restore_recording(self, recording_id: str):
        """Alias for :meth:`include_recording` for GUI commands."""
        self.include_recording(recording_id)

    def exclude_recording(self, recording_id: str):
        """
        Exclude a recording by its ID.
        If the recording is not found, log a warning.
        """
        if recording_id not in self.excluded_recordings:
            # Find the recording and set its exclude flag
            for rec in self.get_all_recordings(include_excluded=True):
                if rec.id == recording_id:
                    self.annot.excluded_recordings.append(recording_id)
                    break
            else:
                logger.warning(f"Recording {recording_id} not found in session {self.id}.")
                return

            self.invalidate_selection_results()
            if self.repo is not None:
                self.repo.save(self)
        else:
            logger.warning(f"Recording {recording_id} is already excluded in session {self.id}.")

        # Note: We no longer auto-exclude sessions when all recordings are excluded.
        # This prevents silent state changes that can cause GUI synchronization issues.
        # Sessions with no active recordings will remain visible but show appropriate
        # warnings when attempting to plot. Users can manually exclude the session if desired.

    def restore_session(self):
        """
        Restore the session by including all previously excluded recordings.
        This is a user action that modifies the session's exclude flags.
        """
        self.annot.excluded_recordings = []
        self.invalidate_selection_results()
        if self.repo is not None:
            self.repo.save(self)

    def exclude_session(self):
        """
        Exclude the entire session by marking all recordings as excluded.
        This is a user action that modifies the session's exclude flags.

        Note: This method is typically called BY the dataset's exclude_session() method,
        which handles adding the session to the dataset's excluded list. We don't call
        parent_dataset.exclude_session() here to avoid circular logic.
        """
        self.annot.excluded_recordings = [rec.id for rec in self.get_all_recordings(include_excluded=True)]
        self.invalidate_selection_results()
        if self.repo is not None:
            self.repo.save(self)

    # ──────────────────────────────────────────────────────────────────
    # 3) Methods for CLI/Jupyter use only
    # ──────────────────────────────────────────────────────────────────
    def update_window_settings(self):
        """
        Deprecated: This method has been deprecated to maintain separation of concerns.
        The `monstim_signals` package must remain GUI-agnostic.

        To update window settings, please use the main GUI application or modify
        the `session.annot` object directly in your script.
        """
        logger.warning("Session.update_window_settings() is deprecated and has been removed to avoid PySide6 dependencies in domain logic.")

    # ──────────────────────────────────────────────────────────────────
    # 4) Clean-up
    # ──────────────────────────────────────────────────────────────────
    def prepare_cache(self, products, methods=(), progress=None, cancelled=None) -> int:
        """Prepare selected products without importing Qt."""
        requested = set(products)
        selected_methods = tuple(dict.fromkeys(methods))
        need_windows = bool(requested & {"window_results", "amplitudes", "extrema_details", "mmax"})
        if need_windows and not selected_methods:
            selected_methods = (self.default_method,)
        total = (len(self.all_recordings) if "filtered_signals" in requested or need_windows else 0) + (
            len(selected_methods) * self.num_channels if need_windows else 0
        )
        completed = 0

        def stopped() -> bool:
            return bool(cancelled and cancelled())

        def report(detail: str) -> None:
            if progress is not None:
                progress(completed, total, detail)

        if "filtered_signals" in requested or need_windows:
            for recording in self.all_recordings:
                if stopped():
                    return completed
                self._get_signal_recording(recording, "filtered")
                completed += 1
                report(f"{self.id} / {recording.id}")

        if need_windows:
            for method in selected_methods:
                for channel in range(self.num_channels):
                    if stopped():
                        return completed
                    handled = False
                    if "mmax" in requested:
                        try:
                            self.get_m_max(method, channel)
                        except (LatencyWindowNotFoundError, NoCalculableMmaxError, ValueError) as exc:
                            logger.debug("Skipping M-max warm-up for %s channel %s: %s", self.id, channel, exc)
                        handled = True
                    if "amplitudes" in requested:
                        for window in self.latency_windows:
                            self.get_lw_reflex_amplitudes(method, channel, window)
                        self.get_lw_distribution(method, channel)
                        handled = True
                    if not handled:
                        self.get_all_lw_reflex_amplitude_results(method, channel, include_excluded=True)
                    completed += 1
                    report(f"{self.id} / channel {channel + 1} / {method}")
        return completed

    def close(self, force_gc: bool = True):
        """Close all recording HDF5 file handles.

        Args:
            force_gc: If True, force garbage collection after closing.
                     Set to False when closing as part of dataset/experiment.
        """
        self.release_cached_data()
        for rec in self.get_all_recordings(include_excluded=True):
            try:
                rec.close()
            except Exception as e:
                logger.warning(f"Error closing recording {rec.id}: {e}")

        # Force GC when closing session individually (not as part of dataset)
        if force_gc:
            import gc

            gc.collect()

    def __enter__(self) -> Session:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ──────────────────────────────────────────────────────────────────
    # 5) Object representation and reports
    # ──────────────────────────────────────────────────────────────────
    def session_parameters(self) -> dict[str, Any]:
        """
        Logs Session object parameters and returns a dictionary with the session parameters.
        This includes session ID, number of recordings, number of channels, scan rate,
        number of samples, pre-stimulus and post-stimulus acquisition times, stimulus delay,
        stimulus duration, recording interval, and EMG amplifier gains.
        """
        report = [
            f"Session Parameters for '{self.formatted_name}'",
            "===============================",
            f"Session ID: {self.id}",
            f"# of Recordings (including any excluded ones): {self.num_recordings}",
            f"# of Channels: {self.num_channels}",
            f"Scan Rate (Hz): {self.scan_rate}",
            f"Samples/Channel: {self.num_samples}",
            f"Pre-Stim Acq. time (ms): {self.pre_stim_acquired}",
            f"Post-Stim Acq. time (ms): {self.post_stim_acquired}",
            f"Stimulus Delay (ms): {self.stim_delay}",
            f"Stimulus Duration (ms): {self.stim_duration}",
            f"Recording Interval (s): {self.recording_interval if self.recording_interval else 'Not specified'}",
            f"EMG Amp Gains: {self.emg_amp_gains if self.emg_amp_gains else 'Not specified'}",
        ]

        for line in report:
            logger.info(line)
        return report

    def m_max_report(self):
        """
        Logs the M-wave amplitudes for each channel in the session.
        """
        report = [
            f"Session M-max Report for '{self.formatted_name}'",
            "===============================",
        ]
        for i, channel_name in enumerate(self.channel_names):
            try:
                channel_m_max = self.get_m_max(self.default_method, i, return_mmax_stim_range=False)
                line = f"- {channel_name}: M-max amplitude ({self.default_method}) = {channel_m_max:.2f} V"
                report.append(line)
            except TypeError:
                line = f"- Channel {i} does not have a valid M-max amplitude."
                report.append(line)

        for line in report:
            logger.info(line)
        return report

    def __repr__(self):
        return f"Session(session_id={self.id}, num_recordings={self.num_recordings})"

    def __str__(self):
        return f"Session: {self.id} with {self.num_recordings} recordings"

    def __len__(self):
        return self.num_recordings

    def __bool__(self) -> bool:
        """
        Sessions can be valid even if they have zero recordings (e.g., after
        exclusions). Define truthiness explicitly so generic `if session:`
        checks do not treat them as falsy.
        """
        return True

    def set_config(self, config: ResolvedConfig | dict) -> None:
        """
        Update the configuration for this session.
        """
        resolved = config if isinstance(config, ResolvedConfig) else ResolvedConfig(deep_merge(self._config, config))
        changes = resolved.diff(self._config)
        if changes == ConfigChange.NONE:
            return
        recreate_plotter = resolved.plot.construction_fingerprint != self._config.plot.construction_fingerprint
        self._config = resolved
        for rec in self.get_all_recordings(include_excluded=True):
            if hasattr(rec, "set_config"):
                rec.set_config(resolved)

        self.apply_config(changes)
        if recreate_plotter:
            self.plotter = SessionPlotterPyQtGraph(self)
