# monstim_signals/domain/session.py
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from typing import TYPE_CHECKING, Any, List

import numpy as np

from monstim_signals.core import LatencyWindow, SessionAnnot, StimCluster, load_config
from monstim_signals.domain.recording import Recording
from monstim_signals.plotting import SessionPlotterPyQtGraph
from monstim_signals.transform import (
    NoCalculableMmaxError,
    butter_bandpass_filter,
    calculate_emg_amplitude,
    correct_emg_to_baseline,
    get_avg_mmax,
)

if TYPE_CHECKING:
    from monstim_signals.domain.dataset import Dataset
    from monstim_signals.io.repositories import SessionRepository


# ──────────────────────────────────────────────────────────────────
class Session:
    """
    A collection of multiple Recordings, each at a different stimulus amplitude,
    all belonging to one “session” (animal & date).
    """

    def __init__(
        self,
        session_id: str,
        recordings: List["Recording"],
        annot: "SessionAnnot",
        repo: Any = None,
        config: dict = None,
    ):
        self.id: str = session_id
        self._all_recordings: List["Recording"] = recordings
        self.annot: "SessionAnnot" = annot
        self.repo: "SessionRepository" = repo
        self.parent_dataset: "Dataset | None" = None
        self._config = config
        self._load_config_settings()
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
        _config = self._config if self._config is not None else load_config()
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
        self.default_channel_names: List[str] = _config.get("default_channel_names", [])

    def _load_session_parameters(self):
        # ---------- Pull session‐wide parameters from the first recording's meta ----------
        if self.recordings:
            first_meta = self.recordings[0].meta
            self.formatted_name = self.id + "_" + first_meta.recording_id  # e.g., "AA00_0000"
            self.scan_rate = first_meta.scan_rate  # Hz
            self.num_samples = first_meta.num_samples  # samples/channel
            self.num_channels = first_meta.num_channels  # number of channels
            self._channel_types: List[str] = first_meta.channel_types.copy()  # list of channel types

            # Stimulus parameters
            self.stim_clusters: List[StimCluster] = first_meta.stim_clusters.copy()  # list of StimCluster objects
            self.primary_stim: StimCluster = getattr(
                first_meta, "primary_stim", None
            )  # the primary StimCluster for this session
            if self.primary_stim is None:
                logging.warning(
                    f"Session {self.id} does not have a primary stimulus defined. Defaulting to the first StimCluster."
                )
                self.primary_stim = self.stim_clusters[0] if self.stim_clusters else None
                if self.primary_stim is None:
                    logging.error(f"Session {self.id} has no StimClusters defined. Cannot determine primary stimulus.")
                    raise ValueError(f"Session {self.id} has no StimClusters defined. Cannot determine primary stimulus.")
            self.pre_stim_acquired = first_meta.pre_stim_acquired
            self.post_stim_acquired = first_meta.post_stim_acquired
            self.stim_delay = self.primary_stim.stim_delay  # in ms, delay
            self.stim_duration = self.primary_stim.stim_duration
            self.stim_start: float = self.stim_delay + self.pre_stim_acquired

            # Parameters that may sometimes be None
            self.recording_interval: float = getattr(
                first_meta, "recording_interval", None
            )  # in seconds, time between recordings (if applicable)
            self.emg_amp_gains: List[int] = getattr(first_meta, "emg_amp_gains", None)  # default to 1000 if not specified
        else:
            raise ValueError(f"Session {self.id} has no recordings associated with it.")

    def _initialize_annotations(self):
        # Check in case of empty list annot
        if len(self.annot.channels) != self.num_channels:
            from monstim_signals.core import SignalChannel

            logging.warning(
                f"Session {self.id} has {len(self.annot.channels)} channels in annot, but {self.num_channels} channels in recordings. Reinitializing channel annotations."
            )
            self.annot.channels = [
                SignalChannel(
                    name=(self.default_channel_names[i] if i < len(self.default_channel_names) else f"Channel {i+1}"),
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

    def apply_config(self, reset_caches: bool = True) -> None:
        """
        Apply the loaded configuration settings to the session.
        This is called after loading the session or when preferences are changed.
        """
        self._load_config_settings()  # Reload config settings to ensure they are up-to-date

        self.plotter = SessionPlotterPyQtGraph(self)
        for window in self.latency_windows:
            window.linestyle = self.latency_window_style
            window.color = self.m_color if window.name == "M-wave" else window.color
            window.color = self.h_color if window.name == "H-reflex" else window.color

        if reset_caches:
            self.reset_all_caches()

    @property
    def num_recordings(self) -> int:
        return len(self.recordings)

    @property
    def num_all_recordings(self) -> int:
        return len(self.all_recordings)

    @property
    def latency_windows(self) -> List[LatencyWindow]:
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
                if hasattr(m, "primary_stim") and hasattr(first, "primary_stim"):
                    if m.primary_stim.stim_delay != first.primary_stim.stim_delay:
                        warnings.append(
                            f"Recording {rec.id} stim_delay {m.primary_stim.stim_delay} != {first.primary_stim.stim_delay}."
                        )
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
                if any(b < a for a, b in zip(volts, volts[1:])):
                    warnings.append("Stimulus voltages are not sorted non-decreasing.")
        finally:
            # Store for notice system
            self._consistency_warnings = warnings
            for w in warnings:
                logging.warning(f"Session {self.id} consistency: {w}")

    # ------------------------------------------------------------------
    # Latency window helper methods
    # ------------------------------------------------------------------
    def add_latency_window(
        self,
        name: str,
        start_times: List[float],
        durations: List[float],
        color: str | None = None,
        linestyle: str | None = None,
    ) -> None:
        """Add a new :class:`LatencyWindow` to the session."""
        window = LatencyWindow(
            name=name,
            start_times=start_times,
            durations=durations,
            color=color or self.m_color,
            linestyle=linestyle or self.latency_window_style,
        )
        self.annot.latency_windows.append(window)
        self.update_latency_window_parameters()
        if self.repo is not None:
            self.repo.save(self)

    def apply_latency_window_preset(self, preset_name: str) -> None:
        """Replace latency windows using a preset defined in the config file."""
        from monstim_signals.core import load_config

        presets = load_config().get("latency_window_presets", {})
        if preset_name not in presets:
            logging.warning(f"Preset '{preset_name}' not found in config.")
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

        self.update_latency_window_parameters()
        if self.repo is not None:
            self.repo.save(self)

    def remove_latency_window(self, name: str) -> None:
        """Remove a latency window by name."""
        self.annot.latency_windows = [w for w in self.annot.latency_windows if w.name != name]
        self.update_latency_window_parameters()
        if self.repo is not None:
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
        Return a list of stimulus voltages for each recording in the session.
        This assumes that each recording's primary cluster stim_v is the amplitude for that recording.
        """
        return np.array([rec.meta.primary_stim.stim_v for rec in self.recordings])

    @property
    def recordings(self) -> List[Recording]:
        """
        Return a list of active recordings in the session.
        This filters out any recordings that are marked as excluded in the session annotations.
        """
        return self.get_all_recordings(include_excluded=False)

    @property
    def all_recordings(self) -> List[Recording]:
        """
        Return a list of all recordings in the session, including excluded ones.
        """
        return self.get_all_recordings(include_excluded=True)

    def get_all_recordings(self, include_excluded: bool = False) -> List[Recording]:
        """
        Return a list of recordings in the session.

        Args:
            include_excluded (bool): If True, returns all recordings including excluded ones.
                                   If False, returns only active (non-excluded) recordings.

        Returns:
            List[Recording]: The list of recordings based on the include_excluded parameter.
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
                for ch, (start, dur) in enumerate(zip(w.start_times, w.durations)):
                    if dur <= 0:
                        notices.append(
                            {
                                "level": "warning",
                                "code": "zero_or_negative_window",
                                "message": f"Latency window '{w.name}' channel {ch} has non-positive duration {dur}.",
                            }
                        )
            # Missing canonical M-wave window
            if not any(
                (w.name or "").lower() in {"m-wave", "m_wave", "m wave", "mwave", "m-response", "m_response", "m response"}
                for w in self.latency_windows
            ):
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
                for ch, (start, dur) in enumerate(zip(w.start_times, w.durations)):
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
            logging.debug(f"Notice collection error (session {self.id}): {e}")
        return notices

    # ──────────────────────────────────────────────────────────────────
    # 0) Cached properties and cache reset methods
    # ──────────────────────────────────────────────────────────────────
    @cached_property
    def recordings_raw(self) -> List[np.ndarray]:
        """
        Return a list of raw data arrays for each recording.
        Each array is of shape (num_samples, num_channels).
        """
        recordings = []
        for rec in self.recordings:
            raw_data = rec.raw_view()
            for ch in range(rec.meta.num_channels):
                if self.annot.channels[ch].invert:
                    raw_data[:, ch] *= -1.0
            recordings.append(raw_data)
        return recordings

    @cached_property
    def recordings_filtered(self) -> List[np.ndarray]:
        """
        Return a list of processed data arrays for each recording.
        Each array is of shape (num_samples, num_channels).
        This applies a butter bandpass filter to the raw data and inverts if
        indicated in the channel annotations in the session annot.json file.
        """

        def _process_single_recording(rec: Recording) -> np.ndarray:
            """
            Process a single recording's raw data with the butter bandpass filter.
            """
            raw_data = rec.raw_view()
            bf_args = getattr(
                self,
                "butter_filter_args",
                {"lowcut": None, "highcut": None, "order": None},
            )

            filtered_channels = []
            for ch in range(rec.meta.num_channels):
                channel_data = raw_data[:, ch]
                channel_type = self.channel_types[ch].lower()

                if channel_type in ("force", "length"):
                    # Apply specific processing for force and length channels
                    # TODO: Implement specific filtering for force and length channels if needed
                    # Notes/TODOs:
                    # - Force/length channels may need low-pass smoothing or detrending
                    #   rather than the EMG bandpass. Add configuration options and a
                    #   clear API entry point for channel-specific processing.
                    # - Consider adding unit tests that validate the processing for
                    #   force/length channels and ensure these channels are not
                    #   inadvertently rectified/treated as EMG.
                    filtered = correct_emg_to_baseline(channel_data, self.scan_rate, self.stim_delay)
                elif channel_type in ("emg",):
                    # Apply specific processing for EMG channels
                    filtered = butter_bandpass_filter(
                        channel_data,
                        fs=self.scan_rate,
                        lowcut=bf_args["lowcut"],
                        highcut=bf_args["highcut"],
                        order=bf_args["order"],
                    )
                else:
                    logging.warning(f"No specific processing for channel type: {channel_type}")
                    filtered = channel_data

                if self.annot.channels[ch].invert:
                    filtered = -filtered

                filtered_channels.append(filtered)

            return np.column_stack(filtered_channels)

        max_workers = (os.cpu_count() - 1) or 1  # Use all available CPU cores
        filtered_recordings: List[np.ndarray] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit futures and maintain order by storing them in a list
            ordered_futures = []
            for rec in self.recordings:
                future = executor.submit(_process_single_recording, rec)
                ordered_futures.append(future)

            # Get results in the same order as the original recordings
            for future in ordered_futures:
                filtered_array = future.result()  # this blocks until that recording is done
                filtered_recordings.append(filtered_array)

        return filtered_recordings

    @cached_property
    def recordings_rectified_raw(self) -> List[np.ndarray]:
        """
        Return a list of rectified raw data arrays for each recording.
        Each array is of shape (num_samples, num_channels).
        This applies a rectification to the raw data and inverts if indicated in the channel annotations.
        """
        recordings = []
        for rec in self.recordings:
            raw_data = rec.raw_view()
            rectified = np.abs(raw_data)
            recordings.append(rectified)
        return recordings

    @cached_property
    def recordings_rectified_filtered(self) -> List[np.ndarray]:
        """
        Return a list of rectified filtered data arrays for each recording.
        Each array is of shape (num_samples, num_channels).
        This applies a rectification to the filtered data and inverts if indicated in the channel annotations.
        """
        recordings = []
        for rec in self.recordings_filtered:
            rectified = np.abs(rec)
            recordings.append(rectified)
        return recordings

    @cached_property
    def m_max(self) -> List[float]:
        """
        Return the maximum M-wave value for each recording in the session.
        This is computed from the raw data using the M-wave latency windows.
        """
        results = []
        for rec in self.recordings:
            for channel_index in range(self.num_channels):
                try:  # Check if the channel has a valid M-max amplitude.
                    channel_mmax = self.get_m_max(self.default_method, channel_index, return_mmax_stim_range=False)
                    results.append(channel_mmax)
                except NoCalculableMmaxError:
                    logging.info(f"Channel {channel_index} does not have a valid M-max amplitude.")
                    results.append(None)
                except ValueError as e:
                    logging.error(f"Error in calculating M-max amplitude for channel {channel_index}. Error: {str(e)}")
                    results.append(None)
        return results

    def reset_all_caches(self):
        """
        Reset all cached properties in the session.
        This is used after changing any session parameters or excluding/including recordings.
        """
        self.reset_recordings_cache()
        self.reset_cached_reflex_properties()
        self.update_latency_window_parameters()

    # TODO: Caching / consistency
    # - Ensure cached_property names and keys cleared by reset_recordings_cache() match
    #   exactly. Consider automating this via a decorator or metaclass.
    #   — add tests to validate cache invalidation for all cached properties including
    #   `m_max`, `recordings_rectified_filtered`, etc.

    def update_latency_window_parameters(self):
        """
        Update cached M/H-response parameters from latency windows.
        This remains for backwards compatibility. If no M-wave or H-reflex
        windows exist, the corresponding attributes will be set to empty lists.
        """
        self.m_start = [0.0] * self.num_channels
        self.m_duration = [0.0] * self.num_channels
        self.h_start = [0.0] * self.num_channels
        self.h_duration = [0.0] * self.num_channels
        for window in self.latency_windows:
            lname = window.name.lower()
            if lname in {
                "m-wave",
                "m_wave",
                "m wave",
                "mwave",
                "m-response",
                "m_response",
                "m response",
            }:
                self.m_start = window.start_times
                self.m_duration = window.durations
            elif lname in {
                "h-wave",
                "h_wave",
                "h wave",
                "hwave",
                "h-reflex",
                "h_reflex",
                "h reflex",
                "hresponse",
                "h_response",
                "h response",
            }:
                self.h_start = window.start_times
                self.h_duration = window.durations

    def reset_cached_reflex_properties(self):
        """
        Reset the cached M-wave max values.
        This is used after changing the latency windows or excluding/including recordings from the session set.
        """
        if "m_max" in self.__dict__:
            del self.__dict__["m_max"]

    def reset_recordings_cache(self):
        """
        Reset the cached processed recordings.
        This is used after changing the filter parameters or excluding/including recordings from the session set.
        """
        if "recordings" in self.__dict__:
            del self.__dict__["recordings"]
        if "recordings_raw" in self.__dict__:
            del self.__dict__["recordings_raw"]

        if "recordings_filtered" in self.__dict__:
            del self.__dict__["recordings_filtered"]
        if "recordings_rectified_raw" in self.__dict__:
            del self.__dict__["recordings_rectified_raw"]
        # TODO: include recordings_rectified_filtered and the cached m_max if present
        if "recordings_rectified_filtered" in self.__dict__:
            del self.__dict__["recordings_rectified_filtered"]
        if "m_max" in self.__dict__:
            del self.__dict__["m_max"]

    # ──────────────────────────────────────────────────────────────────
    # 1) Properties for GUI & analysis code
    # ──────────────────────────────────────────────────────────────────
    def plot(self, plot_type: str = None, **kwargs):
        """
        Plots EMG data from a single session using the specified plot_type.

        Args:
            - plot_type (str): The type of plot to generate. Options include 'emg', 'suspectedH', 'mmax', 'reflexCurves', 'reflexAverages', and 'mCurvesSmoothened'.
                Plot types are defined in the EMGSessionPlotter class in Plot_EMG.py.
            - channel_names (list): A list of channel names to plot. If None, all channels will be plotted.
            - **kwargs: Additional keyword arguments to pass to the plotting function.

                The most common keyword arguments include:
                - 'data_type' (str): The type of data to plot. Options are 'filtered', 'raw', 'rectified_raw', or 'rectified_filtered'. Default is 'filtered'.
                - 'method' (str): The method to use for calculating the M-wave/reflex amplitudes. Options include 'average_rectified', 'rms', 'peak_to_trough', and 'average_unrectified'. Default method is set in config.yml under 'default_method'.
                - 'relative_to_mmax' (bool): Whether to plot the data proportional to the M-wave amplitude (True) or as the actual recorded amplitude (False). Default is False.
                - 'all_flags' (bool): Whether to plot flags at all windows (True) or not (False). Default is False.

                Less common keyword arguments include:
                - 'm_flags' (bool): Whether to plot flags at the M-wave window (True) or not (False). Default is False.
                - 'h_flags' (bool): Whether to plot flags at the H-reflex window (True) or not (False). Default is False.
                - 'h_threshold' (float): The threshold for detecting the H-reflex in the suspectedH plot. Default is 0.3.
                - 'mmax_report' (bool): Whether to print the details of the M-max calculations (True) or not (False). Default is False.
                - 'manual_mmax' (float): The manually set M-wave amplitude to use for plotting the reflex curves. Default is None.

        Example Usages:
            # Plot filtered EMG data
                session.plot()

            # Plot raw EMG data with flags at the M-wave and H-reflex windows
            session.plot(data_type='raw', all_flags=True)

            # Plot all EMG data with the M-wave and H-reflex windows highlighted
            session.plot(plot_type='suspectedH')

            # Plot M-wave amplitudes for each channel
            session.plot(plot_type='mmax')

            # Plot the reflex curves for each channel
            session.plot(plot_type='reflexCurves')
        """
        # Call the appropriate plotting method from the plotter object
        raw_data = getattr(self.plotter, f'plot_{"emg" if not plot_type else plot_type}')(**kwargs)
        return raw_data

    def get_m_max(self, method, channel_index, return_mmax_stim_range=False):
        """
        Calculates the M-wave amplitude for a specific channel in the session.

        Args:
            method (str): The method to use for calculating the M-wave amplitude. Options include 'average_rectified', 'rms', 'peak_to_trough', and 'average_unrectified'.
            channel_index (int): The index of the channel to calculate the M-wave amplitude for.

        Returns:
            float: The M-wave amplitude for the specified channel.
        """
        m_wave_amplitudes = self.get_m_wave_amplitudes(method, channel_index)

        if return_mmax_stim_range:
            return get_avg_mmax(
                self.stimulus_voltages,
                m_wave_amplitudes,
                **self.m_max_args,
                return_mmax_stim_range=True,
            )
        else:
            return get_avg_mmax(self.stimulus_voltages, m_wave_amplitudes, **self.m_max_args)

    def get_m_wave_amplitudes(self, method, channel_index):
        """Return a list of M-wave amplitudes for each recording."""
        window_start = self.m_start[channel_index] + self.stim_start
        window_end = window_start + self.m_duration[channel_index]

        if window_end - window_start <= 0:
            logging.warning(
                f"Invalid or missing M-wave reflex window for channel {channel_index} in session {self.id}. Start: {window_start}, End: {window_end}."
            )
            raise ValueError(
                f"Invalid or missing M-wave reflex window for channel {channel_index} in session {self.id}. Start: {window_start}, End: {window_end}."
            )

        m_wave_amplitudes = [
            calculate_emg_amplitude(
                recording[:, channel_index],
                window_start,
                window_end,
                self.scan_rate,
                method=method,
            )
            for recording in self.recordings_filtered
        ]
        return m_wave_amplitudes

    def get_h_wave_amplitudes(self, method, channel_index):
        """Return a list of H-reflex amplitudes for each recording."""
        window_start = self.h_start[channel_index] + self.stim_start
        window_end = window_start + self.h_duration[channel_index]
        h_wave_amplitudes = [
            calculate_emg_amplitude(
                recording[:, channel_index],
                window_start,
                window_end,
                self.scan_rate,
                method=method,
            )
            for recording in self.recordings_filtered
        ]
        return h_wave_amplitudes

    def get_lw_reflex_amplitudes(self, method: str, channel_index: int, window: str | LatencyWindow) -> np.ndarray:
        """
        Returns the reflex amplitudes for a specific latency window across all sessions in the dataset.

        The array in the same order as the stimulus voltage of each recording.
        """
        # Convert window to LatencyWindow if it's a string
        if not isinstance(window, LatencyWindow):
            window = self.get_latency_window(window)

        if window is not None:
            # Needs to correct window times to the stimulus start time
            window_start = window.start_times[channel_index] + self.stim_start
            window_end = window.end_times[channel_index] + self.stim_start
        else:
            logging.warning(f"Latency window '{window}' not found.")
            return []

        # Calculate the reflex amplitudes for the specified window
        reflex_amplitudes = [
            calculate_emg_amplitude(
                recording[:, channel_index],
                window_start,
                window_end,
                self.scan_rate,
                method=method,
            )
            for recording in self.recordings_filtered
        ]
        return np.array(reflex_amplitudes)

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
            for i, ch in enumerate(self.annot.channels):
                if ch.name == old_name:
                    ch.name = new_name
                    matched = True
            if matched:
                logging.info(f"Renamed channel '{old_name}' to '{new_name}' in session {self.id}.")
            else:
                logging.warning(f"Channel '{old_name}' not found in session {self.id}. No action taken.")
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
        self.reset_recordings_cache()
        self.reset_cached_reflex_properties()

    def change_reflex_latency_windows(self, m_start, m_duration, h_start, h_duration):
        m_window = self.get_latency_window("M-wave")
        if m_window:
            m_window.start_times = m_start
            m_window.durations = m_duration
        h_window = self.get_latency_window("H-reflex")
        if h_window:
            h_window.start_times = h_start
            h_window.durations = h_duration
        self.update_latency_window_parameters()
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
                logging.warning(f"Recording {recording_id} not found in session {self.id}.")
                return
            self.reset_all_caches()
            if self.repo is not None:
                self.repo.save(self)
        else:
            logging.warning(f"Recording {recording_id} is not excluded from session {self.id}. No action taken.")

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
                logging.warning(f"Recording {recording_id} not found in session {self.id}.")
                return

            self.reset_all_caches()
            if self.repo is not None:
                self.repo.save(self)
        else:
            logging.warning(f"Recording {recording_id} is already excluded in session {self.id}.")

        if not self.recordings:
            # If no recordings remain, mark the session and inform parent dataset
            self.exclude_session()
            if self.parent_dataset is not None:
                self.parent_dataset.exclude_session(self.id)

    def restore_session(self):
        """
        Restore the session by including all previously excluded recordings.
        This is a user action that modifies the session's exclude flags.
        """
        self.annot.excluded_recordings = []
        self.reset_all_caches()
        if self.repo is not None:
            self.repo.save(self)

    def exclude_session(self):
        """
        Exclude the entire session by marking all recordings as excluded.
        This is a user action that modifies the session's exclude flags.
        """
        self.annot.excluded_recordings = [rec.id for rec in self.get_all_recordings(include_excluded=True)]
        if self.recordings == []:
            # If no recordings remain, mark the session as excluded
            self.parent_dataset.exclude_session(self.id)
        self.reset_all_caches()
        if self.repo is not None:
            self.repo.save(self)

    # ──────────────────────────────────────────────────────────────────
    # 3) Methods for CLI/Jupyter use only
    # ──────────────────────────────────────────────────────────────────
    def update_window_settings(self):
        """
        ***ONLY FOR USE IN JUPYTER NOTEBOOKS OR INTERACTIVE PYTHON ENVIRONMENTS***

        Opens a GUI to manually update the M-wave and H-reflex window settings for each channel using PyQt.

        This function should only be used if you are working in a Jupyter notebook or an interactive Python environment. Do not call this function in any other GUI environment.
        """
        import sys

        from PySide6.QtWidgets import (
            QApplication,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )

        class ReflexSettingsDialog(QWidget):
            def __init__(self, parent: Session):
                super().__init__()
                self.parent: Session = parent
                self.initUI()

            def initUI(self):
                self.setWindowTitle(f"Update Reflex Window Settings: Session {self.parent.session_id}")
                layout = QVBoxLayout()

                duration_layout = QHBoxLayout()
                duration_layout.addWidget(QLabel("m_duration:"))
                self.m_duration_entry = QLineEdit(str(self.parent.m_duration[0]))
                duration_layout.addWidget(self.m_duration_entry)

                duration_layout.addWidget(QLabel("h_duration:"))
                self.h_duration_entry = QLineEdit(str(self.parent.h_duration[0]))
                duration_layout.addWidget(self.h_duration_entry)

                layout.addLayout(duration_layout)

                self.entries = []
                for i in range(self.parent.num_channels):
                    channel_layout = QHBoxLayout()
                    channel_layout.addWidget(QLabel(f"Channel {i}:"))

                    channel_layout.addWidget(QLabel("m_start:"))
                    m_start_entry = QLineEdit(str(self.parent.m_start[i]))
                    channel_layout.addWidget(m_start_entry)

                    channel_layout.addWidget(QLabel("h_start:"))
                    h_start_entry = QLineEdit(str(self.parent.h_start[i]))
                    channel_layout.addWidget(h_start_entry)

                    layout.addLayout(channel_layout)
                    self.entries.append((m_start_entry, h_start_entry))

                save_button = QPushButton("Confirm")
                save_button.clicked.connect(self.save_settings)
                layout.addWidget(save_button)

                self.setLayout(layout)

            def save_settings(self):
                try:
                    m_duration = float(self.m_duration_entry.text())
                    h_duration = float(self.h_duration_entry.text())
                except ValueError:
                    logging.error("Invalid input for durations. Please enter valid numbers.")
                    return
                m_start = []
                h_start = []
                for i, (m_start_entry, h_start_entry) in enumerate(self.entries):
                    try:
                        m_start.append(float(m_start_entry.text()))
                        h_start.append(float(h_start_entry.text()))
                    except ValueError:
                        logging.error(f"Invalid input for channel {i}. Skipping.")

                # Update the reflex windows in the parent object based on the new windows.
                try:
                    self.parent.change_reflex_latency_windows(m_start, m_duration, h_start, h_duration)
                    self.parent.reset_all_caches()
                except Exception as e:
                    logging.error(
                        f"Error occurred when trying to save the following reflex settings: m_start: {m_start}\n\tm_duration: {m_duration}\n\th_start: {h_start}\n\th_duration: {h_duration}"
                    )
                    logging.error(f"Error: {str(e)}")
                    return

                self.close()

        app = QApplication.instance()  # Check if there's an existing QApplication instance
        if not app:
            app = QApplication(sys.argv)
            window = ReflexSettingsDialog(self)
            window.show()
            app.exec()
        else:
            window = ReflexSettingsDialog(self)
            window.show()

    # ──────────────────────────────────────────────────────────────────
    # 4) Clean‐up
    # ──────────────────────────────────────────────────────────────────
    def close(self, force_gc: bool = True):
        """Close all recording HDF5 file handles.

        Args:
            force_gc: If True, force garbage collection after closing.
                     Set to False when closing as part of dataset/experiment.
        """
        for rec in self.get_all_recordings(include_excluded=True):
            try:
                rec.close()
            except Exception as e:
                logging.warning(f"Error closing recording {rec.id}: {e}")

        # Force GC when closing session individually (not as part of dataset)
        if force_gc:
            import gc

            gc.collect()

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
            logging.info(line)
        return report

    def m_max_report(self):
        """
        Logs the M-wave amplitudes for each channel in the session.
        """
        report = [
            f"Session M-max Report for '{self.formatted_name}'",
            "===============================",
        ]
        print(self.m_max)
        for i, channel_name in enumerate(self.channel_names):
            try:
                channel_m_max = self.get_m_max(self.default_method, i, return_mmax_stim_range=False)
                line = f"- {channel_name}: M-max amplitude ({self.default_method}) = {channel_m_max:.2f} V"
                report.append(line)
            except TypeError:
                line = f"- Channel {i} does not have a valid M-max amplitude."
                report.append(line)

        for line in report:
            logging.info(line)
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

    def set_config(self, config: dict) -> None:
        """
        Update the configuration for this session.
        """
        self._config = config
        for rec in self.get_all_recordings(include_excluded=True):
            if hasattr(rec, "set_config"):
                rec.set_config(config)

        self.apply_config(reset_caches=True)
