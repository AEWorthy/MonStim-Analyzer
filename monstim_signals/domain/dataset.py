# monstim_signals/domain/dataset.py
import logging
from typing import TYPE_CHECKING, Any, List

import numpy as np

from monstim_signals.core import DatasetAnnot, LatencyWindow, load_config
from monstim_signals.domain.session import Session
from monstim_signals.plotting import DatasetPlotterPyQtGraph

if TYPE_CHECKING:
    from monstim_signals.domain.experiment import Experiment
    from monstim_signals.io.repositories import DatasetRepository


class Dataset:
    """
    A “dataset” = all sessions from one animal replicate.
    E.g. Dataset_1(Animal_A) has sessions AA00, AA01, …
    """

    def __init__(
        self,
        dataset_id: str,
        sessions: List[Session],
        annot: DatasetAnnot,
        repo: Any = None,
        config: dict | None = None,
    ):
        self.id: str = dataset_id
        self._all_sessions: List[Session] = sessions
        self.annot: DatasetAnnot = annot
        self.repo: DatasetRepository = repo
        self.parent_experiment: "Experiment | None" = None
        self._config = config
        for sess in self._all_sessions:
            sess.parent_dataset = self

        self._load_config_settings()

        self.plotter = DatasetPlotterPyQtGraph(self)

        # Ensure all sessions share the same recording parameters
        try:
            self.__check_session_consistency()

            self.scan_rate: int = self.sessions[0].scan_rate
            self.stim_start: float = self.sessions[0].stim_start
        except IndexError:
            logging.warning(
                f"Dataset {self.id} has no sessions to check for consistency. Skipping consistency check. Setting scan_rate and stim_start to zeroes."
            )
            self.scan_rate = 0
            self.stim_start = 0.0

        self.update_latency_window_parameters()
        logging.debug(f"Dataset {self.id} initialized with {len(self.sessions)} sessions.")

    @property
    def is_completed(self) -> bool:
        return getattr(self.annot, "is_completed", False)

    @is_completed.setter
    def is_completed(self, value: bool) -> None:
        self.annot.is_completed = bool(value)
        if self.repo is not None:
            self.repo.save(self)

    def _load_config_settings(self) -> None:
        _config = self._config if self._config is not None else load_config()
        self.bin_size = _config["bin_size"]
        self.default_method = _config["default_method"]
        self.m_color = _config["m_color"]
        self.h_color = _config["h_color"]
        # Plot styling shared with Session objects
        self.title_font_size = _config["title_font_size"]
        self.axis_label_font_size = _config["axis_label_font_size"]
        self.tick_font_size = _config["tick_font_size"]
        self.subplot_adjust_args = _config["subplot_adjust_args"]

    @property
    def date(self) -> str:
        """
        Returns the date of the dataset in 'YYYY-MM-DD' format.
        If the date is not set, returns an empty string.
        """
        return self.annot.date if self.annot.date else "UNDEFINED"

    @property
    def animal_id(self) -> str:
        """
        Returns the animal ID of the dataset.
        If the animal ID is not set, returns an empty string.
        """
        return self.annot.animal_id if self.annot.animal_id else "UNDEFINED"

    @property
    def condition(self) -> str:
        """
        Returns the condition of the dataset.
        If the condition is not set, returns an empty string.
        """
        return self.annot.condition if self.annot.condition else "UNDEFINED"

    @property
    def formatted_name(self) -> str:
        """
        Returns the formatted name of the dataset.
        If all metadata components (date, animal_id, condition) are available, returns formatted metadata.
        If some or all metadata is missing, returns the original dataset folder name.
        """
        # Only use formatted metadata if all components are present and valid
        if self.annot.date and self.annot.animal_id and self.annot.condition:
            return f"{self.annot.date} {self.annot.animal_id} {self.annot.condition}"
        else:
            # Fall back to original folder name if any metadata is missing
            return self.id

    @property
    def num_sessions(self) -> int:
        return len(self.sessions)

    @property
    def excluded_sessions(self) -> set[str]:
        """IDs of sessions excluded from this dataset."""
        return set(self.annot.excluded_sessions)

    @property
    def sessions(self) -> List[Session]:
        return self.get_all_sessions(include_excluded=False)

    @property
    def num_channels(self) -> int:
        if not self.sessions:
            return 0
        return min(session.num_channels for session in self.sessions)

    @property
    def channel_names(self) -> List[str]:
        if not self.sessions:
            return []
        return max((session.channel_names for session in self.sessions), key=len)

    @property
    def latency_windows(self) -> List[LatencyWindow]:
        """Return a representative list of latency windows.

        NOTE: Historically this returned the first session's windows and higher-level code
        assumed uniformity across sessions. The application now permits heterogeneous
        per-session latency windows (sessions may add/remove or customize windows
        independently). This legacy property is retained for backward compatibility with
        existing code paths but should NOT be used for aggregation logic. Use the new
        union/inspection helpers instead:

            - unique_latency_window_names()
            - window_presence_map()
            - iter_latency_window_names()

        Returns the latency windows from the first available session or an empty list if
        there are no sessions.
        """
        if not self.sessions:
            return []
        return self.sessions[0].latency_windows

    # ------------------------------------------------------------------
    # Heterogeneous latency window inspection helpers
    # ------------------------------------------------------------------
    def unique_latency_window_names(self) -> List[str]:
        """Return sorted list of the union of latency window names across sessions.

        Case-insensitive uniqueness; original case of first occurrence is preserved.
        """
        name_map: dict[str, str] = {}
        for sess in self.sessions:
            for w in getattr(sess.annot, "latency_windows", []):
                low = (w.name or "").lower()
                if low and low not in name_map:
                    name_map[low] = w.name
        return [name_map[k] for k in sorted(name_map.keys())]

    # TODO: Aggregation UX
    # - Dataset-level aggregation for latency windows is non-trivial when sessions
    #   have heterogeneous windows. Consider adding a GUI helper that visualizes
    #   the union of windows across sessions and lets the user choose a canonical
    #   mapping (e.g., pick a representative window, merge similar windows, or
    #   define per-session offsets). This would improve dataset-level aggregation
    #   and make downstream reports more consistent.

    def iter_latency_window_names(self):
        """Yield latency window names (union) in sorted order."""
        for name in self.unique_latency_window_names():
            yield name

    def window_presence_map(self) -> dict[str, List[str]]:
        """Return mapping of window name -> list of session IDs that contain it."""
        presence: dict[str, List[str]] = {name: [] for name in self.unique_latency_window_names()}
        for sess in self.sessions:
            sess_names = {w.name for w in getattr(sess.annot, "latency_windows", [])}
            for name in presence.keys():
                if name in sess_names:
                    presence[name].append(sess.id)
        return presence

    @property
    def has_heterogeneous_latency_windows(self) -> bool:
        """True if sessions do not all share identical ordered window name lists."""
        if len(self.sessions) <= 1:
            return False
        first = [w.name for w in self.sessions[0].annot.latency_windows]
        for sess in self.sessions[1:]:
            if [w.name for w in sess.annot.latency_windows] != first:
                return True
        return False

    def get_session_latency_window(self, session: Session, window_name: str) -> LatencyWindow | None:
        """Fetch a latency window by name (case-insensitive) from a specific session."""
        low = window_name.lower()
        for w in getattr(session.annot, "latency_windows", []):
            if (w.name or "").lower() == low:
                return w
        return None

    # ------------------------------------------------------------------
    # Diagnostic / notice helpers (queried by GUI for tooltip indicators)
    # ------------------------------------------------------------------
    def collect_notices(self) -> list[dict[str, str]]:
        """Return a list of structured notices about this dataset.

        Each notice is a dict with keys:
            - level: one of "warning", "info"
            - code: machine-readable short code
            - message: human-readable description

        The GUI can map level->icon color (e.g., red exclamation for warning, grey for info)
        and aggregate tooltip text.
        """
        notices: list[dict[str, str]] = []
        try:
            if self.has_heterogeneous_latency_windows:
                notices.append(
                    {
                        "level": "warning",
                        "code": "heterogeneous_latency_windows",
                        "message": "Sessions in this dataset have differing latency window sets.",
                    }
                )
            # Sample rate incongruence (already prevented normally). Defensive check:
            if len({s.scan_rate for s in self.sessions}) > 1:
                notices.append(
                    {
                        "level": "info",
                        "code": "mixed_scan_rates",
                        "message": "Sessions have differing scan rates; comparisons may be approximate.",
                    }
                )
            # Include any consistency warnings captured during initialization
            for msg in getattr(self, "_consistency_warnings", []):
                code = "generic_consistency"
                if "scan_rate" in msg:
                    code = "inconsistent_scan_rate"
                elif "num_channels" in msg:
                    code = "inconsistent_num_channels"
                elif "stim_start" in msg:
                    code = "inconsistent_stim_start"
                notices.append({"level": "warning", "code": code, "message": msg})

            # Missing M-wave latency window at dataset level (if no session has it)
            if not any(
                any(
                    (w.name or "").lower() in {"m-wave", "m_wave", "m wave", "mwave", "m-response", "m_response", "m response"}
                    for w in sess.latency_windows
                )
                for sess in self.sessions
            ):
                notices.append(
                    {
                        "level": "info",
                        "code": "missing_m_wave_window",
                        "message": "No session in this dataset has an M-wave latency window.",
                    }
                )

            # No active sessions
            if len(self.sessions) == 0:
                notices.append(
                    {
                        "level": "warning",
                        "code": "no_active_session",
                        "message": "Dataset has no active sessions.",
                    }
                )

            # Single session only
            if len(self.sessions) == 1:
                notices.append(
                    {
                        "level": "warning",
                        "code": "single_session_only",
                        "message": "Dataset contains only a single active session.",
                    }
                )

            # NOTE: Too intensiveto check heterogeneous M-max failures here
            # # Heterogeneous M-max failures: some sessions have m_max None and others valid
            # mmax_states = []
            # for sess in self.sessions:
            #     try:
            #         mm = sess.get_m_max(sess.default_method, 0)  # channel 0 heuristic
            #         mmax_states.append(mm is not None)
            #     except Exception:
            #         mmax_states.append(False)
            # if mmax_states and any(mmax_states) and not all(mmax_states):
            #     notices.append({
            #         "level": "info",
            #         "code": "heterogeneous_mmax_failures",
            #         "message": "Some sessions have calculable M-max while others do not.",
            #     })

            # High latency window name churn (fraction of unique names not present in majority)
            all_names = []
            for sess in self.sessions:
                all_names.extend([(w.name or "") for w in sess.latency_windows])
            if all_names:
                from collections import Counter

                counts = Counter(all_names)
                total_sessions = len(self.sessions)
                rare = [n for n, c in counts.items() if c < max(2, int(0.5 * total_sessions))]
                if len(rare) / max(1, len(counts)) > 0.4:  # >40% names are rare
                    notices.append(
                        {
                            "level": "info",
                            "code": "high_latency_window_name_churn",
                            "message": "High diversity of latency window names; aggregation consistency may suffer.",
                        }
                    )

        except Exception as e:
            logging.debug(f"Notice collection error (dataset {self.id}): {e}")
        return notices

    @property
    def stimulus_voltages(self) -> np.ndarray:
        """Return sorted unique stimulus voltages binned by `bin_size`."""
        if not self.sessions:
            return np.array([])
        binned_voltages = set()
        for session in self.sessions:
            vols = np.round(np.array(session.stimulus_voltages) / self.bin_size) * self.bin_size
            binned_voltages.update(vols.tolist())
        return np.array(sorted(binned_voltages))

    def get_all_sessions(self, include_excluded=False):
        """
        Returns a list of all sessions in the dataset.
        If include_excluded is True, includes excluded sessions as well.
        """
        if include_excluded:
            return self._all_sessions
        return [sess for sess in self._all_sessions if sess.id not in self.excluded_sessions]

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
        if not self.sessions:
            logging.warning(f"No sessions available to add latency window '{name}' in dataset {self.id}.")
            return
        for session in self.sessions:
            session.add_latency_window(name, start_times, durations, color=color, linestyle=linestyle)
        self.update_latency_window_parameters()

    def remove_latency_window(self, name: str) -> None:
        if not self.sessions:
            logging.warning(f"No sessions available to remove latency window '{name}' in dataset {self.id}.")
            return
        for session in self.sessions:
            session.remove_latency_window(name)
        self.update_latency_window_parameters()

    def apply_latency_window_preset(self, preset_name: str) -> None:
        """Apply a latency window preset to all sessions in the dataset."""
        if not self.sessions:
            logging.warning(f"No sessions available to apply preset '{preset_name}' in dataset {self.id}.")
            return
        for session in self.sessions:
            session.apply_latency_window_preset(preset_name)
        self.update_latency_window_parameters()

    # ──────────────────────────────────────────────────────────────────
    # 0) Cached properties and cache reset methods
    # ──────────────────────────────────────────────────────────────────
    def reset_all_caches(self):
        """
        Resets all cached properties in the dataset.
        This is useful when the underlying data has changed and you need to refresh the cached values.
        """
        for session in self.sessions:
            session.reset_all_caches()
        self.update_latency_window_parameters()

    def update_latency_window_parameters(self):
        """
        Updates the latency window M/H-reflex parameters for all sessions in the dataset.
        """
        for session in self.sessions:
            session.update_latency_window_parameters()

    def apply_config(self, reset_caches: bool = True) -> None:
        """
        Applies user preferences to the dataset.
        This method is a placeholder for any future preferences that might be added.
        """
        for session in self.get_all_sessions(include_excluded=True):
            session.apply_config()

        self._load_config_settings()
        self.plotter = DatasetPlotterPyQtGraph(self)

        if reset_caches:
            self.reset_all_caches()

    def set_config(self, config: dict) -> None:
        """
        Update the configuration for this dataset and all child sessions.
        """
        self._config = config
        for sess in self.get_all_sessions(include_excluded=True):
            if hasattr(sess, "set_config"):
                sess.set_config(config)

        self.apply_config(reset_caches=True)

    # ──────────────────────────────────────────────────────────────────
    # 1) Useful properties for GUI & analysis code
    # ──────────────────────────────────────────────────────────────────
    def plot(self, plot_type: str | None = None, **kwargs):
        """
        Plots EMG data from a single session using the specified plot_type.

        Args:
            - plot_type (str): The type of plot to generate. Options include 'reflexCurves', 'mmax', and 'maxH'. Default is 'reflexCurves'.
                Plot types are defined in the EMGDatasetPlotter class in Plot_EMG.py.
            - channel_names (list): A list of channel names to plot. If None, all channels will be plotted.
            - **kwargs: Additional keyword arguments to pass to the plotting function.

                The most common keyword arguments include:
                - 'method' (str): The method to use for calculating the M-wave/reflex amplitudes. Options include 'average_rectified', 'rms', 'peak_to_trough', and 'average_unrectified'. Default method is set in config.yml under 'default_method'.
                - 'relative_to_mmax' (bool): Whether to plot the data proportional to the M-wave amplitude (True) or as the actual recorded amplitude (False). Default is False.
                - 'mmax_report' (bool): Whether to print the details of the M-max calculations (True) or not (False). Default is False.
                - 'manual_mmax' (float): The manually set M-wave amplitude to use for plotting the reflex curves. Default is None.

        Example Usages:
            # Plot the reflex curves for each channel.
            dataset.plot()

            # Plot M-wave amplitudes for each channel.
            dataset.plot(plot_type='mmax')

            # Plot the reflex curves for each channel.
            dataset.plot(plot_type='reflexCurves')

            # Plot the reflex curves for each channel and print the M-max details.
            dataset.plot(plot_type='reflexCurves', mmax_report=True)
        """
        # Call the appropriate plotting method from the plotter object
        raw_data = getattr(self.plotter, f'plot_{"reflexCurves" if not plot_type else plot_type}')(**kwargs)
        return raw_data

    def get_avg_m_max(self, method, channel_index, return_avg_mmax_thresholds=False):
        """
        Calculates the average M-wave amplitude for a specific channel in the dataset.

        Args:
            method (str): The method to use for calculating the M-wave amplitude. Options include 'average_rectified', 'rms', 'peak_to_trough', and 'average_unrectified'.
            channel_index (int): The index of the channel to calculate the M-wave amplitude for.
            return_avg_mmax_thresholds (bool): Whether to return the average M-max threshold values along with the amplitude. Default is False.

        Returns:
            float: The average M-wave amplitude for the specified channel.
        """
        m_max_amplitudes = []
        m_max_thresholds = []

        for session in self.sessions:
            try:
                m_max, mmax_low_stim, _ = session.get_m_max(method, channel_index, return_mmax_stim_range=True)
                m_max_amplitudes.append(m_max)
                m_max_thresholds.append(mmax_low_stim)

            except ValueError as e:
                logging.warning(f"M-max could not be calculated for session {session.id} channel {channel_index}: {e}")

        if not m_max_amplitudes:
            if return_avg_mmax_thresholds:
                return None, None
            return None

        # Calculate M-max for dataset level - use mean of all valid sessions
        if len(m_max_amplitudes) == 1:
            # Only one session, use its M-max
            final_mmax = float(m_max_amplitudes[0])
            final_mthresh = float(m_max_thresholds[0])
        else:
            # Multiple sessions: use mean M-max from all sessions
            # This provides proper population-level normalization
            final_mmax = float(np.mean(m_max_amplitudes))
            final_mthresh = float(np.mean(m_max_thresholds))

            logging.debug(f"Dataset M-max: Using mean from {len(m_max_amplitudes)} sessions")
            logging.debug(f"  M-max values: {m_max_amplitudes}")
            logging.debug(f"  Mean M-max: {final_mmax}")

        if return_avg_mmax_thresholds:
            return final_mmax, final_mthresh
        else:
            return final_mmax

    def get_avg_m_wave_amplitudes(self, method: str, channel_index: int):
        """Average M-wave amplitudes for each stimulus bin across sessions."""
        m_wave_bins = {v: [] for v in self.stimulus_voltages}
        for session in self.sessions:
            binned = np.round(np.array(session.stimulus_voltages) / self.bin_size) * self.bin_size
            m_wave = session.get_m_wave_amplitudes(method, channel_index)
            for volt, amp in zip(binned, m_wave):
                m_wave_bins[volt].append(amp)
        avg = [float(np.mean(m_wave_bins[v])) if m_wave_bins[v] else np.nan for v in self.stimulus_voltages]
        sem = [
            (float(np.std(m_wave_bins[v]) / np.sqrt(len(m_wave_bins[v]))) if m_wave_bins[v] else np.nan)
            for v in self.stimulus_voltages
        ]
        return avg, sem

    def get_m_wave_amplitudes_at_voltage(self, method: str, channel_index: int, voltage: float) -> np.ndarray:
        """
        Get M-wave amplitudes at a specific stimulus voltage across all sessions.
        """
        amps = []
        for session in self.sessions:
            binned = np.round(np.array(session.stimulus_voltages) / self.bin_size) * self.bin_size
            if voltage in binned:
                idx = np.where(binned == voltage)[0][0]
                amps.append(session.get_m_wave_amplitudes(method, channel_index)[idx])
        return np.array(amps)

    def get_avg_h_wave_amplitudes(self, method: str, channel_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Average H-reflex amplitudes for each stimulus bin across sessions."""
        h_wave_bins = {v: [] for v in self.stimulus_voltages}
        for session in self.sessions:
            binned = np.round(np.array(session.stimulus_voltages) / self.bin_size) * self.bin_size
            h_wave = session.get_h_wave_amplitudes(method, channel_index)
            for volt, amp in zip(binned, h_wave):
                h_wave_bins[volt].append(amp)
        avg = [float(np.mean(h_wave_bins[v])) if h_wave_bins[v] else np.nan for v in self.stimulus_voltages]
        std = [float(np.std(h_wave_bins[v])) if h_wave_bins[v] else np.nan for v in self.stimulus_voltages]
        return np.array(avg), np.array(std)

    def get_h_wave_amplitudes_at_voltage(self, method: str, channel_index: int, voltage: float) -> np.ndarray:
        amps = []
        for session in self.sessions:
            binned = np.round(np.array(session.stimulus_voltages) / self.bin_size) * self.bin_size
            if voltage in binned:
                idx = np.where(binned == voltage)[0][0]
                amps.append(session.get_h_wave_amplitudes(method, channel_index)[idx])
        return np.array(amps)

    def get_lw_reflex_amplitudes(
        self, method: str, channel_index: int, window: str | LatencyWindow
    ) -> dict[str, List[np.ndarray]]:
        """Return reflex amplitudes for a latency window across sessions.

        The window is resolved per session by name (case-insensitive). Sessions missing
        the window are skipped. This allows heterogeneous per-session latency windows.
        Returns mapping: session_id -> np.ndarray of amplitudes (ordered by that session's
        stimulus sequence).
        """
        if not self.sessions:
            return {}

        window_name: str
        if isinstance(window, LatencyWindow):
            window_name = window.name
        else:
            window_name = str(window)

        result: dict[str, List[np.ndarray]] = {}
        missing_sessions: list[str] = []
        for session in self.sessions:
            if self.get_session_latency_window(session, window_name) is None:
                missing_sessions.append(session.id)
                continue
            result[session.id] = session.get_lw_reflex_amplitudes(method, channel_index, window_name)

        if missing_sessions:
            logging.warning(f"Dataset {self.id}: latency window '{window_name}' absent in sessions: {missing_sessions}.")
        return result

    def get_average_lw_reflex_curve(
        self, method: str, channel_index: int, window: str | LatencyWindow
    ) -> dict[str, np.ndarray]:
        """
        Returns the average reflex curve for a specific latency window across all sessions in the dataset.

        Curve's X-axis is the stimulus voltage (float), Y-axis is the average reflex amplitude (float) for a given channel/window.
        """
        if not self.sessions:
            return {"voltages": np.array([]), "means": np.array([]), "stdevs": np.array([]), "n_sessions": np.array([])}

        window_name: str
        if isinstance(window, LatencyWindow):
            window_name = window.name
        else:
            window_name = str(window)

        # Get all reflex amplitudes across available sessions
        all_amplitudes = self.get_lw_reflex_amplitudes(method, channel_index, window_name)
        if not all_amplitudes:
            return {"voltages": np.array([]), "means": np.array([]), "stdevs": np.array([]), "n_sessions": np.array([])}

        voltages_union = self.stimulus_voltages
        bin_amplitudes: dict[float, list[float]] = {v: [] for v in voltages_union}
        contrib_counts: dict[float, int] = {v: 0 for v in voltages_union}

        for session_id, amps in all_amplitudes.items():
            session = next((s for s in self.sessions if s.id == session_id), None)
            if session is None:
                continue
            binned = np.round(np.array(session.stimulus_voltages) / self.bin_size) * self.bin_size
            # Build a temporary mapping for the session to count contribution per voltage bin
            seen_bins: set[float] = set()
            for v, amp in zip(binned, amps):
                if v in bin_amplitudes:
                    bin_amplitudes[v].append(amp)
                    seen_bins.add(v)
            for v in seen_bins:
                contrib_counts[v] += 1

        sorted_volts = sorted(bin_amplitudes.keys())
        means = [float(np.mean(bin_amplitudes[v])) if bin_amplitudes[v] else np.nan for v in sorted_volts]
        stdevs = [float(np.std(bin_amplitudes[v])) if bin_amplitudes[v] else np.nan for v in sorted_volts]
        n_sessions = [contrib_counts[v] for v in sorted_volts]

        return {
            "voltages": np.array(sorted_volts),
            "means": np.array(means),
            "stdevs": np.array(stdevs),
            "n_sessions": np.array(n_sessions),
        }

    # ──────────────────────────────────────────────────────────────────
    # 2) User actions that update annot files
    # ──────────────────────────────────────────────────────────────────
    def invert_channel_polarity(self, channel_index: int) -> None:
        """
        Inverts the polarity of a specific channel across all sessions in the dataset.

        Args:
            channel_index (int): The index of the channel to invert.
        """
        for session in self.sessions:
            session.invert_channel_polarity(channel_index)
        self.reset_all_caches()
        logging.info(f"Inverted polarity of channel {channel_index} in dataset {self.id}.")

    def exclude_session(self, session_id: str) -> None:
        """Exclude a session from this dataset by its ID."""
        if session_id not in [s.id for s in self.get_all_sessions(include_excluded=True)]:
            logging.warning(f"Session {session_id} not found in dataset {self.id}.")
            return
        if session_id not in self.annot.excluded_sessions:
            self.annot.excluded_sessions.append(session_id)
            self.reset_all_caches()
            if self.repo is not None:
                self.repo.save(self)
        else:
            logging.warning(f"Session {session_id} already excluded in dataset {self.id}.")

        if self.sessions == []:
            logging.info(f"All sessions in dataset {self.id} have been excluded.")
            # If no sessions remain, optionally notify parent experiment
            if self.parent_experiment is not None:
                self.parent_experiment.exclude_dataset(self.id)

    def restore_session(self, session_id: str) -> None:
        """Restore a previously excluded session by its ID."""
        if session_id in self.annot.excluded_sessions:
            self.annot.excluded_sessions.remove(session_id)
            self.reset_all_caches()
            if self.repo is not None:
                self.repo.save(self)
        else:
            logging.warning(f"Session {session_id} is not excluded from dataset {self.id}.")

    def rename_channels(self, new_names: dict[str, str]) -> None:
        """
        Renames channels in the dataset based on the provided mapping.

        Args:
            new_names (dict): A dictionary mapping old channel names to new channel names.
        """
        for session in self.sessions:
            session.rename_channels(new_names)
        logging.info(f"Renamed channels in dataset {self.id} according to mapping: {new_names}.")

    # ──────────────────────────────────────────────────────────────────
    # Utility methods
    # ──────────────────────────────────────────────────────────────────
    def __check_session_consistency(self):
        """
        Checks if all sessions in the dataset have the same parameters (scan rate, num_channels, stim_start).

        Returns:
            tuple: A tuple containing a boolean value indicating whether all sessions have consistent parameters and a message indicating the result.
        """
        # Ensure there are sessions to check and that they contain metadata.
        # Perform strict consistency checks only when session metadata is available.
        if not self.sessions:
            return
        # Find a session with complete metadata to use as reference
        reference_session = None
        for s in self.sessions:
            if (
                getattr(s, "scan_rate", None) is not None
                and getattr(s, "num_channels", None) is not None
                and getattr(s, "primary_stim", None) is not None
            ):
                reference_session = s
                break
        if reference_session is None:
            # No session has complete metadata yet; skip consistency check
            return
        reference_scan_rate = reference_session.scan_rate
        reference_num_channels = reference_session.num_channels
        reference_stim_type = reference_session.primary_stim.stim_type

        for session in self.sessions[1:]:
            if session.scan_rate != reference_scan_rate:
                raise ValueError(
                    f"Inconsistent scan rate for {session.id} in {self.formatted_name}: {session.scan_rate} != {reference_scan_rate}."
                )
            if session.num_channels != reference_num_channels:
                raise ValueError(
                    f"Inconsistent number of channels for {session.id} in {self.formatted_name}: {session.num_channels} != {reference_num_channels}."
                )
            if getattr(session, "primary_stim", None) is None:
                # Session missing primary_stim; skip stimulus consistency for now
                continue
            if session.primary_stim.stim_type != reference_stim_type:
                raise ValueError(
                    f"Inconsistent primary stimulus for {session.id} in {self.formatted_name}: {session.primary_stim.stim_type} != {reference_stim_type}."
                )

    # ──────────────────────────────────────────────────────────────────
    # 2) Clean up
    # ──────────────────────────────────────────────────────────────────
    def close(self, force_gc: bool = True) -> None:
        """Close all sessions in the dataset.

        Args:
            force_gc: If True, force garbage collection after closing.
                     Set to False when closing as part of full experiment.
        """
        for sess in self.sessions:
            if hasattr(sess, "close"):
                # Don't GC at session level when closing dataset/experiment
                sess.close(force_gc=False)
            else:
                raise NotImplementedError(f"Session {sess.id} does not have a close method.")

        # Force GC when closing dataset individually (not as part of experiment)
        if force_gc:
            import gc

            gc.collect()

    # ──────────────────────────────────────────────────────────────────
    # 3) Object representation and reports
    # ──────────────────────────────────────────────────────────────────
    def dataset_parameters(self):
        """
        Logs EMG dataset parameters.
        """
        report = [
            f"Dataset Parameters for '{self.formatted_name}':",
            "===============================",
            f"EMG Sessions ({len(self.sessions)}): {[session.id for session in self.sessions]}.",
            f"Date: {self.date}",
            f"Animal ID: {self.animal_id}",
            f"Condition: {self.condition}",
        ]

        for line in report:
            logging.info(line)
        return report

    def __repr__(self) -> str:
        return f"Dataset(dataset_id={self.id}, num_sessions={self.num_sessions})"

    def __str__(self) -> str:
        return f"Dataset: {self.id} with {self.num_sessions} sessions"

    def __len__(self) -> int:
        return self.num_sessions

    def __bool__(self) -> bool:
        """
        A Dataset instance represents a real dataset, even if it currently has
        zero sessions (e.g., after exclusion or during import). Make it truthy
        to avoid accidental falsy evaluation via __len__.
        """
        return True
