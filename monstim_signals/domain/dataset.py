# monstim_signals/domain/dataset.py
from typing import List, Any, TYPE_CHECKING
import logging
import numpy as np

from monstim_signals.plotting import DatasetPlotterPyQtGraph
from monstim_signals.domain.session import Session
from monstim_signals.core import DatasetAnnot, LatencyWindow, load_config

if TYPE_CHECKING:
    from monstim_signals.io.repositories import DatasetRepository
    from monstim_signals.domain.experiment import Experiment

class Dataset:
    """
    A “dataset” = all sessions from one animal replicate.
    E.g. Dataset_1(Animal_A) has sessions AA00, AA01, …
    """
    def __init__(self, dataset_id: str, sessions: List[Session], annot: DatasetAnnot, repo: Any = None, config: dict | None = None):
        self.id : str = dataset_id
        self._all_sessions : List[Session] = sessions
        self.annot         : DatasetAnnot = annot
        self.repo          : DatasetRepository = repo
        self.parent_experiment : 'Experiment | None' = None
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
            logging.warning(f"Dataset {self.id} has no sessions to check for consistency. Skipping consistency check. Setting scan_rate and stim_start to zeroes.")
            self.scan_rate = 0
            self.stim_start = 0.0

        self.update_latency_window_parameters()
        logging.info(f"Dataset {self.id} initialized with {len(self.sessions)} sessions.")

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
        return self.annot.date if self.annot.date else 'Undefined'
    @property
    def animal_id(self) -> str:
        """
        Returns the animal ID of the dataset.
        If the animal ID is not set, returns an empty string.
        """
        return self.annot.animal_id if self.annot.animal_id else 'Undefined'
    @property
    def condition(self) -> str:
        """
        Returns the condition of the dataset.
        If the condition is not set, returns an empty string.
        """
        return self.annot.condition if self.annot.condition else 'Undefined'
    @property
    def formatted_name(self) -> str:
        """
        Returns the formatted name of the dataset.
        If the date, animal_id, or condition is not set, returns the dataset folder name.
        """
        return f"{self.date} {self.animal_id} {self.condition}" if self.date and self.animal_id and self.condition else self.id
    @property
    def num_sessions(self) -> int:
        return len(self.sessions)
    @property
    def excluded_sessions(self) -> set[str]:
        """IDs of sessions excluded from this dataset."""
        return set(self.annot.excluded_sessions)
    @property
    def sessions(self) -> List[Session]:
        return [sess for sess in self._all_sessions if sess.id not in self.excluded_sessions]
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
        if not self.sessions:
            return []
        return self.sessions[0].latency_windows
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
    # ------------------------------------------------------------------
    # Latency window helper methods
    # ------------------------------------------------------------------
    def add_latency_window(self, name: str, start_times: List[float],
                           durations: List[float], color: str | None = None,
                           linestyle: str | None = None) -> None:
        if not self.sessions:
            logging.warning(f"No sessions available to add latency window '{name}' in dataset {self.id}.")
            return
        for session in self.sessions:
            session.add_latency_window(name, start_times, durations,
                                       color=color, linestyle=linestyle)
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

    def get_latency_window(self, name: str) -> LatencyWindow | None:
        if not self.sessions:
            return None
        return self.sessions[0].get_latency_window(name)
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
        self.m_start = [0.0] * self.num_channels
        self.m_duration = [0.0] * self.num_channels
        self.h_start = [0.0] * self.num_channels
        self.h_duration = [0.0] * self.num_channels
        for window in self.latency_windows:
            lname = window.name.lower()
            if lname in {"m-wave", "m_wave", "m wave", "mwave"}:
                self.m_start = window.start_times
                self.m_duration = window.durations
            elif lname in {"h-reflex", "h_reflex", "h reflex", "hreflex"}:
                self.h_start = window.start_times
                self.h_duration = window.durations
        for session in self.sessions:
            session.update_latency_window_parameters()

    def apply_config(self, reset_caches: bool = True) -> None:
        """
        Applies user preferences to the dataset.
        This method is a placeholder for any future preferences that might be added.
        """
        for session in self._all_sessions:
            session.apply_config()

        self._load_config_settings()
        self.plotter = DatasetPlotterPyQtGraph(self)
        
        if reset_caches:
            self.reset_all_caches()
        if self.repo is not None:
            self.repo.save(self)

    def set_config(self, config: dict) -> None:
        """
        Update the configuration for this dataset and all child sessions.
        """
        self._config = config
        for sess in self._all_sessions:
            if hasattr(sess, 'set_config'):
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
                m_max, mmax_low_stim, _ = session.get_m_max(
                    method, channel_index, return_mmax_stim_range=True
                )
                m_max_amplitudes.append(m_max)
                m_max_thresholds.append(mmax_low_stim)
                
            except ValueError as e:

                logging.warning(
                    f"M-max could not be calculated for session {session.id} channel {channel_index}: {e}"
                )

        if not m_max_amplitudes:
            if return_avg_mmax_thresholds:
                return None, None
            return None

        if return_avg_mmax_thresholds:
            return float(np.mean(m_max_amplitudes)), float(np.mean(m_max_thresholds))
        else:
            return float(np.mean(m_max_amplitudes))

    def get_avg_m_wave_amplitudes(self, method: str, channel_index: int):
        """Average M-wave amplitudes for each stimulus bin across sessions."""
        m_wave_bins = {v: [] for v in self.stimulus_voltages}
        for session in self.sessions:
            binned = np.round(np.array(session.stimulus_voltages) / self.bin_size) * self.bin_size
            m_wave = session.get_m_wave_amplitudes(method, channel_index)
            for volt, amp in zip(binned, m_wave):
                m_wave_bins[volt].append(amp)
        avg = [float(np.mean(m_wave_bins[v])) if m_wave_bins[v] else np.nan for v in self.stimulus_voltages]
        sem = [float(np.std(m_wave_bins[v]) / np.sqrt(len(m_wave_bins[v]))) if m_wave_bins[v] else np.nan for v in self.stimulus_voltages]
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

    def get_lw_reflex_amplitudes(self, method: str, channel_index: int, 
                                      window: str | LatencyWindow) -> dict[str, List[np.ndarray]]:
        """Returns reflex amplitudes for a specific latency window across all sessions in the dataset."""
        if not self.sessions:
            return {}

        if not isinstance(window, LatencyWindow):
            window = self.get_latency_window(window)
            if window is None:
                logging.warning(f"Latency window '{window}' not found in dataset {self.id}.")
                return {}

        result: dict[str, List[np.ndarray]] = {}
        for session in self.sessions:
            result[session.id] = session.get_lw_reflex_amplitudes(method, channel_index, window.name)
        return result

    def get_average_lw_reflex_curve(self, method: str, channel_index: int, window: str | LatencyWindow) -> dict[str, np.ndarray]:
        """
        Returns the average reflex curve for a specific latency window across all sessions in the dataset.

        Curve's X-axis is the stimulus voltage (float), Y-axis is the average reflex amplitude (float) for a given channel/window.
        """
        if not self.sessions:
            return {"voltages": [], "means": [], "stdevs": []}
        if not isinstance(window, LatencyWindow):
            window = self.get_latency_window(window)
            if window is None:
                logging.warning(f"Latency window '{window}' not found in dataset {self.id}.")
                return {"voltages": [], "means": [], "stdevs": []}

        # Get all reflex amplitudes for all sessions using the dataset-level method
        all_amplitudes = self.get_lw_reflex_amplitudes(method, channel_index, window)
        # all_amplitudes: dict[session_id, List[float]]
        # Need to bin by voltage
        voltages = self.stimulus_voltages
        # Build a list of amplitudes for each voltage bin across sessions
        bin_amplitudes = {v: [] for v in voltages}
        for session_id, amps in all_amplitudes.items():
            session = next((s for s in self.sessions if s.id == session_id), None)
            if session is None:
                continue
            binned = np.round(np.array(session.stimulus_voltages) / self.bin_size) * self.bin_size
            for v, amp in zip(binned, amps):
                if v in bin_amplitudes:
                    bin_amplitudes[v].append(amp)
        # Compute the mean and standard deviation for each voltage bin
        voltages = sorted(bin_amplitudes.keys())
        means = [float(np.mean(bin_amplitudes[v])) if bin_amplitudes[v] else np.nan for v in voltages]
        stdevs = [float(np.std(bin_amplitudes[v])) if bin_amplitudes[v] else np.nan for v in voltages]
        return {"voltages": np.array(voltages), "means": np.array(means), "stdevs": np.array(stdevs)}
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

    def change_reflex_latency_windows(self, m_start, m_duration, h_start, h_duration):
        m_window = self.get_latency_window("M-wave")
        if m_window:
            m_window.start_times = m_start
            m_window.durations = m_duration
        h_window = self.get_latency_window("H-reflex")
        if h_window:
            h_window.start_times = h_start
            h_window.durations = h_duration
        for session in self.sessions:
            session.change_reflex_latency_windows(m_start, m_duration, h_start, h_duration)
        self.update_latency_window_parameters()
        logging.info(f"Changed reflex latency windows for dataset {self.id} to M-wave start: {m_start}, duration: {m_duration}, H-reflex start: {h_start}, duration: {h_duration}.")

    def exclude_session(self, session_id: str) -> None:
        """Exclude a session from this dataset by its ID."""
        if session_id not in [s.id for s in self._all_sessions]:
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
            # If no sessions remain, clear list and exclude dataset from parent experiment
            if self.parent_experiment is not None:
                self.parent_experiment.exclude_dataset(self.id)
            self.annot.excluded_sessions.clear()
            logging.info(f"Dataset {self.id} has no remaining sessions and is now excluded from the parent experiment.")

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
        reference_session = self.sessions[0]
        reference_scan_rate = reference_session.scan_rate
        reference_num_channels = reference_session.num_channels
        reference_stim_type = reference_session.primary_stim.stim_type

        for session in self.sessions[1:]:
            if session.scan_rate != reference_scan_rate:
                raise ValueError(f"Inconsistent scan rate for {session.id} in {self.formatted_name}: {session.scan_rate} != {reference_scan_rate}.")
            if session.num_channels != reference_num_channels:
                raise ValueError(f"Inconsistent number of channels for {session.id} in {self.formatted_name}: {session.num_channels} != {reference_num_channels}.")
            if session.primary_stim.stim_type != reference_stim_type:
                raise ValueError(f"Inconsistent primary stimulus for {session.id} in {self.formatted_name}: {session.primary_stim.stim_type} != {reference_stim_type}.")
    # ──────────────────────────────────────────────────────────────────
    # 2) Clean up
    # ──────────────────────────────────────────────────────────────────
    def close(self) -> None:
        """
        Close all sessions in the dataset.
        This is a placeholder for any cleanup logic needed.
        """
        for sess in self.sessions:
            if hasattr(sess, 'close'):
                sess.close()
            else:
                raise NotImplementedError(f"Session {sess.id} does not have a close method.")
    # ──────────────────────────────────────────────────────────────────
    # 3) Object representation and reports
    # ──────────────────────────────────────────────────────────────────
    def dataset_parameters(self):
        """
        Logs EMG dataset parameters.
        """
        report = [f"Dataset Parameters for '{self.formatted_name}':",
                  "===============================",
                  f"EMG Sessions ({len(self.sessions)}): {[session.id for session in self.sessions]}.",
                  f"Date: {self.date}",
                  f"Animal ID: {self.animal_id}",
                  f"Condition: {self.condition}"]

        for line in report:
            logging.info(line)
        return report
    def __repr__(self) -> str:
        return f"Dataset(dataset_id={self.id}, num_sessions={self.num_sessions})"
    def __str__(self) -> str:
        return f"Dataset: {self.id} with {self.num_sessions} sessions"
    def __len__(self) -> int:
        return self.num_sessions

