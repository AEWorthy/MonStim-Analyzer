# monstim_signals/domain/dataset.py
from typing import List, Any, TYPE_CHECKING
import logging
import numpy as np

from monstim_signals.plotting.dataset_plotter import DatasetPlotter
from monstim_signals.domain.session import Session
from monstim_signals.core.data_models import DatasetAnnot, LatencyWindow
from monstim_signals.core.utils import load_config
from monstim_signals.transform import calculate_emg_amplitude

if TYPE_CHECKING:
    
    
    from monstim_signals.io.repositories import DatasetRepository

class Dataset:
    """
    A “dataset” = all sessions from one animal replicate.
    E.g. Dataset_1(Animal_A) has sessions AA00, AA01, …
    """
    def __init__(self, dataset_id: str, sessions: List[Session], annot: DatasetAnnot, repo: Any = None):
        self.id : str = dataset_id
        self._all_sessions : List[Session] = sessions
        self.annot         : DatasetAnnot = annot
        self.repo          : DatasetRepository = repo

        self._load_config_settings()

        self.plotter = DatasetPlotter(self)

        # Ensure all sessions share the same recording parameters
        self.__check_session_consistency()

        self.scan_rate: int = self.sessions[0].scan_rate
        self.stim_start: float = self.sessions[0].stim_start

        self.update_latency_window_parameters()
        logging.info(f"Dataset {self.id} initialized with {len(self.sessions)} sessions.")

    def _load_config_settings(self) -> None:
        _config = load_config()
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
    def excluded_sessions(self) -> List[Session]:
        """
        Returns a list of sessions that are excluded from the dataset.
        This is useful for filtering out sessions that do not meet certain criteria.
        """
        return set(self.annot.excluded_sessions)

    @property
    def sessions(self) -> List[Session]:
        return [sess for sess in self._all_sessions if sess.id not in self.excluded_sessions]

    @property
    def num_channels(self) -> int:
        return min(session.num_channels for session in self.sessions)

    @property
    def channel_names(self) -> List[str]:
        return max((session.channel_names for session in self.sessions), key=len)

    @property
    def latency_windows(self) -> List[LatencyWindow]:
        return self.sessions[0].latency_windows

    @property
    def stimulus_voltages(self) -> np.ndarray:
        """Return sorted unique stimulus voltages binned by `bin_size`."""
        binned_voltages = set()
        for session in self.sessions:
            vols = np.round(np.array(session.stimulus_voltages) / self.bin_size) * self.bin_size
            binned_voltages.update(vols.tolist())
        return np.array(sorted(binned_voltages))
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
        for window in self.latency_windows:
            if window.name == "M-wave":
                self.m_start = window.start_times
                self.m_duration = window.durations
            elif window.name == "H-reflex":
                self.h_start = window.start_times
                self.h_duration = window.durations
        for session in self.sessions:
            session.update_latency_window_parameters()
    # ──────────────────────────────────────────────────────────────────
    # 1) Useful properties for GUI & analysis code
    # ──────────────────────────────────────────────────────────────────
    def plot(self, plot_type: str = None, **kwargs):
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
        avg = [np.mean(m_wave_bins[v]) for v in self.stimulus_voltages]
        sem = [np.std(m_wave_bins[v]) / np.sqrt(len(m_wave_bins[v])) for v in self.stimulus_voltages]
        return avg, sem

    def get_m_wave_amplitudes_at_voltage(self, method: str, channel_index: int, voltage: float) -> List[float]:
        amps = []
        for session in self.sessions:
            binned = np.round(np.array(session.stimulus_voltages) / self.bin_size) * self.bin_size
            if voltage in binned:
                idx = np.where(binned == voltage)[0][0]
                amps.append(session.get_m_wave_amplitudes(method, channel_index)[idx])
        return amps

    def get_avg_h_wave_amplitudes(self, method: str, channel_index: int):
        """Average H-reflex amplitudes for each stimulus bin across sessions."""
        h_wave_bins = {v: [] for v in self.stimulus_voltages}
        for session in self.sessions:
            binned = np.round(np.array(session.stimulus_voltages) / self.bin_size) * self.bin_size
            h_wave = session.get_h_wave_amplitudes(method, channel_index)
            for volt, amp in zip(binned, h_wave):
                h_wave_bins[volt].append(amp)
        avg = [np.mean(h_wave_bins[v]) for v in self.stimulus_voltages]
        std = [np.std(h_wave_bins[v]) for v in self.stimulus_voltages]
        return avg, std

    def get_h_wave_amplitudes_at_voltage(self, method: str, channel_index: int, voltage: float) -> List[float]:
        amps = []
        for session in self.sessions:
            binned = np.round(np.array(session.stimulus_voltages) / self.bin_size) * self.bin_size
            if voltage in binned:
                idx = np.where(binned == voltage)[0][0]
                amps.append(session.get_h_wave_amplitudes(method, channel_index)[idx])
        return amps

    def session_response_map(self, channel: int, window: tuple[float, float]) -> dict[str, list[float]]:
        """Return per-session response amplitudes within ``window``.

        This helper is used by :class:`Experiment` to build aggregated response
        maps across multiple datasets."""
        start_ms, end_ms = window
        result: dict[str, list[float]] = {}
        for session in self.sessions:
            amplitudes = []
            for rec_array in session.recordings_filtered:
                amp = calculate_emg_amplitude(
                    rec_array[:, channel],
                    start_ms + session.stim_start,
                    end_ms + session.stim_start,
                    session.scan_rate,
                    method=self.default_method,
                )
                amplitudes.append(amp)
            result[session.id] = amplitudes
        return result

    # ──────────────────────────────────────────────────────────────────
    # 2) User actions that update annot files
    # ──────────────────────────────────────────────────────────────────
    def change_reflex_latency_windows(self, m_start, m_duration, h_start, h_duration):
        for window in self.latency_windows:
            if window.name == "M-wave":
                window.start_times = m_start
                window.durations = m_duration
            elif window.name == "H-reflex":
                window.start_times = h_start
                window.durations = h_duration
        for session in self.sessions:
            session.change_reflex_latency_windows(m_start, m_duration, h_start, h_duration)
        self.update_latency_window_parameters()
        logging.info(f"Changed reflex latency windows for dataset {self.id} to M-wave start: {m_start}, duration: {m_duration}, H-reflex start: {h_start}, duration: {h_duration}.")
    
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
        report = [f"EMG Sessions ({len(self.sessions)}): {[session.id for session in self.sessions]}.",
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

