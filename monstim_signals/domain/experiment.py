# monstim_signals/domain/experiment.py
from typing import List, Any
import logging
import numpy as np

from monstim_signals.domain.dataset import Dataset
from monstim_signals.plotting.experiment_plotter import EMGExperimentPlotter
from monstim_signals.core.data_models import ExperimentAnnot, LatencyWindow
from monstim_signals.core.utils import load_config

class Experiment:
    """A collection of :class:`Dataset` objects."""

    def __init__(self, expt_id: str, datasets: List[Dataset],
                 annot: ExperimentAnnot | None = None, repo: Any = None):
        self.id = expt_id
        self._all_datasets: List[Dataset] = datasets
        self.annot: ExperimentAnnot = annot or ExperimentAnnot.create_empty()
        self.repo = repo

        self._load_config_settings()
        self.plotter = EMGExperimentPlotter(self)

        self.__check_dataset_consistency()
        if self.datasets:
            self.scan_rate = self.datasets[0].scan_rate
            self.stim_start = self.datasets[0].stim_start
        self.update_latency_window_parameters()

    def __check_dataset_consistency(self) -> None:
        if not self.datasets:
            return
        ref = self.datasets[0]
        ref_rate = ref.scan_rate
        ref_channels = ref.num_channels
        ref_stim = ref.stim_start
        warnings = []
        for ds in self.datasets[1:]:
            if ds.scan_rate != ref_rate:
                warnings.append(f"Inconsistent scan_rate for '{ds.id}': {ds.scan_rate} != {ref_rate}.")
            if ds.num_channels != ref_channels:
                warnings.append(f"Inconsistent num_channels for '{ds.id}': {ds.num_channels} != {ref_channels}.")
            if ds.stim_start != ref_stim:
                warnings.append(f"Inconsistent stim_start for '{ds.id}': {ds.stim_start} != {ref_stim}.")
        for w in warnings:
            logging.warning(w)

    def _load_config_settings(self) -> None:
        _config = load_config()
        self.bin_size = _config["bin_size"]
        self.default_method = _config["default_method"]
        self.m_color = _config["m_color"]
        self.h_color = _config["h_color"]
        self.title_font_size = _config["title_font_size"]
        self.axis_label_font_size = _config["axis_label_font_size"]
        self.tick_font_size = _config["tick_font_size"]
        self.subplot_adjust_args = _config["subplot_adjust_args"]

    @property
    def num_datasets(self) -> int:
        return len(self.datasets)

    @property
    def excluded_datasets(self) -> set[str]:
        return set(self.annot.excluded_datasets)

    @property
    def datasets(self) -> List[Dataset]:
        return [ds for ds in self._all_datasets if ds.id not in self.excluded_datasets]

    @property
    def emg_datasets(self) -> List[Dataset]:
        return self.datasets

    @property
    def num_channels(self) -> int:
        return min(ds.num_channels for ds in self.datasets)

    @property
    def channel_names(self) -> List[str]:
        return max((ds.channel_names for ds in self.datasets), key=len)

    @property
    def latency_windows(self) -> List[LatencyWindow]:
        return max((ds.latency_windows for ds in self.datasets), key=len)

    @property
    def stimulus_voltages(self) -> np.ndarray:
        binned_voltages = set()
        for ds in self.datasets:
            volts = np.round(np.array(ds.stimulus_voltages) / self.bin_size) * self.bin_size
            binned_voltages.update(volts.tolist())
        return np.array(sorted(binned_voltages))

    # ──────────────────────────────────────────────────────────────────
    # 1) Useful properties for GUI & analysis code
    # ──────────────────────────────────────────────────────────────────
    # Example: gather session “H‐reflex curves” for every dataset & session:
    #    Returns a nested dict: { "Animal_A": { "Session_01": [ … ], … }, … }
    # ──────────────────────────────────────────────────────────────────
    def experiment_response_map(self, channel: int, window) -> dict[str, dict[str, list[float]]]:
        result = {}
        for ds in self.datasets:
            ds_map = ds.session_response_map(channel=channel, window=window)
            result[ds.id] = ds_map
        return result

    def plot(self, plot_type: str = None, **kwargs):
        raw_data = getattr(self.plotter, f"plot_{'reflexCurves' if not plot_type else plot_type}")(**kwargs)
        return raw_data

    def invert_channel_polarity(self, channel_index: int) -> None:
        for ds in self.datasets:
            for sess in ds.sessions:
                sess.invert_channel_polarity(channel_index)

    def rename_experiment(self, new_name: str) -> None:
        self.id = new_name.replace(' ', '_')
        logging.info(f"Experiment renamed to '{new_name}'.")

    def add_dataset(self, dataset: Dataset) -> None:
        if dataset.id not in [ds.id for ds in self._all_datasets]:
            self._all_datasets.append(dataset)
            self.reset_all_caches()

    def remove_dataset(self, dataset_id: str) -> None:
        self._all_datasets = [ds for ds in self._all_datasets if ds.id != dataset_id]
        self.reset_all_caches()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _aggregate_wave_amplitudes(self, method: str, channel_index: int,
                                   amplitude_func) -> tuple[list[float], list[float]]:
        """Aggregate amplitudes across datasets.

        Parameters
        ----------
        method:
            Amplitude calculation method.
        channel_index:
            Index of the channel to aggregate.
        amplitude_func:
            Function returning ``(avg, err)`` for a dataset.

        Returns
        -------
        tuple[list[float], list[float]]
            Averaged amplitudes and standard errors for each stimulus bin.
        """
        bins: dict[float, list[float]] = {v: [] for v in self.stimulus_voltages}
        for ds in self.datasets:
            binned_voltages = np.round(np.array(ds.stimulus_voltages) / self.bin_size) * self.bin_size
            amps, _ = amplitude_func(ds)
            for v, amp in zip(binned_voltages, amps):
                bins[v].append(amp)

        avg = [float(np.mean(bins[v])) for v in self.stimulus_voltages]
        sem = [float(np.std(bins[v]) / np.sqrt(len(bins[v]))) if bins[v] else 0.0
               for v in self.stimulus_voltages]
        return avg, sem

    def get_avg_m_wave_amplitudes(self, method: str, channel_index: int):
        return self._aggregate_wave_amplitudes(
            method=method,
            channel_index=channel_index,
            amplitude_func=lambda ds: ds.get_avg_m_wave_amplitudes(method, channel_index)
        )

    def get_m_wave_amplitude_avgs_at_voltage(self, method: str, channel_index: int, voltage: float) -> List[float]:
        amps = []
        for ds in self.datasets:
            if voltage in ds.stimulus_voltages:
                idx = np.where(ds.stimulus_voltages == voltage)[0][0]
                avg, _ = ds.get_avg_m_wave_amplitudes(method, channel_index)
                amps.append(avg[idx])
        return amps

    def get_avg_h_wave_amplitudes(self, method: str, channel_index: int):
        h_wave_bins = {v: [] for v in self.stimulus_voltages}
        for ds in self.datasets:
            binned = np.round(np.array(ds.stimulus_voltages) / self.bin_size) * self.bin_size
            h_wave, _ = ds.get_avg_h_wave_amplitudes(method, channel_index)
            for volt, amp in zip(binned, h_wave):
                h_wave_bins[volt].append(amp)
        avg = [np.mean(h_wave_bins[v]) for v in self.stimulus_voltages]
        std = [np.std(h_wave_bins[v]) for v in self.stimulus_voltages]
        return avg, std

    def get_h_wave_amplitude_avgs_at_voltage(self, method: str, channel_index: int, voltage: float) -> List[float]:
        amps = []
        for ds in self.datasets:
            if voltage in ds.stimulus_voltages:
                idx = np.where(ds.stimulus_voltages == voltage)[0][0]
                avg, _ = ds.get_avg_h_wave_amplitudes(method, channel_index)
                amps.append(avg[idx])
        return amps

    def get_avg_m_max(self, method: str, channel_index: int, return_avg_mmax_thresholds: bool = False):
        m_max, m_thresh = zip(*[
            ds.get_avg_m_max(method, channel_index, return_avg_mmax_thresholds=True)[:2]
            for ds in self.datasets if ds.get_avg_m_max(method, channel_index) is not None
        ])
        if return_avg_mmax_thresholds:
            return np.mean(m_max), np.mean(m_thresh)
        else:
            return np.mean(m_max)

    def reset_all_caches(self):
        for ds in self.datasets:
            ds.reset_all_caches()
        self.update_latency_window_parameters()

    def update_latency_window_parameters(self):
        for window in self.latency_windows:
            if window.name == "M-wave":
                self.m_start = window.start_times
                self.m_duration = window.durations
            elif window.name == "H-reflex":
                self.h_start = window.start_times
                self.h_duration = window.durations
        for ds in self.datasets:
            ds.update_latency_window_parameters()
    
    # ──────────────────────────────────────────────────────────────────
    # 2) Clean up
    # ──────────────────────────────────────────────────────────────────
    def close(self) -> None:
        """
        Close all datasets in the experiment.
        This is a placeholder for any cleanup logic needed.
        """
        for ds in self.datasets:
            if hasattr(ds, 'close_all'):
                ds.close()
            else:
                raise NotImplementedError(f"Dataset {ds.id} does not have a close_all method.")

    def experiment_parameters(self):
        report = [f"Experiment ID: {self.id}",
                  f"Datasets ({len(self.datasets)}): {[ds.id for ds in self.datasets]}."]
        for line in report:
            logging.info(line)
        return report
    
    # ──────────────────────────────────────────────────────────────────
    # 3) Object representation
    # ──────────────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        return f"Experiment(expt_id={self.id}, num_datasets={self.num_datasets})"
    def __str__(self) -> str:
        return f"Experiment: '{self.id}' with {self.num_datasets} datasets"
    def __len__(self) -> int:
        return self.num_datasets

