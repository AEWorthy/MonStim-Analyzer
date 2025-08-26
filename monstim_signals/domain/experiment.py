# monstim_signals/domain/experiment.py
import logging
from typing import TYPE_CHECKING, Any, List

import numpy as np

from monstim_signals.core import ExperimentAnnot, LatencyWindow, load_config
from monstim_signals.domain.dataset import Dataset
from monstim_signals.plotting import ExperimentPlotterPyQtGraph

if TYPE_CHECKING:
    from monstim_signals.io.repositories import ExperimentRepository


class Experiment:
    """A collection of :class:`Dataset` objects."""

    def __init__(
        self,
        expt_id: str,
        datasets: List[Dataset],
        annot: ExperimentAnnot | None = None,
        repo: Any = None,
        config: dict = None,
    ):
        self.id = expt_id
        self._all_datasets: List[Dataset] = datasets
        for ds in self._all_datasets:
            ds.parent_experiment = self
        self.annot: ExperimentAnnot = annot or ExperimentAnnot.create_empty()
        self.repo: "ExperimentRepository" = repo
        self._config = config

        self._load_config_settings()
        self.plotter = ExperimentPlotterPyQtGraph(self)

        self.__check_dataset_consistency()
        if self.datasets:
            self.scan_rate = self.datasets[0].scan_rate
            self.stim_start = self.datasets[0].stim_start
        self.update_latency_window_parameters()

    @property
    def is_completed(self) -> bool:
        return getattr(self.annot, "is_completed", False)

    @is_completed.setter
    def is_completed(self, value: bool) -> None:
        self.annot.is_completed = bool(value)
        if self.repo is not None:
            self.repo.save(self)

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
        _config = self._config if self._config is not None else load_config()
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
        if not self.datasets:
            return 0
        return min(ds.num_channels for ds in self.datasets)

    @property
    def channel_names(self) -> List[str]:
        if not self.datasets:
            return []
        return max((ds.channel_names for ds in self.datasets), key=len)

    @property
    def latency_windows(self) -> List[LatencyWindow]:
        if not self.datasets:
            return []
        return max((ds.latency_windows for ds in self.datasets), key=len)

    @property
    def stimulus_voltages(self) -> np.ndarray:
        if not self.datasets:
            return np.array([])
        # Collect all unique stimulus voltages across datasets, rounded to the bin size
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
    def plot(self, plot_type: str = None, **kwargs):
        raw_data = getattr(self.plotter, f"plot_{'reflexCurves' if not plot_type else plot_type}")(**kwargs)
        return raw_data

    def invert_channel_polarity(self, channel_index: int) -> None:
        for ds in self.datasets:
            ds.invert_channel_polarity(channel_index)
        logging.info(f"Channel {channel_index} polarity inverted for all datasets in experiment '{self.id}'.")

    def add_dataset(self, dataset: Dataset) -> None:
        if dataset.id not in [ds.id for ds in self._all_datasets]:
            self._all_datasets.append(dataset)
            self.reset_all_caches()

    def remove_dataset(self, dataset_id: str) -> None:
        self._all_datasets = [ds for ds in self._all_datasets if ds.id != dataset_id]
        self.reset_all_caches()

    def apply_latency_window_preset(self, preset_name: str) -> None:
        """Apply a latency window preset to every dataset and session."""
        for ds in self._all_datasets:
            ds.apply_latency_window_preset(preset_name)
        self.update_latency_window_parameters()

    def exclude_dataset(self, dataset_id: str) -> None:
        """Exclude a dataset from this experiment by its ID."""
        if dataset_id not in [ds.id for ds in self._all_datasets]:
            logging.warning(f"Dataset {dataset_id} not found in experiment {self.id}.")
            return
        if dataset_id not in self.annot.excluded_datasets:
            self.annot.excluded_datasets.append(dataset_id)
            self.reset_all_caches()
            if self.repo is not None:
                self.repo.save(self)
        else:
            logging.warning(f"Dataset {dataset_id} already excluded in experiment {self.id}.")

        # Reset the exclusion list if all datasets are excluded
        if not self.datasets:
            self.annot.excluded_datasets.clear()
            logging.warning(f"All datasets excluded from experiment {self.id}. Resetting exclusion list.")
            if self.repo is not None:
                self.repo.save(self)

    def restore_dataset(self, dataset_id: str) -> None:
        """Restore a previously excluded dataset by its ID."""
        if dataset_id in self.annot.excluded_datasets:
            self.annot.excluded_datasets.remove(dataset_id)
            for ds in self._all_datasets:
                if ds.id == dataset_id:
                    for sess in ds._all_sessions:
                        # Restore all sessions of the dataset
                        sess.restore_session()
            self.reset_all_caches()
            if self.repo is not None:
                self.repo.save(self)
        else:
            logging.warning(f"Dataset {dataset_id} is not excluded from experiment {self.id}.")

    def get_avg_m_wave_amplitudes(self, method: str, channel_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Average M-wave amplitudes for each stimulus bin across datasets."""
        m_wave_bins = {v: [] for v in self.stimulus_voltages}
        for ds in self.datasets:
            binned_voltages = np.round(np.array(ds.stimulus_voltages) / self.bin_size) * self.bin_size
            m_wave, _ = ds.get_avg_m_wave_amplitudes(method, channel_index)
            for volt, amp in zip(binned_voltages, m_wave):
                m_wave_bins[volt].append(amp)

        avg = [float(np.mean(m_wave_bins[v])) if m_wave_bins[v] else 0.0 for v in self.stimulus_voltages]
        sem = [
            (float(np.std(m_wave_bins[v]) / np.sqrt(len(m_wave_bins[v]))) if m_wave_bins[v] else 0.0)
            for v in self.stimulus_voltages
        ]
        return np.array(avg), np.array(sem)

    def _aggregate_wave_amplitudes(self, method: str, channel_index: int, amplitude_func):
        """Aggregate wave amplitudes across datasets."""
        wave_bins = {v: [] for v in self.stimulus_voltages}
        for ds in self.datasets:
            binned = np.round(np.array(ds.stimulus_voltages) / self.bin_size) * self.bin_size
            avg_vals, _ = amplitude_func(ds)
            for volt, amp in zip(binned, avg_vals):
                wave_bins[volt].append(amp)
        avg = [float(np.mean(wave_bins[v])) if wave_bins[v] else np.nan for v in self.stimulus_voltages]
        sem = [
            (float(np.std(wave_bins[v]) / np.sqrt(len(wave_bins[v]))) if wave_bins[v] else np.nan)
            for v in self.stimulus_voltages
        ]
        return avg, sem

    def get_m_wave_amplitude_avgs_at_voltage(self, method: str, channel_index: int, voltage: float) -> np.ndarray:
        """Get average M-wave amplitudes at a specific voltage across datasets."""
        amps = []
        for ds in self.datasets:
            if voltage in ds.stimulus_voltages:
                idx = np.where(ds.stimulus_voltages == voltage)[0][0]
                avg, _ = ds.get_avg_m_wave_amplitudes(method, channel_index)
                amps.append(avg[idx])
        return np.array(amps)

    def get_avg_h_wave_amplitudes(self, method: str, channel_index: int) -> tuple[np.ndarray, np.ndarray]:
        h_wave_bins = {v: [] for v in self.stimulus_voltages}
        for ds in self.datasets:
            binned = np.round(np.array(ds.stimulus_voltages) / self.bin_size) * self.bin_size
            h_wave, _ = ds.get_avg_h_wave_amplitudes(method, channel_index)
            for volt, amp in zip(binned, h_wave):
                h_wave_bins[volt].append(amp)
        avg = [float(np.mean(h_wave_bins[v])) if h_wave_bins[v] else np.nan for v in self.stimulus_voltages]
        std = [float(np.std(h_wave_bins[v])) if h_wave_bins[v] else np.nan for v in self.stimulus_voltages]
        return np.array(avg), np.array(std)

    def get_h_wave_amplitude_avgs_at_voltage(self, method: str, channel_index: int, voltage: float) -> np.ndarray:
        amps = []
        for ds in self.datasets:
            if voltage in ds.stimulus_voltages:
                idx = np.where(ds.stimulus_voltages == voltage)[0][0]
                avg, _ = ds.get_avg_h_wave_amplitudes(method, channel_index)
                amps.append(avg[idx])
        return np.array(amps)

    def get_avg_m_max(self, method: str, channel_index: int, return_avg_mmax_thresholds: bool = False):
        m_max_list = []
        m_thresh_list = []

        for ds in self.datasets:
            mmax, mthresh = ds.get_avg_m_max(method, channel_index, return_avg_mmax_thresholds=True)
            if mmax is not None:
                m_max_list.append(mmax)
                m_thresh_list.append(mthresh)

        if not m_max_list:
            if return_avg_mmax_thresholds:
                return None, None
            return None

        # Calculate M-max for experiment level - use mean of all datasets
        if len(m_max_list) == 1:
            # Only one dataset, use its M-max
            final_mmax = float(m_max_list[0])
            final_mthresh = float(m_thresh_list[0])
        else:
            # Multiple datasets: use mean M-max from all datasets
            # This provides proper population-level normalization across mice
            final_mmax = float(np.mean(m_max_list))
            final_mthresh = float(np.mean(m_thresh_list))

            logging.debug(f"Experiment M-max: Using mean from {len(m_max_list)} datasets")
            logging.debug(f"  M-max values: {m_max_list}")
            logging.debug(f"  Mean M-max: {final_mmax}")

        if return_avg_mmax_thresholds:
            return final_mmax, final_mthresh
        else:
            return final_mmax

    def reset_all_caches(self):
        for ds in self.datasets:
            ds.reset_all_caches()
        self.update_latency_window_parameters()

    def apply_config(self, reset_caches: bool = True) -> None:
        """
        Apply user preferences to the experiment.
        This is a placeholder for any logic needed to apply preferences.
        """
        for ds in self._all_datasets:
            ds.apply_config()

        self._load_config_settings()
        self.plotter = ExperimentPlotterPyQtGraph(self)

        if reset_caches:
            self.reset_all_caches()
        if self.repo is not None:
            self.repo.save(self)
        logging.info(f"Preferences successfully applied to experiment '{self.id}'.")

    def set_config(self, config: dict) -> None:
        """
        Update the configuration for this experiment and all child datasets.
        """
        self._config = config
        for ds in self._all_datasets:
            if hasattr(ds, "set_config"):
                ds.set_config(config)
            else:
                logging.warning(f"Dataset {ds.id} does not support set_config method. Skipping.")

        self.apply_config(reset_caches=True)

    # ──────────────────────────────────────────────────────────────────
    # 1) Update annotation parameters
    # ──────────────────────────────────────────────────────────────────
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

    def rename_channels(self, new_names: dict[str, str]) -> None:
        """
        Rename channels in all datasets based on a mapping provided in new_names.
        :param new_names: A dictionary mapping old channel names to new names.
        """
        for ds in self.datasets:
            ds.rename_channels(new_names)
        logging.info(f"Channels renamed in experiment '{self.id}' according to provided mapping.")

    # ──────────────────────────────────────────────────────────────────
    # 2) Clean up
    # ──────────────────────────────────────────────────────────────────
    def save(self) -> None:
        """
        Save the experiment to the repository.
        This is a placeholder for any save logic needed.
        """
        if self.repo is not None:
            self.repo.save(self)
        else:
            raise NotImplementedError("No repository defined for saving the experiment.")

    def close(self) -> None:
        """
        Close all datasets in the experiment.
        This is a placeholder for any cleanup logic needed.
        """
        for ds in self.datasets:
            ds.close()

    # ──────────────────────────────────────────────────────────────────
    # 3) Object representation and reports
    # ──────────────────────────────────────────────────────────────────
    def experiment_parameters(self):
        report = [
            f" Experiment Parameters for Experiment '{self.id}':",
            "===============================",
            f"Datasets ({len(self.datasets)}): {[ds.id for ds in self.datasets]}.",
        ]
        for line in report:
            logging.info(line)
        return report

    def __repr__(self) -> str:
        return f"Experiment(expt_id={self.id}, num_datasets={self.num_datasets})"

    def __str__(self) -> str:
        return f"Experiment: '{self.id}' with {self.num_datasets} datasets"

    def __len__(self) -> int:
        return self.num_datasets
