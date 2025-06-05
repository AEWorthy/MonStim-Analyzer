# monstim_signals/domain/session.py
import os
import logging
from typing import List, Any
from functools import cached_property
from concurrent.futures import ThreadPoolExecutor
import numpy as np


from monstim_signals.core.data_models import SessionAnnot, LatencyWindow, StimCluster
from monstim_signals.core.utils import load_config
from monstim_signals.domain.recording import Recording
from monstim_signals.Transform_EMG import butter_bandpass_filter, calculate_emg_amplitude, get_avg_mmax, NoCalculableMmaxError
from monstim_signals.Plot_EMG import EMGSessionPlotter
# ──────────────────────────────────────────────────────────────────
class Session:
    """
    A collection of multiple Recordings, each at a different stimulus amplitude,
    all belonging to one “session” (animal & date).
    """
    def __init__(self, session_id : str, recordings : List[Recording], annot : SessionAnnot, repo : Any = None):
        self.id               = session_id
        self._all_recordings  = recordings
        self.annot            = annot  # SessionAnnot, holds user edits like exclude flags, etc.
        self.repo             = repo  # back‐pointer to SessionRepository

        self._load_config_settings()
        self._load_session_parameters()
        self._initialize_annotations()
        self.plotter = EMGSessionPlotter(self)
        self.update_latency_window_parameters()

    def _load_config_settings(self):
        _config = load_config()
        self.default_m_start : List[float] = _config['m_start']
        self.default_m_duration : List[float] = [_config['m_duration'] for _ in range(len(self.default_m_start))]
        self.default_h_start : List[float] = _config['h_start']
        self.default_h_duration : List[float] = [_config['h_duration'] for _ in range(len(self.default_h_start))]
        
        self.time_window_ms : float = _config['time_window']
        self.bin_size : float = _config['bin_size']
        self.latency_window_style : str = _config['latency_window_style']
        self.m_color : str = _config['m_color']
        self.h_color : str = _config['h_color']
        self.title_font_size : int = _config['title_font_size']
        self.axis_label_font_size : int = _config['axis_label_font_size']
        self.tick_font_size : int = _config['tick_font_size']
        self.subplot_adjust_args = _config['subplot_adjust_args']
        self.m_max_args = _config['m_max_args']
        self.butter_filter_args = _config['butter_filter_args']
        self.default_method : str = _config['default_method']
        self.default_channel_names : List[str] = _config['default_channel_names'] 

    def _load_session_parameters(self):
        # ---------- Pull session‐wide parameters from the first recording's meta ----------
        first_meta = self.recordings[0].meta
        self.formatted_name       = self.id + '_' + first_meta.recording_id # e.g., "AA00_0000"
        self.scan_rate            = first_meta.scan_rate      # Hz
        self.num_samples          = first_meta.num_samples    # samples/channel
        self.num_channels         = first_meta.num_channels   # number of channels
        self._channel_types : List[str] = first_meta.channel_types.copy()  # list of channel types

        # Stimulus parameters
        self.stim_clusters : List[StimCluster] = first_meta.stim_clusters.copy()  # list of StimCluster objects
        self.pre_stim_acquired    = first_meta.pre_stim_acquired
        self.post_stim_acquired   = first_meta.post_stim_acquired
        self.primary_stim : StimCluster = first_meta.primary_stim  # the primary StimCluster for this session
        self.stim_delay           = self.primary_stim.stim_delay # in ms, delay
        self.stim_duration        = self.primary_stim.stim_duration
        self.stim_start : float   = self.stim_delay + self.pre_stim_acquired
        
        # Parameters that may sometimes be None
        self.stim_interval : float = getattr(first_meta, 'stim_interval', None) # in seconds, time between recordings (if applicable)
        self.emg_amp_gains        = getattr(first_meta, 'emg_amp_gains', None)  # default to 1000 if not specified
  
    def _initialize_annotations(self):
        self.channel_names = [self.annot.channels[i].name for i in range(self.num_channels)]
        self.channel_units = [self.annot.channels[i].unit for i in range(self.num_channels)]
        self.channel_types = [self.annot.channels[i].type_override
                              if self.annot.channels[i].type_override is not None
                              else self._channel_types[i]
                              if i < len(self._channel_types)
                              else "SIGNAL"
                              for i in range(self.num_channels)]

        # Initialize the latency windows for each channel
        if not self.annot.latency_windows:
            m_window = LatencyWindow(
                name="M-wave",
                start_times=self.default_m_start,
                durations=self.default_m_duration,
                color=self.m_color,
                linestyle=self.latency_window_style
            )
            h_window = LatencyWindow(
                name="H-reflex",
                start_times=self.default_h_start,
                durations=self.default_h_duration,
                color=self.h_color,
                linestyle=self.latency_window_style
            )
            self.annot.latency_windows = [m_window, h_window]
            
            # Save changes
            if self.repo is not None:
                self.repo.save(self)

    def _apply_config(self, reset_caches: bool = True):
        """
        Apply the loaded configuration settings to the session.
        This is called after loading the session or when preferences are changed.
        """
        self._load_config_settings()  # Reload config settings to ensure they are up-to-date

        self.plotter = EMGSessionPlotter(self)
        for window in self.latency_windows:
            window.linestyle = self.latency_window_style
            window.color = self.m_color if window.name == "M-wave" else window.color
            window.color = self.h_color if window.name == "H-reflex" else window.color
        
        if reset_caches:
            self.reset_all_caches()
        if self.repo is not None:
            self.repo.save(self)

    @property
    def num_recordings(self) -> int:
        return len(self.recordings)
    @property
    def latency_windows(self) -> List[LatencyWindow]:
        """
        Return the list of latency windows defined in the session annotations.
        This includes M-wave and H-reflex windows.
        """
        return self.annot.latency_windows
    @property
    def excluded_recordings(self):
        return set(self.annot.excluded_recordings)
    @property
    def stimulus_voltages(self) -> List[float]:
        """
        Return a list of stimulus voltages for each recording in the session.
        This assumes that each recording's primary cluster stim_v is the amplitude for that recording.
        """
        return [rec.meta.primary_stim.stim_v for rec in self.recordings]
    # ──────────────────────────────────────────────────────────────────
    # 1) Useful properties for GUI & analysis code
    # ──────────────────────────────────────────────────────────────────
    @cached_property
    def recordings(self) -> List[Recording]:
        """
        Return a list of active recordings in the session.
        This filters out any recordings that are marked as excluded in the session annotations.
        """
        return [
            rec for rec in self._all_recordings 
            if rec.id not in self.excluded_recordings
        ]
    @cached_property
    def recordings_raw(self) -> List[np.ndarray]:
        """
        Return a list of raw data arrays for each recording.
        Each array is of shape (num_samples, num_channels).
        """
        recordings = []
        for rec in self.recordings:
            raw_data = [rec.raw_view() for rec in self.recordings]
            for ch in range(rec.meta.num_channels):
                if self.annot.channels[ch].invert:
                    raw_data[:, ch] *= -1.0
            recordings.append(raw_data)
    @cached_property
    def recordings_processed(self) -> List[np.ndarray]:
        """
        Return a list of processed data arrays for each recording.
        Each array is of shape (num_samples, num_channels).
        This applies a butter bandpass filter to the raw data and inverts if
        indicated in the channel annotations in the session annot.json file.
        """
        def _process_single_recording(self, rec: Recording) -> np.ndarray:
            """
            Process a single recording's raw data with the butter bandpass filter.
            """
            raw_data = rec.raw_view()

            bf_args = getattr(self, "butter_filter_args", {"lowcut":None, "highcut":None, "order":None})
            filtered = butter_bandpass_filter(
                raw_data,
                fs=self.scan_rate,
                lowcut=bf_args["lowcut"],
                highcut=bf_args["highcut"],
                order=bf_args["order"]
            )

            # apply inversion flags
            for ch in range(rec.meta.num_channels):
                if self.annot.channels[ch].invert:
                    filtered[:, ch] *= -1.0
            return filtered
       
        max_workers = (os.cpu_count() - 1) or 1 # Use all available CPU cores
        processed_list : List[np.ndarray] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_single_recording, (self,rec)): rec for rec in self.recordings}
            
            for future in futures:
                filtered_array = future.result()  # this blocks until that recording is done
                processed_list.append(filtered_array)

        return processed_list
    @cached_property
    def m_max(self) -> List[float]:
        """
        Return the maximum M-wave value for each recording in the session.
        This is computed from the raw data using the M-wave latency windows.
        """
        results = []
        for rec in self.recordings:
            for channel_index in range(self.num_channels):
                try: # Check if the channel has a valid M-max amplitude.
                    channel_mmax = self.get_m_max(self.default_method, channel_index, return_mmax_stim_range=False)
                    results.append(channel_mmax)
                except NoCalculableMmaxError:
                    logging.info(f"Channel {channel_index} does not have a valid M-max amplitude.")
                    results.append(None)
                except ValueError as e:
                    logging.error(f"Error in calculating M-max amplitude for channel {channel_index}. Error: {str(e)}")
                    results.append(None)
        return results
    
    

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
            return get_avg_mmax(self.stimulus_voltages, m_wave_amplitudes, **self.m_max_args, return_mmax_stim_range=True)
        else:
            return get_avg_mmax(self.stimulus_voltages, m_wave_amplitudes, **self.m_max_args)
    
    def get_m_wave_amplitudes(self, method, channel_index):
        m_wave_amplitudes = [calculate_emg_amplitude(recording['channel_data'][channel_index], 
                                                                    (self.m_start[channel_index] + self.stim_start),
                                                                    (self.m_start[channel_index] + self.stim_start + self.m_duration[channel_index]), 
                                                                    self.scan_rate, 
                                                                    method=method) 
                                                                    for recording in self.recordings_processed]
        np.array(m_wave_amplitudes)
        return m_wave_amplitudes
    
    def get_h_wave_amplitudes(self, method, channel_index):
        h_wave_amplitudes = [calculate_emg_amplitude(recording['channel_data'][channel_index], 
                                                                   (self.h_start[channel_index] + self.stim_start),
                                                                   (self.h_start[channel_index] + self.stim_start + self.h_duration[channel_index]), 
                                                                   self.scan_rate, 
                                                                   method=method) 
                                                                   for recording in self.recordings_processed]
        np.array(h_wave_amplitudes)
        return h_wave_amplitudes

    #WIP

    def response_curve(self, channel: int, window: LatencyWindow) -> List[float]:
        # """
        # Example: for a given channel and a LatencyWindow (e.g. H-reflex),
        # compute the max value in that window minus baseline, for each recording.
        # Returns a list of [resp_at_stim1, resp_at_stim2, …] sorted by stim amplitude.
        # """
        # results = []
        # for rec in self.recordings:
        #     # 1) get the raw slice for that channel & window
        #     start = window.start_times[channel]
        #     end   = start + window.durations[channel]
        #     # Convert ms → sample index: (ms/1000)*scan_rate
        #     i0 = int(start/1000 * rec.scan_rate)
        #     i1 = int(end/1000 * rec.scan_rate)
        #     tr = rec.raw_view(ch=channel, t=slice(i0, i1))
        #     baseline = np.mean(rec.raw_view(ch=channel, t=slice(0, int(rec.scan_rate*window.start_times[channel]/1000))))
        #     resp = np.max(tr) - baseline
        #     results.append(resp)
        # return results
        raise NotImplementedError("response_curve method is not implemented yet.")

    @property
    def stim_amplitudes(self) -> List[float]:
        """
        Return a list of stimulus amplitudes for each recording in the session.
        This assumes that each recording's primary cluster stim_v is the amplitude for that recording.
        """
        return [rec.stim_amplitude for rec in self.recordings]

    # ──────────────────────────────────────────────────────────────────
    # 2) User actions that update annot files
    # ──────────────────────────────────────────────────────────────────
    def rename_channel(self, new_names : dict[str]):
        for i, new_name in new_names.items():
            if 0 <= i < len(self.annot.channels):
                self.annot.channels[i].name = new_name
            else:
                logging.warning(f"Channel index {i} out of range for renaming.")
        # Optionally update cached names and save
        self.channel_names = [ch.name for ch in self.annot.channels]
        if self.repo is not None:
            self.repo.save(self)
    
    def change_reflex_latency_windows(self, m_start, m_duration, h_start, h_duration):
        for window in self.latency_windows:
            if window.name == "M-wave":
                window.start_times = m_start
                window.durations = m_duration
            elif window.name == "H-reflex":
                window.start_times = h_start
                window.durations = h_duration
        self.update_latency_window_parameters()
        if self.repo is not None:
            self.repo.save(self)

    def include_recording(self, recording_id: str):
        """
        Include a previously excluded recording by its ID.
        If the recording is not found, log a warning.
        """
        if recording_id in self.excluded_recordings:
            for rec in self._all_recordings:
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
    
    def exclude_recording(self, recording_id: str):
        """
        Exclude a recording by its ID.
        If the recording is not found, log a warning.
        """
        if recording_id not in self.excluded_recordings:
            # Find the recording and set its exclude flag
            for rec in self._all_recordings:
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
        self.annot.excluded_recordings = [rec.id for rec in self._all_recordings]
        self.reset_all_caches()
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

    # ──────────────────────────────────────────────────────────────────
    # 3) Methods for CLI/Jupyter use only
    # ──────────────────────────────────────────────────────────────────
    def update_window_settings(self):
        """
        ***ONLY FOR USE IN JUPYTER NOTEBOOKS OR INTERACTIVE PYTHON ENVIRONMENTS***

        Opens a GUI to manually update the M-wave and H-reflex window settings for each channel using PyQt.

        This function should only be used if you are working in a Jupyter notebook or an interactive Python environment. Do not call this function in any other GUI environment.
        """
        from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton
        import sys
        class ReflexSettingsDialog(QWidget):
            def __init__(self, parent : Session):
                super().__init__()
                self.parent : Session = parent
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
                    logging.error(f"Error occurred when trying to save the following reflex settings: m_start: {m_start}\n\tm_duration: {m_duration}\n\th_start: {h_start}\n\th_duration: {h_duration}")
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
    # 3) Update/reset methods
    # ──────────────────────────────────────────────────────────────────
    def reset_all_caches(self):
        """
        Reset all cached properties in the session.
        This is used after changing any session parameters or excluding/including recordings.
        """
        self.reset_recordings_cache()
        self.reset_cached_reflex_properties()
        self.update_latency_window_parameters()
        
    def update_latency_window_parameters(self):
        '''
        Update the M-wave and H-reflex start times and durations from the latency windows.
        This is called after loading the session or when latency windows are modified.
        '''
        for window in self.latency_windows:
            if window.name == "M-wave":
                self.m_start = window.start_times
                self.m_duration = window.durations
            elif window.name == "H-reflex":
                self.h_start = window.start_times
                self.h_duration = window.durations

    def reset_cached_reflex_properties(self):
        """
        Reset the cached M-wave max values.
        This is used after changing the latency windows or excluding/including recordings from the session set.
        """
        if "m_max" in self.__dict__:
            del self.__dict__["m_max"]

    def reset_recordings_cache (self):
        """
        Reset the cached processed recordings.
        This is used after changing the filter parameters or excluding/including recordings from the session set.
        """
        if "recordings" in self.__dict__:
            del self.__dict__["recordings"]
        if "recordings_raw" in self.__dict__:
            del self.__dict__["recordings_raw"]
        if "recordings_processed" in self.__dict__:
            del self.__dict__["recordings_processed"]
    # ──────────────────────────────────────────────────────────────────
    # 3) Clean‐up
    # ──────────────────────────────────────────────────────────────────
    def close(self):
        for rec in self.recordings:
            rec.close()
    # ──────────────────────────────────────────────────────────────────
    # 4) Object representation and reports
    # ──────────────────────────────────────────────────────────────────
    def session_parameters(self) -> dict[str, Any]:
        """
        Logs EMG recording session parameters from a Pickle file.
        """
        report = [f"Session Parameters for {self.formatted_name}",
                  "========================================",
                  f"Session ID: {self.id}",
                  f"# of Recordings (including any excluded ones): {self.num_recordings}", 
                  f"# of Channels: {self.num_channels}",
                  f"Scan Rate (Hz): {self.scan_rate}",
                  f"Samples/Channel: {self.num_samples}",
                  f"Pre-Stim Acq. time (ms): {self.pre_stim_acquired}",
                  f"Post-Stim Acq. time (ms): {self.post_stim_acquired}",
                  f"Stimulus Delay (ms): {self.stim_delay}",
                  f"Stimulus Duration (ms): {self.stim_duration}",
                  f"Stimulus Interval (s): {self.stim_interval if self.stim_interval else 'Not specified'}",
                  f"EMG Amp Gains: {self.emg_amp_gains if self.emg_amp_gains else 'Not specified'}",]        
        
        for line in report:
            logging.info(line)
        return report
    def __repr__(self):
        return f"Session(session_id={self.id}, num_recordings={self.num_recordings})"
    def __str__(self):
        return f"Session: {self.id} with {self.num_recordings} recordings"
    def __len__(self):
        return self.num_recordings
    
