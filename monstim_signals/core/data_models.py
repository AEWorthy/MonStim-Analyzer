# monstim_signals/core/data_models.py
from dataclasses import dataclass, field, fields
from typing import List, Dict, Any
import logging
from monstim_signals.core.utils import DATA_VERSION

# -----------------------------------------------------------------------
# Basic data models for MonStim Signals
# -----------------------------------------------------------------------
@dataclass # TODO: Add a method to create dataset latency window objects for each session in the dataset. Make the default windows be the m-wave and h-reflex windows.
class LatencyWindow:
    name: str
    color: str
    start_times: List[float] # one per channel
    durations: List[float] # one per channel
    linestyle: str = '--'
    window_version: str = DATA_VERSION

    @property
    def end_times(self):
        return [start + dur for start, dur in zip(self.start_times, self.durations)]

    def get_legend_element(self, stylized=True):
        from matplotlib.lines import Line2D
        if stylized:
            return Line2D([0], [0], color=self.color, linestyle=self.linestyle, label=self.name)
        else:
            return Line2D([0], [0], color=self.color, linestyle='-', label=self.name)

@dataclass
class StimCluster:
    stim_delay:    float  # train start delay in ms
    stim_duration: float  # total train duration in ms.
    stim_type:     str    # e.g. "Electrical", "OFF", "Motor Length"
    stim_v:        float  # initial amplitude in V
    stim_min_v:    float  # minimum amplitude that the cluster can reach in V
    stim_max_v:    float  # maximum amplitude that the cluster can reach in V
    pulse_shape:   str    # e.g. "Square", "Triangle", "Sine"
    num_pulses:    int    # number of pulses in the train
    pulse_period:  float  # total pulse period in ms
    peak_duration: float # ms, time from start of pulse to peak amplitude
    ramp_duration: float  # ms, time to ramp up to peak amplitude
    def __post_init__(self):
        if self.ramp_duration is None:
            self.ramp_duration = 0.0
        if self.stim_duration is None:
            self.stim_duration = (
                # if stim_duration is not specified, compute it from pulses/ramp
                self.stim_delay 
                + ((self.pulse_period + self.ramp_duration*2 + self.peak_duration) 
                * self.num_pulses
                - (self.pulse_period + self.ramp_duration))
                )
        if self.pulse_shape is None:
            self.pulse_shape = "Square"  # Default to Square if not specified
        if self.ramp_duration is None:
            self.ramp_duration = 0.0  # Default to 0 if not specified

@dataclass
class SignalChannel:
    invert : bool = False
    name : str = "[NAME]"  # e.g. "TA", "SOL", "VL"
    unit : str = "unit(s)"
    type_override : str | None = None  # e.g. "EMG", "Force", "Accelerometer"

    @staticmethod
    def create_empty() -> 'SignalChannel':
        """
        Create an empty ChannelAnnot with default values.
        """
        return SignalChannel(
            invert=False,
            name="[NAME]",
            unit="unit(s)",
            type_override=None
        )

# -----------------------------------------------------------------------
# Recording permanent metadata
# This is stored in the .meta.json file alongside the .raw.h5 file.
# -----------------------------------------------------------------------
@dataclass
class RecordingMeta:
    """
    Describes a single recording (one .raw.h5 + .meta.json).
    """
    recording_id         : str          # e.g. "WT41-0000"
    num_channels         : int          # number of channels in the recording
    scan_rate            : int          # in Hz or samples/sec
    pre_stim_acquired    : int          # number of ms acquired before the first stimulus
    post_stim_acquired   : int          # number of ms acquired after the last stimulus
    recording_interval   : float        # in seconds, time between consecutive recordings/stimuli
    channel_types        : List[str]    # e.g. ["EMG", "Force", "Accelerometer"]
    emg_amp_gains        : List[int]    # e.g. [1000, 1000, 1000] (gain for each EMG channel)
    stim_clusters        : List[StimCluster]                # list of StimCluster objects, one per stimulus cluster
    primary_stim         : StimCluster | int | None = None  # (1-based index) primary stimulus cluster
    num_samples          : int | None = None                # filled lazily
    meta_version         : str = DATA_VERSION               # version of the meta format, e.g. "0.0.1"
    def __post_init__(self):
        if self.primary_stim and not isinstance(self.primary_stim, StimCluster):
            if isinstance(self.primary_stim, int):
                self.primary_stim = self.stim_clusters[self.primary_stim - 1] if self.primary_stim > 0 else None
            else:
                logging.warning(f"primary_stim should be a StimCluster, got {type(self.primary_stim)}. Setting to None.")
    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> 'RecordingMeta':
        """
        Build a RecordingMeta from a JSON dict that may contain extra keys,
        and convert nested stim_clusters from dicts → StimCluster.
        """
        # 1) Filter out unexpected keys
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in valid}

        # 2) Convert stim_clusters if present
        sc_list = filtered.get("stim_clusters", [])
        filtered["stim_clusters"] = [StimCluster(**d) for d in sc_list]

        # 3) Now call the real constructor
        return cls(**filtered)

# -----------------------------------------------------------------------
# Annotation data structures for recordings and sessions
# -----------------------------------------------------------------------
@dataclass
class RecordingAnnot:
    """
    Holds user‐editable flags for one recording.
    e.g., which channels to invert, exclude, or cached computations.
    """
    cache    : Dict[str, Any] = field(default_factory=dict)
    version  : str = "0.0.0"

    @staticmethod
    def create_empty() -> 'RecordingAnnot':
        """
        Create an empty RecordingAnnot with default values.
        """
        return RecordingAnnot(
            version=DATA_VERSION,
            cache={}
            )
       
    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> 'RecordingAnnot':
        """
        Build a RecordingAnnot from a JSON dict that may contain extra keys.
        """
        # 1) Filter out unexpected keys
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in valid}
        for invalid_key in raw.keys() - valid:
            logging.warning(f"Invalid key '{invalid_key}' found in RecordingAnnot dict. Ignoring it.")

        # 2) Now call the real constructor
        return RecordingAnnot(**filtered)
    
@dataclass
class SessionAnnot:
    """
    Holds all user edits for a Session:
      - Which recordings to exclude
      - Custom latency windows (persisting user tweaks)
      - Optional cached m_max values (small arrays)
      - Custom channel_names (if user renamed channels)
    """
    excluded_recordings  : List[str] = field(default_factory=list)
    latency_windows      : List[LatencyWindow] = field(default_factory=list)
    channels             : List[SignalChannel] = field(default_factory=list)
    m_max_values         : List[float] = field(default_factory=list)
    version              : str = "0.0.0"

    @staticmethod
    def create_empty(num_channels : int = 0) -> 'SessionAnnot':
        """
        Create an empty SessionAnnot with default values.
        """
        return SessionAnnot(
            excluded_recordings=[],
            latency_windows=[],
            channels=[SignalChannel.create_empty() for _ in range(num_channels)],
            m_max_values=[],
            version=DATA_VERSION
        )
    
    @classmethod
    def from_meta(cls, recording_meta: RecordingMeta) -> 'SessionAnnot':
        """
        Create a RecordingAnnot from a RecordingMeta object.
        Initializes channels based on the number of channels in the meta.
        """
        annot = cls.create_empty(num_channels=recording_meta.num_channels)
        
        # Fill in channel names and units based on meta
        for i in range(recording_meta.num_channels):
            channel = annot.channels[i] if i < len(annot.channels) else SignalChannel.create_empty()
            channel_type = recording_meta.channel_types[i] if i < len(recording_meta.channel_types) else None
            channel.name = channel_type if (channel_type not in (None, 'unknown', 'emg')) else f"Ch{i}"
            channel.unit = 'V'
            annot.channels[i] = channel

        return annot

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> 'SessionAnnot':
        """
        Build a SessionAnnot from a JSON dict that may contain extra keys.
        """
        # 1) Filter out unexpected keys
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in valid}
        for invalid_key in raw.keys() - valid:
            logging.warning(f"Invalid key '{invalid_key}' found in SessionAnnot dict. Ignoring it.")

        # 2) Convert latency_windows and channels if present
        lw_list = filtered.get("latency_windows", [])
        filtered["latency_windows"] = [LatencyWindow(**d) for d in lw_list]
        ch_list = filtered.get("channels", [])
        filtered["channels"] = [SignalChannel(**d) for d in ch_list]

        # 3) Now call the real constructor
        return SessionAnnot(**filtered)
    
@dataclass
class DatasetAnnot:
    """
    Holds all user edits for a Dataset:
      - Custom latency windows (persisting user tweaks)
      - Optional cached m_max values (small arrays)
      - Custom channel_names (if user renamed channels)
    """
    date: str |None = None  # Date of dataset collection: e.g., "240829" for 29 Aug 2024
    animal_id: str = None # e.g., "C328.1"
    condition: str = None # e.g., "post-dec mcurve_long-"
    excluded_sessions: List[str] = field(default_factory=list)
    version: str = "0.0.0"

    @staticmethod
    def create_empty(num_channels: int = 0) -> 'DatasetAnnot':
        """
        Create an empty DatasetAnnot with default values.
        """
        return DatasetAnnot(
            excluded_sessions=[],
            version=DATA_VERSION
        )
    def from_ds_name(cls, dataset_name: str) -> 'DatasetAnnot':
        """
        Create a DatasetAnnot from a dataset name.
        This is useful for initializing an annotation object for a new dataset.
        """
        from monstim_signals.io.string_parser import parse_dataset_name
        try:
            # TODO make date formatting configurable
            date, animal_id, condition = parse_dataset_name(dataset_name, preferred_date_format='YYYYMMDD') 
        except ValueError:
            date = animal_id = condition = None
        
        return DatasetAnnot(
            date=date,
            animal_id=animal_id,
            condition=condition,
            excluded_sessions=[],
            version=DATA_VERSION
        )
    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> 'DatasetAnnot':
        """
        Build a DatasetAnnot from a JSON dict that may contain extra keys.
        """
        # 1) Filter out unexpected keys
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in valid}
        for invalid_key in raw.keys() - valid:
            logging.warning(f"Invalid key '{invalid_key}' found in DatasetAnnot dict. Ignoring it.")

        # 2) Now call the real constructor
        return DatasetAnnot(**filtered)

