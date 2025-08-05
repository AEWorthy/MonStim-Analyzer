# monstim_signals/core/data_models.py
import logging
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List

from monstim_signals.version import DATA_VERSION


# -----------------------------------------------------------------------
# Basic data models for MonStim Signals
# -----------------------------------------------------------------------
@dataclass
class LatencyWindow:
    name: str
    color: str
    start_times: List[float]  # one per channel; ms, relative to stimulus start time
    durations: List[float]  # one per channel; ms
    linestyle: str = "--"

    @property
    def end_times(self):
        return [start + dur for start, dur in zip(self.start_times, self.durations)]

    @property
    def label(self):
        return self.name if self.name else "Latency Window"

    def get_legend_element(self, stylized=True):
        from matplotlib.lines import Line2D

        if stylized:
            return Line2D(
                [0], [0], color=self.color, linestyle=self.linestyle, label=self.name
            )
        else:
            return Line2D([0], [0], color=self.color, linestyle="-", label=self.name)

    def __str__(self):
        return self.name


@dataclass
class StimCluster:
    stim_delay: float  # train start delay in ms
    stim_duration: float  # total train duration in ms.
    stim_type: str  # e.g. "Electrical", "OFF", "Motor Length"
    stim_v: float  # initial amplitude in V
    stim_min_v: float  # minimum amplitude that the cluster can reach in V
    stim_max_v: float  # maximum amplitude that the cluster can reach in V
    pulse_shape: str  # e.g. "Square", "Triangle", "Sine"
    num_pulses: int  # number of pulses in the train
    pulse_period: float  # total pulse period in ms
    peak_duration: float  # ms, time from start of pulse to peak amplitude
    ramp_duration: float  # ms, time to ramp up to peak amplitude

    def __post_init__(self):
        if self.ramp_duration is None:
            self.ramp_duration = 0.0
        if self.stim_duration is None:
            self.stim_duration = (
                # if stim_duration is not specified, compute it from pulses/ramp
                self.stim_delay
                + (
                    (self.pulse_period + self.ramp_duration * 2 + self.peak_duration)
                    * self.num_pulses
                    - (self.pulse_period + self.ramp_duration)
                )
            )
        if self.pulse_shape is None:
            self.pulse_shape = "Square"  # Default to Square if not specified
        if self.ramp_duration is None:
            self.ramp_duration = 0.0  # Default to 0 if not specified

    @classmethod
    def from_meta(cls, meta: Dict[str, Any]) -> "StimCluster":
        """
        Create a StimCluster from a metadata dictionary.
        This is useful for converting from JSON or other formats.
        """
        # Filter out unexpected keys
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in meta.items() if k in valid}
        return cls(**filtered)


@dataclass
class SignalChannel:
    invert: bool = False
    name: str = "[NAME]"  # e.g. "TA", "SOL", "VL"
    unit: str = "unit(s)"
    type_override: str | None = None  # e.g. "EMG", "Force", "Accelerometer"

    @staticmethod
    def create_empty() -> "SignalChannel":
        """
        Create an empty ChannelAnnot with default values.
        """
        return SignalChannel(
            invert=False, name="[NAME]", unit="unit(s)", type_override=None
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

    recording_id: str  # e.g. "WT41-0000"
    num_channels: int  # number of channels in the recording
    scan_rate: int  # in Hz or samples/sec
    pre_stim_acquired: int  # number of ms acquired before the first stimulus
    post_stim_acquired: int  # number of ms acquired after the last stimulus
    recording_interval: float  # in seconds, time between consecutive recordings/stimuli
    channel_types: List[str]  # e.g. ["EMG", "Force", "Accelerometer"]
    emg_amp_gains: List[int]  # e.g. [1000, 1000, 1000] (gain for each EMG channel)
    stim_clusters: List[
        StimCluster
    ]  # list of StimCluster objects, one per stimulus cluster
    primary_stim: StimCluster | int | None = (
        None  # (1-based index) primary stimulus cluster
    )
    num_samples: int | None = None  # filled lazily
    data_version: str = "0.0.0"  # version of the meta format, e.g. "0.0.1"

    def __post_init__(self):
        if self.primary_stim and not isinstance(self.primary_stim, StimCluster):
            if isinstance(self.primary_stim, int):
                self.primary_stim = (
                    self.stim_clusters[self.primary_stim - 1]
                    if self.primary_stim > 0
                    else None
                )
            else:
                logging.warning(
                    f"primary_stim should be a StimCluster, got {type(self.primary_stim)}. Setting to None."
                )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RecordingMeta":
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

    cache: Dict[str, Any] = field(default_factory=dict)
    data_version: str = "0.0.0"

    @staticmethod
    def create_empty() -> "RecordingAnnot":
        """
        Create an empty RecordingAnnot with default values.
        """
        return RecordingAnnot(data_version=DATA_VERSION, cache={})

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RecordingAnnot":
        """
        Build a RecordingAnnot from a JSON dict that may contain extra keys.
        """
        # 1) Filter out unexpected keys
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in valid}
        for invalid_key in raw.keys() - valid:
            logging.warning(
                f"Invalid key '{invalid_key}' found in RecordingAnnot dict. Ignoring it."
            )

        # 2) Now call the real constructor
        return RecordingAnnot(**filtered)


@dataclass
class SessionAnnot:
    """
    SessionAnnot is a data class that encapsulates all user edits and annotations for a session. It tracks which recordings are excluded, custom latency windows, optional cached m_max values, and custom channel names (if the user renamed channels). The class provides methods to create empty annotations, initialize from recording metadata, and construct from a dictionary (e.g., loaded from JSON), handling extra or unexpected keys gracefully.
    Attributes:
        excluded_recordings (List[str]): List of recording IDs to exclude from analysis.
        latency_windows (List[LatencyWindow]): List of custom latency windows, reflecting user adjustments.
        channels (List[SignalChannel]): List of signal channels, possibly renamed by the user.
        m_max_values (List[float]): Optional cached m_max values for each channel.
        is_completed (bool): Indicates if the session annotation is finalized.
        version (str): Version string for the annotation data format.
    Methods:
        create_empty(num_channels: int = 0) -> 'SessionAnnot':
            Creates an empty SessionAnnot instance with the specified number of channels, all initialized to default values.
        from_meta(cls, recording_meta: RecordingMeta) -> 'SessionAnnot':
            Constructs a SessionAnnot from a RecordingMeta object, initializing channels and their names/units based on the metadata.
        from_dict(cls, raw: dict[str, Any]) -> 'SessionAnnot':
            Builds a SessionAnnot from a dictionary (such as one loaded from JSON), filtering out unexpected keys and converting nested structures as needed.
    """

    excluded_recordings: List[str] = field(
        default_factory=list
    )  # list of recording IDs to exclude
    latency_windows: List[LatencyWindow] = field(default_factory=list)
    channels: List[SignalChannel] = field(default_factory=list)
    m_max_values: List[float] = field(default_factory=list)
    is_completed: bool = False
    data_version: str = "0.0.0"

    @staticmethod
    def create_empty(num_channels: int = 0) -> "SessionAnnot":
        """
        Create an empty SessionAnnot with default values.
        """
        return SessionAnnot(
            excluded_recordings=[],
            latency_windows=[],
            channels=[
                SignalChannel.create_empty()
                for _ in range(num_channels)
                if num_channels > 0
            ],
            m_max_values=[],
            is_completed=False,
            data_version=DATA_VERSION,
        )

    @classmethod
    def from_meta(cls, recording_meta: RecordingMeta) -> "SessionAnnot":
        """
        Create a RecordingAnnot from a RecordingMeta object.
        Initializes channels based on the number of channels in the meta.
        """
        annot = cls.create_empty(num_channels=recording_meta.num_channels)

        # Fill in channel names and units based on meta
        for i in range(recording_meta.num_channels):
            channel = (
                annot.channels[i]
                if i < len(annot.channels)
                else SignalChannel.create_empty()
            )
            channel_type = (
                recording_meta.channel_types[i]
                if i < len(recording_meta.channel_types)
                else None
            )
            channel.name = (
                channel_type
                if (channel_type not in (None, "unknown", "emg"))
                else f"Ch{i}"
            )
            channel.unit = "V"
            annot.channels[i] = channel

        return annot

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SessionAnnot":
        """
        Build a SessionAnnot from a JSON dict that may contain extra keys.
        """
        # 1) Filter out unexpected keys
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in valid}
        for invalid_key in raw.keys() - valid:
            logging.warning(
                f"Invalid key '{invalid_key}' found in SessionAnnot dict. Ignoring it."
            )

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

    date: str | None = (
        None  # Date of dataset collection: e.g., "240829" for 29 Aug 2024
    )
    animal_id: str = None  # e.g., "C328.1"
    condition: str = None  # e.g., "post-dec mcurve_long-"
    excluded_sessions: List[str] = field(default_factory=list)
    is_completed: bool = False
    data_version: str = "0.0.0"

    @staticmethod
    def create_empty(num_channels: int = 0) -> "DatasetAnnot":
        """
        Create an empty DatasetAnnot with default values.
        """
        return DatasetAnnot(
            excluded_sessions=[], is_completed=False, data_version=DATA_VERSION
        )

    @classmethod
    def from_ds_name(cls, dataset_name: str) -> "DatasetAnnot":
        """
        Create a DatasetAnnot from a dataset name.
        This is useful for initializing an annotation object for a new dataset.
        """
        from monstim_signals.core import load_config
        from monstim_signals.io.string_parser import parse_dataset_name

        cfg = load_config()
        preferred_format = cfg.get("preferred_date_format", "YYMMDD")
        try:
            date, animal_id, condition = parse_dataset_name(
                dataset_name, preferred_date_format=preferred_format
            )
        except ValueError:
            date = animal_id = condition = None

        return DatasetAnnot(
            date=date,
            animal_id=animal_id,
            condition=condition,
            excluded_sessions=[],
            is_completed=False,
            data_version=DATA_VERSION,
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DatasetAnnot":
        """
        Build a DatasetAnnot from a JSON dict that may contain extra keys.
        """
        # 1) Filter out unexpected keys
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in valid}
        for invalid_key in raw.keys() - valid:
            logging.warning(
                f"Invalid key '{invalid_key}' found in DatasetAnnot dict. Ignoring it."
            )

        # 2) Now call the real constructor
        return DatasetAnnot(**filtered)


# -----------------------------------------------------------------------
# Experiment annotation
# -----------------------------------------------------------------------


@dataclass
class ExperimentAnnot:
    """Annotation information for an :class:`Experiment`."""

    excluded_datasets: List[str] = field(default_factory=list)
    is_completed: bool = False
    data_version: str = DATA_VERSION

    @staticmethod
    def create_empty() -> "ExperimentAnnot":
        """Return a blank annotation object."""
        return ExperimentAnnot(
            excluded_datasets=[], is_completed=False, data_version=DATA_VERSION
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentAnnot":
        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in raw.items() if k in valid}
        for invalid_key in raw.keys() - valid:
            logging.warning(
                f"Invalid key '{invalid_key}' found in ExperimentAnnot dict. Ignoring it."
            )
        return ExperimentAnnot(**filtered)
