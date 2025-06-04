# monstim_signals/domain/recording.py
import functools
import logging
import numpy as np
import h5py
from typing import Any

from monstim_signals.core.data_models import RecordingMeta, RecordingAnnot, LatencyWindow
from monstim_signals.Transform_EMG import butter_bandpass_filter

class Recording:
    """
    Pure‐Python object representing one recording (one stimulus amplitude).
    Holds:
      - meta: RecordingMeta (immutable facts, e.g. scan_rate, stim_v, etc.)
      - annot: RecordingAnnot (user edits, e.g. invert flags)
      - raw: either an h5py.Dataset (lazy) or a 2D np.ndarray [samples × channels]
      - repo: back‐pointer to its RecordingRepository
    """
    def __init__(
        self,
        meta : RecordingMeta,
        annot: RecordingAnnot,
        raw  : h5py.Dataset | np.ndarray,
        repo : Any = None
    ):
        self.meta   = meta
        self.annot  = annot
        self._raw   = raw
        self.repo   = repo

        # default latency windows per channel
        self.latency_windows: list[LatencyWindow] = [
            LatencyWindow("M‐wave",   "red",   [1.1]*meta.num_channels, [5.0]*meta.num_channels),
            LatencyWindow("H‐reflex","blue",  [6.5]*meta.num_channels, [6.0]*meta.num_channels),
        ]
    
    # ──────────────────────────────────────────────────────────────────
    # 1) Simple properties (GUI & analysis code expects these)
    # ──────────────────────────────────────────────────────────────────
    @property
    def id(self) -> str:
        return self.meta.recording_id
    @property
    def num_channels(self) -> int:
        return self.meta.num_channels
    @property
    def scan_rate(self) -> int:
        return self.meta.scan_rate
    @property
    def num_samples(self) -> int:
        return self.meta.num_samples
    @property
    def stim_amplitude(self) -> float:
        # Assume the primary StimCluster’s stim_v is the amplitude for this recording
        return self.meta.primary_stim.stim_v
    
    # ──────────────────────────────────────────────────────────────────
    # 2) Raw vs. Filtered views of signal data
    # ──────────────────────────────────────────────────────────────────
    def raw_view(self, ch: int | slice | list[int] = slice(None), 
                       t: slice = slice(None)) -> np.ndarray:
        """
        Return a NumPy view (or slice) of raw data [time, channels].
        If self._raw is an h5py.Dataset, this will not load the entire array,
        only the requested slice.
        """
        return self._raw[t, ch]

    @functools.cached_property
    def filtered(self) -> np.ndarray:
        """
        Apply a Butterworth bandpass filter “once” and cache it. 
        (The first time you call `recording.filtered`, it'll compute and store it.)
        """
        data = self.raw_view()  # full [samples × channels]
        fs   = self.scan_rate
        lowcut, highcut, order = 100, 3500, 4 #adjust to config values
        logging.warning(f"Applying bandpass filter with hardcoded values: lowcut={lowcut}, highcut={highcut}, order={order}")
        return butter_bandpass_filter(data, fs=fs, lowcut=lowcut, highcut=highcut, order=order)
    
    # ──────────────────────────────────────────────────────────────────
    # 3) User actions (invert, exclude, etc.). Changes annotation.
    # ──────────────────────────────────────────────────────────────────
    def invert_channel(self, ch: int) -> None:
        """
        Flip the invert flag for channel `ch`, then save annotation to disk.
        """
        self.annot.channels[ch].invert = not self.annot.channels[ch].invert
        if self.repo is not None:
            self.repo.save(self)

    def exclude_recording(self, do_exclude: bool = True) -> None:
        """
        Mark this entire recording as excluded (or re‐include if do_exclude=False).
        """
        self.annot.excluded = do_exclude
        if self.repo is not None:
            self.repo.save(self)

    # ──────────────────────────────────────────────────────────────────
    # 4) Clean‐up (close HDF5 file when you’re done)
    # ──────────────────────────────────────────────────────────────────
    def close(self) -> None:
        if isinstance(self._raw, h5py.Dataset):
            try:
                self._raw.file.close()
            except Exception as exception:
                logging.exception(f"Failed to close HDF5 file for recording '{self.id}': {exception}")
                pass
    
    # ──────────────────────────────────────────────────────────────────
    # 5) Object representation
    # ──────────────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        return f"Recording(id={self.id}, num_channels={self.num_channels}, scan_rate={self.scan_rate})"
    def __str__(self) -> str:
        return f"Recording: {self.id} with {self.num_channels} channels at {self.scan_rate} Hz"
    def __len__(self) -> int:
        return self.num_samples