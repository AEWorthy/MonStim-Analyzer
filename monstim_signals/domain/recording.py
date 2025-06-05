# monstim_signals/domain/recording.py
import logging
import numpy as np
import h5py
from typing import Any

from monstim_signals.core.data_models import RecordingMeta, RecordingAnnot

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