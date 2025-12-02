# monstim_signals/domain/recording.py
import logging
from typing import Any

import h5py
import numpy as np

from monstim_signals.core import RecordingAnnot, RecordingMeta, StimCluster


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
        meta: RecordingMeta,
        annot: RecordingAnnot,
        raw: h5py.Dataset | np.ndarray,
        repo: Any = None,
        config: dict = None,
    ):
        self.meta = meta
        self.annot = annot
        self._raw = raw
        self.repo = repo
        self._config = config or {}

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
    def channel_types(self) -> list[str]:
        """
        Return the list of channel types for this recording.
        This is a list of strings, e.g. ["EMG", "Force", "Length", "ElectricalStimulus"].
        """
        return self.meta.channel_types

    @property
    def scan_rate(self) -> int:
        return self.meta.scan_rate

    @property
    def num_samples(self) -> int:
        return self.meta.num_samples

    @property
    def stim_clusters(self) -> list[StimCluster]:
        """
        Return the list of StimCluster objects for this recording.
        """
        return [StimCluster.from_meta(cluster) for cluster in self.meta.stim_clusters]

    @property
    def stim_amplitude(self) -> float:
        # Assume the primary StimCluster’s stim_v is the amplitude for this recording
        return self.meta.primary_stim.stim_v

    # ──────────────────────────────────────────────────────────────────
    # 2) Raw vs. Filtered views of signal data
    # ──────────────────────────────────────────────────────────────────
    def _ensure_raw_open(self) -> None:
        """Ensure the underlying HDF5 dataset is open.

        If the recording was previously closed (self._raw is None) and we have a
        repository pointing to the files, reopen the HDF5 file in read mode and
        patch self._raw to the 'raw' dataset. This makes the object usable after
        a close/rename cycle without requiring a full application restart.
        """
        if self._raw is not None:
            return
        if self.repo is None:
            logging.debug(f"Recording '{self.id}' has no repo to reopen raw data.")
            return
        try:
            raw_path = getattr(self.repo, "raw_h5", None)
            if raw_path is None:
                logging.debug(f"Recording '{self.id}' repo has no raw_h5 attribute.")
                return
            raw_path_str = str(raw_path)
            # Check file exists before attempting to open
            try:
                from pathlib import Path

                if not Path(raw_path_str).exists():
                    logging.warning(f"Raw HDF5 not found for recording '{getattr(self, 'id', '<unknown>')}': {raw_path_str}")
                    return
            except Exception:
                pass

            # Open file and keep the dataset handle for lazy, slice-based access
            # NOTE: We intentionally do NOT read the whole dataset into memory.
            #       Holding an h5py.Dataset allows efficient slicing during plotting/analysis.
            #       The file handle will be closed in `close()`.
            h5file = h5py.File(raw_path_str, "r")
            try:
                self._raw = h5file["raw"]  # type: ignore[assignment]
            except Exception:
                # Ensure file is closed if dataset access fails
                try:
                    h5file.close()
                except Exception:
                    logging.exception("Failed to close HDF5 file after dataset access failure.")
                    pass
                raise
            # Update num_samples in metadata in case it changed
            try:
                self.meta.num_samples = int(self._raw.shape[0])
            except Exception:
                pass
            logging.debug(f"Reopened HDF5 for recording '{self.id}' from {raw_path_str}")
        except Exception as err:
            logging.exception(f"Failed to reopen HDF5 for recording '{getattr(self, 'id', '<unknown>')}': {err}")

    def raw_view(self, ch: int | slice | list[int] = slice(None), t: slice = slice(None)) -> np.ndarray:
        """
        Return a NumPy view (or slice) of raw data [time, channels].
        If the underlying dataset was closed (self._raw is None), attempt to
        lazily reopen it from the repository path.
        """
        # Lazily reopen HDF5 dataset if it was previously closed
        if self._raw is None:
            self._ensure_raw_open()

        if self._raw is None:
            raise RuntimeError(f"Raw data for recording '{getattr(self, 'id', '<unknown>')}' is not available.")

        return self._raw[t, ch]

    # ──────────────────────────────────────────────────────────────────
    # 3) Configuration (future extensibility)
    # ──────────────────────────────────────────────────────────────────
    def set_config(self, config: dict) -> None:
        """
        Update the configuration for this recording (future extensibility).
        """
        self._config = config or {}

    # ──────────────────────────────────────────────────────────────────
    # 4) Clean‐up (close HDF5 file when you’re done)
    # ──────────────────────────────────────────────────────────────────
    def close(self) -> None:
        if isinstance(self._raw, h5py.Dataset):
            try:
                # Close the underlying HDF5 file
                self._raw.file.close()
            except Exception as exception:
                logging.exception(f"Failed to close HDF5 file for recording '{self.id}': {exception}")
                pass
            finally:
                # Release the reference so the h5py objects can be garbage collected
                try:
                    self._raw = None
                except Exception:
                    # Ensure we don't raise while cleaning up
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
