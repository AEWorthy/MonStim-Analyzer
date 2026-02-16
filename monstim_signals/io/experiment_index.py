from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

INDEX_FILENAME = ".index.json"
INDEX_VERSION = 1


@dataclass
class FileInfo:
    path: str
    size: int
    mtime: float


@dataclass
class RecordingIndex:
    path: str
    meta_path: Optional[str] = None
    size: Optional[int] = None
    mtime: Optional[float] = None
    # Metadata extracted from meta.json for fast loading
    meta_size: Optional[int] = None  # size of meta.json file
    meta_mtime: Optional[float] = None  # mtime of meta.json file
    recording_id: Optional[str] = None
    num_channels: Optional[int] = None
    num_samples: Optional[int] = None
    scan_rate: Optional[int] = None
    primary_stim_v: Optional[float] = None  # For sorting recordings


@dataclass
class SessionIndex:
    id: str
    path: str
    recordings: List[RecordingIndex]
    meta_path: Optional[str] = None
    size: Optional[int] = None
    mtime: Optional[float] = None


@dataclass
class DatasetIndex:
    id: str
    path: str
    sessions: List[SessionIndex]
    meta_path: Optional[str] = None
    size: Optional[int] = None
    mtime: Optional[float] = None


@dataclass
class ExperimentIndex:
    id: str
    path: str
    datasets: List[DatasetIndex]
    version: int = INDEX_VERSION
    generated_at: float = 0.0


def _stat_safe(p: Path) -> Tuple[Optional[int], Optional[float]]:
    try:
        st = p.stat()
        return st.st_size, st.st_mtime
    except FileNotFoundError:
        return None, None


def _dump_json(data: dict, file_path: Path) -> None:
    file_path.write_text(json.dumps(data), encoding="utf-8")


def _load_json(file_path: Path) -> dict:
    return json.loads(file_path.read_text(encoding="utf-8"))


def index_path(exp_path: Path) -> Path:
    return exp_path / INDEX_FILENAME


def build_experiment_index(exp_id: str, exp_path: Path, progress_cb: Optional[callable] = None) -> ExperimentIndex:
    datasets: List[DatasetIndex] = []
    ds_names = sorted([d.name for d in exp_path.iterdir() if d.is_dir()])
    total_ds = len(ds_names)
    for i, ds_name in enumerate(ds_names, start=1):
        ds_path = exp_path / ds_name
        ds_meta = ds_path / "dataset.annot.json"
        ds_size, ds_mtime = _stat_safe(ds_meta)

        sessions: List[SessionIndex] = []
        for sess_name in sorted([s.name for s in ds_path.iterdir() if s.is_dir()]):
            sess_path = ds_path / sess_name
            sess_meta = sess_path / "session.annot.json"
            sess_size, sess_mtime = _stat_safe(sess_meta)

            recordings: List[RecordingIndex] = []
            # Recordings can be files or subfolders depending on pipeline; include known extensions
            items = sorted(sess_path.iterdir())
            for item in items:
                if item.is_file():
                    # HDF5/NPY/CSV payloads potentially
                    if item.suffix.lower() in {".h5", ".hdf5", ".npy", ".npz", ".csv"}:
                        r_size, r_mtime = _stat_safe(item)
                        # Extract metadata for fast loading
                        stem = item.with_suffix("")
                        meta_file = stem.with_suffix(".meta.json")
                        meta_size, meta_mtime = _stat_safe(meta_file)
                        # Read minimal metadata from meta.json if it exists
                        recording_id = None
                        num_channels = None
                        num_samples = None
                        scan_rate = None
                        primary_stim_v = None
                        if meta_file.exists():
                            try:
                                meta_dict = _load_json(meta_file)
                                recording_id = meta_dict.get("recording_id")
                                num_channels = meta_dict.get("num_channels")
                                num_samples = meta_dict.get("num_samples")
                                scan_rate = meta_dict.get("scan_rate")
                                # Extract primary_stim.stim_v for sorting
                                primary_stim = meta_dict.get("primary_stim")
                                if isinstance(primary_stim, dict):
                                    primary_stim_v = primary_stim.get("stim_v")
                                elif isinstance(primary_stim, int) and primary_stim > 0:
                                    # primary_stim is 1-based index into stim_clusters
                                    stim_clusters = meta_dict.get("stim_clusters", [])
                                    if 0 <= primary_stim - 1 < len(stim_clusters):
                                        primary_stim_v = stim_clusters[primary_stim - 1].get("stim_v")
                            except Exception:
                                # If metadata extraction fails, continue without it
                                logging.debug(f"Failed to extract metadata from {meta_file}", exc_info=True)
                        recordings.append(
                            RecordingIndex(
                                path=str(item),
                                meta_path=str(meta_file) if meta_file.exists() else None,
                                size=r_size,
                                mtime=r_mtime,
                                meta_size=meta_size,
                                meta_mtime=meta_mtime,
                                recording_id=recording_id,
                                num_channels=num_channels,
                                num_samples=num_samples,
                                scan_rate=scan_rate,
                                primary_stim_v=primary_stim_v,
                            )
                        )
                elif item.is_dir():
                    # Some pipelines store per-recording folders, include marker files if present
                    meta = item / "recording.annot.json"
                    r_size, r_mtime = _stat_safe(meta)
                    recordings.append(
                        RecordingIndex(
                            path=str(item),
                            meta_path=str(meta) if meta.exists() else None,
                            size=r_size,
                            mtime=r_mtime,
                        )
                    )

            sessions.append(
                SessionIndex(
                    id=sess_name,
                    path=str(sess_path),
                    recordings=recordings,
                    meta_path=str(sess_meta) if sess_meta.exists() else None,
                    size=sess_size,
                    mtime=sess_mtime,
                )
            )

        datasets.append(
            DatasetIndex(
                id=ds_name,
                path=str(ds_path),
                sessions=sessions,
                meta_path=str(ds_meta) if ds_meta.exists() else None,
                size=ds_size,
                mtime=ds_mtime,
            )
        )
        # Emit simple dataset-level progress (0-100 across datasets)
        if progress_cb:
            try:
                pct = int((i / max(total_ds, 1)) * 100)
                progress_cb("index", i, total_ds, ds_name, pct)
            except Exception:
                # Silently ignore callback errors to ensure index building continues
                # even if the progress callback fails (e.g., disconnected, GUI closed)
                logging.debug("Progress callback failed during index build.", exc_info=True)

    idx = ExperimentIndex(id=exp_id, path=str(exp_path), datasets=datasets, generated_at=time.time())
    return idx


def save_experiment_index(index: ExperimentIndex) -> None:
    p = Path(index.path)
    dest = index_path(p)
    _dump_json(asdict(index), dest)


def load_experiment_index(exp_path: Path) -> Optional[ExperimentIndex]:
    p = index_path(exp_path)
    if not p.exists():
        return None
    data = _load_json(p)
    try:
        # Basic schema check
        if data.get("version") != INDEX_VERSION:
            return None
        return ExperimentIndex(
            id=data["id"],
            path=data["path"],
            datasets=[
                DatasetIndex(
                    id=ds["id"],
                    path=ds["path"],
                    sessions=[
                        SessionIndex(
                            id=s["id"],
                            path=s["path"],
                            recordings=[
                                RecordingIndex(
                                    path=r["path"],
                                    meta_path=r.get("meta_path"),
                                    size=r.get("size"),
                                    mtime=r.get("mtime"),
                                    meta_size=r.get("meta_size"),
                                    meta_mtime=r.get("meta_mtime"),
                                    recording_id=r.get("recording_id"),
                                    num_channels=r.get("num_channels"),
                                    num_samples=r.get("num_samples"),
                                    scan_rate=r.get("scan_rate"),
                                    primary_stim_v=r.get("primary_stim_v"),
                                )
                                for r in s["recordings"]
                            ],
                            meta_path=s.get("meta_path"),
                            size=s.get("size"),
                            mtime=s.get("mtime"),
                        )
                        for s in ds["sessions"]
                    ],
                    meta_path=ds.get("meta_path"),
                    size=ds.get("size"),
                    mtime=ds.get("mtime"),
                )
                for ds in data["datasets"]
            ],
            version=data.get("version", INDEX_VERSION),
            generated_at=data.get("generated_at", 0.0),
        )
    except Exception:
        return None


def is_index_stale(index: ExperimentIndex) -> bool:
    # Compare stored sizes/mtimes against current filesystem

    # First, check if the number of datasets on filesystem matches the index
    exp_path = Path(index.path)
    try:
        current_ds_folders = [d for d in exp_path.iterdir() if d.is_dir()]
        if len(current_ds_folders) != len(index.datasets):
            return True  # Dataset added or removed
    except Exception:
        return True  # Can't list directory, assume stale

    # Define helper function once outside the loop for better performance
    def is_recording_item(p: Path) -> bool:
        return (p.is_file() and p.suffix.lower() in {".h5", ".hdf5", ".npy", ".npz", ".csv"}) or p.is_dir()

    for ds in index.datasets:
        ds_meta = Path(ds.meta_path) if ds.meta_path else None
        if ds_meta and ds.size is not None and ds.mtime is not None:
            size, mtime = _stat_safe(ds_meta)
            if size != ds.size or mtime != ds.mtime:
                return True
        ds_path = Path(ds.path)
        if not ds_path.exists():
            return True

        for s in ds.sessions:
            s_meta = Path(s.meta_path) if s.meta_path else None
            if s_meta and s.size is not None and s.mtime is not None:
                size, mtime = _stat_safe(s_meta)
                if size != s.size or mtime != s.mtime:
                    return True
            s_path = Path(s.path)
            if not s_path.exists():
                return True
            # Directory-level change detection: new or removed recordings
            try:
                current_items = sorted(s_path.iterdir())
                current_recordings = [p for p in current_items if is_recording_item(p)]
                if len(current_recordings) != len(s.recordings):
                    return True
            except Exception:
                # If we cannot list, assume stale to be safe
                return True
            # Check a subset of recordings to keep it fast; if any mismatch, rebuild
            for r in s.recordings[:50]:  # sample up to 50
                r_path = Path(r.path)
                if not r_path.exists():
                    return True
                # Check raw data file if tracked
                if r.size is not None or r.mtime is not None:
                    size, mtime = _stat_safe(r_path if r_path.is_file() else Path(r.meta_path) if r.meta_path else r_path)
                    if size != r.size or mtime != r.mtime:
                        return True
                # CRITICAL: Check meta.json file for changes to ensure GUI data is never stale
                if r.meta_path and (r.meta_size is not None or r.meta_mtime is not None):
                    meta_file = Path(r.meta_path)
                    if meta_file.exists():
                        meta_size, meta_mtime = _stat_safe(meta_file)
                        if meta_size != r.meta_size or meta_mtime != r.meta_mtime:
                            return True
                    else:
                        # meta.json was deleted - rebuild index
                        return True
    return False


def ensure_fresh_index(exp_id: str, exp_path: Path, progress_cb: Optional[callable] = None) -> ExperimentIndex:
    idx = load_experiment_index(exp_path)
    if idx is None or is_index_stale(idx):
        idx = build_experiment_index(exp_id, exp_path, progress_cb=progress_cb)
        save_experiment_index(idx)
    return idx


# Helper for repositories: provide lightweight tree without opening payloads


def get_lazy_tree(exp_id: str, exp_path: Path, progress_cb: Optional[callable] = None) -> ExperimentIndex:
    """
    Return an ExperimentIndex ensuring freshness. Intended to be used by repository
    loaders to build domain objects lazily: datasets/sessions/recordings with metadata
    only, deferring any payload open until later.
    """
    return ensure_fresh_index(exp_id, exp_path, progress_cb=progress_cb)


def find_session_index(exp_path: Path, session_path: Path) -> Optional[SessionIndex]:
    """Return SessionIndex for a given session folder using the experiment index.

    Falls back to None if index missing or session not found.
    """
    idx = load_experiment_index(exp_path)
    if idx is None:
        return None
    sess_str = str(session_path)
    for ds in idx.datasets:
        for s in ds.sessions:
            if s.path == sess_str:
                return s
    return None
