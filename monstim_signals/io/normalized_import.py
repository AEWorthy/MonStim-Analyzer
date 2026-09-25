"""Canonical, transactional storage for built-in and add-on importers.

Importers must never write into a user's experiment directory themselves.
They describe normalized recordings and this module validates, stages, and
atomically activates the managed-store representation.
"""

from __future__ import annotations

import os
import re
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from monstim_signals.core import RecordingAnnot
from monstim_signals.io.repositories import RecordingRepository
from monstim_signals.version import DATA_VERSION


class CanonicalImportError(ValueError):
    """A source cannot be represented safely in MonStim's managed store."""


_SAFE_COMPONENT = re.compile(r'^[^<>:"/\\|?*\x00-\x1f]+$')


@dataclass(frozen=True)
class NormalizedRecording:
    """One fully decoded recording ready for canonical validation."""

    dataset_id: str
    session_id: str
    recording_id: str
    samples: np.ndarray
    metadata: dict[str, Any]
    source_provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizedExperiment:
    """A collection of recordings that will become one managed experiment."""

    recordings: tuple[NormalizedRecording, ...]
    importer_id: str
    importer_version: str


def _safe_component(value: str, label: str) -> str:
    value = str(value).strip()
    if not value or value in {".", ".."} or not _SAFE_COMPONENT.fullmatch(value):
        raise CanonicalImportError(f"{label} must be a non-empty safe file-name component.")
    return value


def validate_recording(recording: NormalizedRecording) -> dict[str, Any]:
    """Return validated metadata without changing the source recording."""
    _safe_component(recording.dataset_id, "Dataset ID")
    _safe_component(recording.session_id, "Session ID")
    _safe_component(recording.recording_id, "Recording ID")
    samples = np.asarray(recording.samples)
    if samples.ndim != 2 or not samples.shape[0] or not samples.shape[1]:
        raise CanonicalImportError("Recording samples must be a non-empty two-dimensional array (samples x channels).")
    if not np.issubdtype(samples.dtype, np.number) or not np.isfinite(samples).all():
        raise CanonicalImportError("Recording samples must contain only finite numeric values.")
    metadata = dict(recording.metadata)
    try:
        scan_rate = float(metadata["scan_rate"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CanonicalImportError("Recording metadata requires a positive numeric scan_rate.") from exc
    if not np.isfinite(scan_rate) or scan_rate <= 0:
        raise CanonicalImportError("Recording metadata scan_rate must be positive.")
    channel_types = metadata.get("channel_types")
    valid_channel_types = isinstance(channel_types, list) and all(isinstance(item, str) and item for item in channel_types)
    if not valid_channel_types or len(channel_types) != samples.shape[1]:
        raise CanonicalImportError("channel_types must contain one non-empty type for every sample channel.")
    supplied_channels = metadata.get("num_channels", samples.shape[1])
    if int(supplied_channels) != samples.shape[1]:
        raise CanonicalImportError("num_channels does not match the imported sample matrix.")
    supplied_samples = metadata.get("num_samples", samples.shape[0])
    if int(supplied_samples) != samples.shape[0]:
        raise CanonicalImportError("num_samples does not match the imported sample matrix.")
    clusters = metadata.get("stim_clusters", [])
    if not isinstance(clusters, list) or not all(isinstance(cluster, dict) for cluster in clusters):
        raise CanonicalImportError("stim_clusters must be a list of metadata objects.")
    metadata.update(
        {
            "data_version": DATA_VERSION,
            "session_id": recording.session_id,
            "recording_id": recording.recording_id,
            "scan_rate": scan_rate,
            "num_channels": int(samples.shape[1]),
            "num_samples": int(samples.shape[0]),
            "channel_types": channel_types,
        }
    )
    return metadata


def _recording_stem(recording: NormalizedRecording) -> str:
    return f"{recording.session_id}-{recording.recording_id}"


def write_recording(recording: NormalizedRecording, destination: Path, *, overwrite: bool = False) -> Path:
    """Write one validated recording triple to *destination* and return its stem."""
    metadata = validate_recording(recording)
    destination.mkdir(parents=True, exist_ok=True)
    stem = destination / _recording_stem(recording)
    targets = [stem.with_suffix(suffix) for suffix in (".raw.h5", ".meta.json", ".annot.json")]
    if any(path.exists() for path in targets) and not overwrite:
        raise CanonicalImportError(f"Managed recording already exists: {stem.name}")
    samples = np.asarray(recording.samples, dtype=np.float32)
    with h5py.File(targets[0], "w") as store:
        store.create_dataset("raw", data=samples, chunks=(min(30000, samples.shape[0]), samples.shape[1]), compression="gzip")
        for key in ("scan_rate", "num_channels", "channel_types", "num_samples"):
            store.attrs[key] = metadata[key]
    provenance = {
        "importer": recording.source_provenance.get("importer", {}),
        "source": {key: value for key, value in recording.source_provenance.items() if key != "importer"},
    }
    if provenance["importer"] or provenance["source"]:
        metadata["import_provenance"] = provenance
    RecordingRepository(stem).save_metadata(metadata)
    RecordingRepository(stem).save_annotation(RecordingAnnot.create_empty(), refresh_catalog=False)
    return stem


def validate_managed_experiment(path: Path) -> None:
    """Verify every staged recording has its three canonical files and raw data."""
    raw_files = list(path.rglob("*.raw.h5"))
    if not raw_files:
        raise CanonicalImportError("The importer did not produce any recordings.")
    for raw_path in raw_files:
        stem = raw_path.with_suffix("").with_suffix("")
        for suffix in (".meta.json", ".annot.json"):
            if not stem.with_suffix(suffix).is_file():
                raise CanonicalImportError(f"Staged recording is incomplete: {raw_path.name}")
        with h5py.File(raw_path, "r") as store:
            if "raw" not in store or store["raw"].ndim != 2:
                raise CanonicalImportError(f"Staged recording has invalid raw data: {raw_path.name}")


def write_transactional_experiment(
    experiment: NormalizedExperiment,
    destination: Path,
    *,
    progress_callback: Callable[[int], None] = lambda _value: None,
    is_canceled: Callable[[], bool] = lambda: False,
) -> None:
    """Stage an import beside *destination* and atomically activate it on success.

    Existing user experiments are never overwritten. A cancellation or error
    removes only the unique staging directory and leaves sources untouched.
    """
    if not experiment.recordings:
        raise CanonicalImportError("An importer must return at least one recording.")
    if destination.exists():
        raise CanonicalImportError(f"Experiment already exists and will not be overwritten: {destination.name}")
    _safe_component(destination.name, "Experiment name")
    staging = destination.parent / f".{destination.name}.importing-{uuid.uuid4().hex}"
    try:
        staging.mkdir(parents=True, exist_ok=False)
        total = len(experiment.recordings)
        seen = set()
        for index, recording in enumerate(experiment.recordings, start=1):
            if is_canceled():
                raise InterruptedError("Importer canceled; no data was activated.")
            key = (recording.dataset_id, recording.session_id, recording.recording_id)
            if key in seen:
                raise CanonicalImportError("Importer returned duplicate dataset/session/recording identifiers.")
            seen.add(key)
            provenance = dict(recording.source_provenance)
            provenance["importer"] = {"id": experiment.importer_id, "version": experiment.importer_version}
            staged_recording = NormalizedRecording(
                recording.dataset_id, recording.session_id, recording.recording_id, recording.samples, recording.metadata, provenance
            )
            write_recording(staged_recording, staging / recording.dataset_id / recording.session_id)
            progress_callback(int(index / total * 100))
        validate_managed_experiment(staging)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
