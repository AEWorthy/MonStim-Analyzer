"""Working reference importer for a simple, synthetic delimited EMG stream.

This is an SDK example, not an acquisition-system compatibility claim.  It
shows the minimum normalized files an official importer must create.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from monstim_signals.io.normalized_import import NormalizedExperiment, NormalizedRecording


class DelimitedEmgImporter:
    def probe_source(self, source: Path) -> int:
        if source.suffix.lower() != ".csv":
            return 0
        try:
            self.validate_source(source)
        except OSError, ValueError:
            return 0
        return 100

    def validate_source(self, source: Path) -> None:
        with source.open(newline="", encoding="utf-8") as stream:
            fields = set(csv.DictReader(stream).fieldnames or [])
        missing = {"time_ms", "emg_1", "stimulus_v"} - fields
        if missing:
            raise ValueError("Expected columns: time_ms, emg_1, stimulus_v. Missing: " + ", ".join(sorted(missing)))

    def normalize_experiment(self, source: Path, is_canceled) -> NormalizedExperiment:
        self.validate_source(source)
        rows = []
        with source.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if is_canceled():
                    raise InterruptedError("Import canceled before normalization completed.")
                rows.append(row)
        if len(rows) < 2:
            raise ValueError("The source must contain at least two samples.")
        time_ms = np.array([float(row["time_ms"]) for row in rows], dtype=np.float32)
        signal = np.array([float(row["emg_1"]) for row in rows], dtype=np.float32)
        stimulus = float(rows[0]["stimulus_v"])
        scan_rate = float(1000.0 / np.median(np.diff(time_ms)))
        metadata = {
            "data_version": "2.1.0",
            "monstim_version": "external-example",
            "session_id": "ExampleSession",
            "recording_id": "0001",
            "scan_rate": scan_rate,
            "num_channels": 1,
            "num_samples": len(signal),
            "num_emg_channels": 1,
            "channel_types": ["emg"],
            "stim_clusters": [{"stim_v": stimulus, "stim_type": "External example"}],
        }
        return NormalizedExperiment(
            importer_id="example-delimited-emg",
            importer_version="0.1.0",
            recordings=(
                NormalizedRecording(
                    dataset_id="Example dataset",
                    session_id="ExampleSession",
                    recording_id="0001",
                    samples=signal[:, np.newaxis],
                    metadata=metadata,
                    source_provenance={"source_name": source.name},
                ),
            ),
        )


PLUGIN = DelimitedEmgImporter()
