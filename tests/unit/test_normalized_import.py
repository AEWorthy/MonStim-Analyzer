from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from monstim_signals.io.normalized_import import (
    CanonicalImportError,
    NormalizedExperiment,
    NormalizedRecording,
    write_transactional_experiment,
)


def _recording(**updates):
    values = {
        "dataset_id": "Dataset A",
        "session_id": "AB12",
        "recording_id": "0001",
        "samples": np.array([[0.0], [1.0]], dtype=np.float32),
        "metadata": {"scan_rate": 1000, "num_channels": 1, "num_samples": 2, "channel_types": ["emg"], "stim_clusters": []},
    }
    values.update(updates)
    return NormalizedRecording(**values)


def _experiment(*recordings):
    return NormalizedExperiment(tuple(recordings), "test-importer", "1.0.0")


def test_transactional_import_writes_canonical_files_and_provenance(tmp_path):
    target = tmp_path / "Imported experiment"

    write_transactional_experiment(_experiment(_recording()), target)

    stem = target / "Dataset A" / "AB12" / "AB12-0001"
    assert stem.with_suffix(".raw.h5").is_file()
    metadata = json.loads(stem.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert metadata["import_provenance"]["importer"] == {"id": "test-importer", "version": "1.0.0"}


def test_transactional_import_refuses_existing_experiment_without_touching_it(tmp_path):
    target = tmp_path / "Existing"
    target.mkdir()
    sentinel = target / "do-not-touch.txt"
    sentinel.write_text("user research data", encoding="utf-8")

    with pytest.raises(CanonicalImportError, match="will not be overwritten"):
        write_transactional_experiment(_experiment(_recording()), target)

    assert sentinel.read_text(encoding="utf-8") == "user research data"


def test_failed_import_removes_only_staging_directory(tmp_path):
    target = tmp_path / "Broken"
    invalid = _recording(metadata={"scan_rate": 0, "num_channels": 1, "num_samples": 2, "channel_types": ["emg"]})

    with pytest.raises(CanonicalImportError, match="scan_rate"):
        write_transactional_experiment(_experiment(invalid), target)

    assert not target.exists()
    assert not list(tmp_path.glob(".Broken.importing-*"))


def test_synthetic_example_importer_normalizes_then_uses_transactional_writer(tmp_path):
    root = Path(__file__).resolve().parents[2]
    importer_path = root / "examples" / "importers" / "delimited_emg" / "importer.py"
    source = root / "examples" / "importers" / "delimited_emg" / "synthetic_example.csv"
    spec = importlib.util.spec_from_file_location("synthetic_importer", importer_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    normalized = module.PLUGIN.normalize_experiment(source, lambda: False)
    target = tmp_path / "Example import"
    write_transactional_experiment(normalized, target)

    assert (target / "Example dataset" / "ExampleSession" / "ExampleSession-0001.raw.h5").is_file()
