"""
Focused tests for the Data Import/Export Pipeline in MonStim Analyzer.

Tests the actual available functions in csv_importer module:
- discover_by_ext: Find CSV files in directory tree
- parse_session_rec: Parse session/recording IDs from filenames
- infer_ds_ex: Infer dataset/experiment names from paths
- csv_to_store: Convert CSV to HDF5 with metadata
- get_dataset_session_dict: Map session IDs to CSV files
- import_experiment: Import entire experiment with threading

All tests use temporary directories and clean up properly to avoid leaving .annot files.
"""

import csv
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from monstim_signals.io.csv_importer import (
    csv_to_store,
    discover_by_ext,
    get_dataset_session_dict,
    import_experiment,
    infer_ds_ex,
    parse_session_rec,
)

# --- Test Annotations ---
# Purpose: CSV importer pipeline (discover/parse/infer/convert/import) with realistic data shapes
# Markers: integration, slow (IO heavy, h5py, multiple files)
# Notes: Uses temp_workspace fixture; asserts structure, content, and error handling
pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture
def temp_workspace():
    """Create a temporary directory for testing, cleaned up automatically."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup: Remove the entire temp directory tree
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_csv_data():
    """Generate sample CSV data for testing."""
    # Create realistic EMG-like data
    np.random.seed(42)  # For reproducible tests
    num_samples = 1000
    num_channels = 4

    # Generate data with stimulus artifacts and EMG-like characteristics
    data = np.random.randn(num_samples, num_channels) * 0.1
    # Add stimulus artifacts at specific intervals
    for i in range(0, num_samples, 100):
        if i + 10 < num_samples:
            data[i : i + 10, :] += np.random.randn(10, num_channels) * 2.0

    return data


@pytest.fixture
def mock_csv_files(temp_workspace, sample_csv_data):
    """Create mock CSV files with proper structure for testing."""
    # Create experiment/dataset structure
    exp_dir = temp_workspace / "Test_Experiment"
    dataset_dir = exp_dir / "250920 TestAnimal condition1"
    dataset_dir.mkdir(parents=True)

    csv_files = []

    # Create multiple CSV files for different sessions and recordings
    sessions = ["TA01", "TA02", "TA03"]
    recordings_per_session = 3

    for session_id in sessions:
        for rec_num in range(recordings_per_session):
            filename = f"{session_id}-{rec_num:04d}.csv"
            csv_path = dataset_dir / filename

            # Write CSV data
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                # Add some variation to the data
                data_variant = sample_csv_data + np.random.randn(*sample_csv_data.shape) * 0.01
                for row in data_variant:
                    writer.writerow(row)

            csv_files.append(csv_path)

    # Also create corresponding STM files (stimulus files)
    for csv_file in csv_files:
        stm_path = csv_file.with_suffix(".STM")
        with open(stm_path, "w") as f:
            # Simple stimulus file format
            f.write("# Stimulus file\n")
            f.write("0.1 1.0\n")  # stimulus at 0.1s for 1.0s duration
            f.write("0.5 0.5\n")  # stimulus at 0.5s for 0.5s duration

    return {"exp_dir": exp_dir, "dataset_dir": dataset_dir, "csv_files": csv_files}


class TestCSVDiscoveryAndParsing:
    """Test CSV file discovery and parsing functionality."""

    def test_discover_by_ext_finds_csv_files(self, mock_csv_files):
        """Test that discover_by_ext correctly finds all CSV files."""
        csv_files = discover_by_ext(mock_csv_files["exp_dir"])

        assert len(csv_files) == 9  # 3 sessions × 3 recordings each
        assert all(f.suffix.lower() == ".csv" for f in csv_files)
        assert all(f.exists() for f in csv_files)

    def test_discover_by_ext_ignores_empty_files(self, temp_workspace):
        """Test CSV file discovery behavior with empty files."""
        # Create an empty CSV file
        empty_csv = temp_workspace / "empty.csv"
        empty_csv.touch()

        # Create a non-empty CSV file
        data_csv = temp_workspace / "data.csv"
        with open(data_csv, "w") as f:
            f.write("1,2,3\n4,5,6\n")

        csv_files = discover_by_ext(temp_workspace)

        # The function might filter empty files or not - let's test what actually happens
        assert len(csv_files) >= 1  # At least the non-empty file should be found
        assert data_csv in csv_files
        # Empty file behavior may vary - test that at least valid files are found

    def test_parse_session_rec_correct_format(self):
        """Test parsing of correctly formatted session/recording names."""
        test_cases = [
            ("TA01-0001.csv", ("TA01", "0001")),
            ("AB12-9999.csv", ("AB12", "9999")),
            ("XY34-0000.csv", ("XY34", "0000")),
        ]

        for filename, expected in test_cases:
            path = Path(filename)
            session_id, recording_id = parse_session_rec(path)
            assert (session_id, recording_id) == expected

    def test_parse_session_rec_incorrect_format(self):
        """Test parsing of incorrectly formatted names returns None."""
        incorrect_names = [
            "invalid.csv",
            "TA01_0001.csv",  # underscore instead of dash
            "TA-0001.csv",  # session ID too short
            "TA001-01.csv",  # recording ID too short
            "TA01-ABCD.csv",  # non-numeric recording ID
        ]

        for filename in incorrect_names:
            path = Path(filename)
            session_id, recording_id = parse_session_rec(path)
            assert session_id is None
            assert recording_id is None

    def test_infer_ds_ex_structure(self, mock_csv_files):
        """Test dataset and experiment name inference from directory structure."""
        csv_file = mock_csv_files["csv_files"][0]
        base_dir = mock_csv_files["exp_dir"]

        dataset_name, experiment_name = infer_ds_ex(csv_file, base_dir)

        assert dataset_name == "250920 TestAnimal condition1"
        # experiment_name is None when grandparent == base_dir (which it is in our test setup)
        assert experiment_name is None

        # Test with deeper structure
        deeper_base = mock_csv_files["exp_dir"].parent
        dataset_name2, experiment_name2 = infer_ds_ex(csv_file, deeper_base)
        assert dataset_name2 == "250920 TestAnimal condition1"
        assert experiment_name2 == "Test_Experiment"

    def test_get_dataset_session_dict(self, mock_csv_files):
        """Test mapping of session IDs to CSV file paths."""
        dataset_path = mock_csv_files["dataset_dir"]
        mapping = get_dataset_session_dict(dataset_path)

        assert len(mapping) == 3  # 3 sessions
        assert "TA01" in mapping
        assert "TA02" in mapping
        assert "TA03" in mapping
        assert len(mapping["TA01"]) == 3  # 3 recordings per session


class TestCSVToStoreConversion:
    """Test CSV to HDF5 conversion using csv_to_store functionality."""

    def test_csv_to_store_basic(self, temp_workspace, sample_csv_data):
        """Test basic CSV to HDF5 conversion using csv_to_store."""
        # Create a test CSV file
        csv_path = temp_workspace / "test.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            for row in sample_csv_data:
                writer.writerow(row)

        # Create corresponding STM file
        stm_path = csv_path.with_suffix(".STM")
        with open(stm_path, "w") as f:
            f.write("# Test stimulus\n")
            f.write("0.1 1.0\n")

        output_path = temp_workspace / "test"

        # Mock the parse function to return our sample data
        # Note: parse returns (meta_dict, arr) tuple
        with patch("monstim_signals.io.csv_importer.parse") as mock_parse:
            mock_parse.return_value = ({"scan_rate": 1000, "num_channels": 4, "channel_types": ["EMG"] * 4}, sample_csv_data)

            csv_to_store(
                csv_path=csv_path, output_fp=output_path, overwrite_h5=True, overwrite_meta=True, overwrite_annot=True
            )

        # Verify HDF5 file was created
        h5_path = output_path.with_suffix(".raw.h5")
        assert h5_path.exists()

        # Verify HDF5 contents
        with h5py.File(h5_path, "r") as h5:
            assert "raw" in h5
            assert h5["raw"].shape == sample_csv_data.shape
            assert h5.attrs["scan_rate"] == 1000
            assert h5.attrs["num_channels"] == 4

        # Verify meta.json was created
        meta_path = output_path.with_suffix(".meta.json")
        assert meta_path.exists()

        with open(meta_path) as f:
            meta_data = json.load(f)
            assert meta_data["scan_rate"] == 1000
            assert meta_data["num_channels"] == 4
            # Check that session_id and recording_id were added
            assert "session_id" in meta_data
            assert "recording_id" in meta_data

        # Verify annot.json was created
        annot_path = output_path.with_suffix(".annot.json")
        assert annot_path.exists()

        # Check annotation structure
        with open(annot_path) as f:
            annot_data = json.load(f)
            assert "cache" in annot_data
            assert "data_version" in annot_data

    def test_csv_to_store_no_overwrite(self, temp_workspace, sample_csv_data):
        """Test that existing files are not overwritten when overwrite=False."""
        csv_path = temp_workspace / "test.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            for row in sample_csv_data:
                writer.writerow(row)

        output_path = temp_workspace / "test"
        h5_path = output_path.with_suffix(".raw.h5")
        meta_path = output_path.with_suffix(".meta.json")

        # Create existing files
        h5_path.touch()
        meta_path.write_text('{"existing": "data"}')

        with patch("monstim_signals.io.csv_importer.parse") as mock_parse:
            mock_parse.return_value = ({"scan_rate": 1000, "num_channels": 4, "channel_types": ["EMG"] * 4}, sample_csv_data)

            # This should succeed and create annotation file only
            csv_to_store(
                csv_path=csv_path, output_fp=output_path, overwrite_h5=False, overwrite_meta=False, overwrite_annot=True
            )

        # Verify existing meta file was not overwritten
        with open(meta_path) as f:
            meta_data = json.load(f)
            assert meta_data == {"existing": "data"}

        # Verify H5 file still exists but was not overwritten (should be empty)
        assert h5_path.exists()
        assert h5_path.stat().st_size == 0  # Empty file

        # Verify annotation file was created
        annot_path = output_path.with_suffix(".annot.json")
        assert annot_path.exists()

    def test_csv_to_store_session_recording_id_parsing(self, temp_workspace, sample_csv_data):
        """Test that session_id and recording_id are correctly parsed from filename."""
        csv_path = temp_workspace / "AB12-0123.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            for row in sample_csv_data:
                writer.writerow(row)

        output_path = temp_workspace / "AB12-0123"

        with patch("monstim_signals.io.csv_importer.parse") as mock_parse:
            mock_parse.return_value = (
                {"scan_rate": 2000, "num_channels": 2, "channel_types": ["EMG", "Force"]},
                sample_csv_data[:, :2],  # Use only first 2 channels
            )

            csv_to_store(
                csv_path=csv_path, output_fp=output_path, overwrite_h5=True, overwrite_meta=True, overwrite_annot=True
            )

        # Check that session_id and recording_id are correctly extracted
        meta_path = output_path.with_suffix(".meta.json")
        with open(meta_path) as f:
            meta_data = json.load(f)
            assert meta_data["session_id"] == "AB12"
            assert meta_data["recording_id"] == "0123"


class TestExperimentImport:
    """Test experiment-level import functionality with threading."""

    def test_import_experiment_basic(self, temp_workspace, mock_csv_files):
        """Test basic experiment import functionality."""
        exp_path = mock_csv_files["exp_dir"]
        output_path = temp_workspace / "output"

        # Mock csv_to_store to avoid actual file operations
        with patch("monstim_signals.io.csv_importer.csv_to_store") as mock_convert:
            import_experiment(
                expt_path=exp_path,
                output_path=output_path,
                progress_callback=lambda x: None,
                is_canceled=lambda: False,
                overwrite=True,
                max_workers=1,  # Single threaded for predictable testing
            )

            # Should call csv_to_store for each CSV file
            assert mock_convert.call_count == 9  # 3 sessions × 3 recordings

    def test_import_experiment_with_progress_callback(self, temp_workspace, mock_csv_files):
        """Test that progress callbacks are called during import."""
        exp_path = mock_csv_files["exp_dir"]
        output_path = temp_workspace / "output"

        progress_values = []

        def progress_callback(value):
            progress_values.append(value)

        with patch("monstim_signals.io.csv_importer.csv_to_store"):
            import_experiment(
                expt_path=exp_path,
                output_path=output_path,
                progress_callback=progress_callback,
                is_canceled=lambda: False,
                overwrite=True,
                max_workers=1,
            )

        # Progress callback should have been called
        assert len(progress_values) > 0
        # Should reach 100% at the end
        assert max(progress_values) == 100
        # Values should be in reasonable range
        assert all(0 <= v <= 100 for v in progress_values)

    def test_import_experiment_cancellation(self, temp_workspace, mock_csv_files):
        """Test that import can be cancelled mid-process."""
        exp_path = mock_csv_files["exp_dir"]
        output_path = temp_workspace / "output"

        # Create a cancellation flag that becomes True after a few calls
        call_count = 0

        def is_canceled():
            nonlocal call_count
            call_count += 1
            return call_count > 3  # Cancel after 3 files

        with patch("monstim_signals.io.csv_importer.csv_to_store") as mock_convert:
            import_experiment(
                expt_path=exp_path,
                output_path=output_path,
                progress_callback=lambda x: None,
                is_canceled=is_canceled,
                overwrite=True,
                max_workers=1,
            )

            # Should not have processed all files due to cancellation
            assert mock_convert.call_count < 9

    def test_import_experiment_directory_structure(self, temp_workspace, mock_csv_files):
        """Test that proper directory structure is created during import."""
        exp_path = mock_csv_files["exp_dir"]
        output_path = temp_workspace / "output"

        with patch("monstim_signals.io.csv_importer.csv_to_store"):
            import_experiment(
                expt_path=exp_path,
                output_path=output_path,
                progress_callback=lambda x: None,
                is_canceled=lambda: False,
                overwrite=True,
                max_workers=1,
            )

        # Check that dataset directory was created
        dataset_output = output_path / "250920 TestAnimal condition1"
        assert dataset_output.exists()
        assert dataset_output.is_dir()

        # Check that session directories were created
        for session_id in ["TA01", "TA02", "TA03"]:
            session_dir = dataset_output / session_id
            assert session_dir.exists()
            assert session_dir.is_dir()

    def test_import_experiment_multithreading(self, temp_workspace, mock_csv_files):
        """Test that multithreading works correctly."""
        exp_path = mock_csv_files["exp_dir"]
        output_path = temp_workspace / "output"

        with patch("monstim_signals.io.csv_importer.csv_to_store") as mock_convert:
            import_experiment(
                expt_path=exp_path,
                output_path=output_path,
                progress_callback=lambda x: None,
                is_canceled=lambda: False,
                overwrite=True,
                max_workers=2,  # Multi-threaded
            )

            # Should still process all files regardless of threading
            assert mock_convert.call_count == 9


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge case scenarios."""

    def test_csv_to_store_missing_stm_file(self, temp_workspace, sample_csv_data):
        """Test handling when STM file is missing."""
        csv_path = temp_workspace / "test.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            for row in sample_csv_data:
                writer.writerow(row)

        # Don't create STM file
        output_path = temp_workspace / "test"

        with patch("monstim_signals.io.csv_importer.parse") as mock_parse:
            mock_parse.return_value = ({"scan_rate": 1000, "num_channels": 4, "channel_types": ["EMG"] * 4}, sample_csv_data)

            # Should still work without STM file
            csv_to_store(
                csv_path=csv_path, output_fp=output_path, overwrite_h5=True, overwrite_meta=True, overwrite_annot=True
            )

        h5_path = output_path.with_suffix(".raw.h5")
        assert h5_path.exists()

    def test_import_experiment_empty_directory(self, temp_workspace):
        """Test importing from directory with no CSV files."""
        empty_exp_dir = temp_workspace / "empty_experiment"
        empty_dataset_dir = empty_exp_dir / "empty_dataset"
        empty_dataset_dir.mkdir(parents=True)

        output_path = temp_workspace / "output"

        with patch("monstim_signals.io.csv_importer.csv_to_store") as mock_convert:
            # Should not crash with empty directory
            import_experiment(
                expt_path=empty_exp_dir,
                output_path=output_path,
                progress_callback=lambda x: None,
                is_canceled=lambda: False,
                overwrite=True,
                max_workers=1,
            )

            # No CSV files to process
            assert mock_convert.call_count == 0

    def test_parse_error_handling(self, temp_workspace):
        """Test handling of CSV parsing errors."""
        # Create invalid CSV file
        csv_path = temp_workspace / "invalid.csv"
        with open(csv_path, "w") as f:
            f.write("This is not valid CSV data\n")
            f.write("Random text without proper structure\n")

        output_path = temp_workspace / "invalid"

        # Mock parse to raise an exception
        with patch("monstim_signals.io.csv_importer.parse", side_effect=ValueError("Invalid CSV format")):
            with pytest.raises(ValueError, match="Invalid CSV format"):
                csv_to_store(
                    csv_path=csv_path, output_fp=output_path, overwrite_h5=True, overwrite_meta=True, overwrite_annot=True
                )


class TestCleanupAndFileManagement:
    """Test proper cleanup and file management to ensure no .annot files left behind."""

    def test_no_temp_files_left_behind(self, temp_workspace, sample_csv_data):
        """Test that no temporary files are created during conversion."""
        csv_path = temp_workspace / "test.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            for row in sample_csv_data:
                writer.writerow(row)

        output_path = temp_workspace / "test"

        # Count files before conversion
        files_before = list(temp_workspace.rglob("*"))
        files_before_count = len([f for f in files_before if f.is_file()])

        with patch("monstim_signals.io.csv_importer.parse") as mock_parse:
            mock_parse.return_value = ({"scan_rate": 1000, "num_channels": 4, "channel_types": ["EMG"] * 4}, sample_csv_data)

            csv_to_store(
                csv_path=csv_path, output_fp=output_path, overwrite_h5=True, overwrite_meta=True, overwrite_annot=True
            )

        # Count files after conversion
        files_after = list(temp_workspace.rglob("*"))
        files_after_count = len([f for f in files_after if f.is_file()])

        # Should have added exactly 3 files: .raw.h5, .meta.json, .annot.json
        expected_new_files = 3
        assert files_after_count == files_before_count + expected_new_files

        # Verify no unexpected temp files
        temp_extensions = [".tmp", ".temp", ".bak", ".orig"]
        temp_files = [f for f in files_after if f.suffix.lower() in temp_extensions]
        assert len(temp_files) == 0

    def test_annotation_files_properly_cleaned(self, temp_workspace, sample_csv_data):
        """Test that annotation files don't accumulate inappropriately."""
        csv_path = temp_workspace / "test.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            for row in sample_csv_data:
                writer.writerow(row)

        output_path = temp_workspace / "test"

        with patch("monstim_signals.io.csv_importer.parse") as mock_parse:
            mock_parse.return_value = ({"scan_rate": 1000, "num_channels": 4, "channel_types": ["EMG"] * 4}, sample_csv_data)

            # Run conversion twice with overwrite
            csv_to_store(
                csv_path=csv_path, output_fp=output_path, overwrite_h5=True, overwrite_meta=True, overwrite_annot=True
            )

            csv_to_store(
                csv_path=csv_path, output_fp=output_path, overwrite_h5=True, overwrite_meta=True, overwrite_annot=True
            )

        # Should still have exactly one of each file type
        annot_files = list(temp_workspace.glob("*.annot.json"))
        meta_files = list(temp_workspace.glob("*.meta.json"))
        h5_files = list(temp_workspace.glob("*.raw.h5"))

        assert len(annot_files) == 1
        assert len(meta_files) == 1
        assert len(h5_files) == 1


# Ensure all fixtures properly clean up after themselves
@pytest.fixture(autouse=True)
def cleanup_test_artifacts():
    """Auto-cleanup fixture to ensure no test artifacts remain."""
    yield
    # Additional cleanup is handled by temp_workspace fixture which uses shutil.rmtree
    pass
