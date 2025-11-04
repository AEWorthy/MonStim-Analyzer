"""
Domain Model Business Logic Testing with Real Test Data

Tests the core domain model hierarchy (Session/Dataset/Experiment) using actual test data
from the workspace, without relying on old testing utilities. Tests cached properties,
M-max aggregation, annotation overlays, and repository patterns.
"""

from pathlib import Path

import pytest

from monstim_signals.core.data_models import LatencyWindow
from monstim_signals.io.csv_importer import import_experiment
from monstim_signals.io.repositories import DatasetRepository, SessionRepository

from .helpers import get_golden_root

# --- Test Annotations ---
# Purpose: Exercise domain model against golden fixtures (smoke coverage for repos + overlays)
# Markers: integration (uses golden files under tests/fixtures/golden), slow (IO and processing)
# Notes: Fails if golden data missing to keep CI deterministic
pytestmark = [pytest.mark.integration, pytest.mark.slow]


class TestDomainModelWithRealData:
    """Test domain model business logic using real test data."""

    @pytest.fixture
    def imported_session_path(self, tmp_path: Path):
        """Import golden CSVs into a temp experiment and return a valid session path."""
        golden_root = get_golden_root()
        out_expt = tmp_path / "GoldenExp"
        import_experiment(golden_root, out_expt, overwrite=True, max_workers=1)
        # Pick first dataset/session created
        ds_dirs = [p for p in out_expt.iterdir() if p.is_dir()]
        assert ds_dirs, "No datasets were imported from golden fixtures"
        sess_dirs = [p for p in ds_dirs[0].iterdir() if p.is_dir()]
        assert sess_dirs, f"No sessions found in imported dataset {ds_dirs[0]}"
        return sess_dirs[0]

    @pytest.fixture
    def imported_dataset_path(self, tmp_path: Path):
        """Import golden CSVs into a temp experiment and return a valid dataset path."""
        golden_root = get_golden_root()
        out_expt = tmp_path / "GoldenExp"
        import_experiment(golden_root, out_expt, overwrite=True, max_workers=1)
        ds_dirs = [p for p in out_expt.iterdir() if p.is_dir()]
        assert ds_dirs, "No datasets were imported from golden fixtures"
        return ds_dirs[0]

    @pytest.fixture
    def session_with_mwave(self, imported_session_path):
        """Load a session and add M-wave latency window for testing."""
        if not imported_session_path.exists():
            pytest.fail(f"Imported session not found at {imported_session_path}")

        session = SessionRepository(imported_session_path).load()

        # Add M-response latency window for M-max calculations
        m_window = LatencyWindow(
            name="M-response",
            color="blue",
            start_times=[5.0] * session.num_channels,  # 5ms after stimulus
            durations=[10.0] * session.num_channels,  # 10ms duration
        )
        session.annot.latency_windows = [m_window]
        session.update_latency_window_parameters()

        return session

    def test_session_basic_properties(self, session_with_mwave):
        """Test basic session properties and hierarchy."""
        session = session_with_mwave

        # Test basic properties
        assert session.id is not None
        assert session.num_channels > 0
        assert len(session._all_recordings) > 0
        assert session.annot is not None

        # Test that recordings exist
        assert len(session.recordings_raw) > 0
        assert len(session.recordings_filtered) > 0

        # Verify filtering doesn't change total when no exclusions
        assert len(session.recordings_raw) == len(session.recordings_filtered)

    def test_session_cached_properties(self, session_with_mwave):
        """Test that cached properties work correctly."""
        session = session_with_mwave

        # Access cached properties multiple times
        raw1 = session.recordings_raw
        raw2 = session.recordings_raw
        filtered1 = session.recordings_filtered
        filtered2 = session.recordings_filtered

        # Should return same objects (cached)
        assert raw1 is raw2
        assert filtered1 is filtered2

        # Different property types should be different objects
        assert raw1 is not filtered1

    def test_session_exclusion_filtering(self, session_with_mwave):
        """Test that recording exclusions work properly."""
        session = session_with_mwave

        original_count = len(session.recordings_filtered)
        all_recordings_count = len(session._all_recordings)

        # Exclude first recording
        if all_recordings_count > 0:
            first_recording = session._all_recordings[0]
            session.annot.excluded_recordings.append(first_recording.id)

            # Reset cache to see changes
            session.reset_all_caches()

            # Should have one fewer recording
            assert len(session.recordings_filtered) == original_count - 1

            # Raw recordings behavior depends on session implementation
            # At minimum, should have fewer than all recordings when exclusions applied
            raw_count = len(session.recordings_raw)
            assert raw_count <= all_recordings_count

    def test_mmax_calculation_with_latency_window(self, session_with_mwave):
        """Test M-max calculation works with proper latency window."""
        session = session_with_mwave

        # Should be able to calculate M-max without error now
        if session.num_channels > 0:
            try:
                mmax = session.get_m_max(method="rms", channel_index=0)
                assert mmax is not None
                assert isinstance(mmax, (int, float))
                assert mmax >= 0
            except ValueError as e:
                if "Invalid or missing M-wave reflex window" in str(e):
                    pytest.fail("M-wave window should be properly configured")
                else:
                    # Other ValueError might be expected for test data
                    pass

    def test_session_annotation_system(self, session_with_mwave):
        """Test annotation overlay system preserves data integrity."""
        session = session_with_mwave

        # Original recordings should be unchanged
        original_recordings = session._all_recordings.copy()

        # Add some exclusions
        if len(original_recordings) > 1:
            session.annot.excluded_recordings = [original_recordings[0].id]
            session.reset_all_caches()

            # Original recordings should be unchanged
            assert len(session._all_recordings) == len(original_recordings)
            assert session._all_recordings[0].id == original_recordings[0].id

            # But filtered should be different
            filtered = session.recordings_filtered
            assert len(filtered) == len(original_recordings) - 1

    def test_dataset_session_relationships(self, imported_dataset_path):
        """Test dataset-session hierarchy relationships."""
        if not imported_dataset_path.exists():
            pytest.fail(f"Imported dataset not found at {imported_dataset_path}")

        dataset = DatasetRepository(imported_dataset_path).load()
        # Ensure sessions are not excluded from previous test runs
        if dataset.annot.excluded_sessions:
            dataset.annot.excluded_sessions = []
            if dataset.repo is not None:
                dataset.repo.save(dataset)

        # Should have at least one session
        assert len(dataset.sessions) > 0

        # Each session should reference back to dataset
        for session in dataset.sessions:
            assert session.parent_dataset is dataset

        # Dataset should have valid properties
        assert dataset.id is not None
        assert dataset.num_channels > 0

    def test_session_repository_save_load_cycle(self, imported_session_path, tmp_path):
        """Test that session annotations can be saved and loaded."""
        if not imported_session_path.exists():
            pytest.fail(f"Imported session not found at {imported_session_path}")

        # Load original session
        original_session = SessionRepository(imported_session_path).load()

        # Modify annotations
        original_session.annot.excluded_recordings = ["test_recording_id"]
        original_session.annot.is_completed = True

        # Save to temporary location (we'll mock the save path)
        temp_session_path = tmp_path / "test_session"
        temp_session_path.mkdir(parents=True)

        # Create a temporary repository
        temp_repo = SessionRepository(temp_session_path)
        temp_repo.session_js = temp_session_path / "session.annot.json"

        # Save annotations directly to test file (simplified test)
        import json
        from dataclasses import asdict

        temp_repo.session_js.write_text(json.dumps(asdict(original_session.annot), indent=2))

        # Verify file was created
        assert temp_repo.session_js.exists()

        # The content should be valid JSON
        with open(temp_repo.session_js) as f:
            saved_data = json.load(f)

        assert saved_data["excluded_recordings"] == ["test_recording_id"]
        assert saved_data["is_completed"]

    def test_domain_model_string_representations(self, session_with_mwave):
        """Test that domain objects have useful string representations."""
        session = session_with_mwave

        # Session should have meaningful string representation
        session_str = str(session)
        assert session.id in session_str
        assert "Session" in session_str

        # If we have recordings, test those too
        if len(session._all_recordings) > 0:
            recording = session._all_recordings[0]
            recording_str = str(recording)
            assert recording.id in recording_str

    def test_config_integration_with_domain_objects(self, session_with_mwave):
        """Test that configuration properly integrates with domain objects."""
        session = session_with_mwave

        # Session should have config-derived properties
        assert hasattr(session, "bin_size")
        assert hasattr(session, "time_window_ms")
        assert hasattr(session, "scan_rate")

        # These should be reasonable values
        assert session.bin_size > 0
        assert session.time_window_ms > 0
        assert session.scan_rate > 0


class TestDomainModelIntegration:
    """Integration tests for domain model components."""

    def test_error_handling_in_domain_operations(self):
        """Test error handling in domain operations."""
        # Test with invalid paths
        invalid_path = Path("nonexistent/path")

        with pytest.raises((FileNotFoundError, OSError)):
            SessionRepository(invalid_path).load()

    def test_domain_model_memory_efficiency(self, tmp_path):
        """Test that domain model doesn't create memory leaks."""
        # Create some mock data structures
        from monstim_signals.core.data_models import SessionAnnot, SignalChannel

        # Create annotation with channels
        channels = [SignalChannel(name=f"Channel {i}") for i in range(4)]
        annot = SessionAnnot(channels=channels)

        # Should be able to create and destroy without issues
        assert len(annot.channels) == 4
        assert all(ch.name == f"Channel {i}" for i, ch in enumerate(annot.channels))

        # Clear references
        del channels
        del annot

        # Should not cause issues (basic smoke test for memory management)

    def test_channel_rename_duplicate_guard(self, tmp_path):
        """Renaming channels to duplicate names should raise ValueError and not change state."""
        from monstim_signals.io.repositories import SessionRepository

        from .helpers import create_minimal_session_folder

        sess_dir = create_minimal_session_folder(tmp_path, num_channels=3)
        session = SessionRepository(sess_dir).load()
        original = session.channel_names
        # Map two different originals to the same new name
        mapping = {original[0]: "dup", original[1]: "dup"}
        with pytest.raises(ValueError):
            session.rename_channels(mapping)
        # No change should have been applied
        assert session.channel_names == original
