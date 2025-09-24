from __future__ import annotations

import time
from pathlib import Path

import pytest

from monstim_signals.io.csv_importer import import_experiment, parse_session_rec
from monstim_signals.io.repositories import ExperimentRepository

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _scan_golden_sessions(ds_dir: Path) -> dict[str, set[str]]:
    """Return mapping: session_id -> set of recording stems for CSVs under ds_dir."""
    mapping: dict[str, set[str]] = {}
    for csv in ds_dir.glob("*.csv"):
        sess, _rec = parse_session_rec(csv)
        if not sess:
            continue
        mapping.setdefault(sess, set()).add(csv.stem)
    return mapping


def _get_golden_root() -> Path:
    """Locate golden fixtures, preferring populated directories."""
    base = Path(__file__).parent
    candidates = [base / "fixtures" / "golden", base / "golden"]

    # Prefer directories with actual datasets
    for p in candidates:
        if p.exists():
            datasets = [child for child in p.iterdir() if child.is_dir() and child.name != "invalid"]
            if datasets:
                return p

    # Fallback to any existing directory
    for p in candidates:
        if p.exists():
            return p

    raise AssertionError("Missing golden dataset in tests/fixtures/golden or tests/golden")


def _count_csv_files(root: Path) -> int:
    """Count total CSV files in valid datasets (excluding invalid subdirectory)."""
    count = 0
    for ds_path in root.iterdir():
        if ds_path.is_dir() and ds_path.name != "invalid":
            count += len(list(ds_path.glob("**/*.csv")))
    return count


@pytest.mark.usefixtures("temp_output_dir")
class TestGoldenCSVImport:
    def test_import_structure_and_integrity(self, temp_output_dir: Path):
        """Test CSV import creates correct file structure and preserves content integrity."""
        golden_root = _get_golden_root()

        # Skip invalid subdirectory for positive tests
        ds_names = [p.name for p in golden_root.iterdir() if p.is_dir() and p.name != "invalid"]
        assert len(ds_names) >= 2, f"Expected at least 2 valid datasets in golden, found: {ds_names}"

        # Build expectation by scanning CSVs
        expected_layout = {ds: _scan_golden_sessions(golden_root / ds) for ds in ds_names}
        total_csvs = sum(len(stems) for stems in expected_layout.values() for stems in stems.values())

        # Import into temporary experiment folder
        out_expt = temp_output_dir / "GoldenExp"
        import_experiment(golden_root, out_expt, overwrite=True, max_workers=1)

        # Verify per-dataset and per-session files exist and counts match
        total_outputs = 0
        for ds, sess_map in expected_layout.items():
            ds_out = out_expt / ds
            assert ds_out.exists() and ds_out.is_dir(), f"Dataset output folder missing: {ds_out}"

            for sess, stems in sess_map.items():
                sdir = ds_out / sess
                assert sdir.exists() and sdir.is_dir(), f"Session output folder missing: {sdir}"

                # For each stem, check triple of files written by importer
                for stem in stems:
                    base = sdir / stem
                    h5_file = base.with_suffix(".raw.h5")
                    meta_file = base.with_suffix(".meta.json")
                    annot_file = base.with_suffix(".annot.json")

                    assert h5_file.exists(), f"Missing H5 file: {h5_file}"
                    assert meta_file.exists(), f"Missing meta file: {meta_file}"
                    assert annot_file.exists(), f"Missing annotation file: {annot_file}"

                    # Verify files are non-empty
                    assert h5_file.stat().st_size > 0, f"Empty H5 file: {h5_file}"
                    assert meta_file.stat().st_size > 0, f"Empty meta file: {meta_file}"
                    assert annot_file.stat().st_size > 0, f"Empty annotation file: {annot_file}"

                    total_outputs += 1

        assert total_outputs == total_csvs, f"Expected {total_csvs} output sets, got {total_outputs}"

    def test_invalid_csv_handling(self, temp_output_dir: Path):
        """Test that invalid CSVs are handled gracefully without crashing import."""
        golden_root = _get_golden_root()
        invalid_root = golden_root / "invalid"

        if not invalid_root.exists():
            pytest.skip("No invalid test cases found")

        out_expt = temp_output_dir / "InvalidTest"

        # Import should handle exceptions gracefully and continue processing
        # Invalid CSVs should be logged as errors but not crash the entire import
        import_experiment(invalid_root, out_expt, overwrite=True, max_workers=1)

        # Verify no H5 files were created from invalid inputs
        if out_expt.exists():
            h5_files = list(out_expt.rglob("*.raw.h5"))
            assert len(h5_files) == 0, f"Invalid inputs should not produce H5 files, but found: {h5_files}"

    def test_domain_object_consistency(self, temp_output_dir: Path):
        """Test domain objects load correctly and match CSV structure."""
        golden_root = _get_golden_root()
        ds_names = [p.name for p in golden_root.iterdir() if p.is_dir() and p.name != "invalid"]
        expected_layout = {ds: _scan_golden_sessions(golden_root / ds) for ds in ds_names}

        out_expt = temp_output_dir / "GoldenDomain"
        import_experiment(golden_root, out_expt, overwrite=True, max_workers=1)

        # Load domain objects and verify structure
        expt = ExperimentRepository(out_expt).load()
        assert len(expt.datasets) == len(
            expected_layout
        ), f"Dataset count mismatch: expected {len(expected_layout)}, got {len(expt.datasets)}"

        # Check sessions and recording counts per dataset
        got_by_id = {ds.id: ds for ds in expt.datasets}
        for ds_name, sessions_map in expected_layout.items():
            assert ds_name in got_by_id, f"Missing dataset in domain objects: {ds_name}"
            ds = got_by_id[ds_name]

            # Domain exposes sessions via get_all_sessions
            sessions = ds.get_all_sessions(include_excluded=True)
            assert len(sessions) == len(
                sessions_map
            ), f"Session count mismatch for {ds_name}: expected {len(sessions_map)}, got {len(sessions)}"

            for sess in sessions:
                want_stems = sessions_map.get(sess.id)
                assert want_stems is not None, f"Unexpected session {sess.id} in {ds_name}"

                # 1:1 mapping of recordings to stems
                got_stems = {rec.repo.stem.stem for rec in sess.recordings}
                assert (
                    got_stems == want_stems
                ), f"Recording mismatch in {ds_name}/{sess.id}: expected {want_stems}, got {got_stems}"

    def test_import_performance_baseline(self, temp_output_dir: Path, request):
        """Performance baseline test to catch import regressions."""
        golden_root = _get_golden_root()
        csv_count = _count_csv_files(golden_root)

        def run_import():
            out_expt = temp_output_dir / "PerfTest"
            start_time = time.time()
            import_experiment(golden_root, out_expt, overwrite=True, max_workers=1)
            return time.time() - start_time

        # Try pytest-benchmark if available
        try:
            benchmark = request.getfixturevalue("benchmark")
            benchmark(run_import)
        except Exception:
            # Fallback: manual timing with loose performance check
            elapsed = run_import()
            # Rough check: shouldn't take more than 0.5 seconds per CSV
            max_expected = max(csv_count * 0.5, 10)  # At least 10s budget
            assert elapsed < max_expected, f"Import took {elapsed:.2f}s for {csv_count} CSVs (expected < {max_expected}s)"

    def test_threading_consistency(self, temp_output_dir: Path):
        """Test that multi-threaded import produces consistent results."""
        golden_root = _get_golden_root()

        # Import with single thread
        out_single = temp_output_dir / "SingleThread"
        import_experiment(golden_root, out_single, overwrite=True, max_workers=1)

        # Import with multiple threads
        out_multi = temp_output_dir / "MultiThread"
        import_experiment(golden_root, out_multi, overwrite=True, max_workers=4)

        # Compare results - both should have same structure
        single_files = sorted(out_single.rglob("*.h5"))
        multi_files = sorted(out_multi.rglob("*.h5"))

        assert len(single_files) == len(multi_files), "Thread count affects output file count"

        # Compare relative paths (should be identical)
        single_paths = [f.relative_to(out_single) for f in single_files]
        multi_paths = [f.relative_to(out_multi) for f in multi_files]
        assert single_paths == multi_paths, "Threading affects file organization"

        # Spot check: file sizes should match
        for single_f, multi_f in zip(single_files[:5], multi_files[:5]):  # Check first 5
            assert single_f.stat().st_size == multi_f.stat().st_size, f"Size mismatch: {single_f.name}"
