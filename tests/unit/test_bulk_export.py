"""Tests for the bulk data export feature.

Covers:
  - BulkExportConfig construction
  - Pure computation helpers (_compute_avg_reflex_curves, _compute_mmax, _compute_max_h)
  - _write_object_export (file writing with tempdir)
  - run_bulk_export with a mocked ExperimentRepository
  - _sanitize_path_component edge cases
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.unit


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_config(
    data_level="dataset",
    data_types=None,
    methods=None,
    channel_indices=None,
    output_path="",
    selected_objects=None,
    experiment_paths=None,
    normalize_to_mmax=False,
):
    from monstim_gui.managers.bulk_export_manager import BulkExportConfig

    return BulkExportConfig(
        data_level=data_level,
        selected_objects=selected_objects or {"Expt1": ["DS1"]},
        data_types=data_types or ["avg_reflex_curves"],
        methods=methods or ["rms"],
        channel_indices=channel_indices or [0, 1],
        output_path=output_path,
        experiment_paths=experiment_paths or {"Expt1": "/fake/path"},
        normalize_to_mmax=normalize_to_mmax,
    )


class _MockObj:
    """Minimal stand-in for a Dataset or Experiment domain object."""

    def __init__(self, n_channels=2, n_voltages=5):
        self.channel_names = [f"Ch{i}" for i in range(n_channels)]
        self.stimulus_voltages = np.linspace(0.1, 0.5, n_voltages)
        self._n_voltages = n_voltages

    def unique_latency_window_names(self):
        return ["M-wave", "H-reflex"]

    def get_average_lw_reflex_curve(self, method, channel_index, window):
        v = self.stimulus_voltages
        return {
            "voltages": v,
            "means": np.ones(len(v)) * 0.1,
            "stdevs": np.ones(len(v)) * 0.01,
            "n_sessions": np.ones(len(v), dtype=int) * 3,
        }

    def get_avg_m_max(self, method, channel_index, return_avg_mmax_thresholds=False):
        if return_avg_mmax_thresholds:
            return 0.5, 1.2
        return 0.5

    def get_avg_h_wave_amplitudes(self, method, channel_index):
        v = self.stimulus_voltages
        return np.ones(len(v)) * 0.3, np.ones(len(v)) * 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Tests: BulkExportConfig
# ─────────────────────────────────────────────────────────────────────────────


class TestBulkExportConfig:
    def test_construction_dataset_level(self):
        from monstim_gui.managers.bulk_export_manager import BulkExportConfig

        cfg = BulkExportConfig(
            data_level="dataset",
            selected_objects={"Expt1": ["DS_A", "DS_B"]},
            data_types=["avg_reflex_curves", "mmax"],
            methods=["rms", "auc"],
            channel_indices=[0, 1, 2],
            output_path="/out",
            experiment_paths={"Expt1": "/data/Expt1"},
        )

        assert cfg.data_level == "dataset"
        assert cfg.selected_objects == {"Expt1": ["DS_A", "DS_B"]}
        assert "avg_reflex_curves" in cfg.data_types
        assert "rms" in cfg.methods
        assert cfg.channel_indices == [0, 1, 2]
        assert cfg.output_path == "/out"

    def test_construction_experiment_level(self):
        from monstim_gui.managers.bulk_export_manager import BulkExportConfig

        cfg = BulkExportConfig(
            data_level="experiment",
            selected_objects={"Expt1": [], "Expt2": []},
            data_types=["mmax"],
            methods=["rms"],
            channel_indices=[0],
            output_path="/out",
        )
        assert cfg.data_level == "experiment"
        assert cfg.experiment_paths == {}

    def test_default_experiment_paths_is_empty(self):
        from monstim_gui.managers.bulk_export_manager import BulkExportConfig

        cfg = BulkExportConfig(
            data_level="dataset",
            selected_objects={},
            data_types=[],
            methods=[],
            channel_indices=[],
            output_path="",
        )
        assert cfg.experiment_paths == {}


# ─────────────────────────────────────────────────────────────────────────────
# Tests: _sanitize_path_component
# ─────────────────────────────────────────────────────────────────────────────


class TestSanitizePathComponent:
    def test_normal_name(self):
        from monstim_gui.managers.bulk_export_manager import _sanitize_path_component

        assert _sanitize_path_component("Expt1") == "Expt1"

    def test_slash_replaced(self):
        from monstim_gui.managers.bulk_export_manager import _sanitize_path_component

        assert "/" not in _sanitize_path_component("a/b")
        assert "\\" not in _sanitize_path_component("a\\b")

    def test_colon_replaced(self):
        from monstim_gui.managers.bulk_export_manager import _sanitize_path_component

        result = _sanitize_path_component("C:drive")
        assert ":" not in result

    def test_empty_becomes_unnamed(self):
        from monstim_gui.managers.bulk_export_manager import _sanitize_path_component

        assert _sanitize_path_component("") == "unnamed"

    def test_spaces_in_name(self):
        from monstim_gui.managers.bulk_export_manager import _sanitize_path_component

        result = _sanitize_path_component("My Experiment")
        assert len(result) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests: _compute_avg_reflex_curves
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeAvgReflexCurves:
    def test_returns_dataframe(self):
        from monstim_gui.managers.bulk_export_manager import _compute_avg_reflex_curves

        obj = _MockObj(n_channels=2, n_voltages=5)
        config = _make_config(methods=["rms"], channel_indices=[0, 1])
        df = _compute_avg_reflex_curves(obj, config)

        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_expected_columns(self):
        from monstim_gui.managers.bulk_export_manager import _compute_avg_reflex_curves

        obj = _MockObj(n_channels=1, n_voltages=3)
        config = _make_config(methods=["rms", "auc"], channel_indices=[0])
        df = _compute_avg_reflex_curves(obj, config)

        assert "voltage" in df.columns
        assert "channel" in df.columns
        assert "window" in df.columns
        assert "mean_amplitude_rms" in df.columns
        assert "stdev_amplitude_rms" in df.columns
        assert "n_sessions_rms" in df.columns
        assert "mean_amplitude_auc" in df.columns

    def test_row_count(self):
        from monstim_gui.managers.bulk_export_manager import _compute_avg_reflex_curves

        # 2 channels × 2 windows × 5 voltages = 20 rows
        obj = _MockObj(n_channels=2, n_voltages=5)
        config = _make_config(methods=["rms"], channel_indices=[0, 1])
        df = _compute_avg_reflex_curves(obj, config)
        assert len(df) == 2 * 2 * 5

    def test_no_windows_returns_empty(self):
        from monstim_gui.managers.bulk_export_manager import _compute_avg_reflex_curves

        obj = _MockObj()
        obj.unique_latency_window_names = lambda: []
        config = _make_config(methods=["rms"], channel_indices=[0])
        df = _compute_avg_reflex_curves(obj, config)
        assert df.empty

    def test_method_error_handled_gracefully(self):
        from monstim_gui.managers.bulk_export_manager import _compute_avg_reflex_curves

        obj = _MockObj(n_channels=1, n_voltages=3)

        def _bad_curve(*_a, **_kw):
            raise RuntimeError("simulated error")

        obj.get_average_lw_reflex_curve = _bad_curve
        config = _make_config(methods=["rms"], channel_indices=[0])
        # Should not raise; may return empty
        df = _compute_avg_reflex_curves(obj, config)
        assert isinstance(df, pd.DataFrame)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: _compute_mmax
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeMmax:
    def test_returns_dataframe(self):
        from monstim_gui.managers.bulk_export_manager import _compute_mmax

        obj = _MockObj(n_channels=2)
        config = _make_config(methods=["rms"], channel_indices=[0, 1])
        df = _compute_mmax(obj, config)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_expected_columns(self):
        from monstim_gui.managers.bulk_export_manager import _compute_mmax

        obj = _MockObj(n_channels=1)
        config = _make_config(methods=["rms", "auc"], channel_indices=[0])
        df = _compute_mmax(obj, config)

        assert "channel" in df.columns
        assert "channel_index" in df.columns
        assert "mmax_rms" in df.columns
        assert "mmax_threshold_rms" in df.columns
        assert "mmax_auc" in df.columns

    def test_values_correct(self):
        from monstim_gui.managers.bulk_export_manager import _compute_mmax

        obj = _MockObj(n_channels=1)
        config = _make_config(methods=["rms"], channel_indices=[0])
        df = _compute_mmax(obj, config)

        assert float(df["mmax_rms"].iloc[0]) == pytest.approx(0.5)
        assert float(df["mmax_threshold_rms"].iloc[0]) == pytest.approx(1.2)

    def test_channel_out_of_range_skipped(self):
        from monstim_gui.managers.bulk_export_manager import _compute_mmax

        obj = _MockObj(n_channels=1)
        config = _make_config(methods=["rms"], channel_indices=[0, 5])
        df = _compute_mmax(obj, config)
        # channel 5 doesn't exist → only 1 row (channel 0)
        assert len(df) == 1

    def test_error_returns_none_for_method(self):
        from monstim_gui.managers.bulk_export_manager import _compute_mmax

        obj = _MockObj(n_channels=1)

        def _raise(*_a, **_kw):
            raise ValueError("no mmax")

        obj.get_avg_m_max = _raise
        config = _make_config(methods=["rms"], channel_indices=[0])
        df = _compute_mmax(obj, config)
        assert isinstance(df, pd.DataFrame)
        assert df["mmax_rms"].iloc[0] is None


# ─────────────────────────────────────────────────────────────────────────────
# Tests: _compute_max_h
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeMaxH:
    def test_returns_dataframe(self):
        from monstim_gui.managers.bulk_export_manager import _compute_max_h

        obj = _MockObj(n_channels=2, n_voltages=4)
        config = _make_config(methods=["rms"], channel_indices=[0, 1])
        df = _compute_max_h(obj, config)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_expected_columns(self):
        from monstim_gui.managers.bulk_export_manager import _compute_max_h

        obj = _MockObj(n_channels=1, n_voltages=3)
        config = _make_config(methods=["rms", "auc"], channel_indices=[0])
        df = _compute_max_h(obj, config)
        assert "voltage" in df.columns
        assert "channel" in df.columns
        assert "avg_h_amplitude_rms" in df.columns
        assert "std_h_amplitude_rms" in df.columns
        assert "avg_h_amplitude_auc" in df.columns

    def test_row_count(self):
        from monstim_gui.managers.bulk_export_manager import _compute_max_h

        # 2 channels × 5 voltages = 10 rows
        obj = _MockObj(n_channels=2, n_voltages=5)
        config = _make_config(methods=["rms"], channel_indices=[0, 1])
        df = _compute_max_h(obj, config)
        assert len(df) == 2 * 5

    def test_no_voltages_returns_empty(self):
        from monstim_gui.managers.bulk_export_manager import _compute_max_h

        obj = _MockObj(n_voltages=5)
        obj.stimulus_voltages = np.array([])
        config = _make_config(methods=["rms"], channel_indices=[0])
        df = _compute_max_h(obj, config)
        assert df.empty


# ─────────────────────────────────────────────────────────────────────────────
# Tests: _write_object_export (file writing)
# ─────────────────────────────────────────────────────────────────────────────


class TestWriteObjectExport:
    def test_creates_xlsx(self, tmp_path):
        from monstim_gui.managers.bulk_export_manager import _write_object_export

        obj = _MockObj(n_channels=2, n_voltages=4)
        config = _make_config(
            data_types=["avg_reflex_curves", "mmax", "max_h"],
            methods=["rms"],
            channel_indices=[0, 1],
            output_path=str(tmp_path),
        )
        out_file = _write_object_export(obj, "Expt1", "DS1", config)
        assert out_file.exists()
        assert out_file.suffix == ".xlsx"

    def test_sheets_match_data_types(self, tmp_path):
        from monstim_gui.managers.bulk_export_manager import DATA_TYPE_LABELS, _write_object_export

        obj = _MockObj(n_channels=1, n_voltages=3)
        config = _make_config(
            data_types=["mmax", "max_h"],
            methods=["rms"],
            channel_indices=[0],
            output_path=str(tmp_path),
        )
        out_file = _write_object_export(obj, "E1", "D1", config)
        wb = pd.ExcelFile(out_file)
        assert DATA_TYPE_LABELS["mmax"] in wb.sheet_names
        assert DATA_TYPE_LABELS["max_h"] in wb.sheet_names

    def test_experiment_subfolder_created(self, tmp_path):
        from monstim_gui.managers.bulk_export_manager import _write_object_export

        obj = _MockObj(n_channels=1, n_voltages=2)
        out_root = tmp_path / "exports"
        config = _make_config(
            data_types=["mmax"],
            methods=["rms"],
            channel_indices=[0],
            output_path=str(out_root),
        )
        out_file = _write_object_export(obj, "MyExperiment", "DS_A", config)
        assert (out_root / "MyExperiment").is_dir()
        assert out_file.parent.name == "MyExperiment"

    def test_empty_output_file_removed(self, tmp_path):
        """If all data types produce empty DataFrames, the output file should be removed."""
        from monstim_gui.managers.bulk_export_manager import _write_object_export

        obj = _MockObj(n_channels=1, n_voltages=3)
        # Force avg_reflex_curves to return no windows → empty DF
        obj.unique_latency_window_names = lambda: []
        obj.stimulus_voltages = np.array([])  # empty voltages → empty max_h and mmax skipped

        config = _make_config(
            data_types=["avg_reflex_curves"],
            methods=["rms"],
            channel_indices=[0],
            output_path=str(tmp_path),
        )
        out_file = _write_object_export(obj, "E", "D", config)
        # File should have been removed because no sheets were written
        assert not out_file.exists()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: run_bulk_export (integration with mocked ExperimentRepository)
# ─────────────────────────────────────────────────────────────────────────────


class TestRunBulkExport:
    def test_dataset_level_writes_files(self, tmp_path, monkeypatch):
        from monstim_gui.managers import bulk_export_manager

        ds1 = _MockObj(n_channels=2, n_voltages=4)
        ds1.id = "DS1"

        class FakeDatasetRepo:
            def __init__(self, folder):
                pass

            def load(self, **kwargs):
                return ds1

        # Dataset-level now loads DatasetRepository directly (not ExperimentRepository)
        import monstim_signals.io.repositories as repos_mod

        monkeypatch.setattr(repos_mod, "DatasetRepository", FakeDatasetRepo)

        # Create a fake DS1 sub-folder so the folder existence check passes
        (tmp_path / "DS1").mkdir()

        config = _make_config(
            data_level="dataset",
            selected_objects={"Expt1": ["DS1"]},
            data_types=["mmax"],
            methods=["rms"],
            channel_indices=[0, 1],
            output_path=str(tmp_path),
            experiment_paths={"Expt1": str(tmp_path)},
        )

        written = bulk_export_manager.run_bulk_export(config)
        assert len(written) == 1
        assert "DS1_bulk_export.xlsx" in written[0]

    def test_experiment_level_writes_files(self, tmp_path, monkeypatch):
        import monstim_signals.io.repositories as repos_mod
        from monstim_gui.managers import bulk_export_manager

        mock_experiment = _MockObj(n_channels=2, n_voltages=4)
        mock_experiment.id = "Expt1"

        class FakeRepo:
            def __init__(self, folder):
                pass

            def load(self, **kwargs):
                return mock_experiment

        monkeypatch.setattr(repos_mod, "ExperimentRepository", FakeRepo)

        config = _make_config(
            data_level="experiment",
            selected_objects={"Expt1": []},
            data_types=["mmax"],
            methods=["rms"],
            channel_indices=[0],
            output_path=str(tmp_path),
            experiment_paths={"Expt1": str(tmp_path)},
        )

        written = bulk_export_manager.run_bulk_export(config)
        assert len(written) == 1
        assert "Expt1_bulk_export.xlsx" in written[0]

    def test_cancellation(self, tmp_path, monkeypatch):
        import monstim_signals.io.repositories as repos_mod
        from monstim_gui.managers import bulk_export_manager

        mock_experiment = _MockObj(n_channels=1, n_voltages=3)
        mock_experiment.datasets = []

        class FakeRepo:
            def __init__(self, folder):
                pass

            def load(self, **kwargs):
                return mock_experiment

        monkeypatch.setattr(repos_mod, "ExperimentRepository", FakeRepo)

        config = _make_config(
            data_level="experiment",
            selected_objects={"E1": [], "E2": [], "E3": []},
            data_types=["mmax"],
            methods=["rms"],
            channel_indices=[0],
            output_path=str(tmp_path),
            experiment_paths={"E1": str(tmp_path), "E2": str(tmp_path), "E3": str(tmp_path)},
        )

        # Cancel immediately
        written = bulk_export_manager.run_bulk_export(config, is_canceled=lambda: True)
        assert written == []

    def test_missing_experiment_path_skipped(self, tmp_path):
        from monstim_gui.managers import bulk_export_manager

        config = _make_config(
            data_level="experiment",
            selected_objects={"GhostExpt": []},
            data_types=["mmax"],
            methods=["rms"],
            channel_indices=[0],
            output_path=str(tmp_path),
            experiment_paths={},  # no path for GhostExpt
        )
        written = bulk_export_manager.run_bulk_export(config)
        assert written == []

    def test_progress_callback_called(self, tmp_path, monkeypatch):
        import monstim_signals.io.repositories as repos_mod
        from monstim_gui.managers import bulk_export_manager

        mock_experiment = _MockObj(n_channels=1, n_voltages=2)
        mock_experiment.datasets = []

        class FakeRepo:
            def __init__(self, folder):
                pass

            def load(self, **kwargs):
                return mock_experiment

        monkeypatch.setattr(repos_mod, "ExperimentRepository", FakeRepo)

        calls = []
        config = _make_config(
            data_level="experiment",
            selected_objects={"E1": []},
            data_types=["mmax"],
            methods=["rms"],
            channel_indices=[0],
            output_path=str(tmp_path),
            experiment_paths={"E1": str(tmp_path)},
        )
        bulk_export_manager.run_bulk_export(config, progress_callback=lambda c, t, m: calls.append((c, t, m)))
        assert len(calls) >= 1
        assert calls[-1][0] == 1
        assert calls[-1][1] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Tests: normalize_to_mmax flag
# ─────────────────────────────────────────────────────────────────────────────


class TestNormalizeToMmax:
    """Tests for the normalize_to_mmax plot option."""

    def test_flag_defaults_false(self):
        config = _make_config()
        assert config.normalize_to_mmax is False

    def test_flag_set_true(self):
        config = _make_config(normalize_to_mmax=True)
        assert config.normalize_to_mmax is True

    # -- avg_reflex_curves --

    def test_avg_reflex_no_norm_columns_when_false(self):
        from monstim_gui.managers.bulk_export_manager import _compute_avg_reflex_curves

        obj = _MockObj(n_channels=1, n_voltages=3)
        config = _make_config(
            data_types=["avg_reflex_curves"],
            methods=["rms"],
            channel_indices=[0],
            normalize_to_mmax=False,
        )
        df = _compute_avg_reflex_curves(obj, config)
        norm_cols = [c for c in df.columns if "_norm_mmax_" in c]
        assert norm_cols == [], f"Unexpected norm columns: {norm_cols}"

    def test_avg_reflex_adds_norm_columns_when_true(self):
        from monstim_gui.managers.bulk_export_manager import _compute_avg_reflex_curves

        obj = _MockObj(n_channels=1, n_voltages=3)
        config = _make_config(
            data_types=["avg_reflex_curves"],
            methods=["rms"],
            channel_indices=[0],
            normalize_to_mmax=True,
        )
        df = _compute_avg_reflex_curves(obj, config)
        assert "mean_amplitude_norm_mmax_rms" in df.columns
        assert "stdev_amplitude_norm_mmax_rms" in df.columns

    def test_avg_reflex_norm_values_correct(self):
        from monstim_gui.managers.bulk_export_manager import _compute_avg_reflex_curves

        # Mock returns means=0.1, mmax=0.5 → normalised = 0.2
        obj = _MockObj(n_channels=1, n_voltages=3)
        config = _make_config(
            data_types=["avg_reflex_curves"],
            methods=["rms"],
            channel_indices=[0],
            normalize_to_mmax=True,
        )
        df = _compute_avg_reflex_curves(obj, config)
        # Only rows for channel 0 and window "M-wave"/"H-reflex"
        norm_means = df["mean_amplitude_norm_mmax_rms"].dropna()
        expected = 0.1 / 0.5  # raw mean / M-max
        np.testing.assert_allclose(norm_means.values, expected, rtol=1e-6)

    def test_avg_reflex_multiple_methods_norm_columns(self):
        from monstim_gui.managers.bulk_export_manager import _compute_avg_reflex_curves

        obj = _MockObj(n_channels=1, n_voltages=3)
        config = _make_config(
            data_types=["avg_reflex_curves"],
            methods=["rms", "average_rectified"],
            channel_indices=[0],
            normalize_to_mmax=True,
        )
        df = _compute_avg_reflex_curves(obj, config)
        assert "mean_amplitude_norm_mmax_rms" in df.columns
        assert "mean_amplitude_norm_mmax_average_rectified" in df.columns

    # -- max_h --

    def test_max_h_no_norm_columns_when_false(self):
        from monstim_gui.managers.bulk_export_manager import _compute_max_h

        obj = _MockObj(n_channels=1, n_voltages=3)
        config = _make_config(
            data_types=["max_h"],
            methods=["rms"],
            channel_indices=[0],
            normalize_to_mmax=False,
        )
        df = _compute_max_h(obj, config)
        norm_cols = [c for c in df.columns if "_norm_mmax_" in c]
        assert norm_cols == [], f"Unexpected norm columns: {norm_cols}"

    def test_max_h_adds_norm_columns_when_true(self):
        from monstim_gui.managers.bulk_export_manager import _compute_max_h

        obj = _MockObj(n_channels=1, n_voltages=3)
        config = _make_config(
            data_types=["max_h"],
            methods=["rms"],
            channel_indices=[0],
            normalize_to_mmax=True,
        )
        df = _compute_max_h(obj, config)
        assert "avg_h_amplitude_norm_mmax_rms" in df.columns
        assert "std_h_amplitude_norm_mmax_rms" in df.columns

    def test_max_h_norm_values_correct(self):
        from monstim_gui.managers.bulk_export_manager import _compute_max_h

        # Mock returns avg_h=0.3, mmax=0.5 → normalised = 0.6
        obj = _MockObj(n_channels=1, n_voltages=3)
        config = _make_config(
            data_types=["max_h"],
            methods=["rms"],
            channel_indices=[0],
            normalize_to_mmax=True,
        )
        df = _compute_max_h(obj, config)
        norm_avgs = df["avg_h_amplitude_norm_mmax_rms"].dropna()
        expected = 0.3 / 0.5
        np.testing.assert_allclose(norm_avgs.values, expected, rtol=1e-6)

    # -- mmax sheet is unchanged --

    def test_mmax_sheet_has_no_norm_columns(self):
        from monstim_gui.managers.bulk_export_manager import _compute_mmax

        obj = _MockObj(n_channels=1, n_voltages=3)
        config = _make_config(
            data_types=["mmax"],
            methods=["rms"],
            channel_indices=[0],
            normalize_to_mmax=True,
        )
        df = _compute_mmax(obj, config)
        norm_cols = [c for c in df.columns if "_norm_mmax_" in c]
        assert norm_cols == [], "M-max summary sheet must not contain _norm_mmax_ columns"

    # -- graceful handling of missing / zero M-max --

    def test_avg_reflex_zero_mmax_no_norm_columns(self):
        from monstim_gui.managers.bulk_export_manager import _compute_avg_reflex_curves

        obj = _MockObj(n_channels=1, n_voltages=3)
        # Override get_avg_m_max to return 0 so normalization is skipped
        obj.get_avg_m_max = lambda method, ch, return_avg_mmax_thresholds=False: (
            (0.0, 1.2) if return_avg_mmax_thresholds else 0.0
        )
        config = _make_config(
            data_types=["avg_reflex_curves"],
            methods=["rms"],
            channel_indices=[0],
            normalize_to_mmax=True,
        )
        df = _compute_avg_reflex_curves(obj, config)
        assert "mean_amplitude_norm_mmax_rms" not in df.columns

    def test_avg_reflex_none_mmax_no_norm_columns(self):
        from monstim_gui.managers.bulk_export_manager import _compute_avg_reflex_curves

        obj = _MockObj(n_channels=1, n_voltages=3)
        obj.get_avg_m_max = lambda method, ch, return_avg_mmax_thresholds=False: (
            (None, None) if return_avg_mmax_thresholds else None
        )
        config = _make_config(
            data_types=["avg_reflex_curves"],
            methods=["rms"],
            channel_indices=[0],
            normalize_to_mmax=True,
        )
        df = _compute_avg_reflex_curves(obj, config)
        assert "mean_amplitude_norm_mmax_rms" not in df.columns

    def test_max_h_zero_mmax_no_norm_columns(self):
        from monstim_gui.managers.bulk_export_manager import _compute_max_h

        obj = _MockObj(n_channels=1, n_voltages=3)
        obj.get_avg_m_max = lambda method, ch, return_avg_mmax_thresholds=False: (
            (0.0, 1.2) if return_avg_mmax_thresholds else 0.0
        )
        config = _make_config(
            data_types=["max_h"],
            methods=["rms"],
            channel_indices=[0],
            normalize_to_mmax=True,
        )
        df = _compute_max_h(obj, config)
        assert "avg_h_amplitude_norm_mmax_rms" not in df.columns

    # -- write produces normalized columns in xlsx --

    def test_write_object_export_with_norm_mmax(self, tmp_path):
        from monstim_gui.managers.bulk_export_manager import _write_object_export

        obj = _MockObj(n_channels=1, n_voltages=3)
        config = _make_config(
            data_types=["avg_reflex_curves", "max_h"],
            methods=["rms"],
            channel_indices=[0],
            output_path=str(tmp_path),
            normalize_to_mmax=True,
        )
        # _write_object_export(obj, expt_name, obj_id, config)
        # It writes to: output_path / expt_name / obj_id.xlsx
        out_file = _write_object_export(obj, "Expt1", "DS1", config)
        assert out_file is not None
        assert out_file.exists()
        import openpyxl

        wb = openpyxl.load_workbook(str(out_file))
        ws_arc = wb["Avg Reflex Curves"]
        headers = [cell.value for cell in ws_arc[1]]
        assert "mean_amplitude_norm_mmax_rms" in headers
        assert "stdev_amplitude_norm_mmax_rms" in headers
        ws_mh = wb["Max H-Reflex"]
        headers_mh = [cell.value for cell in ws_mh[1]]
        assert "avg_h_amplitude_norm_mmax_rms" in headers_mh
        assert "std_h_amplitude_norm_mmax_rms" in headers_mh
