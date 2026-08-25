from pathlib import Path

import pytest
import yaml

from monstim_signals.core.configuration import ConfigChange, ConfigResolver


def _config():
    return {
        "bin_size": 0.01,
        "time_window": 8.0,
        "pre_stim_time": 2.0,
        "default_method": "rms",
        "butter_filter_args": {"lowcut": 100, "highcut": 3500, "order": 4},
        "m_max_args": {"min_window_size": 2, "max_window_size": 15, "threshold": 0.3},
        "subplot_adjust_args": {"left": 0.1, "right": 0.9},
    }


def test_profile_nested_values_deep_merge_and_yaml_is_cached(tmp_path: Path, monkeypatch):
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(_config()), encoding="utf-8")
    reads = 0
    original = Path.read_text

    def counted(self, *args, **kwargs):
        nonlocal reads
        reads += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counted)
    resolver = ConfigResolver(path)
    resolved = resolver.resolve({"analysis_parameters": {"butter_filter_args": {"order": 6}}})
    assert resolved["butter_filter_args"] == {"lowcut": 100, "highcut": 3500, "order": 6}
    resolver.resolve()
    resolver.resolve()
    assert reads == 1


def test_config_diff_is_dependency_scoped(tmp_path: Path):
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(_config()), encoding="utf-8")
    resolver = ConfigResolver(path)
    base = resolver.resolve()
    signal = resolver.resolve({"analysis_parameters": {"butter_filter_args": {"order": 5}}})
    analysis = resolver.resolve({"analysis_parameters": {"bin_size": 0.02}})
    plot = resolver.resolve({"analysis_parameters": {"time_window": 12.0}})
    assert signal.diff(base) == ConfigChange.SIGNAL
    assert analysis.diff(base) == ConfigChange.ANALYSIS
    assert plot.diff(base) == ConfigChange.PLOT


def test_unknown_profile_key_fails_actionably(tmp_path: Path):
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(_config()), encoding="utf-8")
    with pytest.raises(ValueError, match=r"Unknown analysis profile keys: butter_filter_args\.unknown"):
        ConfigResolver(path).resolve({"analysis_parameters": {"butter_filter_args": {"unknown": 2}}})
