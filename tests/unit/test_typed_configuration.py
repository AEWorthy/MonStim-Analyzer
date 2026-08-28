from pathlib import Path

import pytest
import yaml

from monstim_signals.core.configuration import ConfigChange, ConfigResolver, ResolvedConfig


def _config():
    return {
        "bin_size": 0.01,
        "time_window": 8.0,
        "pre_stim_time": 2.0,
        "default_method": "rms",
        "m_wave_window_names": ["M-wave", "M_response"],
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
    m_wave_config = _config()
    m_wave_config["m_wave_window_names"] = ["Motor response"]
    m_wave_names = ResolvedConfig(m_wave_config)
    plot = resolver.resolve({"analysis_parameters": {"time_window": 12.0}})
    assert signal.diff(base) == ConfigChange.SIGNAL
    assert analysis.diff(base) == ConfigChange.ANALYSIS
    assert m_wave_names.diff(base) == ConfigChange.ANALYSIS
    assert plot.diff(base) == ConfigChange.PLOT


def test_unknown_profile_key_fails_actionably(tmp_path: Path):
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(_config()), encoding="utf-8")
    with pytest.raises(ValueError, match=r"Unknown analysis profile keys: butter_filter_args\.unknown"):
        ConfigResolver(path).resolve({"analysis_parameters": {"butter_filter_args": {"unknown": 2}}})


@pytest.mark.parametrize(
    ("names", "message"),
    [
        ("M-wave", "m_wave_window_names must be a list of strings"),
        (["M-wave", "  "], "m_wave_window_names cannot contain blank names"),
        (["M-wave", "m-WAVE"], "m_wave_window_names cannot contain duplicate names ignoring case"),
    ],
)
def test_m_wave_window_names_validation(tmp_path: Path, names, message: str):
    path = tmp_path / "config.yml"
    config = _config()
    config["m_wave_window_names"] = names
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        ConfigResolver(path).resolve()


def test_m_wave_window_names_are_trimmed_and_may_be_empty(tmp_path: Path):
    path = tmp_path / "config.yml"
    config = _config()
    config["m_wave_window_names"] = ["  Motor response  "]
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    assert ConfigResolver(path).resolve().analysis.m_wave_window_names == ("Motor response",)

    config["m_wave_window_names"] = []
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert ConfigResolver(path).resolve().analysis.m_wave_window_names == ()


def test_m_wave_window_names_cannot_be_overridden_by_a_profile(tmp_path: Path):
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(_config()), encoding="utf-8")

    with pytest.raises(ValueError, match="Global-only analysis profile keys: m_wave_window_names"):
        ConfigResolver(path).resolve({"analysis_parameters": {"m_wave_window_names": ["Motor response"]}})


def test_user_config_replaces_shipped_m_wave_window_names(tmp_path: Path):
    default_path = tmp_path / "config.yml"
    user_path = tmp_path / "config-user.yml"
    default_path.write_text(yaml.safe_dump(_config()), encoding="utf-8")
    user_path.write_text(yaml.safe_dump({"m_wave_window_names": ["Protocol M"]}), encoding="utf-8")

    resolved = ConfigResolver(default_path, user_path).resolve()

    assert resolved.analysis.m_wave_window_names == ("Protocol M",)
