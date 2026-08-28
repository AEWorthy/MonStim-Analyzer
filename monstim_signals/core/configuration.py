"""Typed, immutable analysis configuration and dependency-aware resolution."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Flag, auto
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Any

import yaml

CALCULATION_METHODS = frozenset({"peak_to_trough", "extrema_ptt", "exclusive_extrema_ptt", "rms", "average_rectified", "average_unrectified", "auc"})
GLOBAL_ONLY_PROFILE_KEYS = frozenset({"m_wave_window_names"})


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _fingerprint(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _fingerprint(item)) for key, item in value.items()))
    if isinstance(value, list | tuple):
        return tuple(_fingerprint(item) for item in value)
    return value


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recursive merge without mutating either input."""
    result = {key: _thaw(value) for key, value in base.items()}
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = _thaw(value)
    return result


def _unknown_paths(reference: Mapping[str, Any], candidate: Mapping[str, Any], prefix: str = "") -> list[str]:
    unknown: list[str] = []
    for key, value in candidate.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if key not in reference:
            unknown.append(path)
        elif isinstance(value, Mapping) and isinstance(reference[key], Mapping):
            unknown.extend(_unknown_paths(reference[key], value, path))
    return unknown


class ConfigChange(Flag):
    NONE = 0
    SIGNAL = auto()
    ANALYSIS = auto()
    PLOT = auto()
    PRESETS = auto()


@dataclass(frozen=True)
class SignalProcessingConfig:
    butter_filter_args: Mapping[str, Any]
    close_raw_after_filter: bool = True
    signal_processing_workers: int | None = None

    @property
    def fingerprint(self) -> tuple[Any, ...]:
        # Worker count and handle-closing policy affect execution, not values.
        return (_fingerprint(self.butter_filter_args),)


@dataclass(frozen=True)
class AnalysisConfig:
    bin_size: float
    default_method: str
    m_max_args: Mapping[str, Any]
    m_wave_window_names: tuple[str, ...]

    @property
    def fingerprint(self) -> tuple[Any, ...]:
        return (self.bin_size, self.default_method, _fingerprint(self.m_max_args), self.m_wave_window_names)


@dataclass(frozen=True)
class PlotConfig:
    time_window: float
    pre_stim_time: float
    values: Mapping[str, Any]

    @property
    def fingerprint(self) -> Any:
        return _fingerprint(self.values)

    @property
    def construction_fingerprint(self) -> Any:
        plotting = self.values.get("plotting", {})
        keys = {"enable_decimation", "max_points_per_curve", "decimation_strategy", "min_decimation_factor"}
        return _fingerprint({key: plotting[key] for key in keys if key in plotting})


@dataclass(frozen=True)
class PresetConfig:
    latency_window_presets: Mapping[str, Any]
    default_channel_names: tuple[str, ...]
    active_latency_window_preset: str | None = None

    @property
    def fingerprint(self) -> tuple[Any, ...]:
        return (_fingerprint(self.latency_window_presets), self.default_channel_names, self.active_latency_window_preset)


class ResolvedConfig(Mapping[str, Any]):
    """Immutable mapping plus typed sections used for precise invalidation."""

    def __init__(self, values: Mapping[str, Any]):
        validated = self._validate(values)
        self._values = _freeze(validated)
        self.signal = SignalProcessingConfig(
            butter_filter_args=self._values["butter_filter_args"],
            close_raw_after_filter=bool(self._values.get("close_raw_after_filter", True)),
            signal_processing_workers=self._values.get("signal_processing_workers"),
        )
        self.analysis = AnalysisConfig(
            bin_size=float(self._values["bin_size"]),
            default_method=str(self._values["default_method"]),
            m_max_args=self._values["m_max_args"],
            m_wave_window_names=tuple(self._values["m_wave_window_names"]),
        )
        plot_keys = {
            "time_window",
            "pre_stim_time",
            "title_font_size",
            "axis_label_font_size",
            "tick_font_size",
            "m_color",
            "h_color",
            "latency_window_style",
            "subplot_adjust_args",
            "plotting",
        }
        self.plot = PlotConfig(
            time_window=float(self._values["time_window"]),
            pre_stim_time=float(self._values["pre_stim_time"]),
            values=MappingProxyType({key: self._values[key] for key in plot_keys if key in self._values}),
        )
        self.presets = PresetConfig(
            latency_window_presets=self._values.get("latency_window_presets", MappingProxyType({})),
            default_channel_names=tuple(self._values.get("default_channel_names", ())),
            active_latency_window_preset=self._values.get("latency_window_preset"),
        )

    @staticmethod
    def _validate(values: Mapping[str, Any]) -> dict[str, Any]:
        result = {key: _thaw(value) for key, value in values.items()}
        if isinstance(result.get("default_channel_names"), str):
            result["default_channel_names"] = [name.strip() for name in result["default_channel_names"].split(",") if name.strip()]
        required = {"bin_size", "time_window", "pre_stim_time", "default_method", "butter_filter_args", "m_max_args", "m_wave_window_names"}
        missing = sorted(required - result.keys())
        if missing:
            raise ValueError(f"Missing required configuration keys: {', '.join(missing)}")
        if float(result["bin_size"]) <= 0:
            raise ValueError("bin_size must be greater than zero")
        if float(result["time_window"]) < 0 or float(result["pre_stim_time"]) < 0:
            raise ValueError("time_window and pre_stim_time must be non-negative")
        method = str(result["default_method"])
        if method not in CALCULATION_METHODS:
            raise ValueError(f"Invalid default_method '{method}'")
        result["default_method"] = method
        m_wave_names = result["m_wave_window_names"]
        if not isinstance(m_wave_names, list) or not all(isinstance(name, str) for name in m_wave_names):
            raise ValueError("m_wave_window_names must be a list of strings")
        normalized_m_wave_names = [name.strip() for name in m_wave_names]
        if any(not name for name in normalized_m_wave_names):
            raise ValueError("m_wave_window_names cannot contain blank names")
        if len({name.casefold() for name in normalized_m_wave_names}) != len(normalized_m_wave_names):
            raise ValueError("m_wave_window_names cannot contain duplicate names ignoring case")
        result["m_wave_window_names"] = normalized_m_wave_names
        for key in ("bin_size", "time_window", "pre_stim_time"):
            if key in result:
                result[key] = float(result[key])
        if not isinstance(result["butter_filter_args"], Mapping) or not isinstance(result["m_max_args"], Mapping):
            raise ValueError("butter_filter_args and m_max_args must be mappings")
        filter_args = result["butter_filter_args"]
        required_filter = {"lowcut", "highcut", "order"}
        if required_filter - filter_args.keys():
            raise ValueError("butter_filter_args requires lowcut, highcut, and order")
        if float(filter_args["lowcut"]) < 0 or float(filter_args["highcut"]) <= float(filter_args["lowcut"]):
            raise ValueError("butter_filter_args requires 0 <= lowcut < highcut")
        if int(filter_args["order"]) < 1:
            raise ValueError("butter_filter_args.order must be at least 1")
        for key in ("lowcut", "highcut"):
            if isinstance(filter_args[key], str):
                number = float(filter_args[key])
                filter_args[key] = int(number) if number.is_integer() else number
        filter_args["order"] = int(filter_args["order"])
        mmax = result["m_max_args"]
        if int(mmax.get("min_window_size", 1)) < 1 or int(mmax.get("max_window_size", 1)) < int(mmax.get("min_window_size", 1)):
            raise ValueError("m_max_args requires 1 <= min_window_size <= max_window_size")
        if "savgol_window_ratio" in mmax and not 0 < float(mmax["savgol_window_ratio"]) <= 1:
            raise ValueError("m_max_args.savgol_window_ratio must be in (0, 1]")
        for key in ("max_window_size", "min_window_size", "savgol_window_length"):
            if key in mmax and mmax[key] is not None:
                mmax[key] = int(mmax[key])
        for key in ("threshold", "validation_tolerance", "savgol_window_ratio"):
            if key in mmax:
                mmax[key] = float(mmax[key])
        for key in ("title_font_size", "axis_label_font_size", "tick_font_size", "signal_processing_workers"):
            if key in result and result[key] is not None:
                result[key] = int(result[key])
        presets = result.get("latency_window_presets", {})
        if not isinstance(presets, Mapping):
            raise ValueError("latency_window_presets must be a mapping")
        for name, windows in presets.items():
            if not isinstance(windows, list):
                raise ValueError(f"latency_window_presets.{name} must be a list")
            for index, window in enumerate(windows):
                if not isinstance(window, Mapping) or not {"name", "start", "duration"} <= window.keys():
                    raise ValueError(f"latency_window_presets.{name}[{index}] requires name, start, and duration")
                if float(window["duration"]) <= 0:
                    raise ValueError(f"latency_window_presets.{name}[{index}].duration must be positive")
                window["name"] = str(window["name"])
                window["start"] = float(window["start"])
                window["duration"] = float(window["duration"])
        channel_names = result.get("default_channel_names", [])
        if not isinstance(channel_names, list) or not all(isinstance(name, str) and name for name in channel_names):
            raise ValueError("default_channel_names must be a list of non-empty strings")
        if "plotting" in result and not isinstance(result["plotting"], Mapping):
            raise ValueError("plotting must be a mapping")
        return result

    def diff(self, previous: ResolvedConfig | Mapping[str, Any] | None) -> ConfigChange:
        if previous is None:
            return ConfigChange.SIGNAL | ConfigChange.ANALYSIS | ConfigChange.PLOT | ConfigChange.PRESETS
        other = previous if isinstance(previous, ResolvedConfig) else ResolvedConfig(previous)
        changed = ConfigChange.NONE
        if self.signal.fingerprint != other.signal.fingerprint:
            changed |= ConfigChange.SIGNAL
        if self.analysis.fingerprint != other.analysis.fingerprint:
            changed |= ConfigChange.ANALYSIS
        if self.plot.fingerprint != other.plot.fingerprint:
            changed |= ConfigChange.PLOT
        if self.presets.fingerprint != other.presets.fingerprint:
            changed |= ConfigChange.PRESETS
        return changed

    def to_dict(self) -> dict[str, Any]:
        return _thaw(self._values)

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


class ConfigResolver:
    """Load base YAML once and resolve profile overlays without selection-time I/O."""

    def __init__(self, default_path: str | Path, user_path: str | Path | None = None):
        self.default_path = Path(default_path)
        self.user_path = Path(user_path) if user_path is not None else self.default_path.with_name("config-user.yml")
        self._base: dict[str, Any] | None = None
        self._lock = RLock()

    def invalidate(self) -> None:
        with self._lock:
            self._base = None

    def _load_base(self) -> dict[str, Any]:
        with self._lock:
            if self._base is None:
                defaults = yaml.safe_load(self.default_path.read_text(encoding="utf-8")) or {}
                user = (yaml.safe_load(self.user_path.read_text(encoding="utf-8")) or {}) if self.user_path.exists() else {}
                self._base = deep_merge(defaults, user)
            return self._base

    def load_raw(self) -> dict[str, Any]:
        """Return a detached copy of the merged YAML configuration.

        The GUI configuration repository historically exposed arbitrary
        application/UI keys through ``read_config``.  Keep that storage-level
        API independent from domain validation while typed consumers use
        :meth:`resolve` at their boundary.
        """
        return deep_merge(self._load_base(), {})

    def resolve(self, profile: Mapping[str, Any] | None = None) -> ResolvedConfig:
        overlay: Mapping[str, Any] = {}
        if profile:
            overlay = profile.get("analysis_parameters", {})
            global_only = sorted(GLOBAL_ONLY_PROFILE_KEYS & overlay.keys())
            if global_only:
                raise ValueError(f"Global-only analysis profile keys: {', '.join(global_only)}")
            unknown = sorted(_unknown_paths(self._load_base(), overlay))
            if unknown:
                raise ValueError(f"Unknown analysis profile keys: {', '.join(unknown)}")
            extra = {key: profile[key] for key in ("latency_window_preset",) if key in profile}
            overlay = deep_merge(overlay, extra)
        return ResolvedConfig(deep_merge(self._load_base(), overlay))
