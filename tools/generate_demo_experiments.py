"""Build the bundled, entirely synthetic MonStim stretch and vibration demos.

The archive is deliberately small and deterministic.  It contains native MonStim
experiment folders, not an export of laboratory recordings, so it is safe to
ship with the application and useful for exploring force/length and bulk-EMG
plotting without exposing research data.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np

from monstim_signals.core import DatasetAnnot, ExperimentAnnot, LatencyWindow, RecordingAnnot, SessionAnnot, SignalChannel, StimCluster
from monstim_signals.io.experiment_catalog import build_catalog
from monstim_signals.io.repositories import (
    DatasetRepository,
    ExperimentRepository,
    RecordingRepository,
    SessionRepository,
)

SCAN_RATE_HZ = 8_000
PRE_STIM_MS = 250
POST_STIM_MS = 1_000
INTENSITIES = (0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00)
SESSION_IDS = ("Trial 1", "Trial 2", "Trial 3")
CONDITIONS = (
    ("Condition A - Reduced response", 0.70),
    ("Condition B - Reference response", 1.00),
    ("Condition C - Enhanced response", 1.30),
)
DEMO_ARCHIVE = Path("docs/resources/demo_experiments/monstim-synthetic-protocol-demos.zip")


def _stimulus(*, kind: str, intensity: float, duration_ms: float, ramp_ms: float) -> StimCluster:
    return StimCluster(
        stim_delay=0.0,
        stim_duration=duration_ms,
        stim_type=kind,
        stim_v=intensity,
        stim_min_v=intensity,
        stim_max_v=intensity,
        pulse_shape="Ramp-Hold-Release" if kind == "Motor Length" else ("Sine" if kind == "Vibration" else "Square"),
        num_pulses=round(duration_ms / 10) if kind == "Vibration" else 1,
        pulse_period=10.0 if kind == "Vibration" else duration_ms,
        peak_duration=max(0.0, duration_ms - 2 * ramp_ms),
        ramp_duration=ramp_ms,
    )


def _burst(time_ms: np.ndarray, start_ms: float, end_ms: float, amplitude: float, frequency_hz: float) -> np.ndarray:
    """Smooth, deterministic rectifiable-like EMG burst without a step edge."""
    active = (time_ms >= start_ms) & (time_ms <= end_ms)
    phase = 2 * np.pi * frequency_hz * (time_ms - start_ms) / 1_000
    envelope = np.zeros_like(time_ms)
    envelope[active] = np.sin(np.pi * (time_ms[active] - start_ms) / max(end_ms - start_ms, 1)) ** 0.6
    return amplitude * envelope * (0.72 * np.sin(phase) + 0.28 * np.sin(2.1 * phase + 0.7))


def _vibration_recording(time_ms: np.ndarray, intensity: float, rng: np.random.Generator, response_scale: float = 1.0) -> np.ndarray:
    """Fictional TA/LG bulk EMG plus force and length during 100 Hz vibration."""
    data = rng.normal(0, 0.012, (len(time_ms), 4))
    duration = 500.0
    carrier = np.sin(2 * np.pi * 100 * time_ms / 1_000)
    gate = ((time_ms >= 0) & (time_ms <= duration)).astype(float)
    taper = np.minimum(np.minimum(np.maximum(time_ms, 0) / 25, np.maximum(duration - time_ms, 0) / 25), 1)
    taper = np.clip(taper, 0, 1) * gate
    # Fictional vibration example: LG has the larger response than TA.
    for channel, (scale, phase) in enumerate(((0.62, 0.0), (1.0, 0.8))):
        response = (0.040 + 0.29 * intensity**1.25) * scale * response_scale
        data[:, channel] += taper * response * (carrier + 0.22 * np.sin(2 * np.pi * 205 * time_ms / 1_000 + phase))
        data[:, channel] += _burst(time_ms, 510, 650, 0.050 * intensity * scale, 145 + channel * 18)
    # A small, tapered 100 Hz displacement and force response make the
    # mechanical channels available for the same bulk-EMG inspection view.
    displacement = taper * intensity * (0.15 + 0.10 * carrier)
    force = taper * intensity * (0.22 + 0.08 * np.sin(2 * np.pi * 100 * time_ms / 1_000 + 0.35))
    data[:, 2] = force + rng.normal(0, 0.003, len(time_ms))
    data[:, 3] = displacement + rng.normal(0, 0.002, len(time_ms))
    return data.astype(np.float32)


def _stretch_recording(time_ms: np.ndarray, intensity: float, rng: np.random.Generator, response_scale: float = 1.0) -> np.ndarray:
    """TA/LG plus fictional force and length for a ramp-hold-release stimulus."""
    data = rng.normal(0, 0.010, (len(time_ms), 4))
    ramp, hold_end, release_end = 150.0, 550.0, 700.0
    length = np.zeros_like(time_ms)
    rising = (time_ms >= 0) & (time_ms < ramp)
    holding = (time_ms >= ramp) & (time_ms < hold_end)
    falling = (time_ms >= hold_end) & (time_ms <= release_end)
    length[rising] = intensity * (time_ms[rising] / ramp)
    length[holding] = intensity
    length[falling] = intensity * (1 - (time_ms[falling] - hold_end) / (release_end - hold_end))
    force = 0.65 * length + 0.08 * np.gradient(length) * SCAN_RATE_HZ / 1_000
    # Fictional WT pattern: LG is the strongly stretch-sensitive muscle. TA
    # has a small, sparse onset/offset response rather than a sustained burst.
    data[:, 0] += _burst(time_ms, 45, 82, (0.035 + 0.055 * intensity) * response_scale, 155)
    data[:, 0] += _burst(time_ms, 570, 615, (0.025 + 0.040 * intensity) * response_scale, 165)
    data[:, 1] += _burst(time_ms, 20, 190, (0.12 + 0.46 * intensity) * response_scale, 145)
    data[:, 1] += _burst(time_ms, 205, 535, (0.025 + 0.12 * intensity) * response_scale, 112)
    data[:, 1] += _burst(time_ms, 545, 735, (0.10 + 0.36 * intensity) * response_scale, 155)
    data[:, 2] += force
    data[:, 3] += length
    return data.astype(np.float32)


def _h_reflex_recording(time_ms: np.ndarray, intensity: float, rng: np.random.Generator, h_reflex_scale: float = 1.0) -> np.ndarray:
    """Two-channel fictional M/H recruitment recording.

    Demo conditions model a change in the H-reflex pathway only.  The direct
    M-response is therefore held constant between conditions.
    """
    data = rng.normal(0, 0.012, (len(time_ms), 2))
    m_amplitude = 1.15 / (1 + np.exp(-1.15 * (intensity - 2.4)))
    h_amplitude = h_reflex_scale * 0.50 * np.exp(-((intensity - 4.2) ** 2) / (2 * 1.35**2))
    for channel, scale in enumerate((1.0, 0.72)):
        data[:, channel] += _burst(time_ms, 4.0, 9.5, m_amplitude * scale, 240)
        data[:, channel] += _burst(time_ms, 24.0, 32.0, h_amplitude * scale, 115)
    return data.astype(np.float32)


def _write_h_reflex_experiment(root: Path) -> None:
    """Write the third native demo using the same layout as the other protocols."""
    experiment = root / "Synthetic H-reflex Recruitment"
    experiment.mkdir(parents=True)
    ExperimentRepository(experiment).save_annotation(ExperimentAnnot.create_empty())
    scan_rate, pre_stim_ms, post_stim_ms = 8_000, 10, 40
    time_ms = np.arange(int((pre_stim_ms + post_stim_ms) * scan_rate / 1_000)) * 1_000 / scan_rate - pre_stim_ms
    channels = [SignalChannel(name="TA", unit="mV", type_override="EMG"), SignalChannel(name="LG", unit="mV", type_override="EMG")]
    intensities = np.linspace(0.5, 9.0, 18)
    for condition_index, (condition_name, response_scale) in enumerate(CONDITIONS, start=1):
        dataset = experiment / condition_name
        dataset.mkdir()
        annotation = DatasetAnnot.create_empty()
        annotation.date, annotation.animal_id, annotation.condition = "260101", f"SYN-HREFLEX-C{condition_index}", condition_name
        DatasetRepository(dataset).save_annotation(annotation, refresh_catalog=False)
        for session_index, session_id in enumerate(SESSION_IDS):
            session = dataset / session_id
            session.mkdir()
            session_annotation = SessionAnnot.create_empty(len(channels))
            session_annotation.channels = channels
            session_annotation.latency_windows = [
                LatencyWindow(name="M-wave", color="#dc2626", start_times=[3.5, 3.5], durations=[5.0, 5.0]),
                LatencyWindow(name="H-reflex", color="#2563eb", start_times=[23.0, 23.0], durations=[8.0, 8.0]),
            ]
            SessionRepository(session).save_annotation(session_annotation, refresh_catalog=False)
            for index, nominal_intensity in enumerate(intensities):
                rng = np.random.default_rng(80_000 + condition_index * 10_000 + session_index * 100 + index)
                intensity = float(np.clip(nominal_intensity * rng.normal(1.0, 0.025), 0.1, 9.0))
                stem = session / f"SYNHC{condition_index}T{session_index + 1}-{index:04d}"
                raw = _h_reflex_recording(time_ms, intensity, rng, response_scale)
                with h5py.File(stem.with_suffix(".raw.h5"), "w") as h5:
                    h5.create_dataset("raw", data=raw, compression="gzip", compression_opts=4)
                stim = _stimulus(kind="Electrical", intensity=intensity, duration_ms=0.2, ramp_ms=0.0)
                meta = {
                    "recording_id": stem.name,
                    "num_channels": 2,
                    "scan_rate": scan_rate,
                    "pre_stim_acquired": pre_stim_ms,
                    "post_stim_acquired": post_stim_ms,
                    "recording_interval": 3.0,
                    "channel_types": ["EMG", "EMG"],
                    "emg_amp_gains": [1000, 1000],
                    "stim_clusters": [asdict(stim)],
                    "primary_stim": 1,
                    "num_samples": raw.shape[0],
                }
                recording_repository = RecordingRepository(stem)
                recording_repository.save_metadata(meta)
                recording_repository.save_annotation(RecordingAnnot.create_empty(), refresh_catalog=False)
    build_catalog(experiment)


def _write_experiment(root: Path, *, name: str, protocol: str, seed: int) -> None:
    experiment = root / name
    experiment.mkdir(parents=True)
    ExperimentRepository(experiment).save_annotation(ExperimentAnnot.create_empty())
    time_ms = np.arange(int((PRE_STIM_MS + POST_STIM_MS) * SCAN_RATE_HZ / 1_000)) * 1_000 / SCAN_RATE_HZ - PRE_STIM_MS
    is_stretch = protocol == "stretch"
    stim_kind = "Motor Length" if is_stretch else "Vibration"
    duration, ramp = (700.0, 150.0) if is_stretch else (500.0, 0.0)
    channels = [
        SignalChannel(name="TA", unit="mV", type_override="EMG"),
        SignalChannel(name="LG", unit="mV", type_override="EMG"),
        SignalChannel(name="Force", unit="N", type_override="Force"),
        SignalChannel(name="Length", unit="mm", type_override="Length"),
    ]
    channel_types = ["EMG", "EMG", "Force", "Length"]
    for condition_index, (condition_name, response_scale) in enumerate(CONDITIONS, start=1):
        dataset = experiment / condition_name
        dataset.mkdir()
        annotation = DatasetAnnot.create_empty()
        annotation.date = "260101"
        annotation.animal_id = f"SYN-{protocol.upper()}-C{condition_index}"
        annotation.condition = condition_name
        DatasetRepository(dataset).save_annotation(annotation, refresh_catalog=False)
        for session_index, session_id in enumerate(SESSION_IDS):
            session = dataset / session_id
            session.mkdir()
            session_annotation = SessionAnnot.create_empty(len(channels))
            session_annotation.channels = channels
            session_annotation.latency_windows = [
                LatencyWindow(name="Background", color="#6b7280", start_times=[-200.0] * len(channels), durations=[150.0] * len(channels)),
                LatencyWindow(name="Response", color="#2563eb", start_times=[0.0] * len(channels), durations=[duration] * len(channels)),
            ]
            SessionRepository(session).save_annotation(session_annotation, refresh_catalog=False)
            for index, nominal_intensity in enumerate(INTENSITIES):
                rng = np.random.default_rng(seed + condition_index * 10_000 + session_index * 100 + index)
                intensity = float(np.clip(nominal_intensity * rng.normal(1.0, 0.035), 0.05, 1.0))
                raw = (
                    _stretch_recording(time_ms, intensity, rng, response_scale)
                    if is_stretch
                    else _vibration_recording(time_ms, intensity, rng, response_scale)
                )
                stem = session / f"SYN{protocol[:3].upper()}C{condition_index}T{session_index + 1}-{index:04d}"
                with h5py.File(stem.with_suffix(".raw.h5"), "w") as h5:
                    h5.create_dataset("raw", data=raw, compression="gzip", compression_opts=4)
                stim = _stimulus(kind=stim_kind, intensity=intensity, duration_ms=duration, ramp_ms=ramp)
                meta = {
                    "recording_id": stem.name,
                    "num_channels": raw.shape[1],
                    "scan_rate": SCAN_RATE_HZ,
                    "pre_stim_acquired": PRE_STIM_MS,
                    "post_stim_acquired": POST_STIM_MS,
                    "recording_interval": 4.0,
                    "channel_types": channel_types,
                    "emg_amp_gains": [1000, 1000, None, None],
                    "stim_clusters": [asdict(stim)],
                    "primary_stim": 1,
                    "num_samples": raw.shape[0],
                }
                recording_repository = RecordingRepository(stem)
                recording_repository.save_metadata(meta)
                recording_repository.save_annotation(RecordingAnnot.create_empty(), refresh_catalog=False)
    build_catalog(experiment)


def build_demo_archive(output: Path = DEMO_ARCHIVE) -> Path:
    """Create a deterministic archive containing all bundled synthetic experiments."""
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Some managed Windows profiles deny creation beneath ``%TEMP%`` and
    # cloud-synchronised source trees.  C:\\tmp is the project's documented
    # writable test location; other platforms retain tempfile's normal choice.
    configured_build_root = os.environ.get("MONSTIM_DEMO_BUILD_ROOT")
    default_build_parent = Path(r"C:\tmp") if os.name == "nt" and Path(r"C:\tmp").is_dir() else None
    build_parent = Path(configured_build_root) if configured_build_root else default_build_parent
    temporary_root = Path(tempfile.mkdtemp(prefix="monstim-synthetic-demos-", dir=build_parent))
    try:
        _write_h_reflex_experiment(temporary_root)
        _write_experiment(temporary_root, name="Synthetic 100Hz Vibration", protocol="vibration", seed=81_000)
        _write_experiment(temporary_root, name="Synthetic Stretch Ramp Hold Release", protocol="stretch", seed=82_000)
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(temporary_root.rglob("*")):
                # The SQLite catalog stores absolute source paths, so it would
                # be stale the moment the archive is extracted. MonStim safely
                # rebuilds this derived cache on first discovery.
                if path.is_file() and path.name != ".monstim-cache.sqlite":
                    info = zipfile.ZipInfo(path.relative_to(temporary_root).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(info, path.read_bytes())
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Build synthetic native MonStim stretch and vibration demo experiments.")
    parser.add_argument("--output", type=Path, default=DEMO_ARCHIVE, help="Destination ZIP archive")
    args = parser.parse_args()
    output = build_demo_archive(args.output)
    print(f"Wrote {output} ({output.stat().st_size / 1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
