"""Capture deterministic, non-sensitive product screenshots from MonStim Analyzer.

The tool builds an in-memory, physiologically plausible H-reflex experiment and
uses the application's real plotting and latency-window UI.  It never reads a
user experiment or writes generated data into the repository.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PySide6.QtCore import QCoreApplication, QSettings, Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from monstim_gui.core.application_state import app_state
from monstim_gui.dialogs.latency import LatencyWindowsDialog
from monstim_gui.gui_main import MonstimGUI
from monstim_signals.core import (
    DatasetAnnot,
    ExperimentAnnot,
    LatencyWindow,
    RecordingAnnot,
    RecordingMeta,
    SessionAnnot,
    SignalChannel,
    StimCluster,
)
from monstim_signals.domain.dataset import Dataset
from monstim_signals.domain.experiment import Experiment
from monstim_signals.domain.recording import Recording
from monstim_signals.domain.session import Session

SCREENS = (
    "session-emg",
    "single-recording",
    "reflex-curves",
    "recruitment-curves",
    "m-max",
    "latency-windows",
    "vibration-emg",
    "stretch-emg",
)
_DEMO_PROFILE = {
    "name": "Demo H-reflex",
    "description": "Synthetic H-reflex screenshot profile: 5 ms pre-stimulus and 35 ms post-stimulus display.",
    "analysis_parameters": {"pre_stim_time": 5.0, "time_window": 35.0},
}
_SCAN_RATE_HZ = 30_000
_PRE_STIM_MS = 20
_POST_STIM_MS = 45


def _biphasic_wave(time_ms: np.ndarray, center_ms: float, width_ms: float, frequency_hz: float, amplitude_mv: float) -> np.ndarray:
    relative_seconds = (time_ms - center_ms) / 1000
    envelope = np.exp(-0.5 * ((time_ms - center_ms) / width_ms) ** 2)
    return amplitude_mv * envelope * np.sin(2 * np.pi * frequency_hz * relative_seconds)


def _make_recording(recording_index: int, stimulus_ma: float, session_seed: int) -> Recording:
    """Create one two-channel EMG recording with recruitment-like responses."""
    samples = int((_PRE_STIM_MS + _POST_STIM_MS) * _SCAN_RATE_HZ / 1000)
    time_ms = np.arange(samples) * 1000 / _SCAN_RATE_HZ
    stim_start_ms = float(_PRE_STIM_MS)
    rng = np.random.default_rng(session_seed * 100 + recording_index)
    data = rng.normal(0, 0.012, size=(samples, 2))

    m_amplitude = 1.15 / (1 + np.exp(-1.15 * (stimulus_ma - 2.4)))
    h_amplitude = 0.50 * np.exp(-((stimulus_ma - 4.2) ** 2) / (2 * 1.35**2))
    for channel, scale in enumerate((1.0, 0.72)):
        data[:, channel] += _biphasic_wave(time_ms, stim_start_ms + 5.9, 0.95, 240, m_amplitude * scale)
        data[:, channel] += _biphasic_wave(time_ms, stim_start_ms + 27.4, 1.75, 115, h_amplitude * scale)

    stim_index = int(stim_start_ms * _SCAN_RATE_HZ / 1000)
    artifact_amplitude = 0.7 + stimulus_ma * 0.12
    data[stim_index - 1 : stim_index + 2, :] += np.array([[0.25], [1.0], [-0.48]]) * artifact_amplitude

    stim = StimCluster(
        stim_delay=0.0,
        stim_duration=0.2,
        stim_type="Electrical",
        stim_v=float(stimulus_ma),
        stim_min_v=float(stimulus_ma),
        stim_max_v=float(stimulus_ma),
        pulse_shape="Square",
        num_pulses=1,
        pulse_period=1.0,
        peak_duration=0.2,
        ramp_duration=0.0,
    )
    meta = RecordingMeta(
        recording_id=f"SYN{session_seed:02d}-{recording_index:04d}",
        num_channels=2,
        scan_rate=_SCAN_RATE_HZ,
        pre_stim_acquired=_PRE_STIM_MS,
        post_stim_acquired=_POST_STIM_MS,
        recording_interval=3.0,
        channel_types=["EMG", "EMG"],
        emg_amp_gains=[1000, 1000],
        stim_clusters=[stim],
        primary_stim=stim,
        num_samples=samples,
    )
    return Recording(meta=meta, annot=RecordingAnnot.create_empty(), raw=data.astype(np.float32))


def _make_session(session_id: str, seed: int) -> Session:
    stimuli = np.linspace(0.5, 9.0, 18)
    recordings = [_make_recording(index, float(stimulus), seed) for index, stimulus in enumerate(stimuli)]
    annotation = SessionAnnot.create_empty(num_channels=2)
    annotation.channels = [
        SignalChannel(name="Tibialis anterior", unit="mV", type_override="EMG"),
        SignalChannel(name="Soleus", unit="mV", type_override="EMG"),
    ]
    annotation.latency_windows = [
        LatencyWindow(name="M-wave", color="tab:red", start_times=[3.5, 3.5], durations=[5.0, 5.0]),
        LatencyWindow(name="H-reflex", color="tab:blue", start_times=[23.0, 23.0], durations=[8.0, 8.0]),
    ]
    return Session(session_id=session_id, recordings=recordings, annot=annotation)


def build_synthetic_experiment() -> tuple[Experiment, Dataset, Session]:
    """Build the reusable in-memory fixture used by every marketing screen."""
    sessions = [_make_session(f"Trial {index}", index) for index in range(1, 4)]
    dataset_annotation = DatasetAnnot.create_empty()
    dataset_annotation.date = "2026-01-15"
    dataset_annotation.animal_id = "DEMO-01"
    dataset_annotation.condition = "Condition B - Reference response"
    dataset = Dataset("Condition B - Reference response", sessions, dataset_annotation)
    experiment = Experiment("Synthetic H-reflex recruitment", [dataset], ExperimentAnnot.create_empty())
    return experiment, dataset, sessions[1]


def _make_protocol_recording(recording_index: int, intensity: float, protocol: str) -> Recording:
    """Create a fictional TA/LG bulk-EMG protocol recording for documentation."""
    # Must exceed twice the shipped 3.5 kHz EMG high-cutoff so the real
    # filtered-EMG plotting path can render this fixture.
    scan_rate, pre_stim_ms, post_stim_ms = 8_000, 250, 1_000
    samples = int((pre_stim_ms + post_stim_ms) * scan_rate / 1_000)
    time_ms = np.arange(samples) * 1_000 / scan_rate - pre_stim_ms
    rng = np.random.default_rng(90_000 + recording_index + (1 if protocol == "vibration" else 100))
    is_stretch = protocol == "stretch"
    data = rng.normal(0, 0.010, (samples, 4))
    if is_stretch:
        rising = np.clip(time_ms / 150, 0, 1)
        falling = np.clip((700 - time_ms) / 150, 0, 1)
        length = intensity * np.minimum(rising, falling) * ((time_ms >= 0) & (time_ms <= 700))
        data[:, 2] += 0.65 * length + 0.08 * np.gradient(length) * scan_rate / 1_000
        data[:, 3] += length
        data[:, 0] += _biphasic_wave(time_ms, 63, 10, 155, 0.035 + 0.055 * intensity)
        data[:, 0] += _biphasic_wave(time_ms, 592, 12, 165, 0.025 + 0.040 * intensity)
        phases = ((20, 190, 0.12 + 0.46 * intensity), (205, 535, 0.025 + 0.12 * intensity), (545, 735, 0.10 + 0.36 * intensity))
        duration, stim_type, shape, ramp = 700.0, "Motor Length", "Ramp-Hold-Release", 150.0
    else:
        taper = np.clip(np.minimum(np.maximum(time_ms, 0) / 25, np.maximum(500 - time_ms, 0) / 25), 0, 1)
        for channel, scale in enumerate((0.62, 1.0)):
            data[:, channel] += taper * (0.04 + 0.29 * intensity**1.25) * scale * np.sin(2 * np.pi * 100 * time_ms / 1_000)
        data[:, 2] += taper * intensity * (0.22 + 0.08 * np.sin(2 * np.pi * 100 * time_ms / 1_000 + 0.35))
        data[:, 3] += taper * intensity * (0.15 + 0.10 * np.sin(2 * np.pi * 100 * time_ms / 1_000))
        phases = ((510, 650, 0.05),)
        duration, stim_type, shape, ramp = 500.0, "Vibration", "Sine", 0.0
    channel_scales = (0.0, 1.0) if is_stretch else (1.0, 0.74)
    for channel, scale in enumerate(channel_scales):
        for start, end, scale_amplitude in phases:
            data[:, channel] += _biphasic_wave(time_ms, (start + end) / 2, (end - start) / 4, 135 + channel * 20, scale_amplitude * intensity * scale)
    stim = StimCluster(
        stim_delay=0.0,
        stim_duration=duration,
        stim_type=stim_type,
        stim_v=intensity,
        stim_min_v=intensity,
        stim_max_v=intensity,
        pulse_shape=shape,
        num_pulses=50 if protocol == "vibration" else 1,
        pulse_period=10.0 if protocol == "vibration" else duration,
        peak_duration=max(0.0, duration - 2 * ramp),
        ramp_duration=ramp,
    )
    meta = RecordingMeta(
        recording_id=f"SYN-{protocol[:3].upper()}-{recording_index:04d}",
        num_channels=data.shape[1],
        scan_rate=scan_rate,
        pre_stim_acquired=pre_stim_ms,
        post_stim_acquired=post_stim_ms,
        recording_interval=4.0,
        channel_types=["EMG", "EMG", "Force", "Length"],
        emg_amp_gains=[1000, 1000, None, None],
        stim_clusters=[stim],
        primary_stim=stim,
        num_samples=samples,
    )
    return Recording(meta=meta, annot=RecordingAnnot.create_empty(), raw=data.astype(np.float32))


def build_synthetic_protocol_experiment(protocol: str) -> tuple[Experiment, Dataset, Session]:
    """Build an in-memory synthetic protocol fixture; no user data is read."""
    names = [SignalChannel(name="TA", unit="mV", type_override="EMG"), SignalChannel(name="LG", unit="mV", type_override="EMG")]
    names.extend((SignalChannel(name="Force", unit="N", type_override="Force"), SignalChannel(name="Length", unit="mm", type_override="Length")))
    sessions = []
    for index in range(1, 4):
        annotation = SessionAnnot.create_empty(len(names))
        annotation.channels = [SignalChannel(name=channel.name, unit=channel.unit, type_override=channel.type_override) for channel in names]
        annotation.latency_windows = [
            LatencyWindow(name="Background", color="#6b7280", start_times=[-200.0] * len(names), durations=[150.0] * len(names)),
            LatencyWindow(
                name="Response",
                color="#2563eb",
                start_times=[0.0] * len(names),
                durations=[700.0 if protocol == "stretch" else 500.0] * len(names),
            ),
        ]
        recordings = [_make_protocol_recording(i, intensity, protocol) for i, intensity in enumerate(np.linspace(0.1, 1.0, 7))]
        sessions.append(Session(f"Trial {index}", recordings, annotation))
    label = "Stretch ramp-hold-release" if protocol == "stretch" else "100 Hz vibration"
    dataset_annotation = DatasetAnnot.create_empty()
    dataset_annotation.condition = "Condition B - Reference response"
    dataset = Dataset("Condition B - Reference response", sessions, dataset_annotation)
    experiment = Experiment(f"Synthetic {label} intensity series", [dataset], ExperimentAnnot.create_empty())
    return experiment, dataset, sessions[1]


def _configure_window(window: MonstimGUI, experiment: Experiment, dataset: Dataset, session: Session) -> None:
    """Connect the real GUI to the in-memory hierarchy without filesystem loading."""
    window.headless = True
    window.current_experiment = experiment
    window.current_dataset = dataset
    window.current_session = session
    window.channel_names = session.channel_names
    # Do not point the selection widget at a synthetic filesystem path: its
    # normal refresh path intentionally consults on-disk experiment metadata.
    window.expts_dict = {}
    window.expts_dict_keys = []
    window.data_selection_widget.update()
    experiment_combo = window.data_selection_widget.experiment_combo
    experiment_combo.blockSignals(True)
    experiment_combo.clear()
    experiment_combo.addItem(experiment.id)
    experiment_combo.setCurrentIndex(0)
    experiment_combo.blockSignals(False)
    # Keep a protocol-appropriate profile in memory so the whole synthetic
    # stimulus is visible without creating or changing a user profile.
    stim_type = session.recordings[0].meta.primary_stim.stim_type
    profile = _DEMO_PROFILE
    if stim_type in {"Vibration", "Motor Length"}:
        profile = {
            "name": f"Demo {stim_type}",
            "description": f"Synthetic {stim_type.lower()} screenshot profile.",
            "analysis_parameters": {"pre_stim_time": 250.0, "time_window": 1000.0},
        }
    window._profile_list.append((profile["name"], "[in-memory demo profile]", profile))
    profile_combo = window.profile_selector_combo
    profile_combo.blockSignals(True)
    profile_combo.addItem(profile["name"], userData="[in-memory demo profile]")
    profile_combo.setItemData(
        profile_combo.count() - 1,
        profile["description"],
        role=Qt.ItemDataRole.ToolTipRole,
    )
    profile_combo.blockSignals(False)
    profile_combo.setCurrentIndex(profile_combo.count() - 1)
    window.plot_widget.on_data_selection_changed()
    window.resize(1600, 980)
    window.show()
    QCoreApplication.processEvents()


def _select_plot(window: MonstimGUI, level: str, plot_name: str) -> None:
    radios = {
        "session": window.plot_widget.session_radio,
        "dataset": window.plot_widget.dataset_radio,
        "experiment": window.plot_widget.experiment_radio,
    }
    radios[level].setChecked(True)
    window.plot_widget.plot_type_combo.setCurrentText(plot_name)
    QCoreApplication.processEvents()


def _render(window: MonstimGUI) -> None:
    window.plot_controller.plot_data()
    # Let the deferred graphics-layout update and the software renderer finish.
    QTest.qWait(150)
    QCoreApplication.processEvents()


def _capture_widget(widget, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not widget.grab().save(str(output)):
        raise RuntimeError(f"Could not save screenshot: {output}")


def _capture(screen: str, output: Path, window: MonstimGUI, session: Session) -> None:
    if screen == "session-emg":
        _select_plot(window, "session", "EMG")
        options = window.plot_widget.current_option_widget
        options.show_extrema_labels_checkbox.setChecked(True)
        _render(window)
        _capture_widget(window, output)
    elif screen == "single-recording":
        _select_plot(window, "session", "Single EMG Recordings")
        options = window.plot_widget.current_option_widget
        options.recording_cycler.recording_spinbox.setValue(8)
        options.show_extrema_labels_checkbox.setChecked(True)
        _render(window)
        _capture_widget(window, output)
    elif screen == "recruitment-curves":
        _select_plot(window, "dataset", "Average Reflex:Stimulus Curves")
        _render(window)
        _capture_widget(window, output)
    elif screen == "reflex-curves":
        _select_plot(window, "session", "Reflex:Stimulus Curves")
        _render(window)
        _capture_widget(window, output)
    elif screen == "m-max":
        _select_plot(window, "dataset", "M-max")
        _render(window)
        _capture_widget(window, output)
    elif screen == "latency-windows":
        dialog = LatencyWindowsDialog(session, window, config_repo=window.config_repo)
        dialog.resize(1120, 780)
        dialog.show()
        QTest.qWait(75)
        dialog.editor.table.selectRow(0)
        QTest.qWait(25)
        _capture_widget(dialog, output)
        dialog.close()
    elif screen in {"vibration-emg", "stretch-emg"}:
        _select_plot(window, "session", "EMG")
        options = window.plot_widget.current_option_widget
        options.show_extrema_labels_checkbox.setChecked(False)
        if screen in {"vibration-emg", "stretch-emg"}:
            options.channel_selector.set_selected_channels([0, 1, 2, 3])
        _render(window)
        _capture_widget(window, output)
    else:
        raise ValueError(f"Unknown screen: {screen}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture reproducible marketing screenshots using synthetic H-reflex data.")
    parser.add_argument("--screen", choices=SCREENS, help="Capture one named screen")
    parser.add_argument("--all", action="store_true", help="Capture every named screen")
    parser.add_argument("--output", type=Path, help="PNG destination for --screen")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets/demo"), help="Destination directory for --all")
    args = parser.parse_args()
    if args.all == (args.screen is not None):
        parser.error("choose exactly one of --screen or --all")
    if args.screen and args.output is None:
        parser.error("--output is required with --screen")
    if args.all and args.output is not None:
        parser.error("--output is only valid with --screen")

    settings_root = Path(tempfile.mkdtemp(prefix="monstim-marketing-settings-"))
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(settings_root))
    QApplication.setOrganizationName("MonStim Screenshot Capture")
    QApplication.setApplicationName("Marketing Fixture")
    app = QApplication.instance() or QApplication([])
    # The offscreen Qt platform bundled with some Conda installations cannot
    # discover its optional font folder. Register the Windows system font so
    # headless captures remain readable as well as deterministic.
    windows_font = Path(os.environ.get("WINDIR", r"C:\\Windows")) / "Fonts" / "segoeui.ttf"
    if windows_font.is_file():
        QFontDatabase.addApplicationFont(str(windows_font))
    app.setFont(QFont("Segoe UI", 10))
    app_state.reinitialize_settings()
    window: MonstimGUI | None = None
    try:
        experiment, dataset, session = build_synthetic_experiment()
        window = MonstimGUI()
        if args.all:
            for screen in SCREENS:
                if screen == "vibration-emg":
                    experiment, dataset, session = build_synthetic_protocol_experiment("vibration")
                elif screen == "stretch-emg":
                    experiment, dataset, session = build_synthetic_protocol_experiment("stretch")
                else:
                    experiment, dataset, session = build_synthetic_experiment()
                _configure_window(window, experiment, dataset, session)
                _capture(screen, args.output_dir / f"{screen}.png", window, session)
        else:
            if args.screen in {"vibration-emg", "stretch-emg"}:
                experiment, dataset, session = build_synthetic_protocol_experiment(args.screen.removesuffix("-emg"))
            _configure_window(window, experiment, dataset, session)
            _capture(args.screen, args.output, window, session)
    finally:
        if window is not None:
            window.close()
        shutil.rmtree(settings_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
