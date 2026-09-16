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

SCREENS = ("session-emg", "single-recording", "recruitment-curves", "latency-windows")
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
    sessions = [_make_session(f"SYN-{index:03d}", index) for index in range(1, 4)]
    dataset_annotation = DatasetAnnot.create_empty()
    dataset_annotation.date = "2026-01-15"
    dataset_annotation.animal_id = "DEMO-01"
    dataset_annotation.condition = "Baseline"
    dataset = Dataset("Synthetic H-reflex baseline", sessions, dataset_annotation)
    experiment = Experiment("Synthetic H-reflex recruitment", [dataset], ExperimentAnnot.create_empty())
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
    # Keep this profile in memory. It makes the delayed H-reflex visible in
    # the real plotting path without creating a user profile or changing the
    # shipped default profile.
    window._profile_list.append((_DEMO_PROFILE["name"], "[in-memory demo profile]", _DEMO_PROFILE))
    profile_combo = window.profile_selector_combo
    profile_combo.blockSignals(True)
    profile_combo.addItem(_DEMO_PROFILE["name"], userData="[in-memory demo profile]")
    profile_combo.setItemData(
        profile_combo.count() - 1,
        _DEMO_PROFILE["description"],
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
    elif screen == "latency-windows":
        dialog = LatencyWindowsDialog(session, window, config_repo=window.config_repo)
        dialog.resize(1120, 780)
        dialog.show()
        QTest.qWait(75)
        dialog.editor.table.selectRow(0)
        QTest.qWait(25)
        _capture_widget(dialog, output)
        dialog.close()
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
        _configure_window(window, experiment, dataset, session)
        if args.all:
            for screen in SCREENS:
                _capture(screen, args.output_dir / f"{screen}.png", window, session)
        else:
            _capture(args.screen, args.output, window, session)
    finally:
        if window is not None:
            window.close()
        shutil.rmtree(settings_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
