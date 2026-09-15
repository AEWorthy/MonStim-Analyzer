"""Capture non-sensitive documentation screenshots from focused MonStim dialogs."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PySide6.QtWidgets import QApplication, QInputDialog

from monstim_gui.dialogs.addon_manager import AddonManagerDialog
from monstim_gui.dialogs.bulk_export_dialog import BulkExportDialog
from monstim_gui.dialogs.latency import LatencyWindowsDialog
from monstim_gui.dialogs.recording_exclusion_editor import RecordingExclusionEditor
from monstim_gui.dialogs.update_manager import UpdateManagerDialog
from monstim_signals.core import LatencyWindow, SessionAnnot
from monstim_signals.domain.session import Session

SCREENS = (
    "addon-manager",
    "update-manager",
    "addon-source-selection",
    "latency-window-editor",
    "recording-exclusion-editor",
    "bulk-export-dialog",
)


class _SyntheticSession:
    def __init__(self) -> None:
        self.id = "SYN-001"
        self.excluded_recordings: set[str] = set()
        self._config = {}
        self.default_method = "rms"
        time = np.linspace(0, 0.04, 1000, endpoint=False)
        self._recordings = [
            _SyntheticRecording(f"{index:04d}", stimulus, time, scale)
            for index, (stimulus, scale) in enumerate(((0.5, 0.2), (1.0, 0.45), (1.5, 0.8), (2.0, 0.6)), start=1)
        ]

    def get_all_recordings(self, include_excluded: bool = True):
        return list(self._recordings)


class _SyntheticRecording:
    def __init__(self, recording_id: str, stimulus: float, time: np.ndarray, scale: float) -> None:
        self.id = recording_id
        self.stim_amplitude = stimulus
        self.num_channels = 2
        self.num_samples = len(time)
        self.scan_rate = 25_000.0
        self.channel_types = ["emg", "emg"]
        burst = np.exp(-(((time - 0.012) / 0.003) ** 2)) * np.sin(2 * np.pi * 450 * time) * scale
        self._data = np.column_stack((burst, burst * 0.7))

    def raw_view(self, ch: int, t):
        return self._data[t, ch]


class _SyntheticGui:
    def __init__(self, fixture_root: Path) -> None:
        from PySide6.QtWidgets import QWidget

        self.widget = QWidget()
        self.current_session = _SyntheticSession()
        self.current_dataset = None
        self.current_experiment = None
        self.current_experiment = None
        self.command_invoker = SimpleNamespace(execute=lambda _command: None)
        self.status_bar = SimpleNamespace(showMessage=lambda *_args: None)
        self.plot_widget = SimpleNamespace(persistent_channel_selection=[0])
        self.active_profile_data = {}
        self.channel_names = ["Left TA", "Right TA"]
        self.export_path = "User/Data/Path/"
        experiment = fixture_root / "Synthetic EMG Demonstration"
        (experiment / "250101 SYN1 Baseline").mkdir(parents=True, exist_ok=True)
        self.expts_dict = {"Synthetic EMG Demonstration": str(experiment)}
        for name in (
            "current_session",
            "current_dataset",
            "current_experiment",
            "command_invoker",
            "status_bar",
            "plot_widget",
            "active_profile_data",
            "channel_names",
            "export_path",
            "expts_dict",
        ):
            setattr(self.widget, name, getattr(self, name))

    def __getattr__(self, name):
        return getattr(self.widget, name)


class _LatencyConfig:
    def read_config(self):
        return {"m_wave_window_names": ["M-wave", "M_response"], "latency_window_presets": {}}


def _latency_dialog() -> LatencyWindowsDialog:
    session = Session.__new__(Session)
    session.id = "SYN-001"
    session.channel_names = ["Left TA", "Right TA"]
    annotation = SessionAnnot.create_empty()
    annotation.latency_windows = [
        LatencyWindow(name="M-wave", color="blue", start_times=[2.0, 2.2], durations=[2.0, 2.0]),
        LatencyWindow(name="H-reflex", color="red", start_times=[8.0, 8.2], durations=[6.0, 6.0]),
    ]
    session.annot = annotation
    parent = SimpleNamespace(current_experiment=None, current_dataset=None, current_session=session)
    # Qt requires a QWidget parent, while the dialog only needs the selection attributes.
    from PySide6.QtWidgets import QWidget

    widget = QWidget()
    widget.current_experiment = parent.current_experiment
    widget.current_dataset = parent.current_dataset
    widget.current_session = parent.current_session
    dialog = LatencyWindowsDialog(session, widget, config_repo=_LatencyConfig())
    dialog.editor.table.selectRow(0)
    dialog.editor.per_channel_radio.setChecked(True)
    return dialog


def _dialog_for(screen: str, fixture_root: Path):
    if screen == "addon-manager":
        return AddonManagerDialog()
    if screen == "update-manager":
        return UpdateManagerDialog()
    if screen == "addon-source-selection":
        dialog = QInputDialog()
        dialog.setWindowTitle("Import using Add-on")
        dialog.setLabelText("Source type:")
        dialog.setComboBoxItems(["File", "Folder"])
        dialog.setOption(QInputDialog.InputDialogOption.UseListViewForComboBoxItems)
        dialog.resize(500, 170)
        return dialog
    if screen == "latency-window-editor":
        return _latency_dialog()
    if screen == "recording-exclusion-editor":
        gui = _SyntheticGui(fixture_root)
        dialog = RecordingExclusionEditor(gui.widget)
        dialog.stimulus_group.setChecked(True)
        dialog.threshold_type_combo.setCurrentIndex(dialog.threshold_type_combo.findData("above"))
        dialog.threshold_spinbox.setValue(1.25)
        dialog.update_preview()
        return dialog
    if screen == "bulk-export-dialog":
        gui = _SyntheticGui(fixture_root)
        dialog = BulkExportDialog(gui.widget)
        group = dialog._expt_groups[0]
        group._dataset_cbs[0].setChecked(True)
        next(iter(dialog._dtype_cbs.values())).setChecked(True)
        next(iter(dialog._method_cbs.values())).setChecked(True)
        dialog._channel_cbs[0].setChecked(True)
        dialog._refresh_readiness()
        return dialog
    raise ValueError(f"Unknown screen: {screen}")


def _capture(screen: str, output: Path, app: QApplication, fixture_root: Path) -> None:
    dialog = _dialog_for(screen, fixture_root)
    dialog.show()
    app.processEvents()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not dialog.grab().save(str(output)):
        raise RuntimeError(f"Could not save {output}")
    dialog.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", choices=SCREENS, default="addon-manager", help="Single focused dialog to capture")
    parser.add_argument("--output", type=Path, help="Output PNG for one screen (defaults to the historic Add-on Manager mode)")
    parser.add_argument("--all", action="store_true", help="Capture every safe static dialog into --output-dir")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets"), help="Directory used with --all")
    args = parser.parse_args()
    if args.all and args.output:
        parser.error("use either --all or --output, not both")
    if not args.all and args.output is None:
        parser.error("--output is required unless --all is used")
    # Never expose a maintainer's normal plug-in location in documentation.
    temporary_plugin_root = tempfile.mkdtemp(prefix="monstim-docs-plugin-")
    temporary_fixture_root = Path(tempfile.mkdtemp(prefix="monstim-docs-data-"))
    previous_plugin_root = os.environ.get("MONSTIM_PLUGIN_DIR")
    previous_screenshot_mode = os.environ.get("MONSTIM_DOCS_SCREENSHOT")
    os.environ["MONSTIM_PLUGIN_DIR"] = temporary_plugin_root
    os.environ["MONSTIM_DOCS_SCREENSHOT"] = "1"
    app = QApplication.instance() or QApplication([])
    app.setOrganizationName("WorthyLab")
    app.setApplicationName("MonStim Analyzer")
    try:
        if args.all:
            for screen in SCREENS:
                _capture(screen, args.output_dir / f"{screen}.png", app, temporary_fixture_root)
        else:
            _capture(args.screen, args.output, app, temporary_fixture_root)
    finally:
        if previous_plugin_root is None:
            os.environ.pop("MONSTIM_PLUGIN_DIR", None)
        else:
            os.environ["MONSTIM_PLUGIN_DIR"] = previous_plugin_root
        if previous_screenshot_mode is None:
            os.environ.pop("MONSTIM_DOCS_SCREENSHOT", None)
        else:
            os.environ["MONSTIM_DOCS_SCREENSHOT"] = previous_screenshot_mode
        shutil.rmtree(temporary_plugin_root, ignore_errors=True)
        shutil.rmtree(temporary_fixture_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
