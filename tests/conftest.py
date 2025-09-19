import os
import sys
from pathlib import Path
import contextlib
import types
import pytest

# Ensure Qt doesn't try to connect to a display in CI/headless
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Ensure local test helpers can be imported with `import helpers`
tests_dir = os.path.dirname(__file__)
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)

# no direct import of get_output_path here; we override it dynamically inside the context manager


@contextlib.contextmanager
def override_output_path(tmpdir: Path):
    """Temporarily override both monstim_signals.core.utils.get_output_path and monstim_signals.core.get_output_path
    so any code importing either will write into tmpdir.
    """
    import monstim_signals.core.utils as utils
    import monstim_signals.core as core

    # Save originals
    original_utils_get_output_path = utils.get_output_path
    original_core_get_output_path = getattr(core, "get_output_path", None)

    def _get_output_path_override():
        p = str(tmpdir)
        os.makedirs(p, exist_ok=True)
        return p

    try:
        utils.get_output_path = _get_output_path_override
        if original_core_get_output_path is not None:
            core.get_output_path = _get_output_path_override  # type: ignore[attr-defined]
        yield
    finally:
        utils.get_output_path = original_utils_get_output_path
        if original_core_get_output_path is not None:
            core.get_output_path = original_core_get_output_path  # type: ignore[attr-defined]


@pytest.fixture()
def temp_output_dir(tmp_path: Path):
    """Provide a clean temporary directory as data output folder for tests."""
    with override_output_path(tmp_path):
        yield tmp_path


class FakeMenuBar:
    def update_undo_redo_labels(self):
        pass


class FakeStatusBar:
    def showMessage(self, *_args, **_kwargs):
        pass


class FakeDataSelectionWidget:
    def __init__(self):
        # Minimal API used by commands
        self.experiment_combo = types.SimpleNamespace(setCurrentIndex=lambda *_: None, blockSignals=lambda *_: None)
        self.dataset_combo = types.SimpleNamespace(
            setCurrentIndex=lambda *_: None, setEnabled=lambda *_: None, blockSignals=lambda *_: None
        )
        self.session_combo = types.SimpleNamespace(
            setCurrentIndex=lambda *_: None, setEnabled=lambda *_: None, blockSignals=lambda *_: None
        )

    def update_experiment_combo(self):
        pass

    def update_dataset_combo(self):
        pass

    def update_session_combo(self):
        pass

    def sync_combo_selections(self):
        pass


class FakePlotWidget:
    def __init__(self):
        self.current_option_widget = types.SimpleNamespace(
            recording_cycler=types.SimpleNamespace(reset_max_recordings=lambda: None)
        )

    def on_data_selection_changed(self):
        pass


class FakeConfigRepo:
    def read_config(self):
        return {}


class FakeGUI:
    """A minimal MonstimGUI stand-in for filesystem-level curation commands."""

    def __init__(self, output_dir: Path):
        self.output_path = str(output_dir)
        self.expts_dict = {}
        self.expts_dict_keys = []
        self.current_experiment = None
        self.current_dataset = None
        self.current_session = None
        self.menu_bar = FakeMenuBar()
        self.status_bar = FakeStatusBar()
        self.plot_widget = FakePlotWidget()
        self.data_selection_widget = FakeDataSelectionWidget()
        self.config_repo = FakeConfigRepo()
        self.has_unsaved_changes = False
        # populated by DataManager.refresh_existing_experiments
        self.profile_selector_combo = types.SimpleNamespace(currentText=lambda: "")

    # The following helpers mirror MonstimGUI API used by DataManager
    def set_current_experiment(self, expt):
        self.current_experiment = expt

    def set_current_dataset(self, ds):
        self.current_dataset = ds

    def set_current_session(self, s):
        self.current_session = s


@pytest.fixture()
def fake_gui(temp_output_dir: Path):
    # Ensure directory exists and initial expts mapping is empty
    gui = FakeGUI(temp_output_dir)
    # Seed expts_dict by scanning folder
    from monstim_gui.managers.data_manager import DataManager

    dm = DataManager(gui)
    dm.refresh_existing_experiments()
    gui.data_manager = dm
    return gui
