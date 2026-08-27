import contextlib
import os
import shutil
import sys
import types
from pathlib import Path

import pytest

# Ensure Qt doesn't try to connect to a display in CI/headless
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# VS Code launches the selected Conda interpreter directly for pytest instead
# of activating the environment first.  On Windows that leaves its base
# environment's DLL directories ahead of this environment's BLAS/LAPACK
# runtime, which can make SciPy fail natively at its first LAPACK call.
# Keep the handle alive so this environment-local directory remains available
# for the entire test process.  This runs before pytest imports test modules.
_conda_library_bin = Path(sys.prefix) / "Library" / "bin"
if sys.platform == "win32" and _conda_library_bin.is_dir():
    _conda_library_bin_text = str(_conda_library_bin)
    _path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if _conda_library_bin_text.casefold() not in {entry.casefold() for entry in _path_entries}:
        os.environ["PATH"] = os.pathsep.join([_conda_library_bin_text, *_path_entries])
    else:
        os.environ["PATH"] = os.pathsep.join(
            [_conda_library_bin_text, *(entry for entry in _path_entries if entry.casefold() != _conda_library_bin_text.casefold())]
        )
    _conda_dll_directory = os.add_dll_directory(_conda_library_bin_text)
else:
    _conda_dll_directory = None


@pytest.fixture(scope="session")
def qapplication():
    """Keep one QApplication alive for the complete Qt-test session.

    PySide owns the native Qt application through its Python wrapper.  A
    throwaway ``QApplication.instance() or QApplication([])`` expression can
    therefore destroy the application while PyQtGraph still owns graphics
    objects, which is prone to intermittent native crashes on Windows.
    """
    from PySide6.QtWidgets import QApplication

    application = QApplication.instance() or QApplication([])
    yield application
    # Do not call quit() or delete the wrapper: widgets may be finalized after
    # individual test teardown, and Qt should own the shutdown sequence.


# Ensure the project root is importable so `monstim_gui` and `monstim_signals` resolve under pytest
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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
    import monstim_signals.core as core
    import monstim_signals.core.utils as utils

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
        self.dataset_combo = types.SimpleNamespace(setCurrentIndex=lambda *_: None, setEnabled=lambda *_: None, blockSignals=lambda *_: None)
        self.session_combo = types.SimpleNamespace(setCurrentIndex=lambda *_: None, setEnabled=lambda *_: None, blockSignals=lambda *_: None)

    # New unified API in real widget
    def update(self, levels: tuple[str, ...] | None = None, preserve_selection: bool = True):
        pass

    def refresh(self, levels: tuple[str, ...] | None = None):
        pass

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
        self.current_option_widget = types.SimpleNamespace(recording_cycler=types.SimpleNamespace(reset_max_recordings=lambda: None))

    def on_data_selection_changed(self):
        pass


class FakeConfigRepo:
    def read_config(self):
        return {}


class FakeGUI:
    """A minimal MonstimGUI stand-in for filesystem-level curation commands."""

    def __init__(self, output_dir: Path):
        self.output_path = str(output_dir)
        # Mark this stand-in as headless so UI code can choose to suppress
        # modal dialogs (e.g., QMessageBox) during tests.
        self.headless = True
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
        # populated by DataManager.unpack_existing_experiments
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
    dm.unpack_existing_experiments()
    gui.data_manager = dm
    return gui


# --- Session-level cleanup for ad-hoc temp dirs ---
@pytest.fixture(scope="session", autouse=True)
def _cleanup_pytest_tmp_golden_check():
    """Delete a stray .pytest-tmp-golden-check folder in repo root after tests.

    Some local runs may create a top-level scratch folder; keep the tree clean by
    removing it once the test session is over.
    """
    yield
    repo_root = Path(__file__).resolve().parents[1]
    stray = repo_root / ".pytest-tmp-golden-check"
    try:
        if stray.exists():
            shutil.rmtree(stray, ignore_errors=True)
    except Exception:
        # Best-effort cleanup; ignore errors to not mask test results
        pass
