from pathlib import Path

import main
from monstim_signals.core import utils


def test_startup_logging_uses_qt_application_data_location(monkeypatch, tmp_path: Path):
    expected = tmp_path / "WorthyLab" / "MonStim Analyzer"
    monkeypatch.setattr(main.QStandardPaths, "writableLocation", lambda _location: str(expected))

    assert Path(main.make_default_log_dir()) == expected / "logs"


def test_shared_log_folder_uses_qt_application_data_location(monkeypatch, tmp_path: Path):
    expected = tmp_path / "WorthyLab" / "MonStim Analyzer"
    monkeypatch.setattr(utils.QStandardPaths, "writableLocation", lambda _location: str(expected))

    assert Path(utils.get_log_dir()) == expected / "logs"
