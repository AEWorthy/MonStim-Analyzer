from types import SimpleNamespace

from PySide6.QtWidgets import QDialog

from monstim_gui.dialogs.data_curation_manager import DataCurationManager, auto_refresh


def test_auto_refresh_preserves_required_boolean_arguments():
    calls = []

    class Target:
        def load_data(self):
            calls.append("refresh")

        @auto_refresh
        def update(self, include):
            calls.append(include)

    target = Target()

    Target.update(target, False)

    assert calls == [False, "refresh"]


def test_auto_refresh_discards_qt_checked_state_for_no_argument_methods():
    calls = []

    class Target:
        def load_data(self):
            calls.append("refresh")

        @auto_refresh
        def update(self):
            calls.append("updated")

    target = Target()

    Target.update(target, False)

    assert calls == ["updated", "refresh"]


def test_auto_refresh_skips_duplicate_reload_after_centralized_refresh():
    calls = []

    class Target:
        def __init__(self):
            self._data_refresh_generation = 0

        def load_data(self):
            calls.append("refresh")

        @auto_refresh
        def update(self):
            self._data_refresh_generation += 1

    target = Target()
    target.update()

    assert calls == []


def test_close_keeps_immediately_applied_changes_without_prompting(qapplication):
    class Command:
        def __init__(self):
            self.undo_called = False

        def undo(self):
            self.undo_called = True

    class Dialog(DataCurationManager):
        def __init__(self):
            QDialog.__init__(self)
            self._changes_made = True
            self.session_commands = [Command()]

    dialog = Dialog()
    changes_emitted = []
    dialog.data_structure_changed.connect(lambda: changes_emitted.append(True))

    dialog.accept()

    assert dialog.result() == QDialog.DialogCode.Accepted
    assert not dialog.session_commands[0].undo_called
    assert changes_emitted == [True]


def test_bulk_dataset_exclusion_keeps_false_include_argument(monkeypatch):
    import monstim_gui.commands as commands_module

    created = []

    class FakeToggleCommand:
        def __init__(self, gui, experiment_id, dataset_id, *, exclude):
            created.append((experiment_id, dataset_id, exclude))

        def execute(self):
            pass

    class FakeBatchCommand:
        def __init__(self, name, commands):
            self.name = name
            self.commands = commands

        def execute(self):
            pass

    monkeypatch.setattr(commands_module, "ToggleDatasetInclusionCommand", FakeToggleCommand)
    monkeypatch.setattr(commands_module, "BatchCommand", FakeBatchCommand)

    manager = SimpleNamespace(
        gui=object(),
        session_commands=[],
        _changes_made=False,
        load_data=lambda: None,
        _selected_dataset_data=lambda: [
            {"experiment_id": "exp", "metadata": {"id": "ds-1"}},
            {"experiment_id": "exp", "metadata": {"id": "ds-2"}},
        ],
    )

    DataCurationManager.set_selected_datasets_included(manager, False)

    assert created == [("exp", "ds-1", True), ("exp", "ds-2", True)]
    assert manager.session_commands[0].name == "Exclude 2 dataset(s)"


def test_failed_context_deletion_is_reported_without_escaping_qt_slot(monkeypatch):
    import monstim_gui.commands as commands_module
    import monstim_gui.dialogs.data_curation_manager as manager_module

    class FailedDeleteCommand:
        def __init__(self, *_args):
            pass

        def execute(self):
            raise PermissionError("directory is locked")

    messages = []
    monkeypatch.setattr(commands_module, "DeleteExperimentCommand", FailedDeleteCommand)
    monkeypatch.setattr(
        manager_module.QMessageBox,
        "question",
        lambda *_args: manager_module.QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(manager_module.QMessageBox, "critical", lambda *_args: messages.append("critical"))

    manager = SimpleNamespace(
        gui=object(),
        session_commands=[],
        _changes_made=False,
        _data_refresh_generation=0,
        load_data=lambda: messages.append("refresh"),
    )

    DataCurationManager.context_delete_experiment(manager, "LockedExperiment")

    assert messages == ["critical", "refresh"]
