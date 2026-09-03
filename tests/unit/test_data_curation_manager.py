from types import SimpleNamespace

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
