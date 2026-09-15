from types import SimpleNamespace

from monstim_gui.commands import DeleteExperimentCommand


def test_delete_experiment_skips_unrelated_catalog_rebuild(monkeypatch):
    refresh_calls = []

    class FakeDataManager:
        def delete_experiment_by_id(self, experiment_id):
            assert experiment_id == "OldExperiment"

        def refresh_data_views(self, *paths, **kwargs):
            refresh_calls.append((paths, kwargs))

    gui = SimpleNamespace(
        data_manager=FakeDataManager(),
        expts_dict={"RemainingExperiment": "C:/data/RemainingExperiment"},
    )

    DeleteExperimentCommand(gui, "OldExperiment").execute()

    assert refresh_calls == [((), {"rebuild_catalogs": False})]


def test_refresh_data_views_does_not_scan_all_experiments_when_rebuild_disabled(monkeypatch):
    from monstim_gui.managers.data_manager import DataManager

    built = []
    monkeypatch.setattr(DataManager, "_cancel_cache_warmup", lambda self: None)
    monkeypatch.setattr(DataManager, "unpack_existing_experiments", lambda self: None)
    monkeypatch.setattr("monstim_signals.io.experiment_catalog.build_catalog", lambda path: built.append(path), raising=False)

    gui = SimpleNamespace(
        expts_dict={"ExperimentA": "C:/data/ExperimentA", "ExperimentB": "C:/data/ExperimentB"},
        expts_dict_keys=["ExperimentA", "ExperimentB"],
        current_experiment=None,
        data_selection_widget=None,
        _data_curation_manager=None,
    )

    DataManager(gui).refresh_data_views(rebuild_catalogs=False)

    assert built == []


def test_delete_experiment_stops_cache_warmup_before_removing_files(tmp_path, monkeypatch):
    from pathlib import Path

    from monstim_gui.managers.data_manager import DataManager

    experiment = tmp_path / "LockedExperiment"
    experiment.mkdir()
    calls = []

    class Warmup:
        def cancel_and_wait(self):
            calls.append("cancel")

    gui = SimpleNamespace(
        output_path=str(tmp_path),
        expts_dict={"LockedExperiment": str(experiment)},
        expts_dict_keys=["LockedExperiment"],
        current_experiment=None,
        current_dataset=None,
        current_session=None,
        cache_warmup=Warmup(),
        set_current_experiment=lambda _value: None,
        set_current_dataset=lambda _value: None,
        set_current_session=lambda _value: None,
    )

    def remove(path):
        assert calls == ["cancel"]
        calls.append("remove")
        Path(path).rmdir()

    monkeypatch.setattr("monstim_gui.managers.data_manager.shutil.rmtree", remove)
    monkeypatch.setattr(DataManager, "_invalidate_catalogs", staticmethod(lambda *_paths: None))

    DataManager(gui).delete_experiment_by_id("LockedExperiment")

    assert calls == ["cancel", "remove"]
    assert "LockedExperiment" not in gui.expts_dict
