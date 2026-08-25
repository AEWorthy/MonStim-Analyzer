from PySide6.QtCore import QObject

from monstim_gui.core.load_policy import LoadPolicy, WarmUpLevelPolicy
from monstim_gui.managers.cache_warmup import CacheWarmUpCoordinator


class _Session:
    def __init__(self, session_id):
        self.id = session_id
        self.default_method = "rms"


class _Dataset:
    def __init__(self, dataset_id, sessions):
        self.id = dataset_id
        self.sessions = sessions


class _Experiment:
    def __init__(self, datasets):
        self.id = "experiment"
        self.datasets = datasets


def _coordinator(monkeypatch, policy):
    current = _Session("current")
    sibling = _Session("sibling")
    remaining = _Session("remaining")
    current_dataset = _Dataset("current_dataset", [current, sibling])
    other_dataset = _Dataset("other_dataset", [remaining])
    gui = QObject()
    gui.current_session = current
    gui.current_dataset = current_dataset
    gui.current_experiment = _Experiment([current_dataset, other_dataset])
    monkeypatch.setattr("monstim_gui.managers.cache_warmup.app_state.get_load_policy", lambda: policy)
    return CacheWarmUpCoordinator(gui)


def test_default_policy_creates_no_warmup_tasks(monkeypatch):
    assert _coordinator(monkeypatch, LoadPolicy()).build_tasks() == ()


def test_overlapping_policies_deduplicate_and_prioritize_current_session(monkeypatch):
    policy = LoadPolicy(
        session=WarmUpLevelPolicy(True, True, ("rms",), False, False),
        dataset=WarmUpLevelPolicy(True, True, ("extrema_ptt",), True, True),
        experiment=WarmUpLevelPolicy(True, True, ("rms",), False, False),
    )
    tasks = _coordinator(monkeypatch, policy).build_tasks()
    session_tasks = [task for task in tasks if isinstance(task.target, _Session)]
    assert [task.target.id for task in session_tasks] == ["current", "sibling", "remaining"]
    assert len({id(task.target) for task in session_tasks}) == 3
    assert set(session_tasks[0].methods) == {"rms", "extrema_ptt"}
    assert {"filtered_signals", "window_results", "mmax", "amplitudes"} <= session_tasks[0].products
    jobs = {methods[0]: products for products, methods in session_tasks[0].jobs}
    assert "mmax" not in jobs["rms"]
    assert {"mmax", "amplitudes"} <= jobs["extrema_ptt"]
