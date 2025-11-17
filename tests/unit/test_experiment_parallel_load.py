from pathlib import Path

from monstim_signals.core.utils import load_config
from monstim_signals.io.repositories import ExperimentRepository
from tests.helpers import create_minimal_dataset_folder


def test_experiment_parallel_load(tmp_path: Path):
    # Create a temporary experiment with two datasets and one session each
    exp = tmp_path / "exp1"
    exp.mkdir()

    # dataset 1
    create_minimal_dataset_folder(exp, dataset_name="dsA", num_recordings=2, num_samples=100)
    # dataset 2
    create_minimal_dataset_folder(exp, dataset_name="dsB", num_recordings=2, num_samples=100)

    # Load with lazy_open and two workers
    cfg = load_config()
    cfg.update({"lazy_open_h5": True, "load_workers": 2})
    repo = ExperimentRepository(exp)
    expt = repo.load(config=cfg)

    assert expt is not None
    assert len(expt.datasets) == 2

    # Recordings should still be accessible (lazy reopen)
    for ds in expt.datasets:
        for session in ds.sessions:
            for rec in session.recordings:
                view = rec.raw_view(t=slice(0, 10))
                assert view.shape[0] == 10
