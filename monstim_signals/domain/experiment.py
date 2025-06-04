# monstim_signals/domain/experiment.py
from typing import List, Any

from monstim_signals.domain.dataset import Dataset

class Experiment:
    """
    An “experiment” = a collection of Datasets (animals).
    E.g. Animal_A, Animal_B, Animal_C, … all under one experiment folder.
    """
    def __init__(self, expt_id: str, datasets: List[Dataset], repo: Any = None):
        self.id   = expt_id
        self.datasets = datasets
        self.repo     = repo

    @property
    def num_datasets(self) -> int:
        return len(self.datasets)

    # ──────────────────────────────────────────────────────────────────
    # 1) Useful properties for GUI & analysis code
    # ──────────────────────────────────────────────────────────────────
    # Example: gather session “H‐reflex curves” for every dataset & session:
    #    Returns a nested dict: { "Animal_A": { "Session_01": [ … ], … }, … }
    # ──────────────────────────────────────────────────────────────────
    def experiment_response_map(self, channel: int, window) -> dict[str, dict[str, list[float]]]:
        result = {}
        for ds in self.datasets:
            ds_map = ds.session_response_map(channel=channel, window=window)
            result[ds.id] = ds_map
        return result
    
    # ──────────────────────────────────────────────────────────────────
    # 2) Clean up
    # ──────────────────────────────────────────────────────────────────
    def close(self) -> None:
        """
        Close all datasets in the experiment.
        This is a placeholder for any cleanup logic needed.
        """
        for ds in self.datasets:
            if hasattr(ds, 'close_all'):
                ds.close()
            else:
                raise NotImplementedError(f"Dataset {ds.id} does not have a close_all method.")
    
    # ──────────────────────────────────────────────────────────────────
    # 3) Object representation
    # ──────────────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        return f"Experiment(expt_id={self.id}, num_datasets={self.num_datasets})"
    def __str__(self) -> str:
        return f"Experiment: '{self.id}' with {self.num_datasets} datasets"
    def __len__(self) -> int:
        return self.num_datasets

