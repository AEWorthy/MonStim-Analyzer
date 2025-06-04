# monstim_signals/domain/dataset.py
from typing import List, Any

from monstim_signals.domain.session import Session

class Dataset:
    """
    A “dataset” = all sessions from one animal replicate.
    E.g. Dataset_1(Animal_A) has sessions AA00, AA01, …
    """
    def __init__(
        self,
        dataset_id: str,
        sessions  : List[Session],
        repo      : Any = None
    ):
        self.id = dataset_id
        self.sessions   = sessions
        self.repo       = repo

    @property
    def num_sessions(self) -> int:
        return len(self.sessions)

    # ──────────────────────────────────────────────────────────────────
    # 1) Useful properties for GUI & analysis code
    # ──────────────────────────────────────────────────────────────────
    # Example: gather all H‐reflex peak amplitudes for channel 0,
    #    across every session & every stimulus group. Returns a dict:
    #        { "Session_01": [resp1, resp2, …], "Session_02": [ … ], … }
    # ──────────────────────────────────────────────────────────────────
    def session_response_map(self, channel: int, window) -> dict[str, list[float]]:
        result = {}
        for sess in self.sessions:
            curve = sess.response_curve(channel=channel, window=window)
            result[sess.id] = curve
        return result
    
    # ──────────────────────────────────────────────────────────────────
    # 2) Clean up
    # ──────────────────────────────────────────────────────────────────
    def close(self) -> None:
        """
        Close all sessions in the dataset.
        This is a placeholder for any cleanup logic needed.
        """
        for sess in self.sessions:
            if hasattr(sess, 'close'):
                sess.close()
            else:
                raise NotImplementedError(f"Session {sess.id} does not have a close method.")
    # ──────────────────────────────────────────────────────────────────
    # 3) Object representation
    # ──────────────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        return f"Dataset(dataset_id={self.id}, num_sessions={self.num_sessions})"
    def __str__(self) -> str:
        return f"Dataset: {self.id} with {self.num_sessions} sessions"
    def __len__(self) -> int:
        return self.num_sessions