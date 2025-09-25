"""Clipboard utilities for transient (in-session) data sharing across dialogs.

Currently provides an in-memory clipboard for `LatencyWindow` objects so a user
can copy latency window configurations from one Session/Dataset/Experiment
scope and paste/apply them at another scope without having to create a preset.

Design constraints:
 - Data should NOT persist across program restarts (no disk writes).
 - Must be deep-copied on set/get to avoid accidental mutation.
 - Lightweight and GUI-agnostic (no PyQt imports here) so it can be tested easily.

Potential future extensions could generalize this to multiple data types or
support a history stack; for now we keep it intentionally minimal.
"""

from __future__ import annotations

import copy
from typing import List, Optional

from monstim_signals.core import LatencyWindow


class LatencyWindowClipboard:
    """Static clipboard for latency windows.

    Usage:
        LatencyWindowClipboard.set(windows)
        if LatencyWindowClipboard.has():
            wins = LatencyWindowClipboard.get()
    """

    _windows: Optional[list[LatencyWindow]] = None

    @classmethod
    def set(cls, windows: List[LatencyWindow]):
        # Store deep copies to protect internal state
        cls._windows = [copy.deepcopy(w) for w in windows]

    @classmethod
    def clear(cls):
        cls._windows = None

    @classmethod
    def has(cls) -> bool:
        return cls._windows is not None and len(cls._windows) > 0

    @classmethod
    def get(cls) -> list[LatencyWindow] | None:
        if cls._windows is None:
            return None
        return [copy.deepcopy(w) for w in cls._windows]

    @classmethod
    def count(cls) -> int:
        return len(cls._windows) if cls._windows else 0
