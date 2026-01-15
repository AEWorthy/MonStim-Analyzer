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
import time
from typing import List, Optional, Literal

from monstim_signals.core import LatencyWindow


class LatencyWindowClipboard:
    """Unified clipboard for latency windows (single or multiple).
    
    Tracks both single-window and multi-window clipboard data along with
    timestamps to determine which was set most recently.
    
    Usage:
        # Set single window
        LatencyWindowClipboard.set_single(window)
        
        # Set multiple windows
        LatencyWindowClipboard.set_multiple(windows)
        
        # Check what's available
        if LatencyWindowClipboard.has_single():
            win = LatencyWindowClipboard.get_single()
        
        # Get most recent
        mode, data = LatencyWindowClipboard.get_most_recent()
    """

    _single_window: Optional[LatencyWindow] = None
    _single_timestamp: float = 0.0
    
    _multiple_windows: Optional[list[LatencyWindow]] = None
    _multiple_timestamp: float = 0.0

    @classmethod
    def set_single(cls, window: LatencyWindow):
        """Store a single window with timestamp."""
        cls._single_window = copy.deepcopy(window)
        cls._single_timestamp = time.time()

    @classmethod
    def set_multiple(cls, windows: List[LatencyWindow]):
        """Store multiple windows with timestamp."""
        cls._multiple_windows = [copy.deepcopy(w) for w in windows]
        cls._multiple_timestamp = time.time()

    @classmethod
    def clear(cls):
        """Clear all clipboard data."""
        cls._single_window = None
        cls._single_timestamp = 0.0
        cls._multiple_windows = None
        cls._multiple_timestamp = 0.0

    @classmethod
    def clear_single(cls):
        """Clear only single-window clipboard."""
        cls._single_window = None
        cls._single_timestamp = 0.0

    @classmethod
    def clear_multiple(cls):
        """Clear only multi-window clipboard."""
        cls._multiple_windows = None
        cls._multiple_timestamp = 0.0

    @classmethod
    def has_single(cls) -> bool:
        """Check if a single window is in clipboard."""
        return cls._single_window is not None

    @classmethod
    def has_multiple(cls) -> bool:
        """Check if multiple windows are in clipboard."""
        return cls._multiple_windows is not None and len(cls._multiple_windows) > 0

    @classmethod
    def has_any(cls) -> bool:
        """Check if any clipboard data exists."""
        return cls.has_single() or cls.has_multiple()

    @classmethod
    def get_single(cls) -> LatencyWindow | None:
        """Get a deep copy of the single window."""
        if cls._single_window is None:
            return None
        return copy.deepcopy(cls._single_window)

    @classmethod
    def get_multiple(cls) -> list[LatencyWindow] | None:
        """Get deep copies of multiple windows."""
        if cls._multiple_windows is None:
            return None
        return [copy.deepcopy(w) for w in cls._multiple_windows]

    @classmethod
    def get_most_recent(cls) -> tuple[Literal["single", "multiple", "none"], LatencyWindow | list[LatencyWindow] | None]:
        """Get the most recently set clipboard data.
        
        Returns:
            Tuple of (mode, data) where:
            - mode: "single", "multiple", or "none"
            - data: LatencyWindow, list[LatencyWindow], or None
        """
        if not cls.has_any():
            return ("none", None)
        
        if cls.has_single() and not cls.has_multiple():
            return ("single", cls.get_single())
        
        if cls.has_multiple() and not cls.has_single():
            return ("multiple", cls.get_multiple())
        
        # Both exist, return most recent
        if cls._single_timestamp > cls._multiple_timestamp:
            return ("single", cls.get_single())
        else:
            return ("multiple", cls.get_multiple())

    @classmethod
    def count_multiple(cls) -> int:
        """Get count of windows in multi-window clipboard."""
        return len(cls._multiple_windows) if cls._multiple_windows else 0

