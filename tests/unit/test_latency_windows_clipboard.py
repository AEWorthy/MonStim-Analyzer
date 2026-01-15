import time

from monstim_gui.core.clipboard import LatencyWindowClipboard
from monstim_signals.core import LatencyWindow


def test_single_window_clipboard():
    """Test single-window clipboard operations."""
    LatencyWindowClipboard.clear()
    assert not LatencyWindowClipboard.has_single()
    assert not LatencyWindowClipboard.has_any()

    window = LatencyWindow(name="H-reflex", color="black", start_times=[0.0, 1.0], durations=[5.0, 5.0])
    LatencyWindowClipboard.set_single(window)
    
    assert LatencyWindowClipboard.has_single()
    assert LatencyWindowClipboard.has_any()
    assert not LatencyWindowClipboard.has_multiple()

    fetched = LatencyWindowClipboard.get_single()
    assert fetched is not window  # should be a deep copy
    assert fetched.name == "H-reflex"
    assert fetched.start_times[0] == 0.0

    # Mutate fetched and ensure original clipboard isn't altered
    fetched.name = "Changed"
    again = LatencyWindowClipboard.get_single()
    assert again.name == "H-reflex"

    LatencyWindowClipboard.clear_single()
    assert not LatencyWindowClipboard.has_single()


def test_multiple_windows_clipboard():
    """Test multi-window clipboard operations."""
    LatencyWindowClipboard.clear()
    assert not LatencyWindowClipboard.has_multiple()
    assert not LatencyWindowClipboard.has_any()

    windows = [
        LatencyWindow(name="W1", color="black", start_times=[0.0, 1.0], durations=[5.0, 5.0]),
        LatencyWindow(name="W2", color="red", start_times=[2.0, 2.5], durations=[3.0, 3.0]),
    ]
    LatencyWindowClipboard.set_multiple(windows)
    
    assert LatencyWindowClipboard.has_multiple()
    assert LatencyWindowClipboard.has_any()
    assert not LatencyWindowClipboard.has_single()
    assert LatencyWindowClipboard.count_multiple() == 2

    fetched = LatencyWindowClipboard.get_multiple()
    assert fetched is not windows  # should be deep copies
    assert fetched[0].name == "W1"
    assert fetched[1].start_times[0] == 2.0

    # Mutate fetched and ensure original clipboard isn't altered
    fetched[0].name = "Changed"
    again = LatencyWindowClipboard.get_multiple()
    assert again[0].name == "W1"

    LatencyWindowClipboard.clear_multiple()
    assert not LatencyWindowClipboard.has_multiple()


def test_get_most_recent_single_only():
    """Test get_most_recent when only single window exists."""
    LatencyWindowClipboard.clear()
    
    window = LatencyWindow(name="Test", color="black", start_times=[0.0], durations=[1.0])
    LatencyWindowClipboard.set_single(window)
    
    mode, data = LatencyWindowClipboard.get_most_recent()
    assert mode == "single"
    assert data.name == "Test"


def test_get_most_recent_multiple_only():
    """Test get_most_recent when only multiple windows exist."""
    LatencyWindowClipboard.clear()
    
    windows = [
        LatencyWindow(name="W1", color="black", start_times=[0.0], durations=[1.0]),
        LatencyWindow(name="W2", color="red", start_times=[0.0], durations=[1.0]),
    ]
    LatencyWindowClipboard.set_multiple(windows)
    
    mode, data = LatencyWindowClipboard.get_most_recent()
    assert mode == "multiple"
    assert len(data) == 2
    assert data[0].name == "W1"


def test_get_most_recent_none():
    """Test get_most_recent when clipboard is empty."""
    LatencyWindowClipboard.clear()
    
    mode, data = LatencyWindowClipboard.get_most_recent()
    assert mode == "none"
    assert data is None


def test_get_most_recent_timestamp_single_first():
    """Test get_most_recent returns single when it was set most recently."""
    LatencyWindowClipboard.clear()
    
    windows = [LatencyWindow(name="W1", color="black", start_times=[0.0], durations=[1.0])]
    LatencyWindowClipboard.set_multiple(windows)
    
    time.sleep(0.01)  # Ensure timestamp difference
    
    single = LatencyWindow(name="Single", color="red", start_times=[0.0], durations=[1.0])
    LatencyWindowClipboard.set_single(single)
    
    mode, data = LatencyWindowClipboard.get_most_recent()
    assert mode == "single"
    assert data.name == "Single"


def test_get_most_recent_timestamp_multiple_first():
    """Test get_most_recent returns multiple when it was set most recently."""
    LatencyWindowClipboard.clear()
    
    single = LatencyWindow(name="Single", color="red", start_times=[0.0], durations=[1.0])
    LatencyWindowClipboard.set_single(single)
    
    time.sleep(0.01)  # Ensure timestamp difference
    
    windows = [LatencyWindow(name="W1", color="black", start_times=[0.0], durations=[1.0])]
    LatencyWindowClipboard.set_multiple(windows)
    
    mode, data = LatencyWindowClipboard.get_most_recent()
    assert mode == "multiple"
    assert len(data) == 1
    assert data[0].name == "W1"


def test_clear_operations():
    """Test various clear operations."""
    LatencyWindowClipboard.clear()
    
    # Set both types
    single = LatencyWindow(name="Single", color="red", start_times=[0.0], durations=[1.0])
    windows = [LatencyWindow(name="W1", color="black", start_times=[0.0], durations=[1.0])]
    
    LatencyWindowClipboard.set_single(single)
    LatencyWindowClipboard.set_multiple(windows)
    
    assert LatencyWindowClipboard.has_single()
    assert LatencyWindowClipboard.has_multiple()
    
    # Clear only single
    LatencyWindowClipboard.clear_single()
    assert not LatencyWindowClipboard.has_single()
    assert LatencyWindowClipboard.has_multiple()
    
    # Restore single
    LatencyWindowClipboard.set_single(single)
    
    # Clear only multiple
    LatencyWindowClipboard.clear_multiple()
    assert LatencyWindowClipboard.has_single()
    assert not LatencyWindowClipboard.has_multiple()
    
    # Restore multiple
    LatencyWindowClipboard.set_multiple(windows)
    
    # Clear all
    LatencyWindowClipboard.clear()
    assert not LatencyWindowClipboard.has_single()
    assert not LatencyWindowClipboard.has_multiple()
    assert not LatencyWindowClipboard.has_any()


def test_deep_copy_protection():
    """Test that clipboard protects against mutation via deep copying."""
    LatencyWindowClipboard.clear()
    
    # Test single window protection
    original_single = LatencyWindow(name="Original", color="black", start_times=[1.0], durations=[2.0])
    LatencyWindowClipboard.set_single(original_single)
    
    # Mutate original after setting
    original_single.name = "Mutated"
    original_single.start_times[0] = 999.0
    
    # Clipboard should still have original values
    fetched = LatencyWindowClipboard.get_single()
    assert fetched.name == "Original"
    assert fetched.start_times[0] == 1.0
    
    # Test multiple windows protection
    original_multiple = [
        LatencyWindow(name="W1", color="black", start_times=[1.0], durations=[2.0]),
        LatencyWindow(name="W2", color="red", start_times=[3.0], durations=[4.0]),
    ]
    LatencyWindowClipboard.set_multiple(original_multiple)
    
    # Mutate original list after setting
    original_multiple[0].name = "Mutated"
    original_multiple.append(LatencyWindow(name="W3", color="blue", start_times=[5.0], durations=[6.0]))
    
    # Clipboard should still have original values
    fetched = LatencyWindowClipboard.get_multiple()
    assert len(fetched) == 2
    assert fetched[0].name == "W1"
    assert fetched[1].name == "W2"

