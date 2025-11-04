from monstim_gui.core.clipboard import LatencyWindowClipboard
from monstim_signals.core import LatencyWindow


def test_latency_window_clipboard_set_get_clear():
    # Ensure clean state
    LatencyWindowClipboard.clear()
    assert not LatencyWindowClipboard.has()

    windows = [
        LatencyWindow(name="W1", color="black", start_times=[0.0, 1.0], durations=[5.0, 5.0]),
        LatencyWindow(name="W2", color="red", start_times=[2.0, 2.5], durations=[3.0, 3.0]),
    ]
    LatencyWindowClipboard.set(windows)
    assert LatencyWindowClipboard.has()
    assert LatencyWindowClipboard.count() == 2

    fetched = LatencyWindowClipboard.get()
    assert fetched is not windows  # should be deep copies
    assert fetched[0].name == "W1"
    assert fetched[1].start_times[0] == 2.0

    # Mutate fetched and ensure original clipboard isn't altered
    fetched[0].name = "Changed"
    again = LatencyWindowClipboard.get()
    assert again[0].name == "W1"

    LatencyWindowClipboard.clear()
    assert not LatencyWindowClipboard.has()
