"""Qt lifetime requirements for GUI tests."""

import pytest


@pytest.fixture(autouse=True)
def _qapplication_for_gui_tests(qapplication):
    """Ensure every GUI test shares the retained session QApplication."""
    yield qapplication
