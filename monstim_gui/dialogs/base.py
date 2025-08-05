from matplotlib import colors as mcolors
from PyQt6.QtWebEngineCore import QWebEnginePage

# Small set of pleasant colors for latency windows
COLOR_OPTIONS = list(mcolors.TABLEAU_COLORS.keys())
TAB_COLOR_NAMES = [c.replace("tab:", "") for c in COLOR_OPTIONS]


class WebEnginePage(QWebEnginePage):
    """Custom WebEnginePage to handle JavaScript messages."""

    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
        print(f"JS: {message}")
