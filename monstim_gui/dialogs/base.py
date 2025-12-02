from matplotlib import colors as mcolors

# Conditional import for WebEngine
try:
    from PySide6.QtWebEngineCore import QWebEnginePage

    WEB_ENGINE_AVAILABLE = True
except ImportError:
    QWebEnginePage = None
    WEB_ENGINE_AVAILABLE = False

# Small set of pleasant colors for latency windows
COLOR_OPTIONS = list(mcolors.TABLEAU_COLORS.keys())
TAB_COLOR_NAMES = [c.replace("tab:", "") for c in COLOR_OPTIONS]


if WEB_ENGINE_AVAILABLE:

    class WebEnginePage(QWebEnginePage):
        """Custom WebEnginePage to handle JavaScript messages."""

        def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
            print(f"JS: {message}")

else:
    # Placeholder class when WebEngine is not available
    class WebEnginePage:
        def __init__(self, *args, **kwargs):
            pass
