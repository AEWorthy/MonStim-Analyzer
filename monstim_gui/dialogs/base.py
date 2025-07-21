import logging
import os
import yaml
import copy

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QDialogButtonBox,
    QMessageBox,
    QHBoxLayout,
    QTextEdit,
    QPushButton,
    QApplication,
    QTextBrowser,
    QWidget,
    QFormLayout,
    QGroupBox,
    QScrollArea,
    QSizePolicy,
    QCheckBox, 
    QTabWidget,
    QComboBox,
    QDoubleSpinBox,
    QLayout
)
from PyQt6.QtGui import QPixmap, QFont, QIcon, QDesktopServices
from PyQt6.QtCore import Qt, QUrl, pyqtSlot, QEvent, QTimer, QSize, QPoint
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineScript
import markdown
from markdown.extensions.codehilite import CodeHiliteExtension
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.tables import TableExtension
from mdx_math import MathExtension
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib import colors as mcolors

from monstim_signals.core import get_source_path, CustomYAMLLoader, LatencyWindow
from monstim_signals.domain.dataset import Dataset
from monstim_signals.domain.session import Session
from monstim_signals.domain.experiment import Experiment
from monstim_gui.commands import SetLatencyWindowsCommand
from monstim_gui.core.splash import SPLASH_INFO

# Small set of pleasant colors for latency windows
COLOR_OPTIONS = list(mcolors.TABLEAU_COLORS.keys())
TAB_COLOR_NAMES = [c.replace("tab:", "") for c in COLOR_OPTIONS]


class WebEnginePage(QWebEnginePage):
    """Custom WebEnginePage to handle JavaScript messages."""

    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
        print(f"JS: {message}")

