"""
Init file for the 'managers' package.
Imports all manager classes for convenient access.
"""

from .data_manager import DataManager
from .plot_controller import PlotController, PlotControllerError
from .profile_manager import ProfileManager
from .report_manager import ReportManager


__all__ = [
    "DataManager",
    "PlotController",
    "ProfileManager",
    "ReportManager",
    "PlotControllerError",
]
