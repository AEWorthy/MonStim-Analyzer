"""
Core UI utilities for responsive design and scaling.
"""

from .responsive_widgets import (
    CollapsibleGroupBox,
    ResponsiveComboBox,
    ResponsiveScrollArea,
)
from .ui_config import ui_config
from .ui_scaling import setup_dpi_awareness, ui_scaling

__all__ = [
    "ui_scaling",
    "setup_dpi_awareness",
    "ui_config",
    "ResponsiveComboBox",
    "ResponsiveScrollArea",
    "CollapsibleGroupBox",
]
