"""
Core UI utilities for responsive design and scaling.
"""

from .ui_scaling import ui_scaling, setup_dpi_awareness
from .ui_config import ui_config
from .responsive_widgets import (
    ResponsiveComboBox,
    ResponsiveScrollArea,
    CollapsibleGroupBox,
)

__all__ = [
    "ui_scaling",
    "setup_dpi_awareness",
    "ui_config",
    "ResponsiveComboBox",
    "ResponsiveScrollArea",
    "CollapsibleGroupBox",
]
