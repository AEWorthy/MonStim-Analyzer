"""
Core UI utilities for responsive design and scaling.
"""

from .responsive_widgets import (
    CollapsibleGroupBox,
    ResponsiveComboBox,
    ResponsiveScrollArea,
)
from .ui_scaling import setup_dpi_awareness

__all__ = [
    "CollapsibleGroupBox",
    "ResponsiveComboBox",
    "ResponsiveScrollArea",
    "setup_dpi_awareness",
]
