"""
UI scaling utilities for handling different screen resolutions and DPI settings.
"""
from typing import Tuple
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtCore import QRect


class UIScaling:
    """Utility class for handling UI scaling across different screen resolutions and DPI settings."""
    
    _instance = None
    _scale_factor = None
    _base_dpi = 96  # Standard Windows DPI
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._scale_factor is None:
            self._calculate_scale_factor()
    
    def _calculate_scale_factor(self):
        """Calculate the overall UI scale factor based on screen DPI and resolution."""
        app = QApplication.instance()
        if not app:
            self._scale_factor = 1.0
            return
            
        screen = app.primaryScreen()
        if not screen:
            self._scale_factor = 1.0
            return
            
        # Get DPI-based scaling
        dpi = screen.logicalDotsPerInch()
        dpi_scale = dpi / self._base_dpi
        
        # Get resolution-based scaling for very high resolution displays
        geometry = screen.geometry()
        width, height = geometry.width(), geometry.height()
        
        # Base resolution scaling on 1920x1080 as reference
        base_width, base_height = 1920, 1080
        resolution_scale = min(width / base_width, height / base_height)
        
        # Combine DPI and resolution scaling with reasonable limits
        combined_scale = max(dpi_scale, resolution_scale)
        
        # Apply reasonable bounds and rounding
        if combined_scale < 1.0:
            self._scale_factor = 1.0
        elif combined_scale > 3.0:
            self._scale_factor = 3.0
        else:
            # Round to nearest 0.25 for more predictable scaling
            self._scale_factor = round(combined_scale * 4) / 4
    
    @property
    def scale_factor(self) -> float:
        """Get the current UI scale factor."""
        if self._scale_factor is None:
            self._calculate_scale_factor()
        return self._scale_factor
    
    def scale_size(self, size: int) -> int:
        """Scale a size value according to the current scale factor."""
        return int(size * self.scale_factor)
    
    def scale_font_size(self, base_size: int) -> int:
        """Scale font size with more conservative scaling to maintain readability."""
        # Font scaling should be more conservative than UI scaling
        font_scale = min(self.scale_factor, 1.5)
        return max(8, int(base_size * font_scale))
    
    def get_optimal_panel_width(self, min_width: int = 300, max_width: int = 600) -> int:
        """Get optimal panel width based on screen size and content needs."""
        screen = QApplication.instance().primaryScreen()
        if not screen:
            return self.scale_size(370)
            
        screen_width = screen.geometry().width()
        
        # Calculate based on screen percentage (20-25% of screen width)
        percentage_width = int(screen_width * 0.22)
        
        # Apply bounds
        scaled_min = self.scale_size(min_width)
        scaled_max = self.scale_size(max_width)
        
        return max(scaled_min, min(scaled_max, percentage_width))
    
    def get_window_geometry(self, base_width: int = 800, base_height: int = 770) -> Tuple[int, int]:
        """Get optimal window size based on screen resolution."""
        screen = QApplication.instance().primaryScreen()
        if not screen:
            return self.scale_size(base_width), self.scale_size(base_height)
        
        screen_rect = screen.availableGeometry()
        screen_width, screen_height = screen_rect.width(), screen_rect.height()
        
        # Use percentage of screen size with scaling
        target_width = min(self.scale_size(base_width), int(screen_width * 0.8))
        target_height = min(self.scale_size(base_height), int(screen_height * 0.8))
        
        return target_width, target_height
    
    def get_centered_geometry(self, width: int, height: int) -> QRect:
        """Get centered window geometry for given dimensions."""
        screen = QApplication.instance().primaryScreen()
        if not screen:
            return QRect(30, 30, width, height)
        
        screen_rect = screen.availableGeometry()
        x = (screen_rect.width() - width) // 2
        y = (screen_rect.height() - height) // 2
        
        # Ensure window isn't positioned off-screen
        x = max(0, min(x, screen_rect.width() - width))
        y = max(0, min(y, screen_rect.height() - height))
        
        return QRect(x, y, width, height)
    
    def apply_responsive_sizing(self, widget: QWidget):
        """Apply responsive sizing policies to a widget."""
        from PyQt6.QtWidgets import QSizePolicy
        
        # Set size policies that allow for flexible sizing
        widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, 
            QSizePolicy.Policy.Expanding
        )
        
        # Set reasonable minimum sizes
        min_width = self.scale_size(200)
        min_height = self.scale_size(150)
        widget.setMinimumSize(min_width, min_height)


# Global instance
ui_scaling = UIScaling()


def setup_dpi_awareness():
    """Set up DPI awareness for the application."""
    import os
    
    # Enable DPI awareness on Windows
    if os.name == 'nt':
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)  # PROCESS_DPI_AWARE
        except Exception:
            pass  # Fallback gracefully if not available
    
    # Qt DPI settings
    app = QApplication.instance()
    if app:
        app.setAttribute(app.ApplicationAttribute.AA_EnableHighDpiScaling, True)
        app.setAttribute(app.ApplicationAttribute.AA_UseHighDpiPixmaps, True)


def get_responsive_margins(base_margin: int = 8) -> Tuple[int, int, int, int]:
    """Get responsive margins based on UI scaling."""
    scaled = ui_scaling.scale_size(base_margin)
    return (scaled, scaled, scaled, scaled)


def get_responsive_spacing(base_spacing: int = 6) -> int:
    """Get responsive spacing based on UI scaling."""
    return ui_scaling.scale_size(base_spacing)
