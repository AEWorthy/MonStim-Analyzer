"""
Configuration file for UI scaling and responsive design settings.
"""
from typing import Dict, Any
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QSettings


class UIConfig:
    """Configuration manager for UI scaling and responsive design settings."""
    
    def __init__(self):
        self.settings = QSettings()
        self._load_defaults()
    
    def _load_defaults(self) -> Dict[str, Any]:
        """Load default UI configuration values."""
        return {
            # Scaling settings
            'auto_scale': True,
            'manual_scale_factor': 1.0,
            'min_scale_factor': 0.8,
            'max_scale_factor': 3.0,
            
            # Panel sizing
            'left_panel_min_width': 300,
            'left_panel_max_width': 600,
            'left_panel_preferred_width_percent': 22,  # % of screen width
            
            # Font settings
            'base_font_size': 9,
            'max_font_scale': 1.5,
            
            # Spacing and margins
            'base_spacing': 6,
            'base_margin': 8,
            'form_horizontal_spacing': 8,
            'form_vertical_spacing': 4,
            
            # Combo box settings
            'combo_min_width': 120,
            'combo_min_contents_length': 15,
            'combo_popup_max_width': 400,
            'combo_tooltip_duration': 3000,
            
            # Window settings
            'window_base_width': 800,
            'window_base_height': 770,
            'window_max_screen_percent': 80,  # Max % of screen to use
            'center_windows': True,
            
            # Scroll area settings
            'scroll_area_min_width': 200,
            'scroll_area_min_height': 100,
            
            # Plot settings
            'plot_pane_min_width': 400,
            'plot_pane_min_height': 300,
        }
    
    def get(self, key: str, default=None):
        """Get a configuration value."""
        defaults = self._load_defaults()
        if default is None:
            default = defaults.get(key)
        
        return self.settings.value(f"UI/{key}", default, type=type(default))
    
    def set(self, key: str, value: Any):
        """Set a configuration value."""
        self.settings.setValue(f"UI/{key}", value)
        self.settings.sync()
    
    def get_scale_factor(self) -> float:
        """Get the current scale factor."""
        if self.get('auto_scale', True):
            from .ui_scaling import ui_scaling
            return ui_scaling.scale_factor
        else:
            return float(self.get('manual_scale_factor', 1.0))
    
    def apply_high_dpi_settings(self):
        """Apply high DPI settings to the application."""
        app = QApplication.instance()
        if not app:
            return
            
        # Enable high DPI scaling
        app.setAttribute(app.ApplicationAttribute.AA_EnableHighDpiScaling, True)
        app.setAttribute(app.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
        
        # Set font DPI scaling
        if hasattr(app, 'setHighDpiScaleFactorRoundingPolicy'):
            # PyQt6 specific
            from PyQt6.QtCore import Qt
            app.setHighDpiScaleFactorRoundingPolicy(
                Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
            )
    
    def get_responsive_size(self, base_size: int, size_type: str = 'general') -> int:
        """Get a responsive size based on the configuration and scale factor."""
        scale = self.get_scale_factor()
        
        # Apply different scaling rules based on size type
        if size_type == 'font':
            max_scale = self.get('max_font_scale', 1.5)
            scale = min(scale, max_scale)
        elif size_type == 'spacing':
            # Spacing scales more conservatively
            scale = 1.0 + (scale - 1.0) * 0.7
        elif size_type == 'margin':
            # Margins scale even more conservatively
            scale = 1.0 + (scale - 1.0) * 0.5
        
        return max(1, int(base_size * scale))
    
    def get_window_geometry(self) -> tuple:
        """Get optimal window geometry based on current settings."""
        from .ui_scaling import ui_scaling
        
        base_width = self.get('window_base_width', 800)
        base_height = self.get('window_base_height', 770)
        
        width, height = ui_scaling.get_window_geometry(base_width, base_height)
        
        if self.get('center_windows', True):
            geometry = ui_scaling.get_centered_geometry(width, height)
            return geometry.x(), geometry.y(), width, height
        else:
            return 30, 30, width, height
    
    def save_window_state(self, window, key: str = 'main_window'):
        """Save window geometry and state."""
        if hasattr(window, 'saveGeometry') and hasattr(window, 'saveState'):
            self.settings.setValue(f"WindowState/{key}/geometry", window.saveGeometry())
            if hasattr(window, 'saveState'):
                self.settings.setValue(f"WindowState/{key}/state", window.saveState())
            self.settings.sync()
    
    def restore_window_state(self, window, key: str = 'main_window') -> bool:
        """Restore window geometry and state. Returns True if successful."""
        try:
            geometry = self.settings.value(f"WindowState/{key}/geometry")
            state = self.settings.value(f"WindowState/{key}/state")
            
            restored = False
            if geometry and hasattr(window, 'restoreGeometry'):
                restored = window.restoreGeometry(geometry)
            
            if state and hasattr(window, 'restoreState'):
                window.restoreState(state)
                restored = True
                
            return restored
        except Exception:
            return False


# Global configuration instance
ui_config = UIConfig()
