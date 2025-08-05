# UI Scaling and Responsive Design Guide

## Overview

This guide explains the comprehensive UI scaling solution implemented to handle different screen resolutions and DPI settings, preventing widget cutoffs and ensuring readable text across various display configurations.

## Problem Statement

The original GUI had several scaling issues:
1. **Fixed widths**: Left panel had a fixed 370px width that didn't adapt to different screen sizes
2. **No DPI awareness**: Application didn't handle high DPI displays properly
3. **Long text compression**: Plot options with long text (like "Reflex Amplitude Calculation Method") became unreadable when compressed
4. **No responsive sizing**: Widget sizes didn't adapt to screen resolution

## Solution Components

### 1. UI Scaling Engine (`monstim_gui/core/ui_scaling.py`)

The `UIScaling` class provides:
- **Automatic DPI detection**: Calculates scale factors based on screen DPI and resolution
- **Reasonable bounds**: Applies scale factors between 1.0x and 3.0x with 0.25 increments
- **Responsive panel sizing**: Calculates optimal panel widths as percentages of screen width
- **Centered window positioning**: Places windows optimally on screen

Key methods:
- `scale_size(size)`: Scale any pixel value appropriately
- `get_optimal_panel_width()`: Calculate responsive panel width
- `get_window_geometry()`: Get optimal window dimensions
- `get_centered_geometry()`: Center windows on screen

### 2. Configuration System (`monstim_gui/core/ui_config.py`)

The `UIConfig` class manages:
- **Persistent settings**: Saves user preferences using QSettings
- **Default values**: Provides sensible defaults for all scaling parameters
- **Window state management**: Saves/restores window positions and sizes
- **Flexible scaling options**: Supports both automatic and manual scaling

Key settings:
- `auto_scale`: Enable/disable automatic DPI scaling
- `left_panel_preferred_width_percent`: Panel width as % of screen
- `base_font_size`: Base application font size
- `max_font_scale`: Maximum font scaling factor

### 3. Responsive Widgets (`monstim_gui/core/responsive_widgets.py`)

Custom widget classes that handle scaling gracefully:

#### `ResponsiveComboBox`
- **Dynamic popup width**: Expands popup to fit longest item text
- **Automatic tooltips**: Shows full text as tooltip when truncated
- **Minimum width scaling**: Scales minimum width based on DPI
- **Text elision**: Handles long text gracefully

#### `ResponsiveScrollArea`
- **Flexible sizing**: Adapts to content with proper scroll bars
- **Minimum size scaling**: Scales minimum dimensions appropriately
- **Frame optimization**: Removes unnecessary frames for cleaner look

#### `CollapsibleGroupBox`
- **Space saving**: Allows collapsing sections to save vertical space
- **Click to toggle**: Simple click interface for expand/collapse
- **Visual indicators**: Clear arrows showing state

### 4. Layout Improvements

Updated layouts throughout the application:
- **Responsive margins and spacing**: Scale based on DPI
- **Flexible panel widths**: Use min/max constraints instead of fixed widths
- **Scroll areas for long content**: Prevent cutoffs in plot options
- **Proper size policies**: Allow widgets to expand/contract appropriately

## Implementation Details

### DPI Awareness Setup

In `main.py`, DPI awareness is enabled before creating the QApplication:

```python
from monstim_gui.core.ui_scaling import setup_dpi_awareness
setup_dpi_awareness()
app = QApplication(sys.argv)
```

This enables:
- High DPI scaling on Windows/macOS/Linux
- Proper pixmap scaling for crisp icons
- Automatic font DPI adjustments

### Window Management

The main window now:
1. **Tries to restore** previous size/position from saved settings
2. **Falls back to responsive sizing** if no saved state exists
3. **Saves state on close** for next application start

### Plot Options Improvements

The most problematic area (plot options with long text) now uses:
- `ResponsiveComboBox` for all dropdown menus
- `ResponsiveScrollArea` wrapping the options content
- Proper tooltips showing full text when truncated
- Dynamic popup sizing to fit content

## Usage Examples

### Scaling a Size Value
```python
from monstim_gui.core.ui_scaling import ui_scaling

# Scale a 10px margin
scaled_margin = ui_scaling.scale_size(10)

# Scale with bounds (for fonts)
scaled_font = ui_scaling.scale_font_size(9)
```

### Using Responsive Widgets
```python
from monstim_gui.core.responsive_widgets import ResponsiveComboBox

# Instead of QComboBox()
combo = ResponsiveComboBox()
combo.addItem("Very Long Text That Might Not Fit")
# Automatically handles tooltips and popup sizing
```

### Getting Optimal Panel Width
```python
from monstim_gui.core.ui_scaling import ui_scaling

# Get width as 22% of screen, between 300-600px scaled
optimal_width = ui_scaling.get_optimal_panel_width(300, 600)
panel.setMinimumWidth(optimal_width)
panel.setMaximumWidth(int(optimal_width * 1.5))
```

## User Preferences

Users can access UI scaling preferences through the main menu:
- **Auto-scale toggle**: Enable/disable automatic DPI detection
- **Manual scale factor**: Set custom scaling when auto-scale is off
- **Panel width preference**: Adjust left panel width percentage
- **Font settings**: Control base font size and maximum scaling

Settings are persistent and take effect after application restart.

## Testing Different Scenarios

The solution handles various scenarios:

### High DPI Displays (150%, 200%, 300% scaling)
- Automatically detects system DPI settings
- Scales all UI elements proportionally
- Maintains readable text and clickable targets

### Very High Resolution Displays (4K, 5K, 8K)
- Uses resolution-based scaling in addition to DPI
- Prevents tiny UI elements on large screens
- Calculates optimal panel sizes as screen percentages

### Small/Low Resolution Displays
- Ensures minimum sizes are maintained
- Uses scroll areas to prevent cutoffs
- Allows collapsing sections to save space

### Mixed DPI Setups (Multiple Monitors)
- Uses primary screen for scaling calculations
- Handles window movement between screens gracefully
- Maintains consistent scaling across sessions

## Troubleshooting

### Text Still Too Small/Large
1. Open UI Preferences dialog
2. Adjust "Base Font Size" setting
3. Modify "Max Font Scale" if needed
4. Restart application

### Panel Too Wide/Narrow
1. Open UI Preferences dialog
2. Adjust "Left Panel Width" percentage
3. Restart application

### Window Opens Off-Screen
- Delete settings file to reset window positions
- Or use UI Preferences to reset to defaults

### Combo Boxes Still Truncated
- Check that `ResponsiveComboBox` is being used instead of `QComboBox`
- Verify popup max width setting in preferences
- Hover over truncated items to see full text in tooltip

## Future Enhancements

Possible improvements for future versions:
1. **Real-time scaling**: Apply scale changes without restart
2. **Per-monitor DPI**: Better handling of mixed DPI setups
3. **Accessibility options**: Support for system accessibility scaling
4. **Theme-aware scaling**: Different scaling for light/dark themes
5. **Custom scaling profiles**: Save different configurations for different use cases

## Files Modified

- `main.py`: Added DPI awareness setup
- `monstim_gui/gui_main.py`: Responsive window sizing and state management
- `monstim_gui/widgets/gui_layout.py`: Flexible panel sizing
- `monstim_gui/plotting/plot_options.py`: ResponsiveComboBox usage
- `monstim_gui/plotting/plotting_widget.py`: Scroll area for options

## New Files Added

- `monstim_gui/core/ui_scaling.py`: Core scaling engine
- `monstim_gui/core/ui_config.py`: Configuration management
- `monstim_gui/core/responsive_widgets.py`: Custom responsive widgets
- `monstim_gui/dialogs/ui_scaling_preferences.py`: User preferences dialog
- `docs/ui_scaling_guide.md`: This documentation file
