"""
Custom GUI elements with improved scaling and text handling capabilities.
"""

from PyQt6.QtWidgets import (
    QComboBox,
    QLabel,
    QScrollArea,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSizePolicy,
    QFrame,
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFontMetrics
from .ui_scaling import ui_scaling


class ResponsiveComboBox(QComboBox):
    """
    A combo box that handles long text gracefully and scales responsively.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setToolTipDuration(3000)  # Show tooltip for 3 seconds
        self._setup_styling()

    def _setup_styling(self):
        """Set up responsive styling for the combo box."""
        # Set minimum width based on UI scaling
        min_width = ui_scaling.scale_size(120)
        self.setMinimumWidth(min_width)

        # Enable text elision and tooltip on hover
        self.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.setMinimumContentsLength(15)  # Show at least 15 characters

    def addItem(self, text, userData=None):
        """Add item with automatic tooltip for long text."""
        super().addItem(text, userData)

        # Add tooltip if text is longer than what can be displayed
        font_metrics = QFontMetrics(self.font())
        available_width = self.width() - ui_scaling.scale_size(
            40
        )  # Account for dropdown arrow

        if font_metrics.horizontalAdvance(text) > available_width:
            self.setItemData(self.count() - 1, text, Qt.ItemDataRole.ToolTipRole)

    def showPopup(self):
        """Override to ensure popup is wide enough for content."""
        super().showPopup()

        # Make popup at least as wide as the longest item
        if self.view():
            max_width = 0
            font_metrics = QFontMetrics(self.font())

            for i in range(self.count()):
                text = self.itemText(i)
                text_width = font_metrics.horizontalAdvance(text)
                max_width = max(max_width, text_width)

            # Add padding and ensure minimum size
            popup_width = max(max_width + ui_scaling.scale_size(40), self.width())
            popup_width = min(
                popup_width, ui_scaling.scale_size(400)
            )  # Cap maximum width

            self.view().setMinimumWidth(popup_width)


class ResponsiveScrollArea(QScrollArea):
    """
    A scroll area that provides better handling of dynamic content sizing.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_scrollarea()

    def _setup_scrollarea(self):
        """Set up the scroll area with responsive properties."""
        self.setWidgetResizable(True)
        self.setFrameStyle(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Set minimum size based on UI scaling
        min_width = ui_scaling.scale_size(200)
        min_height = ui_scaling.scale_size(100)
        self.setMinimumSize(min_width, min_height)

        # Set size policy to expand
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def setWidget(self, widget):
        """Override to ensure proper size policies on the contained widget."""
        if widget:
            widget.setSizePolicy(
                QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
            )
        super().setWidget(widget)


class CollapsibleGroupBox(QFrame):
    """
    A collapsible group box that can expand/collapse to save space.
    """

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self._title = title
        self._collapsed = False
        self._content_widget = None
        self._setup_ui()

    def _setup_ui(self):
        """Set up the collapsible group box UI."""
        self.setFrameStyle(QFrame.Shape.StyledPanel)

        # Main layout
        self._main_layout = QVBoxLayout(self)
        margins = ui_scaling.scale_size(4)
        spacing = ui_scaling.scale_size(2)
        self._main_layout.setContentsMargins(margins, margins, margins, margins)
        self._main_layout.setSpacing(spacing)

        # Title bar (clickable)
        self._title_frame = QFrame()
        self._title_frame.setFrameStyle(QFrame.Shape.Box)
        self._title_frame.setCursor(Qt.CursorShape.PointingHandCursor)

        title_layout = QHBoxLayout(self._title_frame)
        margins = ui_scaling.scale_size(4)
        title_layout.setContentsMargins(margins, margins, margins, margins)

        self._title_label = QLabel(self._title)
        self._title_label.setStyleSheet("font-weight: bold;")

        self._collapse_indicator = QLabel("▼")  # Down arrow when expanded

        title_layout.addWidget(self._title_label)
        title_layout.addStretch()
        title_layout.addWidget(self._collapse_indicator)

        self._main_layout.addWidget(self._title_frame)

        # Content area
        self._content_area = QFrame()
        self._content_layout = QVBoxLayout(self._content_area)
        self._content_layout.setContentsMargins(0, 0, 0, 0)

        self._main_layout.addWidget(self._content_area)

        # Connect click event
        self._title_frame.mousePressEvent = self._toggle_collapsed

    def _toggle_collapsed(self, event):
        """Toggle the collapsed state."""
        self._collapsed = not self._collapsed
        self._content_area.setVisible(not self._collapsed)
        self._collapse_indicator.setText("▶" if self._collapsed else "▼")

        # Emit a signal or trigger parent layout update if needed
        if self.parent():
            self.parent().update()

    def setContentWidget(self, widget: QWidget):
        """Set the content widget."""
        if self._content_widget:
            self._content_layout.removeWidget(self._content_widget)
            self._content_widget.setParent(None)

        self._content_widget = widget
        if widget:
            self._content_layout.addWidget(widget)

    def isCollapsed(self) -> bool:
        """Check if the group box is collapsed."""
        return self._collapsed

    def setCollapsed(self, collapsed: bool):
        """Set the collapsed state."""
        if self._collapsed != collapsed:
            self._toggle_collapsed(None)


class ResponsiveLabel(QLabel):
    """
    A label that handles text wrapping and scaling better.
    """

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self._setup_label()

    def _setup_label(self):
        """Set up responsive label properties."""
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        # Scale font size
        font = self.font()
        scaled_size = ui_scaling.scale_font_size(font.pointSize())
        font.setPointSize(scaled_size)
        self.setFont(font)

    def minimumSizeHint(self) -> QSize:
        """Provide better minimum size hint."""
        font_metrics = QFontMetrics(self.font())
        text_width = font_metrics.horizontalAdvance(self.text())
        text_height = font_metrics.height()

        # Allow for some reasonable wrapping
        max_width = ui_scaling.scale_size(200)
        if text_width > max_width:
            # Estimate wrapped height
            lines = (text_width // max_width) + 1
            text_height *= lines
            text_width = max_width

        return QSize(text_width, text_height)
