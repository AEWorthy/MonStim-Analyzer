"""Shared presentation and interaction primitives for the Qt application."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QPoint, QRect
from PySide6.QtGui import QColor, QPainter, QPalette, QPolygon
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QProxyStyle,
    QStyle,
    QStyleOption,
    QStyleOptionSpinBox,
    QWidget,
)

APPLICATION_STYLESHEET = """
    QMainWindow { background: #20252b; }
    QWidget#mainSidebar {
        background: #242629;
        border: 1px solid #3c434b;
        border-radius: 7px;
    }
    QSplitter#mainContentSplitter::handle:horizontal {
        width: 6px;
        background: #20252b;
    }
    QSplitter#mainContentSplitter::handle:horizontal:hover { background: #4c86b8; }
    QWidget#profileSelectorRow {
        background: #2a3036;
        border: 1px solid #3c434b;
        border-radius: 5px;
    }
    QGroupBox {
        border: 1px solid #3c434b;
        border-radius: 6px;
        margin-top: 16px;
        padding-top: 10px;
        font-weight: 700;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 10px;
        padding: 0 4px;
    }
    QGroupBox::indicator { width: 18px; height: 18px; }
    QCheckBox#plotChannelSelector {
        spacing: 5px;
        padding: 2px 5px 2px 2px;
        color: #cbd4dc;
        border: 1px solid transparent;
        border-radius: 4px;
    }
    QCheckBox#plotChannelSelector:hover {
        color: #ffffff;
        background: #30363d;
        border-color: #4a5159;
    }
    QCheckBox#plotChannelSelector::indicator {
        width: 18px;
        height: 18px;
        border: 1px solid #65717c;
        border-radius: 5px;
        background: #27292c;
    }
    QCheckBox#plotChannelSelector::indicator:checked {
        border-color: #f0a15a;
        background: #e07a3f;
    }
    QCheckBox#plotChannelSelector:disabled { color: #717a82; }
    QCheckBox#plotChannelSelector::indicator:disabled {
        border-color: #4a5159;
        background: #25282b;
    }
    QGroupBox#collapsibleSidebarSection {
        margin-top: 0;
        padding-top: 30px;
    }
    QToolButton#collapsibleSectionHeader {
        padding: 4px 8px;
        border: 0;
        border-radius: 4px;
        color: #ffffff;
        background: #30363d;
        font-weight: 700;
        font-size: 13px;
        text-align: left;
    }
    QToolButton#collapsibleSectionHeader:hover { background: #3c454e; }
    QToolButton#collapsibleSectionHeader:checked { background: #36414b; }
    QTabWidget::pane {
        border: 1px solid #3c434b;
        border-radius: 6px;
        top: -1px;
    }
    QTabBar::tab {
        padding: 8px 13px;
        margin-right: 2px;
        border: 0;
        border-bottom: 3px solid transparent;
        color: #bfc7cf;
        font-weight: 600;
    }
    QTabBar::tab:selected {
        color: #ffffff;
        background: #2c333a;
        border-bottom-color: #e07a3f;
    }
    QTabBar::tab:hover:!selected { background: rgba(255, 255, 255, 0.06); color: #ffffff; }
    QComboBox, QAbstractSpinBox, QLineEdit, QTextEdit, QPlainTextEdit {
        padding: 4px 6px;
        border: 1px solid #3c434b;
        border-radius: 4px;
        background: #27292c;
    }
    QComboBox:hover, QAbstractSpinBox:hover, QLineEdit:hover, QTextEdit:hover {
        border-color: #59646f;
    }
    QComboBox:focus, QAbstractSpinBox:focus, QLineEdit:focus, QTextEdit:focus,
    QPlainTextEdit:focus { border-color: #f0a15a; }
    QPushButton {
        padding: 5px 12px;
        border: 1px solid #4a5159;
        border-radius: 4px;
        background: #34373a;
        font-weight: 600;
    }
    QPushButton:hover { background: #41464b; border-color: #65717c; }
    QPushButton:default { border-color: #e07a3f; }
    QPushButton[plotOptionToggle="true"] {
        min-height: 20px;
        padding: 4px 8px;
        color: #cbd4dc;
        background: #2b2f33;
        border-color: #4a5159;
        text-align: left;
    }
    QPushButton[plotOptionToggle="true"]:hover {
        color: #ffffff;
        background: #394149;
        border-color: #65717c;
    }
    QPushButton[plotOptionToggle="true"]:checked {
        color: #ffffff;
        background: #633b26;
        border-color: #e07a3f;
    }
    QPushButton[plotOptionToggle="true"]:checked:hover { background: #79482d; }
    QPushButton[plotOptionToggle="true"]:disabled {
        color: #717a82;
        background: #25282b;
        border-color: #373c41;
    }
    QTreeView, QTreeWidget, QTableView, QTableWidget, QListWidget {
        border: 1px solid #3c434b;
        border-radius: 5px;
        alternate-background-color: #282b2f;
    }
    QTreeView::item, QTreeWidget::item, QTableView::item, QTableWidget::item, QListWidget::item {
        padding: 5px;
    }
    QTreeView::item:selected, QTreeWidget::item:selected, QTableView::item:selected,
    QTableWidget::item:selected, QListWidget::item:selected { background: #633b26; }
    QHeaderView::section {
        padding: 6px;
        border: 0;
        border-bottom: 1px solid #3c434b;
        background: #303338;
        font-weight: 700;
    }
    QMenuBar, QStatusBar { background: #242629; color: #e6e0db; }
    QMenuBar::item { padding: 6px 10px; }
    QMenuBar::item:selected { background: #633b26; }
    QMenu {
        background: #242629;
        color: #e6e0db;
        border: 1px solid #3c434b;
        padding: 4px;
    }
    QMenu::item { padding: 6px 24px 6px 12px; }
    QMenu::item:selected { color: #ffffff; background: #633b26; }
    QMenu::item:disabled { color: #8d9296; }
    QMenu::separator { height: 1px; background: #3c434b; margin: 4px 8px; }
    QToolTip { border: 1px solid #f0a15a; padding: 5px; }
"""


def _application_palette(base_palette: QPalette) -> QPalette:
    """Return MonStim's fixed dark palette, independent of the OS appearance."""
    palette = QPalette(base_palette)
    colors = {
        QPalette.ColorRole.Window: "#20252b",
        QPalette.ColorRole.WindowText: "#e6e0db",
        QPalette.ColorRole.Base: "#27292c",
        QPalette.ColorRole.AlternateBase: "#2c3035",
        QPalette.ColorRole.ToolTipBase: "#34302d",
        QPalette.ColorRole.ToolTipText: "#fff7f0",
        QPalette.ColorRole.Text: "#e6e0db",
        QPalette.ColorRole.Button: "#34373a",
        QPalette.ColorRole.ButtonText: "#e6e0db",
        QPalette.ColorRole.BrightText: "#fff7f0",
        QPalette.ColorRole.Light: "#59646f",
        QPalette.ColorRole.Midlight: "#4a5159",
        QPalette.ColorRole.Mid: "#3c434b",
        QPalette.ColorRole.Dark: "#1f2022",
        QPalette.ColorRole.Shadow: "#151719",
        QPalette.ColorRole.Highlight: "#e07a3f",
        QPalette.ColorRole.HighlightedText: "#ffffff",
        QPalette.ColorRole.Link: "#4c86b8",
        QPalette.ColorRole.LinkVisited: "#f0a15a",
        QPalette.ColorRole.PlaceholderText: "#9ca3a8",
    }
    for color_group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive, QPalette.ColorGroup.Disabled):
        for role, color in colors.items():
            palette.setColor(color_group, role, QColor(color))

    disabled_colors = {
        QPalette.ColorRole.WindowText: "#8d9296",
        QPalette.ColorRole.Text: "#8d9296",
        QPalette.ColorRole.ButtonText: "#8d9296",
        QPalette.ColorRole.Highlight: "#684936",
        QPalette.ColorRole.HighlightedText: "#c3bdb8",
        QPalette.ColorRole.Link: "#7897ad",
        QPalette.ColorRole.PlaceholderText: "#70777c",
    }
    for role, color in disabled_colors.items():
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(color))
    if hasattr(QPalette.ColorRole, "Accent"):
        for color_group in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive):
            palette.setColor(color_group, QPalette.ColorRole.Accent, QColor("#e07a3f"))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Accent, QColor("#684936"))
    return palette


class SpinBoxControlStyle(QProxyStyle):
    """Give native spinbox controls a practical, consistently sized hit target.

    Qt stylesheets cannot resize a spin-button without suppressing its native
    arrow.  This proxy keeps native button behavior, while making each button
    wide enough to match the application's combo-box affordances.
    """

    BUTTON_WIDTH = 20
    ARROW_SIZE = 10

    @staticmethod
    def _button_rect(option: QStyleOptionSpinBox, up: bool) -> QRect:
        frame = option.rect
        height = max(1, (frame.height() - 2) // 2)
        top = frame.top() + 1 if up else frame.bottom() - height
        return QRect(frame.right() - SpinBoxControlStyle.BUTTON_WIDTH, top, SpinBoxControlStyle.BUTTON_WIDTH, height)

    def subControlRect(self, control, option, sub_control, widget=None):
        if control == QStyle.ComplexControl.CC_SpinBox and isinstance(option, QStyleOptionSpinBox):
            if sub_control == QStyle.SubControl.SC_SpinBoxUp:
                return self._button_rect(option, up=True)
            if sub_control == QStyle.SubControl.SC_SpinBoxDown:
                return self._button_rect(option, up=False)
            if sub_control == QStyle.SubControl.SC_SpinBoxEditField:
                rect = super().subControlRect(control, option, sub_control, widget)
                return QRect(rect.left(), rect.top(), max(0, self._button_rect(option, up=True).left() - rect.left()), rect.height())
        return super().subControlRect(control, option, sub_control, widget)

    def drawComplexControl(self, control, option, painter: QPainter, widget=None):
        super().drawComplexControl(control, option, painter, widget)
        if control != QStyle.ComplexControl.CC_SpinBox or not isinstance(option, QStyleOptionSpinBox):
            return

        for sub_control, up in (
            (QStyle.SubControl.SC_SpinBoxUp, True),
            (QStyle.SubControl.SC_SpinBoxDown, False),
        ):
            button_rect = self._button_rect(option, up)
            button_option = QStyleOption(option)
            button_option.rect = button_rect
            if option.activeSubControls & sub_control:
                button_option.state |= QStyle.StateFlag.State_Sunken
            if option.state & QStyle.StateFlag.State_MouseOver:
                button_option.state |= QStyle.StateFlag.State_MouseOver
            self.drawPrimitive(QStyle.PrimitiveElement.PE_PanelButtonTool, button_option, painter, widget)

            center = button_rect.center()
            half_width = self.ARROW_SIZE // 2
            half_height = max(2, self.ARROW_SIZE // 3)
            if up:
                points = (
                    QPoint(center.x() - half_width, center.y() + half_height),
                    QPoint(center.x() + half_width, center.y() + half_height),
                    QPoint(center.x(), center.y() - half_height),
                )
            else:
                points = (
                    QPoint(center.x() - half_width, center.y() - half_height),
                    QPoint(center.x() + half_width, center.y() - half_height),
                    QPoint(center.x(), center.y() + half_height),
                )
            painter.save()
            painter.setPen(QColor("#dbe4ed"))
            painter.setBrush(QColor("#dbe4ed"))
            painter.drawPolygon(QPolygon(points))
            painter.restore()


class WheelChangeGuard(QObject):
    """Reject wheel events that would silently change a value editor."""

    def eventFilter(self, watched, event):
        if isinstance(watched, (QComboBox, QAbstractSpinBox)) and event.type() == QEvent.Type.Wheel:
            event.ignore()
            return True
        return super().eventFilter(watched, event)


def install_wheel_change_guard(root: QWidget, guard: WheelChangeGuard | None = None) -> WheelChangeGuard:
    """Install a reusable wheel guard on all value-changing controls below *root*.

    Call this after dynamic controls are rebuilt. The returned guard must be
    retained by the caller so Qt keeps the event filter alive.
    """
    guard = guard or WheelChangeGuard(root)
    for editor_type in (QComboBox, QAbstractSpinBox):
        for editor in root.findChildren(editor_type):
            if editor.property("monstim_wheel_guard"):
                continue
            editor.installEventFilter(guard)
            editor.setProperty("monstim_wheel_guard", True)
    return guard


def apply_application_theme(application: QApplication | None = None) -> None:
    """Apply MonStim's dark theme consistently across OS appearance modes."""
    application = application or QApplication.instance()
    if application is None or application.property("monstim_theme_applied"):
        return
    # Install the proxy before the application stylesheet.  Qt then wraps it
    # in its stylesheet style, while spinbox geometry and native primitives
    # continue to flow through this proxy.
    application.setPalette(_application_palette(application.palette()))
    application.setStyle(SpinBoxControlStyle(application.style()))
    application.setStyleSheet(f"{application.styleSheet()}\n{APPLICATION_STYLESHEET}")
    application.setProperty("monstim_theme_applied", True)
