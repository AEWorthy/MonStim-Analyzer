"""Shared presentation and interaction primitives for the Qt application."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, QPoint, QRect
from PySide6.QtGui import QColor, QPainter, QPolygon
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
    QMainWindow { background: #1f2022; }
    QWidget#mainSidebar {
        background: #242629;
        border: 1px solid #3c434b;
        border-radius: 7px;
    }
    QSplitter#mainContentSplitter::handle:horizontal {
        width: 6px;
        background: #1f2022;
    }
    QSplitter#mainContentSplitter::handle:horizontal:hover { background: #566673; }
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
        border-bottom-color: #e7785b;
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
    QPushButton {
        padding: 5px 12px;
        border: 1px solid #4a5159;
        border-radius: 4px;
        background: #34373a;
        font-weight: 600;
    }
    QPushButton:hover { background: #41464b; border-color: #65717c; }
    QPushButton:default { border-color: #e7785b; }
    QTreeView, QTreeWidget, QTableView, QTableWidget, QListWidget {
        border: 1px solid #3c434b;
        border-radius: 5px;
        alternate-background-color: #282b2f;
    }
    QTreeView::item, QTreeWidget::item, QTableView::item, QTableWidget::item, QListWidget::item {
        padding: 5px;
    }
    QTreeView::item:selected, QTreeWidget::item:selected, QTableView::item:selected,
    QTableWidget::item:selected, QListWidget::item:selected { background: #304b5d; }
    QHeaderView::section {
        padding: 6px;
        border: 0;
        border-bottom: 1px solid #3c434b;
        background: #303338;
        font-weight: 700;
    }
    QMenuBar, QStatusBar { background: #242629; }
    QMenuBar::item { padding: 6px 10px; }
    QMenuBar::item:selected, QMenu::item:selected { background: #304b5d; }
    QToolTip { border: 1px solid #65717c; padding: 5px; }
"""


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
    """Append the shared visual language without discarding an existing theme."""
    application = application or QApplication.instance()
    if application is None or application.property("monstim_theme_applied"):
        return
    # Install the proxy before the application stylesheet.  Qt then wraps it
    # in its stylesheet style, while spinbox geometry and native primitives
    # continue to flow through this proxy.
    application.setStyle(SpinBoxControlStyle(application.style()))
    application.setStyleSheet(f"{application.styleSheet()}\n{APPLICATION_STYLESHEET}")
    application.setProperty("monstim_theme_applied", True)
