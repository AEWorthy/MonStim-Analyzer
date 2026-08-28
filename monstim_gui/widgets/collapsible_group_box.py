"""Collapsible section controls for the main sidebar."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPaintEvent, QPen, QResizeEvent
from PySide6.QtWidgets import QGroupBox, QSizePolicy, QToolButton, QWidget


class _SectionHeaderButton(QToolButton):
    """A restrained disclosure header with a small drawn chevron."""

    def __init__(self, title: str, parent: QWidget) -> None:
        super().__init__(parent)
        self._title = title
        self.setText(title)
        self.setAccessibleName(title)

    def paintEvent(self, event: QPaintEvent) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.isDown():
            background = QColor("#303940")
        elif self.underMouse():
            background = QColor("#30363d")
        else:
            background = QColor("#292e34")
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(background)
        painter.drawRoundedRect(self.rect().adjusted(0, 1, -1, -1), 4, 4)

        chevron_pen = QPen(QColor("#c8d0d7"), 1.5)
        chevron_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        chevron_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(chevron_pen)
        if self.isChecked():
            painter.drawLine(13, 10, 18, 15)
            painter.drawLine(18, 15, 23, 10)
        else:
            painter.drawLine(14, 9, 19, 14)
            painter.drawLine(19, 14, 14, 19)

        painter.setPen(QColor("#f1f3f5"))
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(self.rect().adjusted(32, 0, -8, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self._title)

        if self.hasFocus():
            focus_pen = QPen(QColor("#7c98ad"), 1)
            painter.setPen(focus_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(self.rect().adjusted(1, 2, -3, -3), 3, 3)


class CollapsibleGroupBox(QGroupBox):
    """A group box with a prominent header that hides or reveals its contents."""

    _HEADER_HEIGHT = 28

    def __init__(self, title: str, parent: QWidget | None = None, *, expanded: bool = True) -> None:
        super().__init__("", parent)
        self._collapsed = False
        self._section_title = title
        self._expanded_size_policy = self.sizePolicy()
        self._expanded_minimum_width = 0
        self.setObjectName("collapsibleSidebarSection")

        self.toggle_button = _SectionHeaderButton(title, self)
        self.toggle_button.setObjectName("collapsibleSectionHeader")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(True)
        self.toggle_button.setToolTip(f"Collapse {title}")
        self.toggle_button.toggled.connect(self._on_toggled)

        self.set_expanded(expanded)

    def is_expanded(self) -> bool:
        """Return whether the section contents are currently visible."""
        return not self._collapsed

    def expanded_minimum_width(self) -> int:
        """Return the minimum width required when this section is expanded."""
        if not self._collapsed:
            self._expanded_minimum_width = max(self._expanded_minimum_width, super().minimumSizeHint().width())
        return self._expanded_minimum_width

    def set_expanded(self, expanded: bool) -> None:
        """Show or hide the section contents without changing their state."""
        if self.toggle_button.isChecked() == expanded:
            self._on_toggled(expanded)
        else:
            self.toggle_button.setChecked(expanded)

    def _on_toggled(self, expanded: bool) -> None:
        if not expanded:
            self._expanded_minimum_width = max(self._expanded_minimum_width, super().minimumSizeHint().width())
        self._collapsed = not expanded
        self.toggle_button.setToolTip(f"{'Collapse' if expanded else 'Expand'} {self._section_title}")

        for child in self.findChildren(QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly):
            if child is not self.toggle_button:
                child.setVisible(expanded)

        if expanded:
            self.setMaximumHeight(16777215)  # QWIDGETSIZE_MAX
            self.setSizePolicy(self._expanded_size_policy)
        else:
            self.setMaximumHeight(self._HEADER_HEIGHT + 4)
            self.setSizePolicy(self._expanded_size_policy.horizontalPolicy(), QSizePolicy.Policy.Maximum)

        self.updateGeometry()

    def minimumSizeHint(self) -> QSize:
        if self._collapsed:
            return QSize(super().minimumSizeHint().width(), self._HEADER_HEIGHT + 4)
        return super().minimumSizeHint()

    def sizeHint(self) -> QSize:
        if self._collapsed:
            return QSize(super().sizeHint().width(), self._HEADER_HEIGHT + 4)
        return super().sizeHint()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self.toggle_button.setGeometry(7, 1, max(1, self.width() - 14), self._HEADER_HEIGHT)
