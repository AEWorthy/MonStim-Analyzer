from PyQt6.QtCore import QPoint, QRect, QSize, Qt
from PyQt6.QtWidgets import QLayout


class FlowLayout(QLayout):
    """A flow layout that arranges items horizontally and wraps when needed."""

    def __init__(self, parent=None, margin=0, spacing=-1):
        super().__init__(parent)
        self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)
        self._item_list = []

    def addItem(self, item):
        self._item_list.append(item)

    def count(self):
        return len(self._item_list)

    def itemAt(self, index):
        if 0 <= index < len(self._item_list):
            return self._item_list[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._item_list):
            return self._item_list.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientation(0)  # No expanding directions

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._item_list:
            size = size.expandedTo(item.minimumSize())

        margins = self.contentsMargins()
        size += QSize(margins.left() + margins.right(), margins.top() + margins.bottom())
        return size

    def _do_layout(self, rect, test_only=False):
        m = self.contentsMargins()
        effective_rect = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x = effective_rect.x()
        y = effective_rect.y()
        line_height = 0
        spacing = self.spacing()

        for item in self._item_list:
            widget = item.widget()
            if widget and not widget.isVisible():
                continue

            item_width = item.sizeHint().width()
            item_height = item.sizeHint().height()

            if x + item_width > effective_rect.right() and line_height > 0:
                x = effective_rect.x()
                y += line_height + spacing
                line_height = 0

            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))

            x += item_width + spacing
            line_height = max(line_height, item_height)

        return y + line_height - rect.y() + m.bottom()
