from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget


class MigrationBanner(QWidget):
    run_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("migrationBanner")
        # Style for compact, floating appearance
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMaximumHeight(40)
        self.setMinimumWidth(320)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(8)
        self.label = QLabel("Annotation migrations detected. Run now to update files.")
        self.label.setWordWrap(False)
        self.button = QPushButton("Run now")
        self.button.setFixedHeight(26)
        layout.addWidget(self.label)
        layout.addStretch()
        layout.addWidget(self.button)
        self.button.clicked.connect(self.run_clicked.emit)
        self.hide()  # hidden by default

    def show_message(self, text: str):
        self.label.setText(text)
        # Position bottom-right over the main window content if possible
        try:
            if self.parent() and hasattr(self.parent(), "rect"):
                parent_rect = self.parent().rect()
                self.adjustSize()
                x = parent_rect.right() - self.width() - 16
                y = parent_rect.bottom() - self.height() - 56
                self.move(x, y)
        except Exception:
            pass
        self.show()

    def hide(self):
        super().hide()
