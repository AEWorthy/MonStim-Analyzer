from PySide6.QtCore import Qt
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import QLineEdit


class FloatLineEdit(QLineEdit):
    def __init__(self, default_value=0, parent=None):
        super().__init__(str(default_value), parent)
        self.setValidator(QDoubleValidator())
        self.setAlignment(Qt.AlignmentFlag.AlignRight)

    def get_value(self):
        return float(self.text()) if self.text() else None

    def set_value(self, value):
        self.setText(str(value))
