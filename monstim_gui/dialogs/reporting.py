from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)


class CopyableReportDialog(QDialog):
    def __init__(self, title, report, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setLayout(QVBoxLayout())

        self.text_edit = QTextEdit()
        self.text_edit.setPlainText(report)
        self.text_edit.setReadOnly(True)
        self.layout().addWidget(self.text_edit)

        button_layout = QHBoxLayout()

        copy_button = QPushButton("Copy to Clipboard")
        copy_button.clicked.connect(self.copy_to_clipboard)
        button_layout.addWidget(copy_button)

        done_button = QPushButton("Done")
        done_button.clicked.connect(self.close)
        button_layout.addWidget(done_button)

        self.layout().addLayout(button_layout)

        self.resize(300, 200)

    def copy_to_clipboard(self):
        QApplication.clipboard().setText(self.text_edit.toPlainText())
