import sys
import os
import pandas as pd
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QVBoxLayout,
    QTableView,
    QDialogButtonBox,
    QFileDialog,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QAbstractTableModel

from monstim_signals.core import get_base_path
from monstim_gui.core.application_state import app_state


class PandasModel(QAbstractTableModel):
    def __init__(self, data: pd.DataFrame, parent=None):
        super().__init__(parent)
        self._data = data

    def rowCount(self, parent=None) -> int:
        return self._data.shape[0]

    def columnCount(self, parent=None) -> int:
        return self._data.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            value = self._data.iat[index.row(), index.column()]
            return str(value)
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._data.columns[section])
            if orientation == Qt.Orientation.Vertical:
                idx = self._data.index
                if isinstance(idx, pd.MultiIndex):
                    names = [f"{n}:{v}" for n, v in zip(idx.names, idx[section]) if n]
                    return " | ".join(names)
                else:
                    name = idx.name or ""
                    val = idx[section]
                    return f"{name}:{val}" if name else str(val)
        return None


class DataFrameDialog(QDialog):
    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.df = df
        self.setWindowTitle("Data Preview")
        self.resize(800, 400)

        layout = QVBoxLayout(self)

        # Table view
        self.table_view = QTableView(self)
        self.table_view.setModel(PandasModel(self.df, self))
        self.table_view.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setSortingEnabled(True)
        self.table_view.resizeColumnsToContents()
        self.table_view.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table_view)

        # Button box: Export and Close
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Close,
            Qt.Orientation.Horizontal,
            self,
        )
        # Rename Save to Export CSV...
        save_btn = button_box.button(QDialogButtonBox.StandardButton.Save)
        save_btn.setText("Export CSV...")
        button_box.accepted.connect(self.save_as)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def save_as(self):
        # Get the last used export directory, fallback to base path
        last_export_path = app_state.get_last_export_path()
        if not last_export_path or not os.path.isdir(last_export_path):
            last_export_path = str(get_base_path())

        default_name = os.path.join(last_export_path, "exported_data.csv")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export DataFrame", default_name, "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            # Save the directory for next time
            app_state.save_last_export_path(os.path.dirname(path))
            self.df.to_csv(path, index=True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
    dlg = DataFrameDialog(df)
    dlg.exec()
