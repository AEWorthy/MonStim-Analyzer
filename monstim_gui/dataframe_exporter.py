import sys
import pandas as pd
from PyQt6.QtWidgets import QApplication, QDialog, QVBoxLayout, QTableView, QPushButton, QHBoxLayout, QFileDialog, QScrollArea
from PyQt6.QtCore import Qt, QAbstractTableModel

from monstim_signals.core.utils import get_base_path

class PandasModel(QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._data.columns[section])
            if orientation == Qt.Orientation.Vertical:
                if isinstance(self._data.index, pd.MultiIndex):
                    # MultiIndex: Display the index names alongside the values
                    index_values = [str(i) for i in self._data.index[section]]
                    index_names = self._data.index.names  # Get the names of the index levels
                    return " | ".join([f"{name}: {value}" for name, value in zip(index_names, index_values) if name is not None])
                else:
                    # Single index
                    index_name = self._data.index.name
                    index_value = str(self._data.index[section])
                    return f"{index_name}: {index_value}" if index_name else index_value
        return None


class DataFrameDialog(QDialog):
    def __init__(self, df : pd.DataFrame, parent=None):
        self.df = df
        super().__init__(parent)
        self.setWindowTitle("DataFrame Viewer")
        self.setGeometry(100, 100, 800, 400)

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Make the table view scrollable
        scroll_area = QScrollArea()
        table_view = QTableView()
        scroll_area.setWidget(table_view)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)

        model = PandasModel(self.df)
        table_view.setModel(model)

        # Add "Save As..." and "Close" buttons
        button_layout = QHBoxLayout()
        save_button = QPushButton("Export Dataframe...")
        close_button = QPushButton("Close")

        save_button.clicked.connect(self.save_as)
        close_button.clicked.connect(self.close)

        button_layout.addWidget(save_button)
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)

    def save_as(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Export DataFrame", get_base_path(), "CSV Files (*.csv);;All Files (*)")
        if file_name:
            self.df.to_csv(file_name, index=True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    dialog = DataFrameDialog(df)
    dialog.exec()
