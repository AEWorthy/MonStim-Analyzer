import os
import re
import sys

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QSizePolicy,
    QTableView,
    QVBoxLayout,
)

from monstim_gui.core.application_state import app_state
from monstim_signals.core import get_base_path

# Normalize a few known method names to short suffixes
METHOD_SUFFIX_MAP = {
    "peak_to_trough": "ptt",
    "rms": "rms",
    "average_rectified": "avgrect",
    "average_unrectified": "avgunrect",
    "auc": "auc",
}


def _sanitize_filename(name: str) -> str:
    # Remove characters invalid on Windows and collapse whitespace
    if not name:
        return ""
    # replace path separators and forbidden chars with underscore
    name = re.sub(r'[<>:"/\\|?*]+', "_", name)
    # replace any sequence of non-alphanum._- with underscore
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    # trim
    return name.strip(" _-")


def make_export_filename(df: pd.DataFrame) -> str:
    """Create a descriptive, auto-named filename for DataFrame exports.

    Uses recent application selection (experiment/dataset/session) and
    optional DataFrame attributes to produce a human-readable filename.
    """
    parts = []

    # Determine the data level to include: prefer explicit DataFrame attr
    try:
        data_level_attr = df.attrs.get("data_level") if hasattr(df, "attrs") else None
    except Exception:
        data_level_attr = None

    # Infer session state from app_state as fallback
    session_state = app_state.get_last_session_state() if hasattr(app_state, "get_last_session_state") else {}
    sel = app_state.get_last_selection() if hasattr(app_state, "get_last_selection") else {}

    exp_from_state = session_state.get("experiment") or sel.get("experiment") or sel.get("experiment_id")
    ds_from_state = session_state.get("dataset") or sel.get("dataset") or sel.get("dataset_id")
    sess_from_state = session_state.get("session") or sel.get("session") or sel.get("session_id")

    # Allow DataFrame attrs to override specific ids
    try:
        exp_attr = df.attrs.get("experiment") if hasattr(df, "attrs") else None
        ds_attr = df.attrs.get("dataset") if hasattr(df, "attrs") else None
        sess_attr = df.attrs.get("session") if hasattr(df, "attrs") else None
    except Exception:
        exp_attr = ds_attr = sess_attr = None

    exp = exp_attr or exp_from_state
    ds = ds_attr or ds_from_state
    sess = sess_attr or sess_from_state

    # Decide which single level to include. Respect explicit df.attrs['data_level'] when set.
    level = None
    if data_level_attr:
        level = data_level_attr.lower()
    else:
        # infer: prefer session if available, then dataset, then experiment
        if sess:
            level = "session"
        elif ds:
            level = "dataset"
        elif exp:
            level = "experiment"

    if level == "session" and sess:
        parts.append(str(sess))
    elif level == "dataset" and ds:
        parts.append(str(ds))
    elif level == "experiment" and exp:
        parts.append(str(exp))

    # Prefer explicit DataFrame attributes for plot type and data level
    # Resolve plot type (prefer short aliases)
    PLOT_TYPE_ALIASES = {
        "Average Reflex:Stimulus Curves": "avg_reflex",
        "Average Reflex Stimulus Curves": "avg_reflex",
        "M-max Report (RMS)": "mmax_rms",
        "Session Info. Report": "session_info",
        "Dataset Info. Report": "dataset_info",
        "Experiment Info. Report": "experiment_info",
    }

    def _short_plot_key(s: str) -> str:
        if not s:
            return None
        if s in PLOT_TYPE_ALIASES:
            return PLOT_TYPE_ALIASES[s]
        # fallback: slugify/shorten
        key = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
        return key[:40]

    try:
        raw_plot = None
        if hasattr(df, "attrs"):
            for k in ("plot_type", "plot", "plot_name"):
                raw_plot = raw_plot or df.attrs.get(k)
        raw_plot = raw_plot or getattr(df, "name", None)
    except Exception:
        raw_plot = None
    plot_key = _short_plot_key(raw_plot)

    # Append the plot type key (preferred) or a DataFrame name hint
    dfname = None
    try:
        dfname = getattr(df, "name", None) or (df.attrs.get("name") if hasattr(df, "attrs") else None)
    except Exception:
        dfname = None

    if plot_key:
        parts.append(str(plot_key))

    # If this is an average-reflex style export, append the amplitude
    # calculation method (when available) so files indicate the metric used.
    try:
        method = None
        if hasattr(df, "attrs"):
            # common keys used across the codebase: 'method' from plot options
            method = df.attrs.get("method") or df.attrs.get("amplitude_method") or df.attrs.get("calc_method")
    except Exception:
        method = None

    if method:
        mkey = str(method).lower()
        suf = METHOD_SUFFIX_MAP.get(mkey, re.sub(r"[^a-z0-9]+", "", mkey)[:10])
        # Append as its own part so the filename becomes e.g. '...__avg_reflex__rms'
        parts.append(str(suf))
    elif dfname:
        parts.append(str(dfname))

    filename = "__".join(parts)
    filename = _sanitize_filename(filename)
    if not filename:
        filename = "export"
    return f"{filename}.csv"


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
    def __init__(
        self, df: pd.DataFrame, parent=None, plot_type: str = None, data_level: str = None, plot_options: dict | None = None
    ):
        super().__init__(parent)
        self.df = df
        self.plot_type = plot_type
        self.data_level = data_level
        self.plot_options = plot_options or {}
        self.setWindowTitle("Data Preview")
        self.resize(800, 400)

        layout = QVBoxLayout(self)

        # Table view
        self.table_view = QTableView(self)
        self.table_view.setModel(PandasModel(self.df, self))
        self.table_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setSortingEnabled(True)
        self.table_view.resizeColumnsToContents()
        self.table_view.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table_view)

        # Button box: Export and Close
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Close,
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

        # Set DataFrame attrs with plot metadata for filename generation
        if self.plot_type:
            self.df.attrs["plot_type"] = self.plot_type
        if self.data_level:
            self.df.attrs["data_level"] = self.data_level
        # If plot options include a calculation method, store that too so filenames
        # can indicate which amplitude method was used (e.g., rms, ptt, auc).
        try:
            method = self.plot_options.get("method") if isinstance(self.plot_options, dict) else None
            if method:
                self.df.attrs["method"] = method
        except Exception:
            pass

        # Build a descriptive default filename based on context
        suggested = make_export_filename(self.df)
        default_name = os.path.join(last_export_path, suggested)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export DataFrame",
            default_name,
            "CSV Files (*.csv);;All Files (*)",
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
