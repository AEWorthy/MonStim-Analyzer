from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

# TODO: Fix filter dialog to align with valid search terms and keys. May require revamping the data curation search system.


class FilterDialog(QDialog):
    """
    Dialog to build an advanced filter by selecting field qualifiers and known terms.

    Inputs:
    - qualifiers: list of (key, label) pairs for field qualifiers (e.g., ("animal", "Animal"))
    - terms_by_key: mapping from qualifier key -> dict of value -> count (excludes name)

    Output:
    - Builds a tokenized query string suitable for the manager's `_apply_filter` method.
    """

    def __init__(self, qualifiers: List[Tuple[str, str]], terms_by_key: Dict[str, Dict[str, int]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filter Builder")
        self.resize(560, 500)
        self._qualifiers = qualifiers
        self._terms_by_key = terms_by_key or {}
        self._checkboxes: Dict[str, List[QCheckBox]] = {}

        main = QVBoxLayout(self)
        main.setContentsMargins(10, 10, 10, 10)
        main.setSpacing(8)

        # Quick help
        help_lbl = QLabel("Select terms to filter. Use key:value qualifiers. Combine with free-text tokens below.")
        help_lbl.setWordWrap(True)
        main.addWidget(help_lbl)

        # Free-text tokens entry
        self.free_text = QLineEdit()
        self.free_text.setPlaceholderText('Optional: extra tokens or "phrases" to include')
        main.addWidget(self.free_text)

        # Scrollable area for qualifiers and term checkboxes
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        form = QFormLayout(inner)
        form.setContentsMargins(4, 4, 4, 4)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(6)

        for key, label in self._qualifiers:
            values_map = dict(self._terms_by_key.get(key, {}))
            values = sorted(values_map.keys())
            # Collapsible section with arrow and value count
            count_text = f"{label} ({len(values)} values)"
            gb = QGroupBox(count_text)
            vbox = QVBoxLayout()
            vbox.setContentsMargins(6, 6, 6, 6)
            vbox.setSpacing(4)
            boxes: List[QCheckBox] = []
            for v in values:
                count = values_map.get(v, 0)
                label_text = f"{v} ({count})" if v else f"(empty) ({count})"
                cb = QCheckBox(label_text)
                cb.setTristate(False)
                vbox.addWidget(cb)
                boxes.append(cb)
            gb.setLayout(vbox)
            # Add a header toolbutton to toggle collapse
            header = QToolButton()
            header.setText(count_text)
            header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            header.setArrowType(Qt.ArrowType.RightArrow)
            header.setCheckable(True)
            header.setChecked(False)  # collapsed by default

            container = QWidget()
            container_v = QVBoxLayout()
            container_v.setContentsMargins(0, 0, 0, 0)
            container_v.setSpacing(4)
            container_v.addWidget(header)
            container_v.addWidget(gb)
            gb.setVisible(False)
            container.setLayout(container_v)

            def _toggle(checked=False, section=gb, btn=header):
                section.setVisible(checked)
                btn.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)

            header.toggled.connect(_toggle)

            form.addRow(container)
            self._checkboxes[key] = boxes

        inner.setLayout(form)
        scroll.setWidget(inner)
        main.addWidget(scroll, 1)

        # Buttons
        btns = QHBoxLayout()
        btns.addStretch(1)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_checks)
        btns.addWidget(self.clear_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btns.addWidget(self.cancel_btn)

        self.apply_btn = QPushButton("Apply Filter")
        self.apply_btn.setDefault(True)
        self.apply_btn.clicked.connect(self._apply)
        btns.addWidget(self.apply_btn)

        main.addLayout(btns)

    def _clear_checks(self):
        try:
            for boxes in self._checkboxes.values():
                for cb in boxes:
                    cb.setChecked(False)
            self.free_text.clear()
        except Exception:
            pass

    def _apply(self):
        """Build a query string like 'animal:WT cond:rev-light "extra phrase" foxp2' and accept."""
        try:
            tokens: List[str] = []
            for key, _label in self._qualifiers:
                for cb in self._checkboxes.get(key, []):
                    if cb.isChecked():
                        # Extract original value without the appended count
                        txt = cb.text().strip()
                        if txt.endswith(")") and "(" in txt:
                            val = txt[: txt.rfind("(")].strip()
                        else:
                            val = txt
                        if val:
                            # Quote values that contain spaces
                            if " " in val:
                                tokens.append(f'{key}:"{val}"')
                            else:
                                tokens.append(f"{key}:{val}")
            extra = self.free_text.text().strip()
            if extra:
                tokens.append(extra)
            self._result_query = " ".join(tokens).strip()
        except Exception as e:
            logging.error(f"Failed to build filter tokens: {e}")
            self._result_query = ""
        self.accept()

    def result_query(self) -> str:
        return getattr(self, "_result_query", "")
