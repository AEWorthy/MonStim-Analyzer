"""Reusable, typed controls for draft-backed settings pages."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QLineEdit, QSizePolicy, QSpinBox, QWidget

from monstim_signals.core.configuration import CALCULATION_METHODS

TAB_COLORS = ("blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan")
LINE_STYLES = (":", "-", "--", "-.")


def _label(key: str) -> str:
    return key.replace("_", " ").replace("args", "settings").title()


class DraftField(QWidget):
    """A typed setting editor with stable draft/value/reset semantics.

    Mappings are composed from the same scalar controls so users edit named
    values, rather than error-prone serialized YAML.
    """

    changed = Signal()

    def __init__(self, label: str, value: Any, parent=None, *, key: str = "", help_text: str = ""):
        super().__init__(parent)
        self.label = label
        self.key = key
        self._reference = copy.deepcopy(value)
        self._mapping_fields: dict[str, DraftField] = {}
        self._is_list = isinstance(value, list | tuple)
        self._is_mapping = isinstance(value, Mapping)
        self._layout = QFormLayout(self)
        self._layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setVerticalSpacing(4)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        if self._is_mapping:
            for subkey, subvalue in value.items():
                child = DraftField("", subvalue, self, key=f"{key}.{subkey}")
                self._mapping_fields[str(subkey)] = child
                self._layout.addRow(_label(str(subkey)), child)
                child.changed.connect(self.changed)
            self.editor = None
        else:
            self.editor = self._create_editor(value)
            self.editor.setMinimumHeight(26)
            self._layout.addRow(label, self.editor)
            if help_text:
                self.setToolTip(help_text)
                self.editor.setToolTip(help_text)

    def _create_editor(self, value: Any) -> QWidget:
        if isinstance(value, bool):
            editor = QCheckBox(self)
            editor.setChecked(value)
            editor.toggled.connect(self.changed)
            return editor
        if isinstance(value, int):
            editor = QSpinBox(self)
            editor.setRange(-1_000_000, 1_000_000)
            editor.setValue(value)
            editor.setSuffix(self._numeric_suffix())
            editor.valueChanged.connect(self.changed)
            return editor
        if isinstance(value, float):
            editor = QDoubleSpinBox(self)
            editor.setRange(-1_000_000.0, 1_000_000.0)
            editor.setDecimals(6)
            editor.setSingleStep(0.1 if abs(value) >= 1 else 0.01)
            editor.setValue(value)
            editor.setSuffix(self._numeric_suffix())
            editor.valueChanged.connect(self.changed)
            return editor
        if self._is_list:
            editor = QLineEdit(", ".join(map(str, value)), self)
            editor.setPlaceholderText("Comma-separated values")
            editor.textEdited.connect(self.changed)
            return editor
        if self.key.endswith("default_method"):
            editor = QComboBox(self)
            editor.addItems(sorted(CALCULATION_METHODS))
            editor.setCurrentText(str(value))
            editor.currentTextChanged.connect(self.changed)
            return editor
        if self.key.split(".")[-1] in {"m_color", "h_color"}:
            editor = QComboBox(self)
            for color in TAB_COLORS:
                editor.addItem(color, f"tab:{color}")
            editor.setEditable(True)
            index = editor.findData(str(value))
            if index >= 0:
                editor.setCurrentIndex(index)
            else:
                editor.setCurrentText(str(value).removeprefix("tab:"))
            editor.currentTextChanged.connect(self.changed)
            return editor
        if self.key.split(".")[-1] == "latency_window_style":
            editor = QComboBox(self)
            editor.addItems(LINE_STYLES)
            editor.setEditable(True)
            editor.setCurrentText(str(value))
            editor.currentTextChanged.connect(self.changed)
            return editor
        editor = QLineEdit(str(value), self)
        editor.textEdited.connect(self.changed)
        return editor

    def _numeric_suffix(self) -> str:
        leaf = self.key.split(".")[-1]
        if leaf in {"time_window", "pre_stim_time", "duration"}:
            return " ms"
        if leaf == "bin_size":
            return " V"
        if leaf in {"lowcut", "highcut"}:
            return " Hz"
        if leaf in {"title_font_size", "axis_label_font_size", "tick_font_size"}:
            return " pt"
        return ""

    def value(self) -> Any:
        if self._is_mapping:
            return {key: field.value() for key, field in self._mapping_fields.items()}
        assert self.editor is not None
        if isinstance(self._reference, bool):
            return self.editor.isChecked()  # type: ignore[union-attr]
        if isinstance(self._reference, int):
            return self.editor.value()  # type: ignore[union-attr]
        if isinstance(self._reference, float):
            return self.editor.value()  # type: ignore[union-attr]
        if self._is_list:
            text = self.editor.text()  # type: ignore[union-attr]
            return [item.strip() for item in text.split(",") if item.strip()]
        # String settings, including ':' line styles, are always literal text.
        if isinstance(self.editor, QComboBox):
            return self.editor.currentData() or self.editor.currentText()
        return self.editor.text()

    def set_value(self, value: Any) -> None:
        self._reference = copy.deepcopy(value)
        if self._is_mapping:
            for key, child in self._mapping_fields.items():
                if key in value:
                    child.set_value(value[key])
            return
        assert self.editor is not None
        if isinstance(self._reference, bool):
            self.editor.setChecked(value)  # type: ignore[union-attr]
        elif isinstance(self._reference, int | float):
            self.editor.setValue(value)  # type: ignore[union-attr]
        elif self._is_list:
            self.editor.setText(", ".join(map(str, value)))  # type: ignore[union-attr]
        elif isinstance(self.editor, QComboBox):
            index = self.editor.findData(str(value))
            if index >= 0:
                self.editor.setCurrentIndex(index)
            else:
                self.editor.setCurrentText(str(value).removeprefix("tab:"))
        else:
            self.editor.setText(str(value))  # type: ignore[union-attr]

    def reset(self) -> None:
        self.set_value(self._reference)


class OverrideField(QWidget):
    """A reusable inherited-or-overridden profile setting field."""

    changed = Signal()

    def __init__(self, label: str, global_value: Any, override: Any = None, *, overridden: bool = False, parent=None, key: str = ""):
        super().__init__(parent)
        self.global_value = copy.deepcopy(global_value)
        self.override_box = QCheckBox("Override global value", self)
        # A missing override value is never meaningful; show the effective
        # global value until the user supplies an explicit replacement.
        initial_value = override if overridden and override is not None else global_value
        self.field = DraftField(label, initial_value, self, key=key)
        self.inherited_hint = QLineEdit(self)
        self.inherited_hint.setReadOnly(True)
        self.inherited_hint.setText(f"Global value: {self._display_global_value()}")
        layout = QFormLayout(self)
        layout.addRow(self.override_box)
        layout.addRow(self.field)
        layout.addRow(self.inherited_hint)
        self.override_box.setChecked(overridden)
        self.field.setEnabled(overridden)
        self.override_box.toggled.connect(self._on_override_toggled)
        self.field.changed.connect(self.changed)

    def _display_global_value(self) -> str:
        if isinstance(self.global_value, Mapping):
            return "See the global Analysis settings"
        if isinstance(self.global_value, list | tuple):
            return ", ".join(map(str, self.global_value))
        return str(self.global_value)

    def _on_override_toggled(self, enabled: bool) -> None:
        self.field.setEnabled(enabled)
        if not enabled:
            self.field.set_value(self.global_value)
        self.changed.emit()

    def value(self) -> tuple[bool, Any]:
        if not self.override_box.isChecked():
            return False, copy.deepcopy(self.global_value)
        return True, self.field.value()

    def reset_to_inherited(self) -> None:
        self.override_box.setChecked(False)
