"""Regression coverage for reusable Settings Center controls."""

from PySide6.QtCore import QEvent, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QSpinBox, QStyle, QStyleOptionSpinBox, QWidget

from monstim_gui.core.ui_theme import SpinBoxControlStyle, apply_application_theme, install_wheel_change_guard
from monstim_gui.managers.profile_manager import ProfileRecord
from monstim_gui.widgets.profile_library import ProfileLibraryWidget
from monstim_gui.widgets.settings_controls import DraftField, OverrideField


def test_inactive_override_does_not_parse_stale_editor_text():
    """Changing profiles must not parse fields that are inheriting globally."""
    field = OverrideField("Line style", ":", key="latency_window_style")
    field.field.editor.setCurrentText(":")
    field.override_box.setChecked(False)

    overridden, value = field.value()

    assert overridden is False
    assert value == ":"


def test_active_line_style_override_remains_literal_text():
    field = OverrideField("Line style", ":", key="latency_window_style", overridden=True)

    overridden, value = field.value()

    assert overridden is True
    assert value == ":"


def test_color_selector_uses_friendly_label_and_canonical_value():
    field = DraftField("H color", "tab:blue", key="h_color")

    assert field.editor.currentText() == "blue"
    assert field.value() == "tab:blue"


def test_profile_library_uses_a_selectable_source_table():
    library = ProfileLibraryWidget()
    built_in = ProfileRecord("Standard", "builtin.yml", {"description": "Shipped profile"}, "Built-in", True)
    user = ProfileRecord("Custom", "custom.yml", {"description": "Editable profile"}, "User", False)

    library.set_records([built_in, user], selected_path=user.path)

    assert library.list_widget.columnCount() == 2
    assert library.current_record() == user
    assert library.list_widget.topLevelItem(0).text(1) == "Built-in"
    assert library.source_label.text() == "User profile — editable"


def test_shared_wheel_guard_blocks_dropdown_and_spinbox_wheels():
    root = QWidget()
    combo = QComboBox(root)
    spinbox = QSpinBox(root)

    guard = install_wheel_change_guard(root)

    assert combo.property("monstim_wheel_guard")
    assert spinbox.property("monstim_wheel_guard")
    assert guard.eventFilter(combo, QEvent(QEvent.Type.Wheel))
    assert guard.eventFilter(spinbox, QEvent(QEvent.Type.Wheel))


def test_spinbox_style_provides_full_size_native_button_hit_targets():
    apply_application_theme(QApplication.instance())
    spinbox = QSpinBox()
    spinbox.resize(140, 36)
    spinbox.show()

    option = QStyleOptionSpinBox()
    spinbox.initStyleOption(option)
    up_button = spinbox.style().subControlRect(QStyle.ComplexControl.CC_SpinBox, option, QStyle.SubControl.SC_SpinBoxUp, spinbox)
    down_button = spinbox.style().subControlRect(QStyle.ComplexControl.CC_SpinBox, option, QStyle.SubControl.SC_SpinBoxDown, spinbox)

    assert up_button.width() == SpinBoxControlStyle.BUTTON_WIDTH
    assert down_button.width() == SpinBoxControlStyle.BUTTON_WIDTH
    QTest.mouseClick(spinbox, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, up_button.center())
    assert spinbox.value() == 1
    QTest.mouseClick(spinbox, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, down_button.center())
    assert spinbox.value() == 0
