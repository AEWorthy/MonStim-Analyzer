"""Regression coverage for reusable Settings Center controls."""

from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QPalette
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QSpinBox, QStyle, QStyleOptionSpinBox, QWidget

from monstim_gui.core.ui_theme import (
    APPLICATION_STYLESHEET,
    SpinBoxControlStyle,
    _application_palette,
    apply_application_theme,
    install_wheel_change_guard,
)
from monstim_gui.dialogs.settings_center import SettingsCenter
from monstim_gui.io.config_repository import ConfigRepository
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


def test_theme_palette_is_dark_and_warm_even_when_built_from_a_light_palette():
    """The application's appearance must not inherit light-mode system colors."""
    light_palette = QPalette()
    light_palette.setColor(QPalette.ColorRole.Window, Qt.GlobalColor.white)
    light_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.black)

    palette = _application_palette(light_palette)

    assert palette.color(QPalette.ColorRole.Window).name() == "#20252b"
    assert palette.color(QPalette.ColorRole.Base).name() == "#27292c"
    assert palette.color(QPalette.ColorRole.Text).name() == "#e6e0db"
    assert palette.color(QPalette.ColorRole.Highlight).name() == "#e07a3f"
    assert palette.color(QPalette.ColorRole.HighlightedText).name() == "#ffffff"
    assert palette.color(QPalette.ColorRole.Link).name() == "#4c86b8"
    assert palette.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text).name() == "#8d9296"
    assert palette.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base).name() == "#27292c"
    assert palette.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Button).name() == "#34373a"


def test_popup_menus_define_dark_surfaces_and_readable_text():
    assert "QMenu {\n        background: #242629;\n        color: #e6e0db;" in APPLICATION_STYLESHEET
    assert "QMenu::item:selected { color: #ffffff; background: #633b26; }" in APPLICATION_STYLESHEET
    assert "QMenu::item:disabled { color: #8d9296; }" in APPLICATION_STYLESHEET


def test_settings_center_uses_the_warm_application_selection_colors(tmp_path):
    default_config = "docs/resources/config.yml"
    config_repo = ConfigRepository(default_config, str(tmp_path / "config-user.yml"))
    center = SettingsCenter(default_config, config_repo=config_repo)

    stylesheet = center.styleSheet()

    assert "background: #633b26" in stylesheet
    assert "border-left: 3px solid #e07a3f" in stylesheet
    assert "border-bottom-color: #e07a3f" in stylesheet
    assert "#304553" not in stylesheet
    assert "#6d9fbe" not in stylesheet
    center.close()
