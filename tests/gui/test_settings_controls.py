"""Regression coverage for reusable Settings Center controls."""

from PySide6.QtCore import QEvent, QSettings, Qt
from PySide6.QtGui import QPalette
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QSpinBox, QStyle, QStyleOptionSpinBox, QWidget

from monstim_gui.core.keyboard_shortcuts import SHORTCUT_DEFINITIONS, KeyboardShortcutController, default_shortcuts, normalize_shortcuts
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


def test_keyboard_shortcut_defaults_are_unique_and_include_plot_without_extraction():
    shortcuts = normalize_shortcuts(default_shortcuts())

    assert len(set(shortcuts.values())) == len(SHORTCUT_DEFINITIONS)
    assert shortcuts["plot"] == "Ctrl+P"
    assert shortcuts["plot_extract"] == "Ctrl+Shift+P"
    assert shortcuts["next_session"] == "Alt+1"
    assert shortcuts["previous_session"] == "Alt+Shift+1"
    assert shortcuts["complete_session"] == "Ctrl+1"
    assert shortcuts["incomplete_session"] == "Ctrl+Shift+1"


def test_keyboard_shortcut_preferences_reject_duplicate_assignments():
    shortcuts = default_shortcuts()
    shortcuts["next_session"] = shortcuts["previous_session"]

    try:
        normalize_shortcuts(shortcuts)
    except ValueError as error:
        assert "assigned more than once" in str(error)
    else:
        raise AssertionError("duplicate shortcuts must be rejected")


def test_keyboard_navigation_auto_plot_preference_defaults_to_enabled_and_persists(tmp_path):
    settings = QSettings(str(tmp_path / "shortcuts.ini"), QSettings.Format.IniFormat)

    assert KeyboardShortcutController.load_auto_plot_on_navigation(settings) is True

    KeyboardShortcutController.save_preferences(default_shortcuts(), auto_plot_on_navigation=False, settings=settings)

    assert KeyboardShortcutController.load_auto_plot_on_navigation(settings) is False


def test_keyboard_navigation_plots_only_after_a_successful_selection_change():
    class PlotController:
        def __init__(self):
            self.calls = 0

        def plot_data(self):
            self.calls += 1

    class Gui:
        def __init__(self, changed):
            self.changed = changed
            self.navigation_calls = []
            self.plot_controller = PlotController()
            self.current_experiment = object()

        def step_data_selection(self, level, direction):
            self.navigation_calls.append((level, direction))
            return self.changed

    controller = object.__new__(KeyboardShortcutController)
    controller.gui = Gui(changed=True)
    controller.auto_plot_on_navigation = True

    controller._dispatch("next_session")

    assert controller.gui.navigation_calls == [("session", 1)]
    assert controller.gui.plot_controller.calls == 1

    controller.gui.changed = False
    controller._dispatch("previous_session")
    controller.auto_plot_on_navigation = False
    controller._dispatch("next_session")

    assert controller.gui.plot_controller.calls == 1


def test_keyboard_navigation_defers_plotting_until_an_experiment_finishes_loading():
    class PlotController:
        def __init__(self):
            self.calls = 0

        def plot_data(self):
            self.calls += 1

    class Gui:
        current_experiment = None

        def __init__(self):
            self.plot_controller = PlotController()

    controller = object.__new__(KeyboardShortcutController)
    controller.gui = Gui()
    controller.auto_plot_on_navigation = True
    controller._plot_when_experiment_ready = False
    controller._pending_experiment_id = None

    controller._plot_after_navigation()

    assert controller._plot_when_experiment_ready is True
    assert controller.gui.plot_controller.calls == 0

    controller.gui.current_experiment = object()
    controller.on_experiment_load_finished()

    assert controller.gui.plot_controller.calls == 1
    assert controller._plot_when_experiment_ready is False


def test_deferred_keyboard_plot_is_discarded_when_a_different_experiment_loads():
    class PlotController:
        def __init__(self):
            self.calls = 0

        def plot_data(self):
            self.calls += 1

    class ExperimentCombo:
        @staticmethod
        def currentData(_role):
            return "requested-experiment"

    class Gui:
        current_experiment = None

        def __init__(self):
            self.plot_controller = PlotController()
            self.data_selection_widget = type("Selection", (), {"experiment_combo": ExperimentCombo()})()

    controller = object.__new__(KeyboardShortcutController)
    controller.gui = Gui()
    controller.auto_plot_on_navigation = True
    controller._plot_when_experiment_ready = False
    controller._pending_experiment_id = None

    controller._plot_after_navigation()
    controller.gui.current_experiment = type("Experiment", (), {"id": "different-experiment"})()
    controller.on_experiment_load_finished()

    assert controller.gui.plot_controller.calls == 0
