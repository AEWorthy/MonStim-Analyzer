"""Configurable main-window keyboard shortcuts for data curation and plotting."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QKeySequence, QShortcut


@dataclass(frozen=True)
class ShortcutDefinition:
    key: str
    label: str
    description: str
    default: str


SHORTCUT_DEFINITIONS = (
    ShortcutDefinition("previous_session", "Previous session", "Select the preceding session in the current dataset.", "Alt+Shift+1"),
    ShortcutDefinition("next_session", "Next session", "Select the following session in the current dataset.", "Alt+1"),
    ShortcutDefinition("previous_dataset", "Previous dataset", "Select the preceding dataset in the current experiment.", "Alt+Shift+2"),
    ShortcutDefinition("next_dataset", "Next dataset", "Select the following dataset in the current experiment.", "Alt+2"),
    ShortcutDefinition("previous_experiment", "Previous experiment", "Select the preceding experiment.", "Alt+Shift+3"),
    ShortcutDefinition("next_experiment", "Next experiment", "Select the following experiment.", "Alt+3"),
    ShortcutDefinition("complete_session", "Mark session complete", "Mark the selected session complete.", "Ctrl+1"),
    ShortcutDefinition("incomplete_session", "Mark session incomplete", "Mark the selected session incomplete.", "Ctrl+Shift+1"),
    ShortcutDefinition("complete_dataset", "Mark dataset complete", "Mark the selected dataset complete.", "Ctrl+2"),
    ShortcutDefinition("incomplete_dataset", "Mark dataset incomplete", "Mark the selected dataset incomplete.", "Ctrl+Shift+2"),
    ShortcutDefinition("complete_experiment", "Mark experiment complete", "Mark the selected experiment complete.", "Ctrl+3"),
    ShortcutDefinition("incomplete_experiment", "Mark experiment incomplete", "Mark the selected experiment incomplete.", "Ctrl+Shift+3"),
    ShortcutDefinition("plot", "Plot", "Run the selected plot without extracting data.", "Ctrl+P"),
    ShortcutDefinition("plot_extract", "Plot and extract data", "Run the selected plot and extract its data.", "Ctrl+Shift+P"),
)

SHORTCUTS_BY_KEY = {definition.key: definition for definition in SHORTCUT_DEFINITIONS}
SETTINGS_GROUP = "KeyboardShortcuts"
AUTO_PLOT_ON_NAVIGATION_KEY = "auto_plot_on_navigation"
DEFAULT_AUTO_PLOT_ON_NAVIGATION = True
LEGACY_DEFAULT_SHORTCUTS = {
    "complete_session": "Ctrl+Alt+1",
    "incomplete_session": "Ctrl+Alt+Shift+1",
    "complete_dataset": "Ctrl+Alt+2",
    "incomplete_dataset": "Ctrl+Alt+Shift+2",
    "complete_experiment": "Ctrl+Alt+3",
    "incomplete_experiment": "Ctrl+Alt+Shift+3",
}


def default_shortcuts() -> dict[str, str]:
    return {definition.key: definition.default for definition in SHORTCUT_DEFINITIONS}


def normalize_shortcuts(shortcuts: dict[str, str]) -> dict[str, str]:
    """Validate and normalize an editable shortcut map to portable Qt text."""
    normalized: dict[str, str] = {}
    seen: set[str] = set()
    for definition in SHORTCUT_DEFINITIONS:
        value = str(shortcuts.get(definition.key, definition.default)).strip()
        sequence = QKeySequence(value)
        portable = sequence.toString(QKeySequence.SequenceFormat.PortableText)
        if value and not portable:
            raise ValueError(f"{definition.label} has an invalid shortcut: {value!r}")
        if portable in seen:
            raise ValueError(f"Shortcut {portable} is assigned more than once.")
        if portable:
            seen.add(portable)
        normalized[definition.key] = portable
    return normalized


class KeyboardShortcutController:
    """Owns main-window shortcuts and their QSettings persistence."""

    def __init__(self, gui):
        self.gui = gui
        self._shortcuts: dict[str, QShortcut] = {}
        self.auto_plot_on_navigation = self.load_auto_plot_on_navigation()
        self._plot_when_experiment_ready = False
        self._pending_experiment_id: str | None = None
        self.apply(self.load())

    @staticmethod
    def load(settings=None) -> dict[str, str]:
        settings = settings or QSettings()
        values = default_shortcuts()
        settings.beginGroup(SETTINGS_GROUP)
        stored_plot = ""
        has_plot_extract = False
        for definition in SHORTCUT_DEFINITIONS:
            stored = settings.value(definition.key, "", type=str)
            if definition.key == "plot":
                stored_plot = stored
            elif definition.key == "plot_extract":
                has_plot_extract = bool(stored)
            if stored:
                values[definition.key] = stored
        settings.endGroup()
        # The initial shortcut release used Ctrl+Shift+P for plot-only. Move
        # that uncustomized legacy default to Ctrl+P before adding extraction.
        if not has_plot_extract and stored_plot == "Ctrl+Shift+P":
            values["plot"] = "Ctrl+P"
        # Move bindings that matched the earlier completion defaults to their
        # simpler replacements, without changing genuinely customized keys.
        for key, legacy_default in LEGACY_DEFAULT_SHORTCUTS.items():
            if values[key] == legacy_default:
                values[key] = default_shortcuts()[key]
        try:
            return normalize_shortcuts(values)
        except ValueError:
            return default_shortcuts()

    @staticmethod
    def save(shortcuts: dict[str, str], settings=None) -> dict[str, str]:
        return KeyboardShortcutController.save_preferences(shortcuts, settings=settings)

    @staticmethod
    def load_auto_plot_on_navigation(settings=None) -> bool:
        """Return whether keyboard navigation should immediately redraw the plot."""
        settings = settings or QSettings()
        settings.beginGroup(SETTINGS_GROUP)
        enabled = settings.value(AUTO_PLOT_ON_NAVIGATION_KEY, DEFAULT_AUTO_PLOT_ON_NAVIGATION, type=bool)
        settings.endGroup()
        return enabled

    @staticmethod
    def save_preferences(
        shortcuts: dict[str, str],
        auto_plot_on_navigation: bool = DEFAULT_AUTO_PLOT_ON_NAVIGATION,
        settings=None,
    ) -> dict[str, str]:
        settings = settings or QSettings()
        normalized = normalize_shortcuts(shortcuts)
        settings.beginGroup(SETTINGS_GROUP)
        for key, value in normalized.items():
            settings.setValue(key, value)
        settings.setValue(AUTO_PLOT_ON_NAVIGATION_KEY, bool(auto_plot_on_navigation))
        settings.endGroup()
        settings.sync()
        return normalized

    def apply(self, shortcuts: dict[str, str], auto_plot_on_navigation: bool | None = None) -> None:
        normalized = normalize_shortcuts(shortcuts)
        if auto_plot_on_navigation is not None:
            self.auto_plot_on_navigation = bool(auto_plot_on_navigation)
        for definition in SHORTCUT_DEFINITIONS:
            shortcut = self._shortcuts.get(definition.key)
            if shortcut is None:
                shortcut = QShortcut(self.gui)
                shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
                shortcut.activated.connect(lambda key=definition.key: self._dispatch(key))
                self._shortcuts[definition.key] = shortcut
            shortcut.setKey(QKeySequence(normalized[definition.key]))
        self._update_control_tooltips(normalized)

    def _update_control_tooltips(self, shortcuts: dict[str, str]) -> None:
        """Keep visible shortcut guidance synchronized with user preferences."""
        selection = getattr(self.gui, "data_selection_widget", None)
        if selection is not None:
            for level, widget in (
                ("session", selection.session_combo),
                ("dataset", selection.dataset_combo),
                ("experiment", selection.experiment_combo),
            ):
                previous = shortcuts[f"previous_{level}"] or "disabled"
                next_item = shortcuts[f"next_{level}"] or "disabled"
                widget.setToolTip(f"Select a {level}. Previous: {previous}; next: {next_item}.")
        plot_widget = getattr(self.gui, "plot_widget", None)
        plot_button = getattr(plot_widget, "plot_button", None)
        if plot_button is not None:
            shortcut = shortcuts["plot"] or "disabled"
            plot_button.setToolTip(f"Run the selected plot without extracting data. Shortcut: {shortcut}.")
        extract_button = getattr(plot_widget, "get_data_button", None)
        if extract_button is not None:
            shortcut = shortcuts["plot_extract"] or "disabled"
            extract_button.setToolTip(f"Run the selected plot and extract its data. Shortcut: {shortcut}.")

    def _dispatch(self, key: str) -> None:
        if key.startswith("previous_") or key.startswith("next_"):
            level = key.removeprefix("previous_").removeprefix("next_")
            direction = -1 if key.startswith("previous_") else 1
            changed = self.gui.step_data_selection(level, direction)
            if changed and self.auto_plot_on_navigation:
                self._plot_after_navigation()
        elif key.startswith("complete_") or key.startswith("incomplete_"):
            level = key.removeprefix("complete_").removeprefix("incomplete_")
            self.gui.set_current_completion_status(level, key.startswith("complete_"))
        elif key == "plot":
            self.gui.plot_controller.plot_data()
        elif key == "plot_extract":
            self.gui.plot_controller.get_raw_data()

    def _plot_after_navigation(self) -> None:
        """Plot now, or wait for an asynchronously loading experiment."""
        if self.gui.current_experiment is None:
            self._plot_when_experiment_ready = True
            selection = getattr(self.gui, "data_selection_widget", None)
            experiment_combo = getattr(selection, "experiment_combo", None)
            if experiment_combo is not None:
                self._pending_experiment_id = experiment_combo.currentData(Qt.ItemDataRole.UserRole)
            return
        self.gui.plot_controller.plot_data()

    def on_experiment_load_finished(self) -> None:
        """Render a keyboard-requested plot after its experiment is ready."""
        if not self._plot_when_experiment_ready:
            return
        self._plot_when_experiment_ready = False
        expected_experiment_id = self._pending_experiment_id
        self._pending_experiment_id = None
        current_experiment = self.gui.current_experiment
        if expected_experiment_id and getattr(current_experiment, "id", None) != expected_experiment_id:
            return
        if self.auto_plot_on_navigation and current_experiment is not None:
            self.gui.plot_controller.plot_data()

    def on_experiment_load_failed(self) -> None:
        """Discard a deferred plot when its requested experiment did not load."""
        self._plot_when_experiment_ready = False
        self._pending_experiment_id = None
