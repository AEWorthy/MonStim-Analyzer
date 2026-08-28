"""Unified, draft-backed home for program, global analysis, and profiles."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from monstim_gui.dialogs.preferences import LatencyWindowPresetEditor, MWaveWindowNamesEditor
from monstim_gui.dialogs.program_settings import ProgramSettingsDialog
from monstim_gui.io.config_repository import ConfigRepository
from monstim_gui.managers.profile_manager import ProfileManager, ProfileRecord
from monstim_gui.widgets.profile_library import ProfileLibraryWidget
from monstim_gui.widgets.settings_controls import DraftField, OverrideField
from monstim_signals.core.configuration import ResolvedConfig

GLOBAL_CATEGORIES = {
    "Plot appearance": {
        "bin_size",
        "time_window",
        "pre_stim_time",
        "default_method",
        "title_font_size",
        "axis_label_font_size",
        "tick_font_size",
        "m_color",
        "h_color",
        "latency_window_style",
        "subplot_adjust_args",
    },
    "Signal processing": {"butter_filter_args"},
    "M-max": {"m_max_args"},
    "Data defaults": {"default_channel_names", "preferred_date_format"},
}

GLOBAL_GROUPS = {
    "Plot appearance": (
        (
            "EMG timing and calculation",
            "Controls the default time range and amplitude method used when displaying EMG recordings.",
            ("bin_size", "time_window", "pre_stim_time", "default_method"),
        ),
        (
            "Annotations and plot style",
            "Controls font sizes, response colors, and the line style used to mark latency windows on plots.",
            ("title_font_size", "axis_label_font_size", "tick_font_size", "m_color", "h_color", "latency_window_style"),
        ),
        (
            "Figure layout",
            "Fine-tune outer margins and spacing when plots are exported or displayed in figure layouts.",
            ("subplot_adjust_args",),
        ),
    ),
    "Signal processing": (
        (
            "EMG band-pass filter",
            "Frequency limits and filter order applied to raw EMG before filtered analyses and plots.",
            ("butter_filter_args",),
        ),
    ),
    "M-max": (
        (
            "M-max estimation",
            "Controls plateau detection used to estimate M-max. Change these only when you understand the analysis consequences.",
            ("m_max_args",),
        ),
    ),
    "Data defaults": (
        (
            "Imported-data defaults",
            "Names and date parsing defaults applied when new datasets are imported.",
            ("default_channel_names", "preferred_date_format"),
        ),
    ),
}

SETTING_HELP = {
    "bin_size": "Stimulus-voltage bin width in volts.",
    "time_window": "Duration of EMG data shown after stimulus, in milliseconds.",
    "pre_stim_time": "EMG data shown before stimulus, in milliseconds.",
    "default_method": "Default amplitude calculation for new analyses.",
    "title_font_size": "Font size used for plot titles, in points.",
    "axis_label_font_size": "Font size used for x- and y-axis labels, in points.",
    "tick_font_size": "Font size used for axis tick labels, in points.",
    "m_color": "Color used for M-response annotations and traces.",
    "h_color": "Color used for H-reflex annotations and traces.",
    "latency_window_style": "Line style used to draw latency-window boundaries.",
    "subplot_adjust_args": "Figure margins and spacing for exported and displayed plots.",
    "butter_filter_args": "Band-pass filter settings used before analysis.",
    "m_max_args": "Parameters used to estimate the M-max plateau.",
    "default_channel_names": "Default names for newly imported EMG channels.",
    "preferred_date_format": "Expected compact date format when parsing datasets.",
}


class SettingsCenter(QDialog):
    """Composable settings shell. Widgets own drafts; repositories own writes."""

    settings_applied = Signal()

    def __init__(self, default_config_file: str, parent=None, config_repo: ConfigRepository | None = None):
        super().__init__(parent)
        self.setWindowTitle("Settings Center")
        self.setModal(True)
        # Longer sections scroll within their page instead of forcing the
        # dialog off screen.
        self.resize(1180, 640)
        self.setMinimumHeight(360)
        self.config_repo = config_repo or ConfigRepository(default_config_file)
        self.global_config = self.config_repo.read_config()
        self.shipped_config = self.config_repo.read_default_config()
        self.profile_manager = ProfileManager(reference_config=self.global_config)
        self._records = self.profile_manager.list_profile_records()
        self._profile_drafts: dict[str, dict] = {}
        self._deleted_profiles: set[str] = set()
        self._current_profile_path: str | None = None
        self._search_targets: list[tuple[int, QTabWidget | None, int | None, QWidget | None, str]] = []
        self.setStyleSheet(
            """
            QLineEdit { padding: 5px 7px; }
            QListWidget#settingsNavigation { background: transparent; border: 0; padding: 4px 8px; outline: 0; }
            QListWidget#settingsNavigation::item { padding: 9px 10px; margin: 2px 0; border-radius: 4px; font-weight: 600; }
            QListWidget#settingsNavigation::item:selected { background: #304553; border-left: 3px solid #e7785b; padding-left: 7px; }
            QListWidget#settingsNavigation::item:hover:!selected { background: rgba(255, 255, 255, 0.06); }
            QTabWidget::pane { border: 1px solid #3c434b; border-radius: 6px; top: -1px; }
            QTabBar::tab { padding: 8px 14px; margin-right: 2px; border: 0; border-bottom: 3px solid transparent; font-weight: 600; color: #bfc7cf; }
            QTabBar::tab:selected { color: #ffffff; background: #2c333a; border-bottom-color: #e7785b; }
            QTabBar::tab:hover:!selected { color: #ffffff; background: rgba(255, 255, 255, 0.06); }
            QGroupBox { font-weight: 700; border: 1px solid #3c434b; border-radius: 6px; margin-top: 12px; padding-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QLabel[sectionNote="true"] { color: #aeb7c1; }
            QLabel#pageTitle, QLabel#profileLibraryTitle { font-size: 15px; font-weight: 700; }
            QLabel#pageDescription, QLabel#profileLibraryDescription, QLabel#profileLibraryStatus { color: #aeb7c1; }
            QTreeWidget#profileLibraryTable { border: 1px solid #3c434b; border-radius: 5px; }
            QTreeWidget#profileLibraryTable::item { padding: 7px 5px; }
            QTreeWidget#profileLibraryTable::item:selected { background: #304553; }
            """
        )
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        self.search = QLineEdit(self)
        self.search.setPlaceholderText("Search settings (for example: font, filter, M-max, latency)…")
        self.search.setToolTip("Find a specific setting by name, category, or description.")
        root.addWidget(self.search)
        body = QHBoxLayout()
        body.setSpacing(14)
        self.navigation = QListWidget(self)
        self.navigation.setObjectName("settingsNavigation")
        self.navigation.setMinimumWidth(210)
        self.navigation.setMaximumWidth(250)
        self.pages = QStackedWidget(self)
        for label, page in (
            ("Program", self._build_program_page()),
            ("Global Analysis", self._build_global_page()),
            ("Profiles", self._build_profiles_page()),
        ):
            self.navigation.addItem(label)
            self.pages.addWidget(page)
        self.navigation.item(0).setToolTip("Application appearance, loading performance, saved-state tracking, and recovery.")
        self.navigation.item(1).setToolTip("Global defaults used by every analysis profile unless a profile overrides a setting.")
        self.navigation.item(2).setToolTip("Create, edit, import, export, and manage named analysis-profile overrides.")
        self._register_search_target(
            2,
            None,
            None,
            None,
            "profiles profile name profile description latency preset analysis overrides "
            "profile import profile export duplicate profile user profile",
        )
        self.navigation.setCurrentRow(0)
        self.navigation.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.search.textChanged.connect(self._filter_navigation)
        body.addWidget(self.navigation, 0)
        body.addWidget(self.pages, 1)
        root.addLayout(body, 1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Apply, self
        )
        self.reset_section_button = QPushButton("Reset Current Section", self)
        self.reset_section_button.setToolTip("Restore the currently visible settings section to its starting values.")
        buttons.addButton(self.reset_section_button, QDialogButtonBox.ButtonRole.ResetRole)
        buttons.accepted.connect(self._accept_after_apply)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply)
        buttons.button(QDialogButtonBox.StandardButton.Apply).setToolTip("Validate and save changes without closing Settings Center.")
        self.reset_section_button.clicked.connect(self._reset_current_section)
        root.addWidget(buttons)
        self._install_combo_wheel_guards(self)

    def _build_program_page(self) -> QWidget:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        host = QWidget(scroll)
        host_layout = QVBoxLayout(host)
        host_layout.setContentsMargins(10, 8, 10, 10)
        host_layout.addWidget(self._page_heading("Program", "Appearance, performance, privacy, and saved application state.", host))
        self.program_page = ProgramSettingsDialog(host, embedded=True)
        host_layout.addWidget(self.program_page, 1)
        scroll.setWidget(host)
        tabs = self.program_page.findChild(QTabWidget)
        if tabs is not None:
            terms = (
                "appearance display window scaling font interface center windows max window screen usage "
                "combo tooltip duration auto scale manual scale factor base font size left panel width",
                "performance opengl loading cache warm-up lazy parallel lazy open raw hdf5 plot cache warm-up",
                "privacy data tracking session restoration recent files import export clear saved data analysis profile tracking",
            )
            for index, text in enumerate(terms):
                self._register_search_target(0, tabs, index, None, text)
        return scroll

    @staticmethod
    def _page_heading(title: str, description: str, parent: QWidget) -> QWidget:
        heading = QWidget(parent)
        heading.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        layout = QVBoxLayout(heading)
        layout.setContentsMargins(2, 0, 2, 4)
        layout.setSpacing(2)
        title_label = QLabel(title, heading)
        title_label.setObjectName("pageTitle")
        description_label = QLabel(description, heading)
        description_label.setObjectName("pageDescription")
        description_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(description_label)
        return heading

    def _register_search_target(
        self,
        page_index: int,
        tabs: QTabWidget | None,
        tab_index: int | None,
        widget: QWidget | None,
        terms: str,
    ) -> None:
        self._search_targets.append((page_index, tabs, tab_index, widget, terms))

    def _build_global_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(8)
        layout.addWidget(
            self._page_heading(
                "Global Analysis",
                "Defaults shared by every analysis profile. Profiles may selectively override eligible values.",
                page,
            )
        )
        tabs = QTabWidget(page)
        self.global_tabs = tabs
        self.global_fields: dict[str, DraftField] = {}
        for category, groups in GLOBAL_GROUPS.items():
            tab = QWidget(tabs)
            tab_layout = QVBoxLayout(tab)
            tab_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
            for title, description, keys in groups:
                group = QGroupBox(title, tab)
                group.setToolTip(description)
                group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
                group_layout = QVBoxLayout(group)
                group_layout.setContentsMargins(10, 8, 10, 8)
                group_layout.setSpacing(7)
                note = QLabel(description, group)
                note.setWordWrap(True)
                note.setToolTip(description)
                note.setProperty("sectionNote", True)
                group_layout.addWidget(note)
                form = QFormLayout()
                form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                form.setContentsMargins(0, 0, 0, 0)
                form.setVerticalSpacing(6)
                for key in keys:
                    if key not in self.global_config:
                        continue
                    field = DraftField(
                        key.replace("_", " ").title(),
                        self.global_config[key],
                        group,
                        key=key,
                        help_text=SETTING_HELP.get(key, ""),
                    )
                    form.addRow(field)
                    self.global_fields[key] = field
                group_layout.addLayout(form)
                group.setMinimumHeight(group.sizeHint().height())
                tab_layout.addWidget(group)
                search_text = " ".join((title, description, *keys, *(SETTING_HELP.get(key, "") for key in keys)))
                self._register_search_target(1, tabs, tabs.count(), group, search_text)
            tab_layout.addStretch()
            tab_scroll = QScrollArea(tabs)
            tab_scroll.setWidgetResizable(True)
            tab_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            tab_scroll.setWidget(tab)
            tabs.addTab(tab_scroll, category)
            tabs.setTabToolTip(tabs.count() - 1, f"Configure {category.casefold()} defaults.")
        special = QWidget(tabs)
        special_layout = QVBoxLayout(special)
        special_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        special_layout.setSpacing(10)

        preset_group = QGroupBox("Latency Window Presets", special)
        preset_layout = QVBoxLayout(preset_group)
        preset_note = QLabel("Create and edit reusable timing-window templates used by experiments and analysis profiles.", preset_group)
        preset_note.setWordWrap(True)
        preset_layout.addWidget(preset_note)
        self.preset_editor = LatencyWindowPresetEditor(self.global_config.get("latency_window_presets", {}), special)
        self.preset_editor.setToolTip("Create reusable latency-window templates for experiments and profiles.")
        preset_layout.addWidget(self.preset_editor, 1)
        special_layout.addWidget(preset_group, 1)

        names_group = QGroupBox("M-wave Recognition Names", special)
        names_layout = QVBoxLayout(names_group)
        self.mwave_editor = MWaveWindowNamesEditor(
            self.global_config.get("m_wave_window_names", []), self.shipped_config.get("m_wave_window_names", []), names_group
        )
        names_layout.addWidget(self.mwave_editor)
        special_layout.addWidget(names_group)
        latency_scroll = QScrollArea(tabs)
        latency_scroll.setWidgetResizable(True)
        latency_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        latency_scroll.setWidget(special)
        tabs.addTab(latency_scroll, "Latency windows")
        self._register_search_target(
            1,
            tabs,
            tabs.count() - 1,
            preset_group,
            "latency windows preset timing start duration color template duplicate copy paste",
        )
        self._register_search_target(
            1,
            tabs,
            tabs.count() - 1,
            names_group,
            "m-wave recognition names m response mmax recognized latency window add remove defaults",
        )
        layout.addWidget(tabs)
        return page

    def _build_profiles_page(self) -> QWidget:
        page = QWidget(self)
        layout = QHBoxLayout(page)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(14)
        self.profile_library = ProfileLibraryWidget(page)
        self.profile_library.selected.connect(self._select_profile)
        self.profile_library.add_requested.connect(self._new_profile)
        self.profile_library.duplicate_requested.connect(self._duplicate_profile)
        self.profile_library.delete_requested.connect(self._delete_profile)
        self.profile_library.import_requested.connect(self._import_profile)
        self.profile_library.export_requested.connect(self._export_profile)
        layout.addWidget(self.profile_library, 1)
        self.profile_editor_holder = QWidget(page)
        self._profile_editor_layout = QVBoxLayout(self.profile_editor_holder)
        self._profile_editor_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.profile_editor_holder, 3)
        self._refresh_profile_library()
        return page

    def _filter_navigation(self, text: str) -> None:
        query = " ".join(text.casefold().replace("_", " ").replace("-", " ").split())
        tabs_with_matches: dict[QTabWidget, set[int]] = {}
        all_tabs: set[QTabWidget] = set()
        first_match: tuple[int, QTabWidget | None, int | None] | None = None
        first_match_rank = -1
        for page_index, tabs, tab_index, widget, terms in self._search_targets:
            searchable = " ".join(terms.casefold().replace("_", " ").replace("-", " ").split())
            # Search phrases must occur together in one setting/category
            # description.  Matching each word independently made requests
            # such as "profile import" jump to unrelated Program settings.
            matches = not query or query in searchable
            if widget is not None:
                widget.setVisible(matches)
            if tabs is not None and tab_index is not None and matches:
                tabs_with_matches.setdefault(tabs, set()).add(tab_index)
            if tabs is not None:
                all_tabs.add(tabs)
            match_rank = 2 if searchable.startswith(query) else 1
            if matches and match_rank > first_match_rank:
                first_match = (page_index, tabs, tab_index)
                first_match_rank = match_rank
        for tabs in all_tabs:
            visible_indices = tabs_with_matches.get(tabs, set())
            for index in range(tabs.count()):
                tabs.setTabVisible(index, not query or index in visible_indices)
        if first_match is not None and query:
            page_index, tabs, tab_index = first_match
            self.navigation.setCurrentRow(page_index)
            if tabs is not None and tab_index is not None:
                tabs.setCurrentIndex(tab_index)

    def _refresh_profile_library(self, selected_path: str | None = None) -> None:
        self.profile_library.set_records(self._records, selected_path or self._current_profile_path)

    def _save_current_profile_draft(self) -> None:
        if self._current_profile_path and hasattr(self, "profile_name"):
            data = {
                "name": self.profile_name.text().strip(),
                "description": self.profile_description.toPlainText().strip(),
                "latency_window_preset": self.profile_preset.currentText(),
                "analysis_parameters": {},
            }
            for key, field in self.profile_override_fields.items():
                overridden, value = field.value()
                if overridden:
                    data["analysis_parameters"][key] = value
            self._profile_drafts[self._current_profile_path] = data

    def _select_profile(self, record: ProfileRecord | None) -> None:
        self._save_current_profile_draft()
        self._current_profile_path = record.path if record else None
        if not record:
            self._set_profile_editor(QWidget(self.profile_editor_holder))
            return
        data = copy.deepcopy(self._profile_drafts.get(record.path, record.data))
        editor = QTabWidget(self.profile_editor_holder)
        overview = QWidget(editor)
        layout = QVBoxLayout(overview)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)
        layout.addWidget(
            self._page_heading(
                data.get("name", record.name),
                "Built-in profiles are read-only; duplicate one to create an editable user profile."
                if record.read_only
                else "Edit this user-owned profile without changing the active main-window profile.",
                overview,
            )
        )
        form = QFormLayout()
        self.profile_name = QLineEdit(str(data.get("name", record.name)), editor)
        self.profile_name.setToolTip("A descriptive name shown in the main profile selector.")
        self.profile_description = QTextEdit(str(data.get("description", "")), editor)
        self.profile_description.setFixedHeight(65)
        self.profile_description.setToolTip("Explain the experimental setup or analysis intent for this profile.")
        self.profile_preset = QComboBox(editor)
        self.profile_preset.addItems(sorted(self.global_config.get("latency_window_presets", {})))
        self.profile_preset.setCurrentText(str(data.get("latency_window_preset", "")))
        self.profile_preset.setToolTip("Choose the saved latency-window template this profile should apply.")
        form.addRow("Name", self.profile_name)
        form.addRow("Description", self.profile_description)
        form.addRow("Latency preset", self.profile_preset)
        layout.addLayout(form)
        self.profile_override_fields: dict[str, OverrideField] = {}
        params = data.get("analysis_parameters", {})
        override_rows = [
            (category, key) for category, keys in GLOBAL_CATEGORIES.items() for key in sorted(keys & params.keys() & self.global_config.keys())
        ]
        summary_group = QGroupBox(f"Overrides ({len(override_rows)})", overview)
        summary_layout = QVBoxLayout(summary_group)
        summary = QLabel(summary_group)
        summary.setWordWrap(True)
        if override_rows:
            summary.setText("Each row compares this profile's explicit value with the effective Global Analysis default.")
            summary_layout.addWidget(summary)
            self.profile_override_summary = QTableWidget(len(override_rows), 4, summary_group)
            self.profile_override_summary.setObjectName("profileOverrideSummary")
            self.profile_override_summary.setHorizontalHeaderLabels(("Category", "Setting", "Profile override", "Global default"))
            self.profile_override_summary.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            self.profile_override_summary.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.profile_override_summary.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            self.profile_override_summary.setAlternatingRowColors(True)
            self.profile_override_summary.verticalHeader().setVisible(False)
            self.profile_override_summary.verticalHeader().setDefaultSectionSize(28)
            header = self.profile_override_summary.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
            for row, (category, key) in enumerate(override_rows):
                values = (
                    category,
                    key.replace("_", " ").title(),
                    self._display_setting_value(params[key]),
                    self._display_setting_value(self.global_config[key]),
                )
                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    item.setToolTip(value)
                    self.profile_override_summary.setItem(row, column, item)
            self.profile_override_summary.setFixedHeight(
                self.profile_override_summary.horizontalHeader().height()
                + self.profile_override_summary.verticalHeader().defaultSectionSize() * len(override_rows)
                + 4
            )
            summary_layout.addWidget(self.profile_override_summary)
        else:
            summary.setText("This profile currently inherits all eligible Global Analysis settings.")
            summary_layout.addWidget(summary)
        layout.addWidget(summary_group)
        explanation = QLabel("Use a category tab to inspect or change inherited analysis settings.", overview)
        explanation.setObjectName("pageDescription")
        layout.addWidget(explanation)
        for category, keys in GLOBAL_CATEGORIES.items():
            category_page = QWidget(editor)
            category_layout = QVBoxLayout(category_page)
            category_layout.setContentsMargins(12, 10, 12, 10)
            category_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
            group = QGroupBox(category, category_page)
            group_layout = QVBoxLayout(group)
            group_layout.setSpacing(5)
            for key in sorted(keys & self.global_config.keys()):
                field = OverrideField(
                    key.replace("_", " ").title(),
                    self.global_config[key],
                    params.get(key),
                    overridden=key in params,
                    parent=group,
                    key=key,
                )
                field.override_box.setEnabled(not record.read_only)
                if record.read_only:
                    field.field.setEnabled(False)
                self.profile_override_fields[key] = field
                group_layout.addWidget(field)
            category_layout.addWidget(group)
            category_layout.addStretch()
            category_scroll = QScrollArea(editor)
            category_scroll.setWidgetResizable(True)
            category_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            category_scroll.setWidget(category_page)
            editor.addTab(category_scroll, category)
        for widget in (self.profile_name, self.profile_description, self.profile_preset):
            widget.setEnabled(not record.read_only)
        layout.addStretch()
        overview_scroll = QScrollArea(editor)
        overview_scroll.setWidgetResizable(True)
        overview_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        overview_scroll.setWidget(overview)
        editor.insertTab(0, overview_scroll, "Overview")
        editor.setCurrentIndex(0)
        self._set_profile_editor(editor)
        self._install_combo_wheel_guards(editor)

    @staticmethod
    def _display_setting_value(value) -> str:
        """Render a compact, user-facing value for the profile comparison table."""
        if isinstance(value, dict):
            return ", ".join(f"{key}: {SettingsCenter._display_setting_value(item)}" for key, item in value.items())
        if isinstance(value, list | tuple):
            return ", ".join(SettingsCenter._display_setting_value(item) for item in value)
        if isinstance(value, str):
            return value.removeprefix("tab:")
        return str(value)

    def _set_profile_editor(self, editor: QWidget) -> None:
        """Replace the profile editor without leaving nested full-page scrollbars."""
        while self._profile_editor_layout.count():
            item = self._profile_editor_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self._profile_editor_layout.addWidget(editor)

    def _install_combo_wheel_guards(self, root: QWidget) -> None:
        """Prevent accidental setting changes while scrolling a page."""
        # PySide6 accepts one Qt type per findChildren() call, unlike Python's
        # isinstance() tuple syntax.  Gather both types explicitly.
        for setting_widget in [
            *root.findChildren(QComboBox),
            *root.findChildren(QAbstractSpinBox),
        ]:
            setting_widget.installEventFilter(self)

    def eventFilter(self, watched, event):
        if isinstance(watched, (QComboBox, QAbstractSpinBox)) and event.type() == QEvent.Type.Wheel:
            event.ignore()
            return True
        return super().eventFilter(watched, event)

    def _new_profile(self) -> None:
        name, accepted = QInputDialog.getText(self, "New Profile", "Profile name:")
        if not accepted or not name.strip():
            return
        self._save_current_profile_draft()
        path = str(Path(self.profile_manager.user_dir) / self.profile_manager._filename_for(name.strip()))
        if any(record.path == path for record in self._records):
            QMessageBox.warning(self, "Profile exists", "Choose a different profile name.")
            return
        record = ProfileRecord(name.strip(), path, {"name": name.strip(), "description": "", "analysis_parameters": {}}, "User", False)
        self._records.append(record)
        self._current_profile_path = None
        self._refresh_profile_library(path)

    def _duplicate_profile(self, record: ProfileRecord | None) -> None:
        if not record:
            return
        name, accepted = QInputDialog.getText(self, "Duplicate Profile", "New profile name:", text=f"{record.name} Copy")
        if not accepted or not name.strip():
            return
        self._save_current_profile_draft()
        path = str(Path(self.profile_manager.user_dir) / self.profile_manager._filename_for(name.strip()))
        data = copy.deepcopy(self._profile_drafts.get(record.path, record.data))
        data["name"] = name.strip()
        clone = ProfileRecord(name.strip(), path, data, "User", False)
        self._records.append(clone)
        self._profile_drafts[path] = data
        self._current_profile_path = None
        self._refresh_profile_library(path)

    def _delete_profile(self, record: ProfileRecord | None) -> None:
        if not record:
            return
        if (
            QMessageBox.question(
                self,
                "Delete profile",
                f"Delete '{record.name}' when settings are applied?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        self._deleted_profiles.add(record.path)
        self._records = [item for item in self._records if item.path != record.path]
        self._profile_drafts.pop(record.path, None)
        self._current_profile_path = None
        self._refresh_profile_library()

    def _import_profile(self) -> None:
        source, _ = QFileDialog.getOpenFileName(self, "Import Profile", filter="YAML profiles (*.yml *.yaml)")
        if not source:
            return
        try:
            data = self.profile_manager.validate_profile(self.profile_manager.load_profile(source))
        except (OSError, ValueError, yaml.YAMLError) as error:
            QMessageBox.critical(self, "Cannot import profile", str(error))
            return
        name = data["name"]
        if any(item.name == name and item.source == "User" for item in self._records):
            prompt = QMessageBox(self)
            prompt.setIcon(QMessageBox.Icon.Question)
            prompt.setWindowTitle("Profile already exists")
            prompt.setText(f"A user profile named '{name}' already exists.")
            replace = prompt.addButton("Replace", QMessageBox.ButtonRole.AcceptRole)
            keep_both = prompt.addButton("Keep Both", QMessageBox.ButtonRole.ActionRole)
            cancel = prompt.addButton(QMessageBox.StandardButton.Cancel)
            prompt.exec()
            if prompt.clickedButton() is cancel:
                return
            if prompt.clickedButton() is keep_both:
                number = 2
                base = name
                while any(item.name == f"{base} {number}" for item in self._records):
                    number += 1
                name = f"{base} {number}"
                data["name"] = name
            elif prompt.clickedButton() is not replace:
                return
        path = str(Path(self.profile_manager.user_dir) / self.profile_manager._filename_for(name))
        self._records = [item for item in self._records if item.path != path]
        self._records.append(ProfileRecord(name, path, data, "User", False))
        self._profile_drafts[path] = data
        self._current_profile_path = None
        self._refresh_profile_library(path)

    def _export_profile(self, record: ProfileRecord | None) -> None:
        if not record:
            return
        self._save_current_profile_draft()
        destination, _ = QFileDialog.getSaveFileName(self, "Export Profile", f"{record.name}.yml", "YAML profiles (*.yml)")
        if not destination:
            return
        data = self._profile_drafts.get(record.path, record.data)
        try:
            self.profile_manager.validate_profile(data)
            with open(destination, "w", encoding="utf-8") as handle:
                yaml.safe_dump(data, handle, sort_keys=False)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Cannot export profile", str(error))

    def _global_draft(self) -> dict:
        data = copy.deepcopy(self.global_config)
        for key, field in self.global_fields.items():
            try:
                data[key] = field.value()
            except (TypeError, ValueError) as error:
                raise ValueError(f"{field.label}: {error}") from error
        data["m_wave_window_names"] = self.mwave_editor.validate()
        data["latency_window_presets"] = self.preset_editor.get_presets()
        return data

    def apply(self) -> bool:
        self._save_current_profile_draft()
        try:
            global_draft = self._global_draft()
            ResolvedConfig(global_draft)
            self.config_repo.write_config(global_draft)
            self.profile_manager.migrate_legacy_profiles()
            for path in self._deleted_profiles:
                self.profile_manager.delete_profile(path)
            for record in self._records:
                if record.read_only:
                    continue
                data = self._profile_drafts.get(record.path, record.data)
                self.profile_manager.save_profile(data, record.path)
        except (OSError, ValueError, yaml.YAMLError) as error:
            QMessageBox.critical(self, "Cannot apply settings", str(error))
            return False
        self.global_config = self.config_repo.read_config()
        self._deleted_profiles.clear()
        self._profile_drafts.clear()
        self._records = self.profile_manager.list_profile_records()
        self.settings_applied.emit()
        return True

    def _accept_after_apply(self) -> None:
        if self.apply():
            super().accept()

    def _reset_current_section(self) -> None:
        index = self.pages.currentIndex()
        if index == 0:
            self.program_page.reset_to_defaults()
        elif index == 1:
            for field in self.global_fields.values():
                field.reset()
        elif index == 2 and self._current_profile_path:
            record = next((item for item in self._records if item.path == self._current_profile_path), None)
            if record:
                self._profile_drafts[self._current_profile_path] = copy.deepcopy(record.data)
                self._select_profile(record)
