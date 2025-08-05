import ast
import copy
import logging

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from monstim_gui.dialogs.base import COLOR_OPTIONS, TAB_COLOR_NAMES
from monstim_gui.io.config_repository import ConfigRepository
from monstim_gui.managers.profile_manager import ProfileManager
from monstim_signals.core import LatencyWindow

STIMULUS_OPTIONS = ["Force", "Length", "Electrical", "Optical"]


class StimulusSelectorWidget(QWidget):
    def __init__(self, selected=None, parent=None):
        super().__init__(parent)
        self.checkboxes = {}
        layout = QHBoxLayout(self)
        for signal in STIMULUS_OPTIONS:
            cb = QCheckBox(signal)
            layout.addWidget(cb)
            self.checkboxes[signal] = cb
        if selected:
            for sig in selected:
                if sig in self.checkboxes:
                    self.checkboxes[sig].setChecked(True)
        layout.addStretch(1)

    def get_selected(self):
        return [sig for sig, cb in self.checkboxes.items() if cb.isChecked()]

    def set_selected(self, selected):
        for sig, cb in self.checkboxes.items():
            cb.setChecked(sig in selected)


class LatencyWindowPresetEditor(QWidget):
    """Widget to create and edit latency window presets."""

    def __init__(self, presets: dict[str, list[dict]] | None = None, parent=None):
        super().__init__(parent)
        self.presets: list[list[LatencyWindow]] = []
        self.preset_combo = QComboBox()
        self.preset_combo.setEditable(True)
        # Prevent edited text from creating new items and keep the list in sync
        self.preset_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Track which preset is currently displayed so we can save it when the
        # selection changes. ``QComboBox.currentIndexChanged`` is emitted after
        # the widget updates its index, so we cannot rely on the combo box to
        # provide the previous index.
        self._current_index: int | None = None
        self.window_entries: list[
            tuple[
                QGroupBox,
                LatencyWindow,
                QLineEdit,
                QDoubleSpinBox,
                QDoubleSpinBox,
                QComboBox,
            ]
        ] = []
        self._init_data(presets or {})
        self._init_ui()

    def _init_data(self, presets: dict[str, list[dict]]) -> None:
        for name, windows in presets.items():
            self.preset_combo.addItem(name)
            win_objs = []
            for win in windows:
                win_objs.append(
                    LatencyWindow(
                        name=win.get("name", "Window"),
                        start_times=[float(win.get("start", 0.0))],
                        durations=[float(win.get("duration", 1.0))],
                        color=win.get("color", "black"),
                        linestyle=win.get("linestyle", ":"),
                    )
                )
            self.presets.append(win_objs)

        if self.preset_combo.count() == 0:
            self.preset_combo.addItem("default")
            self.presets.append([])

        if self.window_entries:
            self._save_current_preset()

    # ------------------------------------------------------------------
    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        control_row = QHBoxLayout()
        control_row.addWidget(QLabel("Preset:"))
        # Ensure preset names are fully visible and don't get clipped
        self.preset_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        # Provide some extra space for descriptive names without forcing the
        # combo box wider than the dialog.
        self.preset_combo.setMinimumContentsLength(20)
        self.preset_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        control_row.addWidget(self.preset_combo, 1)
        add_btn = QPushButton("Add")
        remove_btn = QPushButton("Remove")
        control_row.addWidget(add_btn)
        control_row.addWidget(remove_btn)
        layout.addLayout(control_row)

        self.preset_combo.currentIndexChanged.connect(self.load_preset)
        self.preset_combo.lineEdit().editingFinished.connect(self._commit_preset_name)  # type: ignore
        add_btn.clicked.connect(self.add_preset)
        remove_btn.clicked.connect(self.remove_preset)

        self.scroll: QScrollArea = QScrollArea()  # type: ignore
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.scroll_widget = QWidget()
        self.scroll_widget.setMinimumSize(0, 0)
        self.scroll_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll.setWidget(self.scroll_widget)
        layout.addWidget(self.scroll, 1)

        add_window_btn = QPushButton("Add Window")
        # QAbstractButton.clicked passes a boolean that we don't use. Wrap the
        # call in a lambda so ``add_window_group`` receives no arguments.
        add_window_btn.clicked.connect(lambda: self.add_window_group())
        layout.addWidget(add_window_btn)

        if self.preset_combo.count() > 0:
            self.load_preset(0, save=False)

    # ------------------------------------------------------------------
    # Preset operations
    # ------------------------------------------------------------------
    def add_preset(self) -> None:
        self._commit_preset_name()
        typed = self.preset_combo.currentText().strip()
        existing = {self.preset_combo.itemText(i) for i in range(self.preset_combo.count())}
        if typed and typed not in existing:
            name = typed
        else:
            name = self._unique_name("Preset")
        self.preset_combo.addItem(name)
        self.presets.append([])
        self.preset_combo.setCurrentIndex(self.preset_combo.count() - 1)
        self._current_index = self.preset_combo.currentIndex()

    def remove_preset(self) -> None:
        if self.preset_combo.count() == 0:
            return
        self._commit_preset_name()
        self._save_current_preset()
        idx = self.preset_combo.currentIndex()
        self.preset_combo.removeItem(idx)
        self.presets.pop(idx)
        if self.preset_combo.count() == 0:
            self.add_preset()
        else:
            self.preset_combo.setCurrentIndex(0)

    def load_preset(self, index: int, *, save: bool = True) -> None:
        """Load the preset at ``index`` into the editor."""
        if save and self._current_index is not None:
            self._save_current_preset()
        if index < 0 or index >= len(self.presets):
            return
        self._clear_windows()
        for win in self.presets[index]:
            self.add_window_group(copy.deepcopy(win))
        self._current_index = index

    def _save_current_preset(self) -> None:
        """Save the currently displayed preset back to ``self.presets``."""
        index = self._current_index if self._current_index is not None else -1
        if index < 0 or index >= len(self.presets):
            return
        windows: list[LatencyWindow] = []
        for (
            _,
            window,
            name_edit,
            start_spin,
            dur_spin,
            color_combo,
        ) in self.window_entries:
            window.name = name_edit.text().strip() or "Window"
            window.start_times = [start_spin.value()]
            window.durations = [dur_spin.value()]
            window.color = color_combo.currentData()
            windows.append(copy.deepcopy(window))
        self.presets[index] = windows

    def _unique_name(self, base: str) -> str:
        existing = {self.preset_combo.itemText(i) for i in range(self.preset_combo.count())}
        idx = 1
        name = f"{base} {idx}"
        while name in existing:
            idx += 1
            name = f"{base} {idx}"
        return name

    def _commit_preset_name(self) -> None:
        """Update the current item's text from the line edit."""
        index = self.preset_combo.currentIndex()
        if index >= 0:
            self.preset_combo.setItemText(index, self.preset_combo.currentText())

    # ------------------------------------------------------------------
    # Window operations
    # ------------------------------------------------------------------
    def _clear_windows(self) -> None:
        for grp, *_ in self.window_entries:
            grp.setParent(None)
            grp.deleteLater()
        self.window_entries.clear()

    def add_window_group(self, window: LatencyWindow | None = None, *, checked: bool | None = None) -> None:
        if window is None:
            window = LatencyWindow(
                name=f"Window {len(self.window_entries)+1}",
                start_times=[0.0],
                durations=[1.0],
                color="black",
                linestyle=":",
            )
        group = QGroupBox(window.name)
        group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        form = QFormLayout()
        name_edit = QLineEdit(window.name)
        start_spin = QDoubleSpinBox()
        start_spin.setDecimals(2)
        start_spin.setRange(-1000.0, 1000.0)
        start_spin.setSingleStep(0.05)
        start_spin.setValue(window.start_times[0])
        dur_spin = QDoubleSpinBox()
        dur_spin.setDecimals(2)
        dur_spin.setRange(0.0, 1000.0)
        dur_spin.setSingleStep(0.05)
        dur_spin.setValue(window.durations[0])
        color_combo = QComboBox()
        for color in COLOR_OPTIONS:
            display = color.replace("tab:", "")
            color_combo.addItem(display, userData=color)
        if window.color in COLOR_OPTIONS:
            color_combo.setCurrentIndex(COLOR_OPTIONS.index(window.color))
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(lambda: self._remove_window_group(group))
        form.addRow("Name", name_edit)
        form.addRow("Start", start_spin)
        form.addRow("Duration", dur_spin)
        form.addRow("Color", color_combo)
        form.addRow(remove_btn)
        group.setLayout(form)
        self.scroll_layout.addWidget(group)
        self.window_entries.append((group, window, name_edit, start_spin, dur_spin, color_combo))

    def _remove_window_group(self, group: QGroupBox) -> None:
        for i, (grp, *_) in enumerate(self.window_entries):
            if grp is group:
                self.window_entries.pop(i)
                break
        group.setParent(None)
        group.deleteLater()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_presets(self) -> dict[str, list[dict]]:
        self._save_current_preset()
        result: dict[str, list[dict]] = {}
        for idx in range(self.preset_combo.count()):
            name = self.preset_combo.itemText(idx)
            windows = []
            for win in self.presets[idx]:
                windows.append(
                    {
                        "name": win.name,
                        "start": float(win.start_times[0]),
                        "duration": float(win.durations[0]),
                        "color": win.color,
                        "linestyle": win.linestyle,
                    }
                )
            result[name] = windows
        return result


class PreferencesDialog(QDialog):
    def parse_field_value(self, value: str, global_value=None):
        """
        Try to parse the string value to match the type of the global config value.
        Handles int, float, list, dict, and falls back to string.
        """
        import ast

        if global_value is not None:
            # Try to match type of global config
            if isinstance(global_value, bool):
                return value.lower() in ("1", "true", "yes", "on")
            if isinstance(global_value, int):
                try:
                    return int(value)
                except Exception:
                    pass
            if isinstance(global_value, float):
                try:
                    return float(value)
                except Exception:
                    pass
            if isinstance(global_value, list):
                # Try to parse as list
                try:
                    parsed = ast.literal_eval(value)
                    if isinstance(parsed, list):
                        return [self.parse_field_value(str(v), global_value[0] if global_value else None) for v in parsed]
                except Exception:
                    # fallback: comma split
                    return [
                        self.parse_field_value(v.strip(), global_value[0] if global_value else None)
                        for v in value.split(",")
                        if v.strip()
                    ]
            if isinstance(global_value, dict):
                try:
                    parsed = ast.literal_eval(value)
                    if isinstance(parsed, dict):
                        # Recursively parse values
                        return {k: self.parse_field_value(str(v), global_value.get(k)) for k, v in parsed.items()}
                except Exception:
                    pass
        # Try to parse as int/float if no global type
        try:
            return int(value)
        except Exception:
            pass
        try:
            return float(value)
        except Exception:
            pass
        # Try to parse as list/dict
        try:
            parsed = ast.literal_eval(value)
            return parsed
        except Exception:
            pass
        return value

    def __init__(self, default_config_file, parent=None, config_repo=None):
        super().__init__()
        self.setModal(True)
        self.setWindowTitle("Preferences")
        self.default_config_file = default_config_file
        self.config_repo = config_repo or ConfigRepository(default_config_file)
        self.config = self.config_repo.read_config()
        self.profile_manager = ProfileManager(reference_config=self.config)
        self.active_profile_path = None
        self.active_profile_data = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.fields = {}

        # --- Profile Selector ---
        profile_row = QHBoxLayout()
        profile_row.addWidget(QLabel("Profile:"))
        self.profile_combo = QComboBox()
        self.profile_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.profile_combo.setMinimumContentsLength(20)
        self.profile_combo.setEditable(False)
        self.profile_add_btn = QPushButton("Add")
        self.profile_del_btn = QPushButton("Delete")
        self.profile_dup_btn = QPushButton("Duplicate")
        profile_row.addWidget(self.profile_combo, 1)
        profile_row.addWidget(self.profile_add_btn)
        profile_row.addWidget(self.profile_dup_btn)
        profile_row.addWidget(self.profile_del_btn)
        layout.addLayout(profile_row)

        self.profile_combo.currentIndexChanged.connect(self.rebuild_form_for_profile)
        self.profile_add_btn.clicked.connect(self.on_profile_add)
        self.profile_del_btn.clicked.connect(self.on_profile_delete)
        self.profile_dup_btn.clicked.connect(self.on_profile_duplicate)

        self.reload_profiles()

        self.form_container = QWidget()
        self.form_layout = QVBoxLayout(self.form_container)
        layout.addWidget(self.form_container, 1)

        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_preferences)
        layout.addWidget(save_button, 0)

        self.setLayout(layout)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(400)
        self.setMinimumHeight(600)
        self.rebuild_form_for_profile(0)

    def rebuild_form_for_profile(self, idx):
        # Clear previous form and delete widgets
        while self.form_layout.count():
            item = self.form_layout.takeAt(0)
            widget = item.widget()  # type: ignore
            if widget is not None:
                widget.deleteLater()
            elif item.layout() is not None:  # type: ignore

                def delete_layout(layout):
                    while layout.count():
                        child = layout.takeAt(0)
                        if child.widget():
                            child.widget().deleteLater()
                        elif child.layout():
                            delete_layout(child.layout())

                delete_layout(item.layout())  # type: ignore
        self.fields = {}
        if idx == 0:  # Global defaults config
            sections = {
                "Basic Plotting Parameters": [
                    "bin_size",
                    "time_window",
                    "pre_stim_time",
                    "default_method",
                    "default_channel_names",
                ],
                "EMG Filter Settings": ["butter_filter_args"],
                "'Suspected H-reflex' Plot Settings": ["h_threshold"],
                "M-max Calculation Settings": ["m_max_args"],
                "Plot Style Settings": [
                    "title_font_size",
                    "axis_label_font_size",
                    "tick_font_size",
                    "m_color",
                    "h_color",
                    "latency_window_style",
                    "subplot_adjust_args",
                ],
                "Dataset Parsing Parameters": ["preferred_date_format"],
                "Latency Window Presets": ["latency_window_presets"],
            }
            tabs = {
                "Plot Settings": [
                    "Basic Plotting Parameters",
                    "'Suspected H-reflex' Plot Settings",
                    "Plot Style Settings",
                ],
                "Latency Window Settings": ["Latency Window Presets"],
                "Misc.": [
                    "EMG Filter Settings",
                    "Dataset Parsing Parameters",
                    "M-max Calculation Settings",
                ],
            }
            tab_widget = QTabWidget()
            tab_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            for tab_name, section_names in tabs.items():
                tab_scroll = QScrollArea()
                tab_scroll.setWidgetResizable(True)
                tab_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                tab_content = QWidget()
                tab_content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
                tab_layout = QVBoxLayout(tab_content)
                for section in section_names:
                    if section == "Latency Window Presets":
                        group = QGroupBox(section)
                        group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
                        vbox = QVBoxLayout(group)
                        editor = LatencyWindowPresetEditor(self.config["latency_window_presets"])
                        vbox.addWidget(editor, 1)
                        self.fields["latency_window_presets"] = editor
                        tab_layout.addWidget(group, 1)
                        continue
                    group_box = QGroupBox(section)
                    group_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
                    form_layout = QFormLayout()
                    for key in sections[section]:
                        value = self.config.get(key)
                        if key == "latency_window_presets":
                            field = LatencyWindowPresetEditor(value)
                            form_layout.addRow(field)
                            self.fields[key] = field
                        elif isinstance(value, dict):
                            sub_group = QGroupBox(key)
                            sub_form = QFormLayout()
                            for sub_key, sub_value in value.items():
                                field = QLineEdit(str(sub_value))
                                sub_form.addRow(sub_key, field)
                                self.fields[f"{key}.{sub_key}"] = field
                            sub_group.setLayout(sub_form)
                            form_layout.addRow(sub_group)
                        elif isinstance(value, list):
                            field = QLineEdit(", ".join(map(str, value)))
                            form_layout.addRow(key, field)
                            self.fields[key] = field
                        else:
                            text = str(value)
                            if key in ("m_color", "h_color"):
                                text = text.replace("tab:", "")
                                field = QLineEdit(text)
                                field.setToolTip("Valid colors: " + ", ".join(TAB_COLOR_NAMES))
                            else:
                                field = QLineEdit(text)
                            form_layout.addRow(key, field)
                            self.fields[key] = field
                    group_box.setLayout(form_layout)
                    tab_layout.addWidget(group_box)
                tab_scroll.setWidget(tab_content)
                tab_widget.addTab(tab_scroll, tab_name)
            self.form_layout.addWidget(tab_widget, 1)
        else:
            # Show only profile fields
            name, path, data = self.profiles[idx - 1]
            vbox = QVBoxLayout()
            vbox.setAlignment(Qt.AlignmentFlag.AlignTop)
            # --- Name and Description fields ---
            name_group = QGroupBox("Profile Name and Description")
            name_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
            name_layout = QFormLayout(name_group)
            name_edit = QLineEdit(data.get("name", ""))
            desc_edit = QTextEdit(data.get("description", ""))
            desc_edit.setFixedHeight(50)
            name_layout.addRow("Name:", name_edit)
            name_layout.addRow("Description:", desc_edit)
            self.fields["profile_name"] = name_edit
            self.fields["profile_description"] = desc_edit
            vbox.addWidget(name_group)
            # Stimuli to plot (checkboxes)
            stimuli = data.get("stimuli_to_plot", [])
            stimuli_group = QGroupBox("Stimuli to Plot")
            stimuli_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
            stimuli_layout = QVBoxLayout(stimuli_group)
            stimuli_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
            stimuli_widget = StimulusSelectorWidget(selected=stimuli)
            stimuli_layout.addWidget(stimuli_widget)
            self.fields["stimuli_to_plot"] = stimuli_widget
            vbox.addWidget(stimuli_group)
            # Analysis parameters
            analysis_params = data.get("analysis_parameters", {})
            analysis_group = QGroupBox("Analysis Parameters")
            analysis_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            # Wrap the analysis parameters in a scroll area
            analysis_scroll = QScrollArea()
            analysis_scroll.setWidgetResizable(True)
            analysis_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            analysis_widget = QWidget()
            analysis_layout = QFormLayout(analysis_widget)
            analysis_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
            self.analysis_param_fields = {}  # param_name: (QLineEdit, remove_btn, row_widget)
            # Add button to add new param (always last row)
            add_param_btn = QPushButton("Add Parameter")
            add_param_btn.clicked.connect(lambda: self._on_add_analysis_param(analysis_layout, add_param_btn))
            analysis_layout.addRow(add_param_btn)
            # Add existing params (insert before button)
            for key, value in analysis_params.items():
                self._add_analysis_param_row(analysis_layout, key, value, before_widget=add_param_btn)
            analysis_widget.setLayout(analysis_layout)
            analysis_scroll.setWidget(analysis_widget)
            analysis_group_layout = QVBoxLayout(analysis_group)
            analysis_group_layout.setContentsMargins(0, 0, 0, 0)
            analysis_group_layout.addWidget(analysis_scroll)
            vbox.addWidget(analysis_group)
            self.form_layout.addLayout(vbox)

    def _add_analysis_param_row(self, layout, key, value, before_widget=None):
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        name_label = QLabel(key)
        if key in ("m_color", "h_color"):
            text = self.display_color(str(value))
            field = QLineEdit(text)
            field.setToolTip("Valid colors: " + ", ".join(TAB_COLOR_NAMES))
        else:
            field = QLineEdit(str(value))
        remove_btn = QPushButton("Remove")

        def remove_row():
            for i in range(layout.rowCount()):
                if (
                    layout.itemAt(i, QFormLayout.ItemRole.FieldRole)
                    and layout.itemAt(i, QFormLayout.ItemRole.FieldRole).widget() is row_widget
                ):
                    layout.removeRow(i)
                    break
            self.fields[f"analysis_parameters.{key}"] = None
            self.analysis_param_fields.pop(key, None)
            try:
                row_widget.setParent(None)
            except RuntimeError:
                pass

        remove_btn.clicked.connect(remove_row)
        row_layout.addWidget(name_label)
        row_layout.addWidget(field)
        row_layout.addWidget(remove_btn)
        if before_widget:
            for i in range(layout.rowCount()):
                if (
                    layout.itemAt(i, QFormLayout.ItemRole.FieldRole)
                    and layout.itemAt(i, QFormLayout.ItemRole.FieldRole).widget() is before_widget
                ):
                    layout.insertRow(i, row_widget)
                    break
            else:
                layout.addRow(row_widget)
        else:
            layout.addRow(row_widget)
        self.fields[f"analysis_parameters.{key}"] = field
        self.analysis_param_fields[key] = (field, remove_btn, row_widget)

    def _on_add_analysis_param(self, layout, add_param_btn):
        # Get available global keys (excluding latency window and already present)
        latency_keys = {"latency_window_presets", "latency_window_preset"}
        global_keys = set(self.config.keys()) - latency_keys
        already = set(self.analysis_param_fields.keys())
        available = sorted(list(global_keys - already))
        if not available:
            QMessageBox.information(self, "No Parameters", "No more global parameters available to add.")
            return
        item, ok = QInputDialog.getItem(self, "Add Parameter", "Select parameter:", available, 0, False)
        if ok and item:
            self._add_analysis_param_row(layout, item, self.config.get(item, ""), before_widget=add_param_btn)

    def reload_profiles(self):
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        self.profile_combo.addItem("(default)", userData=None)  # Add a global config option
        self.profiles = self.profile_manager.list_profiles()
        for name, path, data in self.profiles:
            self.profile_combo.addItem(name, userData=path)
        self.profile_combo.blockSignals(False)
        self.profile_combo.setCurrentIndex(0)
        self.on_profile_selected(0)

    def on_profile_selected(self, idx):
        if idx == 0:  # Global config
            self.active_profile_path = None
            self.active_profile_data = None
            # Load all fields from global config
            for key, field in self.fields.items():
                value = self.config.get(key)
                if isinstance(field, QLineEdit):
                    field.setText(str(value) if value is not None else "")
                elif isinstance(field, QDoubleSpinBox):
                    try:
                        field.setValue(float(value))  # type: ignore
                    except (TypeError, ValueError):
                        pass
                # Add more widget types as needed
            # Set latency window preset
            if "latency_window_presets" in self.fields:
                editor = self.fields["latency_window_presets"]
                preset_name = self.config.get("default_latency_window_preset", "default")
                idx = editor.preset_combo.findText(preset_name)
                if idx >= 0:
                    editor.preset_combo.setCurrentIndex(idx)
            return
        name, path, data = self.profiles[idx - 1]  # -1 because of global
        self.active_profile_path = path
        self.active_profile_data = data
        # Update stimuli fields
        stimuli = data.get("stimuli_to_plot", [])
        if "stimuli_to_plot" in self.fields:
            stimuli_widget = self.fields["stimuli_to_plot"]
            stimuli_widget.set_selected(stimuli)
        # Update analysis_parameters fields in the UI
        analysis_params = data.get("analysis_parameters", {})
        if not isinstance(analysis_params, dict):
            analysis_params = {}
        for key, value in analysis_params.items():
            if key in self.fields:
                field = self.fields[key]
                if isinstance(field, QLineEdit):
                    field.setText(str(value))
                elif isinstance(field, QDoubleSpinBox):
                    field.setValue(float(value))
                # Add more widget types as needed

    def on_profile_add(self):
        name, ok = QInputDialog.getText(self, "New Profile", "Profile name:")
        if ok and name:
            new_data = {
                "name": name,
                "description": "",
                "latency_window_preset": "default",
                "stimuli_to_plot": ["Electrical"],
                "analysis_parameters": {},
            }
            self.profile_manager.save_profile(new_data)
            self.reload_profiles()
            idx = self.profile_combo.findText(name)
            if idx >= 0:
                self.profile_combo.setCurrentIndex(idx)

    def on_profile_delete(self):
        idx = self.profile_combo.currentIndex() - 1  # -1 because of global config
        if idx < 0 or idx >= len(self.profiles):
            return
        name, path, _ = self.profiles[idx]
        warning_text = (
            f"You are about to permanently DELETE the profile '\u201c{name}\u201d'.\n\n"
            "This action cannot be undone.\n\n"
            "Are you sure you want to proceed?"
        )
        reply = QMessageBox.warning(
            self,
            "Delete Profile (Irreversible)",
            warning_text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.profile_manager.delete_profile(path)
            self.reload_profiles()
            # Reset the dialog to the global defaults (select global config)
            self.profile_combo.setCurrentIndex(0)
            self.rebuild_form_for_profile(0)

    def on_profile_duplicate(self):
        idx = self.profile_combo.currentIndex() - 1  # -1 because of global config
        if idx < 0 or idx >= len(self.profiles):
            return
        name, _, data = self.profiles[idx]
        new_name, ok = QInputDialog.getText(self, "Duplicate Profile", "New profile name:", text=f"{name} Copy")
        if ok and new_name:
            new_data = copy.deepcopy(data)
            new_data["name"] = new_name
            self.profile_manager.save_profile(new_data)
            self.reload_profiles()
            idx = self.profile_combo.findText(new_name)
            if idx >= 0:
                self.profile_combo.setCurrentIndex(idx)

    def normalize_color(self, val):
        """
        Accepts any color in COLOR_OPTIONS (with or without 'tab:').
        """
        color = val.strip().lower()
        # Accept both 'red' and 'tab:red' if 'tab:red' is in COLOR_OPTIONS
        if color.startswith("tab:"):
            color = color[4:]
        if color in TAB_COLOR_NAMES:
            return f"tab:{color}"
        return None

    def display_color(self, val):
        """
        Accepts 'tab:red' or 'red', returns 'red' for display.
        """
        color = val.strip().lower()
        if color.startswith("tab:"):
            color = color[4:]
        return color

    def save_preferences(self):
        """Save the current configuration (global or profile)."""
        idx = self.profile_combo.currentIndex()
        color_keys = ["m_color", "h_color"]

        if idx > 0:  # Profile mode
            name, path, data = self.profiles[idx - 1]
            name_edit = self.fields.get("profile_name")
            desc_edit = self.fields.get("profile_description")
            profile_name = name_edit.text().strip() if name_edit else name
            profile_desc = desc_edit.toPlainText().strip() if desc_edit else data.get("description", "")
            profile_data = dict(name=profile_name, description=profile_desc)
            stimuli_widget = self.fields.get("stimuli_to_plot")
            if stimuli_widget:
                profile_data["stimuli_to_plot"] = stimuli_widget.get_selected()
            analysis_params = {}
            invalid_colors = []
            for key, field in self.fields.items():
                if key.startswith("analysis_parameters."):
                    param = key.split(".", 1)[1]
                    val = field.text()
                    if param in color_keys:
                        norm = self.normalize_color(val)
                        if not norm:
                            invalid_colors.append((param, val))
                        val = norm if norm else val
                    ref_val = self.config.get(param)
                    try:
                        parsed_val = ast.literal_eval(val)
                    except Exception:
                        parsed_val = val
                    if ref_val is not None:
                        analysis_params[param] = ConfigRepository.coerce_types(parsed_val, ref_val)
                    else:
                        analysis_params[param] = parsed_val
            if invalid_colors:
                msg = "\n".join(
                    [
                        f"'{v}' is not a valid color for '{k}'. Valid options: {', '.join(COLOR_OPTIONS)}"
                        for k, v in invalid_colors
                    ]
                )
                QMessageBox.warning(
                    self,
                    "Invalid Color",
                    f"Cannot save: Invalid color(s) selected.\n{msg}",
                )
                return
            profile_data["analysis_parameters"] = analysis_params
            try:
                self.profile_manager.save_profile(profile_data, filename=path)
                QMessageBox.information(
                    self,
                    "Profile Saved",
                    f"Profile '{profile_name}' saved successfully.",
                )
                self.reload_profiles()
                idx_new = self.profile_combo.findText(profile_name)
                if idx_new >= 0:
                    self.profile_combo.setCurrentIndex(idx_new)
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save profile: {e}")
        else:  # Global config mode
            new_config = copy.deepcopy(self.config)
            invalid_colors = []
            for key, field in self.fields.items():
                if isinstance(field, LatencyWindowPresetEditor):
                    value = field.get_presets()
                elif isinstance(field, QTextEdit):
                    raw = field.toPlainText()
                    value = raw
                else:
                    raw = field.text()
                    value = raw
                if key in color_keys:
                    norm = self.normalize_color(raw)  # type: ignore
                    if not norm:
                        invalid_colors.append((key, raw))  # type: ignore
                    value = norm if norm else value
                if "." in key:
                    main_key, sub_key = key.split(".")
                    ref_val = (
                        self.config.get(main_key, {}).get(sub_key) if isinstance(self.config.get(main_key), dict) else None
                    )
                    try:
                        parsed_val = ast.literal_eval(value)  # type: ignore
                    except Exception:
                        parsed_val = value
                    if ref_val is not None:
                        coerced = ConfigRepository.coerce_types(parsed_val, ref_val)
                    else:
                        coerced = parsed_val
                    if main_key not in new_config or not isinstance(new_config.get(main_key), dict):
                        new_config[main_key] = {}
                    new_config[main_key][sub_key] = coerced
                else:
                    ref_val = self.config.get(key)
                    try:
                        parsed_val = ast.literal_eval(value)  # type: ignore
                    except Exception:
                        parsed_val = value
                    if ref_val is not None:
                        coerced = ConfigRepository.coerce_types(parsed_val, ref_val)
                    else:
                        coerced = parsed_val
                    new_config[key] = coerced
            if invalid_colors:
                msg = "\n".join(
                    [
                        f"'{v}' is not a valid color for '{k}'. Valid options: {', '.join(COLOR_OPTIONS)}"
                        for k, v in invalid_colors
                    ]
                )
                QMessageBox.warning(
                    self,
                    "Invalid Color",
                    f"Cannot save: Invalid color(s) selected.\n{msg}",
                )
                return
            self.config = new_config
            self.config_repo.write_config(self.config)
            logging.info(f"Saved user config: {self.config}")
            self.accept()

    # The parse_value, parse_color, and convert_to_number functions are no longer used for config/profile type handling.
    # All type coercion is now handled by ConfigRepository.coerce_types for consistency and separation of concerns.
