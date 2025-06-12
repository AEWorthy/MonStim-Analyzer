from .base import *
import copy


class LatencyWindowPresetEditor(QWidget):
    """Widget to create and edit latency window presets."""

    def __init__(self, presets: dict[str, list[dict]] | None = None, parent=None):
        super().__init__(parent)
        self.presets: list[list[LatencyWindow]] = []
        self.preset_combo = QComboBox()
        self.preset_combo.setEditable(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # Track which preset is currently displayed so we can save it when the
        # selection changes. ``QComboBox.currentIndexChanged`` is emitted after
        # the widget updates its index, so we cannot rely on the combo box to
        # provide the previous index.
        self._current_index: int | None = None
        self.window_entries: list[tuple[QGroupBox, LatencyWindow, QLineEdit, QDoubleSpinBox, QDoubleSpinBox, QComboBox]] = []
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
        self.preset_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.preset_combo.setMinimumContentsLength(10)
        self.preset_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        control_row.addWidget(self.preset_combo, 1)
        add_btn = QPushButton("Add")
        remove_btn = QPushButton("Remove")
        control_row.addWidget(add_btn)
        control_row.addWidget(remove_btn)
        layout.addLayout(control_row)

        self.preset_combo.currentIndexChanged.connect(self.load_preset)
        add_btn.clicked.connect(self.add_preset)
        remove_btn.clicked.connect(self.remove_preset)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        # Align entries to the top so empty space doesn't appear above them
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
        name = self._unique_name("Preset")
        self.preset_combo.addItem(name)
        self.presets.append([])
        self.preset_combo.setCurrentIndex(self.preset_combo.count() - 1)

    def remove_preset(self) -> None:
        if self.preset_combo.count() == 0:
            return
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
        self.adjustSize()
        self.updateGeometry()
        self._current_index = index

    def _save_current_preset(self) -> None:
        """Save the currently displayed preset back to ``self.presets``."""
        index = self._current_index if self._current_index is not None else -1
        if index < 0 or index >= len(self.presets):
            return
        windows: list[LatencyWindow] = []
        for _, window, name_edit, start_spin, dur_spin, color_combo in self.window_entries:
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

    # ------------------------------------------------------------------
    # Window operations
    # ------------------------------------------------------------------
    def _clear_windows(self) -> None:
        for grp, *_ in self.window_entries:
            grp.setParent(None)
        self.window_entries.clear()

    def add_window_group(self, window: LatencyWindow | None = None, *, checked: bool | None = None) -> None:
        if window is None:
            window = LatencyWindow(
                name=f"Window {len(self.window_entries)+1}",
                start_times=[0.0],
                durations=[1.0],
                color="black",
                linestyle=":"
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
        self.scroll_widget.adjustSize()
        self.window_entries.append((group, window, name_edit, start_spin, dur_spin, color_combo))

    def _remove_window_group(self, group: QGroupBox) -> None:
        for i, (grp, *_ ) in enumerate(self.window_entries):
            if grp is group:
                self.window_entries.pop(i)
                break
        group.setParent(None)

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
    def __init__(self, default_config_file, parent=None):
        super().__init__()
        self.setModal(True)
        self.setWindowTitle("Preferences")

        self.default_config_file = default_config_file
        self.user_config_file = self.get_user_config_file()
        self.config = self.read_config()
        self.init_ui()

    def get_user_config_file(self):
        # Get the directory of the default config file
        config_dir = os.path.dirname(self.default_config_file)
        return os.path.join(config_dir, 'config-user.yml')
    
    def read_config(self):
        # First, read the original config
        with open(self.default_config_file, 'r') as file:
            config = yaml.safe_load(file)
        
        # If user config exists, update the config with user settings
        if os.path.exists(self.user_config_file):
            with open(self.user_config_file, 'r') as file:
                user_config = yaml.load(file, Loader = CustomLoader)
            if user_config:  # Check if user_config is not None
                self.update_nested_dict(config, user_config)
        
        return config

    def update_nested_dict(self, d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self.update_nested_dict(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def init_ui(self):

        layout = QVBoxLayout()

        self.fields = {}

        # Define sections and their corresponding keys
        sections = {
            "Basic Plotting Parameters": ["bin_size", "time_window", "default_method", "default_channel_names"],
            "EMG Filter Settings": ["butter_filter_args"],
            "'Suspected H-reflex' Plot Settings": ["h_threshold"],
            "M-max Calculation Settings": ["m_max_args"],
            "Plot Style Settings": ["title_font_size", "axis_label_font_size", "tick_font_size",
                                      "m_color", "h_color", "latency_window_style", "subplot_adjust_args"],
            "Dataset Parsing Parameters": ["preferred_date_format"],
            "Latency Window Presets": ["latency_window_presets"],
        }

        # Map sections to high-level tab categories
        tabs = {
            "Plot Settings": ["Basic Plotting Parameters", "'Suspected H-reflex' Plot Settings", "Plot Style Settings"],
            "Latency Window Settings": ["Latency Window Presets"],
            "Misc.": ["EMG Filter Settings", "Dataset Parsing Parameters", "M-max Calculation Settings"],
        }

        tab_widget = QTabWidget()

        for tab_name, section_names in tabs.items():
            tab_scroll = QScrollArea()
            tab_scroll.setWidgetResizable(True)
            tab_content = QWidget()
            tab_layout = QVBoxLayout(tab_content)

            for section in section_names:
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
                        field = QLineEdit(', '.join(map(str, value)))
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

            tab_layout.addStretch()

            tab_scroll.setWidget(tab_content)
            tab_widget.addTab(tab_scroll, tab_name)

        layout.addWidget(tab_widget)

        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_config)
        layout.addWidget(save_button)

        self.setLayout(layout)
        self.setWindowTitle("Preferences")
        self.resize(400, 600)  # Set a default size

    def save_config(self):
        """Save the current preferences to ``config-user.yml``.

        All values from the dialog are merged into the existing configuration
        so that the written YAML file contains the complete set of options.
        """

        # Start from the currently loaded configuration
        new_config = copy.deepcopy(self.config)

        for key, field in self.fields.items():
            if isinstance(field, LatencyWindowPresetEditor):
                value = field.get_presets()
            elif isinstance(field, QTextEdit):
                raw = field.toPlainText()
                value = self.parse_value(raw, key)
            else:
                raw = field.text()
                value = self.parse_value(raw, key)
            if '.' in key:
                main_key, sub_key = key.split('.')
                if main_key not in new_config or not isinstance(new_config.get(main_key), dict):
                    new_config[main_key] = {}
                new_config[main_key][sub_key] = value
            else:
                new_config[key] = value

        self.config = new_config

        # Save the entire configuration to the user config file
        with open(self.user_config_file, 'w') as file:
            yaml.safe_dump(self.config, file)

        logging.info(f"Saved user config: {self.config}")
        self.accept()

    def parse_value(self, value, key):
        # List of keys that should be treated as lists
        list_keys = ['default_channel_names', 'm_start', 'h_start']
        color_keys = ['m_color', 'h_color']

        if key in list_keys:
            # Split by comma and strip whitespace
            return [self.convert_to_number(item.strip()) for item in value.split(',')]

        if key in color_keys:
            return self.parse_color(value)

        return self.convert_to_number(value)

    def parse_color(self, value: str) -> str:
        color = value.strip().lower()
        if color.startswith('tab:'):
            color = color[4:]
        if color not in TAB_COLOR_NAMES:
            QMessageBox.warning(
                self,
                'Invalid Color',
                f"'{value}' is not a valid color. Using 'blue'.\n"
                f"Valid options: {', '.join(TAB_COLOR_NAMES)}",
            )
            color = 'blue'
        return f'tab:{color}'

    def convert_to_number(self, value):
        try:
            # Try to convert to int first
            return int(value)
        except ValueError:
            try:
                # If not int, try float
                return float(value)
            except ValueError:
                # If it's not a number, return as is
                return value
