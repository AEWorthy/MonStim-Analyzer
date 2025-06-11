from .base import *


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
            "Default Reflex Window Settings": ["m_start", "m_duration", "h_start", "h_duration"],
            "'Suspected H-reflex' Plot Settings": ["h_threshold"],
            "M-max Calculation Settings": ["m_max_args"],
            "Plot Style Settings": ["title_font_size", "axis_label_font_size", "tick_font_size",
                                      "m_color", "h_color", "latency_window_style", "subplot_adjust_args"],
            "Dataset Parsing Parameters": ["preferred_date_format"],
        }

        # Map sections to high-level tab categories
        tabs = {
            "Plot Settings": ["Basic Plotting Parameters", "'Suspected H-reflex' Plot Settings", "Plot Style Settings"],
            "Latency Window Settings": ["Default Reflex Window Settings", "M-max Calculation Settings"],
            "Misc.": ["EMG Filter Settings", "Dataset Parsing Parameters"],
        }

        tab_widget = QTabWidget()

        for tab_name, section_names in tabs.items():
            tab_scroll = QScrollArea()
            tab_scroll.setWidgetResizable(True)
            tab_content = QWidget()
            tab_layout = QVBoxLayout(tab_content)

            for section in section_names:
                group_box = QGroupBox(section)
                form_layout = QFormLayout()

                for key in sections[section]:
                    value = self.config.get(key)
                    if isinstance(value, dict):
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
        #TODO: Fix saving function because it does not save the entire config
        user_config = {}

        for key, field in self.fields.items():
            value = self.parse_value(field.text(), key)
            if '.' in key:
                main_key, sub_key = key.split('.')
                if main_key not in user_config or not isinstance(user_config.get(main_key), dict):
                    user_config[main_key] = {}
                user_config[main_key][sub_key] = value
            else:
                user_config[key] = value

        # Save all values to the user config file
        with open(self.user_config_file, 'w') as file:
            yaml.dump(user_config, file)

        logging.info(f"Saved user config: {user_config}")
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
