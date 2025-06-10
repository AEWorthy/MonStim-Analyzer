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
                user_config = yaml.safe_load(file)
                config.update(user_config)

        return config

    def write_config(self):
        with open(self.user_config_file, 'w') as file:
            yaml.safe_dump(self.config, file)

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.tab_widget = QTabWidget()
        self.general_tab = QWidget()
        self.colors_tab = QWidget()
        self.tab_widget.addTab(self.general_tab, 'General')
        self.tab_widget.addTab(self.colors_tab, 'Colors')
        layout.addWidget(self.tab_widget)

        # General Tab
        general_layout = QFormLayout(self.general_tab)
        self.default_expt_entry = QLineEdit(self.config.get('default_experiment', ''))
        general_layout.addRow('Default Experiment', self.default_expt_entry)

        # Colors Tab
        colors_layout = QFormLayout(self.colors_tab)
        self.plot_color_entry = QLineEdit(self.config.get('plot_color', 'tab:blue'))
        colors_layout.addRow('Plot Color', self.plot_color_entry)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.save_preferences)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def save_preferences(self):
        self.config['default_experiment'] = self.default_expt_entry.text()
        self.config['plot_color'] = self.validate_color(self.plot_color_entry.text())
        self.write_config()
        self.accept()

    def validate_color(self, value: str) -> str:
        color = value
        if color.startswith('tab:'):
            color = color.replace('tab:', '')
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
