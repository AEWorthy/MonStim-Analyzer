# Utility functions/scripts for the project
import os
import sys
from typing import List
import yaml
from pathlib import Path
from PyQt6.QtCore import QStandardPaths

try:
    from PyQt6.QtWidgets import QApplication
except ImportError:  # Allow headless environments
    QApplication = None
import numpy as np

DIST_PATH = 'dist'
OUTPUT_PATH = 'data'

def to_camel_case(text: str) -> str:
    """Return *text* converted to ``camelCase``."""
    words = text.split()
    camel_case_text = words[0].lower() + ''.join(word.capitalize() for word in words[1:])
    return camel_case_text

def format_report(report: List[str]) -> str:
    """Join a list of strings into a single newline-separated string."""
    formatted_report = ''
    for line in report:
        if line == report[-1]:
            formatted_report += line
        else:
            formatted_report += line + '\n'
    return formatted_report

def get_base_path() -> str:
    """Return the root path of the project installation."""
    if getattr(sys, 'frozen', False):
        if sys.platform == 'darwin':
            exe_path = os.path.dirname(sys.executable)
            base_path = os.path.abspath(os.path.join(exe_path, '..', '..', '..'))
        else:
            base_path = os.path.dirname(sys.executable)
    else:
        base_path = Path(__file__).resolve().parent.parent.parent

    return Path(base_path)

def get_bundle_path() -> str:
    """Return the path to the bundled resources when running a frozen build."""
    if getattr(sys, 'frozen', False):
        bundle_path = sys._MEIPASS
    else:
        bundle_path = os.path.dirname(os.path.abspath(__file__))

    return bundle_path

def get_output_path() -> str:
    """Return the directory used for program output."""
    output_path = os.path.join(get_base_path(), OUTPUT_PATH)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    return output_path

def get_source_path() -> str:
    """Return the path to the ``src`` folder containing resource files."""
    if getattr(sys, 'frozen', False):
        source_path = os.path.join(get_bundle_path(), 'src')
    else:
        source_path = os.path.join(get_base_path(), 'src')
    return source_path

def get_docs_path() -> str:
    """Return the path to bundled documentation files."""
    if getattr(sys, 'frozen', False):
        docs_path = os.path.join(get_bundle_path(), 'docs')
    else:
        docs_path = os.path.join(get_base_path(), 'docs')
    return docs_path

def get_config_path() -> str:
    """Return the location of ``config.yml``."""
    return os.path.join(get_docs_path(), 'config.yml')

def get_output_bin_path() -> str:
    """Directory that stores serialized analysis objects."""
    output_bin_path = os.path.join(get_output_path(), 'bin')
    if not os.path.exists(output_bin_path):
        os.makedirs(output_bin_path)
    return output_bin_path

def get_data_path() -> str:
    """Return the directory containing packaged data files."""
    if getattr(sys, 'frozen', False):
        data_path = get_base_path()
    else:
        data_path = os.path.join(get_base_path(), DIST_PATH)
    return data_path

def get_log_dir() -> str:
    """Return the directory where application log files are stored."""
    env_path = os.environ.get("MONSTIM_LOG_DIR")
    if env_path and os.path.isdir(env_path):
        return env_path

    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    if not base:
        base = os.getenv("APPDATA", r"C:\\Users\\%USERNAME%\\AppData\\Roaming")
    log_dir = os.path.join(base, "logs")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def get_main_window():
    """Return the active :class:`EMGAnalysisGUI` instance if one exists."""
    from monstim_gui.gui_main import MonstimGUI

    active_window = QApplication.activeWindow()

    if isinstance(active_window, MonstimGUI):
        return active_window
    if active_window.__class__.__name__ == 'EMGAnalysisGUI':
        return active_window
    return None

def deep_equal(val1, val2) -> bool:
    """Recursively compare two values for equality."""
    if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
        return np.array_equal(val1, val2)
    if isinstance(val1, dict) and isinstance(val2, dict):
        if val1.keys() != val2.keys():
            return False
        return all(deep_equal(val1[k], val2[k]) for k in val1)
    if isinstance(val1, list) and isinstance(val2, list):
        if len(val1) != len(val2):
            return False
        return all(deep_equal(v1, v2) for v1, v2 in zip(val1, val2))
    return val1 == val2

def load_config(config_file=None):
        """
        Loads the config.yml file into a YAML object that can be used to reference hard-coded configurable constants.

        Args:
            config_file (str): location of the 'config.yml' file.
        """
        if config_file is None:
            default_config_file = get_config_path()
            user_config_file = os.path.join(os.path.dirname(default_config_file), 'config-user.yml')
            # if it exists, get user config file
            if os.path.exists(user_config_file):
                config_file = user_config_file
            else:
                config_file = default_config_file
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config

# Custom YAML loader to handle tuples
class CustomYAMLLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))

CustomYAMLLoader.add_constructor(
    u'tag:yaml.org,2002:python/tuple',
    CustomYAMLLoader.construct_python_tuple)

