import logging
import os
import yaml
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QGridLayout, QLabel, QLineEdit, QDialogButtonBox, QMessageBox, QHBoxLayout,
                             QTextEdit, QPushButton, QApplication, QTextBrowser, QWidget, QFormLayout, QGroupBox, QScrollArea,
                             QSizePolicy, QCheckBox)
from PyQt6.QtGui import QPixmap, QFont, QIcon, QDesktopServices
from PyQt6.QtCore import Qt, QUrl, pyqtSlot, QEvent, QTimer
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineScript
import markdown
from markdown.extensions.codehilite import CodeHiliteExtension
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.tables import TableExtension
from mdx_math import MathExtension
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from monstim_utils import get_source_path, CustomLoader
from monstim_gui.splash import SPLASH_INFO

if TYPE_CHECKING:
    from monstim_analysis import EMGData


class WebEnginePage(QWebEnginePage):
    # Custom WebEnginePage to handle JavaScript messages
    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
        print(f"JS: {message}")

# Define custom dialogs
class ChangeChannelNamesDialog(QDialog):
    def __init__(self, channel_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Change Channel Names")
        self.setModal(True)
        layout = QGridLayout(self)
        
        self.channel_inputs = {}
        for i, channel_name in enumerate(channel_names):
            layout.addWidget(QLabel(f"Channel {i+1}:"), i, 0)
            self.channel_inputs[channel_name] = QLineEdit(channel_name)
            layout.addWidget(self.channel_inputs[channel_name], i, 1)
        
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons, len(channel_names), 0, 1, 2)

    def get_new_names(self):
        return {old: input.text() for old, input in self.channel_inputs.items()}

# class ReflexSettingsDialog(QDialog):
#     def __init__(self, emg_data : 'EMGData', parent=None):
#         super().__init__(parent)
#         self.emg_data = emg_data

#         self.setModal(True)
#         self.setWindowTitle(f"Update Reflex Window Settings: Dataset {self.emg_data.formatted_name}")

#         self.init_ui()

#     def init_ui(self):
#         layout = QVBoxLayout()

#         # Duration
#         duration_layout = QHBoxLayout()
#         duration_layout.addWidget(QLabel("m_duration:"))
#         self.m_duration_entry = QLineEdit(str(self.emg_data.m_duration[0]))
#         duration_layout.addWidget(self.m_duration_entry)

#         duration_layout.addWidget(QLabel("h_duration:"))
#         self.h_duration_entry = QLineEdit(str(self.emg_data.h_duration[0]))
#         duration_layout.addWidget(self.h_duration_entry)

#         layout.addLayout(duration_layout)

#         # Start times
#         self.entries : list[tuple[QLineEdit, QLineEdit]] = []
#         for i in range(len(self.emg_data.m_start)):
#             channel_layout = QHBoxLayout()
#             channel_layout.addWidget(QLabel(f"Channel {i}:"))

#             channel_layout.addWidget(QLabel("m_start:"))
#             m_start_entry = QLineEdit(str(self.emg_data.m_start[i]))
#             channel_layout.addWidget(m_start_entry)

#             channel_layout.addWidget(QLabel("h_start:"))
#             h_start_entry = QLineEdit(str(self.emg_data.h_start[i]))
#             channel_layout.addWidget(h_start_entry)

#             layout.addLayout(channel_layout)
#             self.entries.append((m_start_entry, h_start_entry))

#         # Buttons
#         button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
#         button_box.accepted.connect(self.save_settings)
#         button_box.rejected.connect(self.reject)
#         layout.addWidget(button_box)

#         self.setLayout(layout)

#     def save_settings(self):
#         try:
#             m_duration = [float(self.m_duration_entry.text()) for _ in range(len(self.emg_data.m_start))]
#             h_duration = [float(self.h_duration_entry.text()) for _ in range(len(self.emg_data.m_start))]
#         except ValueError:
#             QMessageBox.warning(self, "Invalid Input", "Invalid input for durations. Please enter valid numbers.")
#             return

#         m_start = []
#         h_start = []
#         for i, (m_start_entry, h_start_entry) in enumerate(self.entries):
#             try:
#                 m_start.append(float(m_start_entry.text()))
#                 h_start.append(float(h_start_entry.text()))
#             except ValueError:
#                 QMessageBox.warning(self, "Invalid Input", f"Invalid input for channel {i}. Skipping.")

#         try:           
#             self.emg_data.update_reflex_latency_windows(m_start, m_duration, h_start, h_duration)
#         except Exception as e:
#             QMessageBox.warning(self, "Error", f"Error saving settings: {str(e)}")
#             logging.error(f"Error saving reflex settings: {str(e)}\n\tdataset: {self.dataset}\n\tm_start: {m_start}\n\tm_duration: {m_duration}\n\th_start: {h_start}\n\th_duration: {h_duration}")
#             return
        
#         self.emg_data.update_reflex_parameters()
#         self.emg_data.reset_properties(recalculate=False)

#         self.accept()
class ReflexSettingsDialog(QDialog):
    def __init__(self, emg_data: 'EMGData', parent=None):
        super().__init__(parent)
        self.emg_data = emg_data

        self.setModal(True)
        self.setWindowTitle(f"Update Reflex Window Settings: Dataset {self.emg_data.formatted_name}")

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Toggle button for global or per-channel settings
        self.toggle_button = QPushButton("Switch to Per-Channel Start Times")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(True)  # Default to global settings
        self.toggle_button.toggled.connect(self.toggle_settings_mode)
        layout.addWidget(self.toggle_button)

        # Global settings layout
        self.global_layout = QHBoxLayout()
        self.global_layout.addWidget(QLabel("m_duration:"))
        self.global_m_duration_entry = QLineEdit(str(self.emg_data.m_duration[0]))
        self.global_m_duration_entry.installEventFilter(self)
        self.global_layout.addWidget(self.global_m_duration_entry)

        self.global_layout.addWidget(QLabel("h_duration:"))
        self.global_h_duration_entry = QLineEdit(str(self.emg_data.h_duration[0]))
        self.global_h_duration_entry.installEventFilter(self)
        self.global_layout.addWidget(self.global_h_duration_entry)

        self.global_layout.addWidget(QLabel("m_start:"))
        self.global_m_start_entry = QLineEdit(str(self.emg_data.m_start[0]))
        self.global_m_start_entry.installEventFilter(self)
        self.global_layout.addWidget(self.global_m_start_entry)

        self.global_layout.addWidget(QLabel("h_start:"))
        self.global_h_start_entry = QLineEdit(str(self.emg_data.h_start[0]))
        self.global_h_start_entry.installEventFilter(self)
        self.global_layout.addWidget(self.global_h_start_entry)

        layout.addLayout(self.global_layout)

        # Per-channel settings layout
        self.per_channel_layout = QVBoxLayout()
        self.entries: list[tuple[QLineEdit, QLineEdit]] = []
        for i in range(len(self.emg_data.m_start)):
            channel_layout = QHBoxLayout()
            channel_layout.addWidget(QLabel(f"Channel {i}:"))

            channel_layout.addWidget(QLabel("m_start:"))
            m_start_entry = QLineEdit(str(self.emg_data.m_start[i]))
            m_start_entry.installEventFilter(self)
            channel_layout.addWidget(m_start_entry)

            channel_layout.addWidget(QLabel("h_start:"))
            h_start_entry = QLineEdit(str(self.emg_data.h_start[i]))
            h_start_entry.installEventFilter(self)
            channel_layout.addWidget(h_start_entry)

            self.per_channel_layout.addLayout(channel_layout)
            self.entries.append((m_start_entry, h_start_entry))

        layout.addLayout(self.per_channel_layout)

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.save_settings)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

        # Start with per-channel settings hidden
        self.toggle_settings_mode(True)

    def toggle_settings_mode(self, checked):
        if checked:
            self.toggle_button.setText("Switch to Per-Channel Start Times")
            self.global_layout.setEnabled(True)
            self.global_m_start_entry.setEnabled(True)
            self.global_h_start_entry.setEnabled(True)
            for entry in self.entries:
                for widget in entry:
                    widget.setEnabled(False)
        else:
            self.toggle_button.setText("Switch to Global Start Times")
            self.global_layout.setEnabled(True)
            self.global_m_start_entry.setEnabled(False)
            self.global_h_start_entry.setEnabled(False)
            for entry in self.entries:
                for widget in entry:
                    widget.setEnabled(True)

    def save_settings(self):
        try:
            m_duration = [float(self.global_m_duration_entry.text()) for _ in range(len(self.emg_data.m_start))]
            h_duration = [float(self.global_h_duration_entry.text()) for _ in range(len(self.emg_data.m_start))]
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Invalid input for durations. Please enter valid numbers.")
            return

        if self.toggle_button.isChecked():
            # Global start times
            try:
                m_start = [float(self.global_m_start_entry.text()) for _ in range(len(self.emg_data.m_start))]
                h_start = [float(self.global_h_start_entry.text()) for _ in range(len(self.emg_data.m_start))]
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Invalid input for global start times. Please enter valid numbers.")
                return
        else:
            # Per-channel start times
            m_start = []
            h_start = []
            for i, (m_start_entry, h_start_entry) in enumerate(self.entries):
                try:
                    m_start.append(float(m_start_entry.text()))
                    h_start.append(float(h_start_entry.text()))
                except ValueError:
                    QMessageBox.warning(self, "Invalid Input", f"Invalid input for channel {i}. Skipping.")
                    return

        try:
            self.emg_data.update_reflex_latency_windows(m_start, m_duration, h_start, h_duration)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error saving settings: {str(e)}")
            logging.error(f"Error saving reflex settings: {str(e)}\n\tdataset: {self.dataset}\n\tm_start: {m_start}\n\tm_duration: {m_duration}\n\th_start: {h_start}\n\th_duration: {h_duration}")
            return

        self.emg_data.update_reflex_parameters()
        self.emg_data.reset_properties(recalculate=False)

        self.accept()

    def eventFilter(self, source, event):
        if event.type() == QEvent.Type.FocusIn and isinstance(source, QLineEdit):
            QTimer.singleShot(0, source.selectAll)
        return super().eventFilter(source, event)

class CopyableReportDialog(QDialog):
    def __init__(self, title, report, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setLayout(QVBoxLayout())

        self.text_edit = QTextEdit()
        self.text_edit.setPlainText(report)
        self.text_edit.setReadOnly(True)
        self.layout().addWidget(self.text_edit)

        button_layout = QHBoxLayout()
        
        copy_button = QPushButton("Copy to Clipboard")
        copy_button.clicked.connect(self.copy_to_clipboard)
        button_layout.addWidget(copy_button)
        
        done_button = QPushButton("Done")
        done_button.clicked.connect(self.close)
        button_layout.addWidget(done_button)
        
        self.layout().addLayout(button_layout)

        self.resize(300, 200)

    def copy_to_clipboard(self):
        QApplication.clipboard().setText(self.text_edit.toPlainText())

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

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

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
        }

        for section, keys in sections.items():
            group_box = QGroupBox(section)
            form_layout = QFormLayout()

            for key in keys:
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
                    field = QLineEdit(str(value))
                    form_layout.addRow(key, field)
                    self.fields[key] = field

            group_box.setLayout(form_layout)
            scroll_layout.addWidget(group_box)

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        save_button = QPushButton("Save")
        save_button.clicked.connect(self.save_config)
        layout.addWidget(save_button)

        self.setLayout(layout)
        self.setWindowTitle("Preferences")
        self.resize(400, 600)  # Set a default size

    def save_config(self):
        user_config = {}
        for key, field in self.fields.items():
            if '.' in key:
                main_key, sub_key = key.split('.')
                if main_key not in user_config:
                    user_config[main_key] = {}
                user_config[main_key][sub_key] = self.parse_value(field.text(), key)
            else:
                user_config[key] = self.parse_value(field.text(), key)

        # Save all values to the user config file
        with open(self.user_config_file, 'w') as file:
            yaml.dump(user_config, file)

        logging.info(f"Saved user config: {user_config}")
        self.accept()

    def parse_value(self, value, key):
        # List of keys that should be treated as lists
        list_keys = ['default_channel_names', 'm_start', 'h_start']
        
        if key in list_keys:
            # Split by comma and strip whitespace
            return [self.convert_to_number(item.strip()) for item in value.split(',')]
        
        return self.convert_to_number(value)

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
        
class HelpWindow(QWidget):
    def __init__(self, markdown_content, title=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title if title else "Help Window")
        self.setWindowIcon(QIcon(os.path.join(get_source_path(), 'info.png')))
        self.resize(600, 400)
        
        layout = QVBoxLayout()
        self.text_browser = QTextBrowser()
        self.text_browser.setHtml(markdown_content)
        layout.addWidget(self.text_browser)
        self.setLayout(layout)

class LatexHelpWindow(QWidget):
    def __init__(self, markdown_content, title=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title if title else "LaTeX Help Window")
        self.setWindowIcon(QIcon(os.path.join(get_source_path(), 'info.png')))
        self.resize(600, 400)

        layout = QVBoxLayout(self)
        self.web_view = QWebEngineView()
        layout.addWidget(self.web_view)

        # Set custom WebEnginePage to handle JavaScript messages
        self.page = WebEnginePage(self.web_view)
        self.web_view.setPage(self.page)

        # Process and display the markdown content
        self.process_content(markdown_content)

    def process_content(self, markdown_content):
        # Convert markdown to HTML
        html_content = self.markdown_to_html(markdown_content)

        # Get the path to your local MathJax installation
        mathjax_path = os.path.abspath(os.path.join(get_source_path(), 'mathjax', 'es5', 'tex-mml-chtml.js'))
        
        if os.path.exists(mathjax_path):
            print("MathJax file exists")
        else:
            print("MathJax file does not exist")

        # Full HTML content (use the HTML from the previous artifact)
        full_html = f"""
        <!DOCTYPE html>
            <html>
            <head>
                <script id="MathJax-script" async src="file:///{mathjax_path}"></script>
                <script>
                    MathJax = {{
                        tex: {{
                            inlineMath: [['$', '$']]
                        }}
                    }};
                </script>
                <style>
                    body {{
                        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                        font-size: 14px;
                        line-height: 1.5;
                        margin: 0;
                        padding: 10px;
                        transition: background-color 0.5s, color 0.5s;
                        overflow-y: auto;
                    }}
                    body.light-mode {{
                        background-color: #ffffff;
                        color: #000000;
                    }}
                    body.dark-mode {{
                        background-color: #2b2b2b;
                        color: #ffffff;
                    }}
                    a {{
                        color: #0000ff;
                        text-decoration: none;
                    }}
                    a:hover {{
                        text-decoration: underline;
                    }}
                    pre {{
                        background-color: #f0f0f0;
                        border: 1px solid #ccc;
                        border-radius: 4px;
                        padding: 10px;
                        white-space: pre-wrap;
                        word-wrap: break-word;
                    }}
                    body.dark-mode pre {{
                        background-color: #3c3c3c;
                        border-color: #555;
                    }}
                    table {{
                        border-collapse: collapse;
                        margin-bottom: 10px;
                    }}
                    th, td {{
                        border: 1px solid #ccc;
                        padding: 5px;
                    }}
                    body.dark-mode th, body.dark-mode td {{
                        border-color: #555;
                    }}
                </style>
                <script>
                    document.addEventListener("DOMContentLoaded", function() {{
                        const isDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
                        document.body.classList.toggle('dark-mode', isDarkMode);
                        document.body.classList.toggle('light-mode', !isDarkMode);

                        // Add click event listener to all links
                        document.querySelectorAll('a').forEach(link => {{
                            link.addEventListener('click', function(event) {{
                                event.preventDefault();
                                window.pyqt.linkClicked(this.href);
                            }});
                        }});
                    }});
                </script>
            </head>
            <body>
                {html_content}
            </body>
            </html>
        """

        # Add the bridge object
        bridge_script = QWebEngineScript()
        bridge_script.setName("pyqt_bridge")
        bridge_script.setSourceCode("""
            var pyqt = {
                linkClicked: function(url) {
                    new QWebChannel(qt.webChannelTransport, function(channel) {
                        channel.objects.pyqt.linkClicked(url);
                    });
                }
            };
        """)
        bridge_script.setWorldId(QWebEngineScript.ScriptWorldId.ApplicationWorld)
        bridge_script.setInjectionPoint(QWebEngineScript.InjectionPoint.DocumentCreation)
        bridge_script.setRunsOnSubFrames(False)
        self.page.scripts().insert(bridge_script)

        # Convert the file path to a QUrl object
        base_url = QUrl.fromLocalFile(os.path.dirname(mathjax_path) + '/')
        self.web_view.setHtml(full_html, baseUrl=base_url)

        # Set up channel to handle link clicks
        self.channel = QWebChannel()
        self.page.setWebChannel(self.channel)
        self.channel.registerObject('pyqt', self)

    @pyqtSlot(str)
    def linkClicked(self, url):
        print(f"Link clicked: {url}")
        # Handle the link click as needed, e.g., open in external browser
        QDesktopServices.openUrl(QUrl(url))

    def markdown_to_html(self, markdown_content):
        md = markdown.Markdown(extensions=[
            TableExtension(),
            FencedCodeExtension(),
            CodeHiliteExtension(guess_lang=False),
            MathExtension(enable_dollar_delimiter=True)
        ])
        return md.convert(markdown_content)
    
class AboutDialog(QWidget):
    logging.debug("Showing 'About' dialog")
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Program Information")
        self.setWindowIcon(QIcon(os.path.join(get_source_path(), 'info.png')))
        self.setFixedSize(400, 300)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Dialog)

        # Set white background
        self.setStyleSheet("background-color: white;")

        layout = QVBoxLayout(self)

        # Add logo
        logo_pixmap = QPixmap(os.path.join(get_source_path(), 'icon.png'))
        max_width = 100  # Set the desired maximum width
        max_height = 100  # Set the desired maximum height
        logo_pixmap = logo_pixmap.scaled(max_width, max_height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        logo_label = QLabel()
        logo_label.setPixmap(logo_pixmap)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(logo_label)

        font = QFont()
        font.setPointSize(12)

        program_name = QLabel(SPLASH_INFO['program_name'])
        program_name.setStyleSheet("font-weight: bold; color: #333333;")
        program_name.setFont(font)
        program_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(program_name)

        version = QLabel(SPLASH_INFO['version'])
        version.setStyleSheet("color: #666666;")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version)

        description = QLabel(SPLASH_INFO['description'])
        description.setStyleSheet("color: #666666;")
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(description)

        copyright = QLabel(SPLASH_INFO['copyright'])
        copyright.setStyleSheet("color: #999999;")
        copyright.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignBottom)
        layout.addWidget(copyright)

        self.setLayout(layout)

    def mousePressEvent(self, event):
        self.close()

class PlotWindowDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plot Window")
        # self.setWindowIcon(QIcon(os.path.join(get_source_path(), 'plot.png')))
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.canvas = None
        self.toolbar = None

    def create_canvas(self):
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure) # Type: FigureCanvas
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.setMinimumSize(400, 200)       
        self.layout.addWidget(self.canvas)
            
    def set_toolbar(self):
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.toolbar)

    def closeEvent(self, event):
        if self.toolbar:
            self.toolbar.deleteLater()
        if self.canvas:
            self.canvas.deleteLater()
        event.accept()

class InvertChannelPolarityDialog(QDialog):
    def __init__(self, emg_data : 'EMGData', parent=None):
        super().__init__(parent)
        self.setModal(True)
        self.setWindowTitle("Invert Channel Polarity")
        
        self.emg_data = emg_data
        self.channel_names = emg_data.channel_names

        self.selected_channels = []
        
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Add checkbox header
        header_layout = QVBoxLayout()
        header_layout.addWidget(QLabel(f"Invert selected channel polarities for\n'{self.emg_data.formatted_name}'"))
        header_layout.addWidget(QLabel("\nSelect channels to invert:"))
        layout.addLayout(header_layout)

        # Add checkboxes for each channel in the dataset
        self.checkboxes = []
        for name in self.channel_names:
            checkbox = QCheckBox(name, self)
            self.checkboxes.append(checkbox)
            layout.addWidget(checkbox)
        
        # Add button box (OK and Cancel buttons)
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        layout.addWidget(button_box)

        # Connect signals
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
        # Final layout setup
        self.setLayout(layout)

    def get_selected_channel_indexes(self):
        # Return the indexes of the channels where checkboxes are checked
        return [i for i, checkbox in enumerate(self.checkboxes) if checkbox.isChecked()]

class SelectChannelsDialog(QDialog):
    def __init__(self, emg_data : 'EMGData', parent=None):
        super().__init__(parent)
        self.setModal(True)
        self.setWindowTitle("Select Channels")
        
        self.emg_data = emg_data
        self.channel_names = emg_data.channel_names

        self.selected_channels = []
        
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Add checkbox header
        header_layout = QVBoxLayout()
        header_layout.addWidget(QLabel(f"Select channels for\n'{self.emg_data.formatted_name}'"))
        header_layout.addWidget(QLabel("\nSelect channels:"))
        layout.addLayout(header_layout)

        # Add checkboxes for each channel in the dataset
        self.checkboxes = []
        for name in self.channel_names:
            checkbox = QCheckBox(name, self)

            # Check the checkbox if the channel is already selected
            if self.emg_data.channel_names.index(name) not in self.emg_data.excluded_channels:
                checkbox.setChecked(True)

            self.checkboxes.append(checkbox)
            layout.addWidget(checkbox)
        
        # Add button box (OK and Cancel buttons)
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        layout.addWidget(button_box)

        # Connect signals
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        
        # Final layout setup
        self.setLayout(layout)

    def get_selected_channel_indexes(self):
        # Return the indexes of the channels where checkboxes are checked
        return [i for i, checkbox in enumerate(self.checkboxes) if checkbox.isChecked()]
        