import os

import markdown
from markdown.extensions.codehilite import CodeHiliteExtension
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.tables import TableExtension
from mdx_math import MathExtension
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QIcon, QPixmap
from PySide6.QtWidgets import QLabel, QTextBrowser, QVBoxLayout, QWidget

# Optional WebEngine for LaTeX via MathJax (graceful fallback).
# Note: Only used if the QtWebEngine module is present at runtime.
WEB_ENGINE_AVAILABLE = False
try:
    import importlib

    _we = importlib.import_module("PySide6.QtWebEngineWidgets")
    QWebEngineView = getattr(_we, "QWebEngineView")
    WEB_ENGINE_AVAILABLE = QWebEngineView is not None
except Exception:
    QWebEngineView = None

from monstim_gui.core.splash import SPLASH_INFO
from monstim_signals.core import get_source_path

# WebEngine-based rendering was removed for simplicity since
# the application does not require dynamic JS bridging here.


class HelpWindow(QWidget):
    def __init__(self, markdown_content, title=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title if title else "Help Window")
        self.setWindowIcon(QIcon(os.path.join(get_source_path(), "info.png")))
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
        self.setWindowIcon(QIcon(os.path.join(get_source_path(), "info.png")))
        self.resize(600, 400)

        layout = QVBoxLayout(self)

        if WEB_ENGINE_AVAILABLE:
            # Minimal WebEngine view with local MathJax for LaTeX rendering
            self.web_view = QWebEngineView()
            layout.addWidget(self.web_view)

            html_content = self.markdown_to_html(markdown_content)
            mathjax_path = os.path.abspath(os.path.join(get_source_path(), "mathjax", "es5", "tex-mml-chtml.js"))
            full_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset=\"utf-8\">
                <script id=\"MathJax-script\" async src=\"file:///{mathjax_path}\"></script>
                <script>
                    window.MathJax = {{ tex: {{ inlineMath: [['$', '$'], ['\\(', '\\)']] }} }};
                </script>
                <style>
                    body {{ font-family: -apple-system, Segoe UI, Roboto, Arial; margin: 0; padding: 10px; }}
                </style>
            </head>
            <body>
                {html_content}
            </body>
            </html>
            """
            # Load with base URL so MathJax can resolve local resources
            from PySide6.QtCore import QUrl

            base_dir = os.path.dirname(mathjax_path) + "/"
            self.web_view.setHtml(full_html, baseUrl=QUrl.fromLocalFile(base_dir))
        else:
            # Simple QTextBrowser fallback (no LaTeX rendering, plain HTML)
            self.text_browser = QTextBrowser()
            self.text_browser.setOpenExternalLinks(True)
            self.text_browser.setHtml(self.markdown_to_html(markdown_content))
            layout.addWidget(self.text_browser)

    def process_content(self, markdown_content):
        # Deprecated: no-op after removal of WebEngine.
        return

    def markdown_to_html(self, markdown_content):
        md = markdown.Markdown(
            extensions=[
                TableExtension(),
                FencedCodeExtension(),
                CodeHiliteExtension(guess_lang=False),
                MathExtension(enable_dollar_delimiter=True),
            ]
        )
        return md.convert(markdown_content)


class AboutDialog(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Program Information")
        self.setWindowIcon(QIcon(os.path.join(get_source_path(), "icon.png")))
        self.setFixedSize(400, 400)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Dialog)

        # Set white background
        self.setStyleSheet("background-color: white;")

        layout = QVBoxLayout(self)

        # Add logo
        logo_pixmap = QPixmap(os.path.join(get_source_path(), "logo.png"))
        max_width = 200
        max_height = 200
        logo_pixmap = logo_pixmap.scaled(
            max_width,
            max_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        logo_label = QLabel()
        logo_label.setPixmap(logo_pixmap)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(logo_label)

        font = QFont()
        font.setPointSize(12)

        program_name = QLabel(SPLASH_INFO["program_name"])
        program_name.setStyleSheet("font-weight: bold; color: #333333;")
        program_name.setFont(font)
        program_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(program_name)

        version = QLabel(SPLASH_INFO["version"])
        version.setStyleSheet("color: #666666;")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version)

        description = QLabel(SPLASH_INFO["description"])
        description.setStyleSheet("color: #666666;")
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(description)

        copyright = QLabel(SPLASH_INFO["copyright"])
        copyright.setStyleSheet("color: #999999;")
        copyright.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignBottom)
        layout.addWidget(copyright)

        self.setLayout(layout)

    def mousePressEvent(self, event):
        self.close()
