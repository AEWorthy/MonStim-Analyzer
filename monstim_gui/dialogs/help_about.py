from .base import *


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
        mathjax_path = os.path.abspath(
            os.path.join(get_source_path(), "mathjax", "es5", "tex-mml-chtml.js")
        )

        # Full HTML content
        full_html = f"""
        <!DOCTYPE html>
            <html>
            <head>
                <script id=\"MathJax-script\" async src=\"file:///{mathjax_path}\"></script>
                <script>
                    MathJax = {{
                        tex: {{
                            inlineMath: [['$', '$']]
                        }}
                    }};
                </script>
                <style>
                    body {{
                        font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, sans-serif;
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
        bridge_script.setSourceCode(
            """
            var pyqt = {
                linkClicked: function(url) {
                    new QWebChannel(qt.webChannelTransport, function(channel) {
                        channel.objects.pyqt.linkClicked(url);
                    });
                }
            };
        """
        )
        bridge_script.setWorldId(QWebEngineScript.ScriptWorldId.ApplicationWorld)
        bridge_script.setInjectionPoint(
            QWebEngineScript.InjectionPoint.DocumentCreation
        )
        bridge_script.setRunsOnSubFrames(False)
        self.page.scripts().insert(bridge_script)

        # Convert the file path to a QUrl object
        base_url = QUrl.fromLocalFile(os.path.dirname(mathjax_path) + "/")
        self.web_view.setHtml(full_html, baseUrl=base_url)

        # Set up channel to handle link clicks
        self.channel = QWebChannel()
        self.page.setWebChannel(self.channel)
        self.channel.registerObject("pyqt", self)

    @pyqtSlot(str)
    def linkClicked(self, url):
        QDesktopServices.openUrl(QUrl(url))

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
        copyright.setAlignment(
            Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignBottom
        )
        layout.addWidget(copyright)

        self.setLayout(layout)

    def mousePressEvent(self, event):
        self.close()
