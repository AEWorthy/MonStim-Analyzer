"""Splash screen displayed during application startup."""

import logging
import os
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPalette, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QSplashScreen, QVBoxLayout

if __name__ == "__main__":
    import sys

    top_level_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if top_level_dir not in sys.path:
        sys.path.insert(0, top_level_dir)

from monstim_gui.version import VERSION

SPLASH_INFO = {
    "program_name": "MonStim EMG Analyzer",
    "version": f"v{VERSION} (beta)",
    "description": "Software for analyzing EMG data\nfrom LabView MonStim experiments.",
    "copyright": "© 2024–2026 Andrew Worthy",  # noqa: RUF001
}

logger = logging.getLogger(__name__)


def get_splash_asset_path(asset_name: str) -> str:
    """Resolve a bundled splash asset without importing the analysis package."""
    resource_root = Path(sys._MEIPASS) if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[2]
    return str(resource_root / "assets" / asset_name)


class SplashScreen(QSplashScreen):
    def __init__(self):
        logger.debug("Creating splash screen.")
        pixmap = QPixmap(400, 400)
        pixmap.fill(QApplication.palette().color(QPalette.ColorRole.Window))

        super().__init__(pixmap, Qt.WindowType.WindowStaysOnTopHint)
        self.setObjectName("splashScreen")

        # Add program information
        layout = self.layout()
        if layout is None:
            layout = QVBoxLayout(self)

        # Add logo
        logo_pixmap = QPixmap(get_splash_asset_path("logo.png"))
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
        program_name.setObjectName("applicationInfoTitle")
        program_name.setFont(font)
        program_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(program_name)

        version = QLabel(SPLASH_INFO["version"])
        version.setObjectName("applicationInfoSecondary")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version)

        description = QLabel(SPLASH_INFO["description"])
        description.setObjectName("applicationInfoSecondary")
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description.setWordWrap(True)
        layout.addWidget(description)

        copyright = QLabel(SPLASH_INFO["copyright"])
        copyright.setObjectName("applicationInfoMuted")
        copyright.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignBottom)
        layout.addWidget(copyright)


if __name__ == "__main__":
    import sys

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    splash = SplashScreen()
    splash.show()
    sys.exit(app.exec())
