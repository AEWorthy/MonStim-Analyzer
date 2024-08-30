import logging
import os

from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtWidgets import QSplashScreen, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt

from monstim_utils import get_source_path

SPLASH_INFO = {
    'program_name': "MonStim EMG Analyzer",
    'version': "Version 0.2.1 (alpha)",
    'description': "Software for analyzing EMG data\nfrom LabView MonStim experiments.\n\n\nClick to dismiss...",
    'copyright': "© 2024 Andrew Worthy"
}

class SplashScreen(QSplashScreen):
    def __init__(self):
        logging.debug("Creating splash screen.")
        pixmap = QPixmap(400, 300)
        pixmap.fill(Qt.GlobalColor.white)
        
        super().__init__(pixmap, Qt.WindowType.WindowStaysOnTopHint)
        
        # Add program information
        layout = self.layout()
        if layout is None:
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