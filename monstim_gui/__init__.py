# monstim_gui/__init__.py
# Never import packages or modules into this file, or the setup.py will break.

__version__ = '0.5.0'

# Metadata
__title__ = 'monstim_gui'
__description__ = 'Main module for MonStim GUI'
__author__ = 'Andrew Worthy'
__email__ = 'aeworth@emory.edu'

# Import functions
from .gui_main import EMGAnalysisGUI
from .splash import SplashScreen

# Define __all__ for module
__all__ = ['EMGAnalysisGUI', 'SplashScreen']