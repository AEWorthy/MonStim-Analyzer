# monstim_gui/__init__.py
# Never import packages or modules into this file, or the setup.py will break.
from .version import VERSION

__version__ = VERSION

# Metadata
__title__ = "monstim_gui"
__description__ = "Main module for MonStim GUI"
__author__ = "Andrew Worthy"
__email__ = "aeworth@emory.edu"


# Lazy imports to avoid PyQt6 dependency during setup
def __getattr__(name):
    if name == "SplashScreen":
        from .core.splash import SplashScreen

        return SplashScreen
    elif name == "MonstimGUI":
        from .gui_main import MonstimGUI

        return MonstimGUI
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Define __all__ for module
__all__ = ["MonstimGUI", "SplashScreen"]
