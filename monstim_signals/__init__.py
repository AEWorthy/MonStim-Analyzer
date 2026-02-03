# monstim_signals/__init__.py

# Version
from .version import DATA_VERSION

# Package version
__package_version__ = "0.5.1"

# Metadata
__title__ = "monstim_signals"
__description__ = "Main module for MonStim analysis tools"
__version__ = DATA_VERSION
__author__ = "Andrew Worthy"
__email__ = "aeworth@emory.edu"

# Import functions
from .domain import Dataset, Experiment, Session

# Define __all__ for module
__all__ = ["Session", "Dataset", "Experiment"]
