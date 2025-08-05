# Monstim_Analysis/__init__.py
from monstim_gui.version import VERSION

# Never import other packages or modules into this file, or the setup.py will break.

__version__ = VERSION
# Version of the monstim_analysis package is defined by the version.py file in the monstim_gui package.
# This version is used to track the GUI application version.
# monstim_signals has its own versioning system for data compatibility.

# Metadata
__title__ = "MonStim Analyzer"
__description__ = '"MonStimAnalyzer: EMG data analysis and plotting tools"'
__author__ = "Andrew Worthy"
__email__ = "aeworth@emory.edu"
