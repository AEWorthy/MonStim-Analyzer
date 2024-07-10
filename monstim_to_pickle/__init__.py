# monstim_to_pickle/__init__.py

__version__ = '1.2.0'

# Metadata
__title__ = 'monstim_to_pickle'
__description__ = 'Convert MonStim CSV files to Pickle files.'
__author__ = 'Andrew Worthy'
__email__ = 'aeworth@emory.edu'

# Import functions
from .csv_to_pickle import pickle_data, DataProcessingThread

# Define __all__ for module
__all__ = ['pickle_data', 'DataProcessingThread']