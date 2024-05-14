"""
Helper functions for loading config files.
"""

import yaml

def load_config(config_file):
    """
    Loads the config.yaml file into a YAML object that can be used to reference hard-coded configurable constants.

    Args:
        config_file (str): location of the 'config.yaml' file.
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config