# Utility functions/scripts for the project
import os
import sys
from typing import List
import yaml

DATA_PATH = 'files_to_process'
OUTPUT_PATH = 'data'
SAVED_DATASETS_PATH = 'datasets'

def to_camel_case(text : str):
    words = text.split()
    camel_case_text = words[0].lower() + ''.join(word.capitalize() for word in words[1:])
    return camel_case_text

def format_report(report : List[str]):
    formatted_report = ''
    for line in report:
        if line == report[-1]:
            formatted_report += line
        else:
            formatted_report += line + '\n'   
    return formatted_report

def get_config_path():
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        base_path = os.path.dirname(sys.executable)
    else:
        # Running in a normal Python environment
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, 'config.yml')

def get_output_path():
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        base_path = os.path.dirname(sys.executable)
    else:
        # Running in a normal Python environment
        base_path = os.path.dirname(os.path.abspath(__file__))

    output_path = os.path.join(base_path, OUTPUT_PATH)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    return output_path

def get_data_path():
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        base_path = os.path.dirname(sys.executable)
        data_path = base_path
    else:
        # Running in a normal Python environment
        base_path = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_path, DATA_PATH)

        if not os.path.exists(data_path):
            os.makedirs(data_path)

    return data_path

def load_config(config_file=None):
        """
        Loads the config.yaml file into a YAML object that can be used to reference hard-coded configurable constants.

        Args:
            config_file (str): location of the 'config.yaml' file.
        """
        if config_file is None:
            config_file = get_config_path()
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
