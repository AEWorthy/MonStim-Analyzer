# Utility functions/scripts for the project
from typing import List

DATA_PATH = 'files_to_analyze'
OUTPUT_PATH = 'pickles'
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
