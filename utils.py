"""
Misc. helper functions.
"""

import yaml
import os

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def unpackPickleOutput (output_path):
    dataset_pickles_dict = {} #k=datasets, v=pickle_filepath(s)

    for dataset in os.listdir(output_path):
        if os.path.isdir(os.path.join(output_path, dataset)):
            pickles = os.listdir(os.path.join(output_path, dataset))
            pickle_paths = [os.path.join(output_path, dataset, pickle).replace('\\', '/') for pickle in pickles]
            dataset_pickles_dict[dataset] = pickle_paths
        else:
            session_name = dataset.split('-')[0]
            dataset_pickles_dict[session_name] = os.path.join(output_path, dataset).replace('\\', '/')
    return dataset_pickles_dict