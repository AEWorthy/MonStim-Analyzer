# MonStim_CSV_Analysis - main.py

from monstim_to_pickle import pickle_data
import utils
import Analyze_EMG

DATA_PATH = 'files_to_analyze'
OUTPUT_PATH = 'output'

#Process CSVs into Pickle files: 'files_to_analyze' --> 'output'
pickle_data(DATA_PATH, OUTPUT_PATH)

# Create dictionaries of Pickle datasets and single sessions that are in the 'output' directory.
dataset_dict, datasets = utils.unpackPickleOutput(OUTPUT_PATH)
for idx, dataset in enumerate(datasets):
    print(f'dataset index {idx}: {dataset}')

# Define dataset of interest for downstream analysis.
dataset_idx = 10
dataset_oi = utils.dataset_oi(dataset_dict, datasets, dataset_idx)
dataset_oi.dataset_parameters()


# Define session of interest for downstream analysis.
session_idx = 0
session_oi = utils.session_oi(dataset_dict, datasets, dataset_idx, session_idx)
session_oi.session_parameters()

# Visualize single EMG session raw and filtered
channel_names = ["LG", "TA"]
# channel_names = ["LG"]

# session_oi.plot_emg(channel_names=channel_names, m_flags=True, h_flags=True, data_type='raw')
session_oi.plot_emg(channel_names=channel_names, m_flags=True, h_flags=True, data_type='filtered')
# session_oi.plot_emg(channel_names=channel_names, m_flags=True, h_flags=True, data_type='rectified')
# session_oi.plot_emg(channel_names=channel_names, m_flags=True, h_flags=True, data_type='rectified_filtered')


# Inspect reflex curves and suspected H-reflex trials
# session_oi.plot_emg_suspectedH(channel_names=channel_names, h_threshold=.7)
session_oi.plot_reflex_curves(channel_names=channel_names, method='rms')

# Analyze EMG dataset
dataset_oi.plot_reflex_curves(channel_names=channel_names, method='rms')