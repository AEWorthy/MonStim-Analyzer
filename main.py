# MonStim_CSV_Analysis - main.py

import csv_to_pickle
import single_session_analysis
import multi_session_analysis

DATA_PATH = 'files_to_analyze'
OUTPUT_PATH = 'output'


## Process CSV files in "\files_to_analyze" into pickle files formatted for downstream analysis.
#csv_to_pickle.pickle_dataset(DATA_PATH, OUTPUT_PATH)


## Analysis for a single session.

# Files for single-session analysis
pickled_test_data = 'output/240404-4_data.pickle' #'output/040224rec1_data.pickle'

single_session_analysis.session_parameters(pickled_test_data)
single_session_analysis.plot_EMG(pickled_test_data)
single_session_analysis.plot_emg_rectified(pickled_test_data)
single_session_analysis.plot_EMG_suspectedH(pickled_test_data, h_threshold=0.05)

## Analysis for multiple sessions.