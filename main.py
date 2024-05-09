# MonStim_CSV_Analysis - main.py

### Things to fix:
    # Detect time recorded before stimulation.
    # Fix start of plots/data so that 0 is actually the moment of stimulation.
    # Make the time before stimulation negative time.

from csv_to_pickle import pickle_data
import EMG_Utils

DATA_PATH = 'files_to_analyze'
OUTPUT_PATH = 'output'

# Process CSV files in "\files_to_analyze" into pickle files formatted for downstream analysis.
pickle_data(DATA_PATH, OUTPUT_PATH)

pickled_test_session = 'output/240304 hcurve1_data.pickle'
pickled_test_dataset = ['output/240304 hcurve1_data.pickle','output/240304 hcurve2_data.pickle','output/240304 h3_data.pickle','output/240304 h4 1k_data.pickle']
# pickled_test_dataset = ['output/240304 mcurve2 ii_data.pickle','output/240304 mcurve3 ii_data.pickle','output/240304 mcurve4 ii_data.pickle']


## Process CSV files in "\files_to_analyze" into pickle files formatted for downstream analysis.
# csv_to_pickle.pickle_dataset(DATA_PATH, OUTPUT_PATH)

## Analysis for a single session using class-based method
session = EMG_Utils.EMGSession(pickled_test_session)
session.session_parameters()
session.plot_emg(channel_names=["LG"], m_flags=True, h_flags=True)
session.plot_emg_rectified(channel_names=["LG"], m_flags=True, h_flags=True)
# session.plot_emg_suspectedH(channel_names=["LG"], h_threshold=0.05)
session.plot_reflex_curves(channel_names=["LG"])

## Analysis for multiple sessions.
sessions = []
for session in pickled_test_dataset:
    sessions.append(EMG_Utils.EMGSession(session))

dataset = EMG_Utils.EMGDataset(sessions)
dataset.plot_reflex_curves(channel_names=["LG"])