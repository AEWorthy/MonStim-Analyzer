# MonStim_CSV_Analysis - main.py

import csv_to_pickle
import EMG_Utils

DATA_PATH = 'files_to_analyze'
OUTPUT_PATH = 'output'

ex_sciatic_stim_session = 'output/240404-4_data.pickle'
ex_tibial_stim_session = 'output/240404-11_data.pickle'
ex_cper_stim_session = 'output/240404-10_data.pickle'

sciatic_dataset = ['output/240404-1_data.pickle','output/240404-2_data.pickle','output/240404-3_data.pickle','output/240404-4_data.pickle','output/240404-5_data.pickle']
tibial_dataset = ['output/240404-11_data.pickle','output/240404-12_data.pickle','output/240404-13_data.pickle','output/240404-14_data.pickle','output/240404-15_data.pickle']
cper_dataset = ['output/240404-6_data.pickle','output/240404-8_data.pickle','output/240404-9_data.pickle','output/240404-10_data.pickle']

## Process CSV files in "\files_to_analyze" into pickle files formatted for downstream analysis.
# csv_to_pickle.pickle_dataset(DATA_PATH, OUTPUT_PATH)

## Analysis for a single session using class-based method
session = EMG_Utils.EMGSession(ex_sciatic_stim_session)
session.session_parameters()
session.plot_emg(channel_names=["LG", "TA"])
session.plot_emg_rectified(channel_names=["LG", "TA"], m_flags=True, h_flags=True)
# session.plot_emg_suspectedH(channel_names=["LG", "TA"], h_threshold=0.05)
# session.plot_reflex_curves(channel_names=["LG", "TA"])

## Analysis for multiple sessions.
sessions = []
for session in sciatic_dataset:
    sessions.append(EMG_Utils.EMGSession(session))

dataset = EMG_Utils.EMGDataset(sessions)
dataset.plot_reflex_curves(channel_names=["LG", "TA"])