# MonStim_CSV_Analysis - example1.py

import csv_to_pickle
import EMG_Utils

DATA_PATH = 'files_to_analyze'
OUTPUT_PATH = 'output'

ex_sciatic_stim_session_m1 = 'output/240404-4_data.pickle'
ex_tibial_stim_session_m1 = 'output/240404-11_data.pickle'
ex_cper_stim_session_m1 = 'output/240404-10_data.pickle'

sciatic_dataset_m1 = ['output/240404-1_data.pickle','output/240404-2_data.pickle','output/240404-3_data.pickle','output/240404-4_data.pickle','output/240404-5_data.pickle']
tibial_dataset_m1 = ['output/240404-11_data.pickle','output/240404-12_data.pickle','output/240404-13_data.pickle','output/240404-14_data.pickle','output/240404-15_data.pickle']
cper_dataset_m1 = ['output/240404-6_data.pickle','output/240404-8_data.pickle','output/240404-9_data.pickle','output/240404-10_data.pickle']


# ex_tibial_stim_session_m2 = 'output/040224rec6_data.pickle'
# ex_cper_stim_session_m1 = 'output/040224rec10_data.pickle'

# tibial_dataset_m2 = ['output/040224rec5_data.pickle','output/040224rec6_data.pickle','output/040224rec7_data.pickle','output/040224rec14_data.pickle','output/040224rec15_data.pickle','output/040224rec16a_data.pickle','output/040224rec16b_data.pickle']
# cper_dataset_m2 = ['output/040224rec10_data.pickle','output/040224rec11_data.pickle','output/040224rec12_data.pickle','output/040224rec13_data.pickle','output/040224rec8_data.pickle','output/040224rec9_data.pickle']

## Process CSV files in "\files_to_analyze" into pickle files formatted for downstream analysis.
# csv_to_pickle.pickle_dataset(DATA_PATH, OUTPUT_PATH)

## Analysis for a single session using class-based method
session = EMG_Utils.EMGSession(ex_cper_stim_session_m1)
session.session_parameters()
session.plot_emg(channel_names=["LG"])
session.plot_emg_rectified(channel_names=["LG"], m_flags=True, h_flags=True)
# session.plot_emg_suspectedH(channel_names=["LG", "TA"], h_threshold=0.05)
session.plot_reflex_curves(channel_names=["LG"])

## Analysis for multiple sessions.
sessions = []
for session in tibial_dataset_m2:
    sessions.append(EMG_Utils.EMGSession(session))

dataset = EMG_Utils.EMGDataset(sessions)
dataset.plot_reflex_curves(channel_names=["LG"])