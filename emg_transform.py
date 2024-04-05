# emg_transform.py

import numpy as np

def rectify_emg(emg_array):
    """
    Rectify EMG data by taking the absolute value.
    """
    return np.abs(emg_array)

def calculate_average_amplitude(emg_data, start_ms, end_ms, scan_rate):
    """
    Calculate the average rectified EMG amplitude between start_ms and end_ms.
    """
    start_index = int(start_ms * scan_rate / 1000)
    end_index = int(end_ms * scan_rate / 1000)
    emg_window = emg_data[start_index:end_index]
    rectified_emg_window = rectify_emg(emg_window)
    return np.mean(rectified_emg_window)