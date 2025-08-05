from .plot_options import (AverageReflexCurvesOptions, AverageSessionReflexOptions,
                           EMGOptions, MaxHReflexOptions, MMaxOptions,
                           SessionReflexCurvesOptions, SingleEMGRecordingOptions)

PLOT_NAME_DICT = {
    "EMG": "emg",
    "Suspected H-reflexes": "suspectedH",
    "Reflex:Stimulus Curves": "reflexCurves",
    "M-max": "mmax",
    "Max H-reflex": "maxH",
    "Average Reflex:Stimulus Curves": "averageReflexCurves",
    "Single EMG Recordings": "singleEMG",
    "Reflex Averages": "reflexAverages",
}

PLOT_OPTIONS_DICT = {
    "session": {
        "EMG": EMGOptions,
        "Single EMG Recordings": SingleEMGRecordingOptions,
        "Reflex Averages": AverageSessionReflexOptions,
        "Reflex:Stimulus Curves": SessionReflexCurvesOptions,
        "Average Reflex:Stimulus Curves": AverageSessionReflexOptions,
        "M-max": MMaxOptions,
    },
    "dataset": {
        "Average Reflex:Stimulus Curves": AverageReflexCurvesOptions,
        "Max H-reflex": MaxHReflexOptions,
        "M-max": MMaxOptions,
    },
    "experiment": {
        "Average Reflex:Stimulus Curves": AverageReflexCurvesOptions,
        "Max H-reflex": MaxHReflexOptions,
        "M-max": MMaxOptions,
    },
}
